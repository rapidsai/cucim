/*
 * Apache License, Version 2.0
 * Copyright 2021 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "image_cache_shared_memory.h"

#include "cucim/cuimage.h"
#include "cucim/memory/memory_manager.h"

#include <boost/make_shared.hpp>
#include <fmt/format.h>


template <>
struct boost::hash<cucim::cache::MapKey>
{
    typedef cucim::cache::MapKey argument_type;
    typedef size_t result_type;
    result_type operator()(argument_type::type& s) const
    {
        std::size_t h1 = std::hash<uint64_t>{}(s->file_hash);
        std::size_t h2 = std::hash<uint64_t>{}(s->location_hash);
        return h1 ^ (h2 << 1); // or use boost::hash_combine
    }

    result_type operator()(const argument_type::type& s) const
    {
        std::size_t h1 = std::hash<uint64_t>{}(s->file_hash);
        std::size_t h2 = std::hash<uint64_t>{}(s->location_hash);
        return h1 ^ (h2 << 1); // or use boost::hash_combine
    }

    result_type operator()(const cucim::cache::ImageCacheKey& s) const
    {
        std::size_t h1 = std::hash<uint64_t>{}(s.file_hash);
        std::size_t h2 = std::hash<uint64_t>{}(s.location_hash);
        return h1 ^ (h2 << 1); // or use boost::hash_combine
    }

    result_type operator()(const std::shared_ptr<cucim::cache::ImageCacheKey>& s) const
    {
        std::size_t h1 = std::hash<uint64_t>{}(s->file_hash);
        std::size_t h2 = std::hash<uint64_t>{}(s->location_hash);
        return h1 ^ (h2 << 1); // or use boost::hash_combine
    }
};

template <>
struct std::equal_to<cucim::cache::MapKey>
{
    typedef cucim::cache::MapKey argument_type;

    bool operator()(const argument_type::type& lhs, const argument_type::type& rhs) const
    {
        return lhs->location_hash == rhs->location_hash && lhs->file_hash == rhs->file_hash;
    }

    bool operator()(const argument_type::type& lhs, const cucim::cache::ImageCacheKey& rhs) const
    {
        return lhs->location_hash == rhs.location_hash && lhs->file_hash == rhs.file_hash;
    }

    bool operator()(const cucim::cache::ImageCacheKey& lhs, const std::shared_ptr<cucim::cache::ImageCacheKey>& rhs) const
    {
        return lhs.location_hash == rhs->location_hash && lhs.file_hash == rhs->file_hash;
    }
};


namespace cucim::cache
{


template <class P>
struct null_deleter
{
private:
    P p_;

public:
    null_deleter(const P& p) : p_(p)
    {
    }
    void operator()(void const*)
    {
        p_.reset();
    }

    P const& get() const
    {
        return p_;
    }
};


template <class T>
shared_mem_deleter<T>::shared_mem_deleter(std::unique_ptr<boost::interprocess::managed_shared_memory>& segment)
    : seg_(segment)
{
}

template <class T>
void shared_mem_deleter<T>::operator()(T* p)
{
    if (seg_)
    {
        seg_->destroy_ptr(p);
    }
}

// Apparently, cache requires about 13MiB + (400 bytes per one capacity) for the data structure (hashmap+vector).
// so allocate additional bytes which are 100MiB (exta) + 512(rough estimation per item) * (capacity) bytes.
// Not having enough segment(shared) memory can cause a memory allocation failure and the process can get stuck.
// https://stackoverflow.com/questions/4166642/how-much-memory-should-managed-shared-memory-allocate-boost
static size_t calc_segment_size(const ImageCacheConfig& config)
{
    return kOneMiB * config.memory_capacity + (config.extra_shared_memory_size * kOneMiB + 512 * config.capacity);
}

template <class T>
using deleter_type = boost::interprocess::shared_ptr<
    T,
    boost::interprocess::allocator<
        void,
        boost::interprocess::segment_manager<char,
                                             boost::interprocess::rbtree_best_fit<boost::interprocess::mutex_family>,
                                             boost::interprocess::iset_index>>,
    boost::interprocess::deleter<
        T,
        boost::interprocess::segment_manager<char,
                                             boost::interprocess::rbtree_best_fit<boost::interprocess::mutex_family>,
                                             boost::interprocess::iset_index>>>;

struct ImageCacheItemDetail
{
    ImageCacheItemDetail(deleter_type<ImageCacheKey>& key, deleter_type<SharedMemoryImageCacheValue>& value)
        : key(key), value(value)
    {
    }
    deleter_type<ImageCacheKey> key;
    deleter_type<SharedMemoryImageCacheValue> value;
};

SharedMemoryImageCacheValue::SharedMemoryImageCacheValue(void* data,
                                                         uint64_t size,
                                                         void* user_obj,
                                                         const cucim::io::DeviceType device_type)
    : ImageCacheValue(data, size, user_obj, device_type){};

SharedMemoryImageCacheValue::~SharedMemoryImageCacheValue()
{
    if (data)
    {
        if (user_obj)
        {
            static_cast<boost::interprocess::managed_shared_memory*>(user_obj)->deallocate(data);
            data = nullptr;
        }
    }
};

SharedMemoryImageCache::SharedMemoryImageCache(const ImageCacheConfig& config, const cucim::io::DeviceType device_type)
    : ImageCache(config, CacheType::kSharedMemory, device_type),
      segment_(create_segment(config)),
      //   mutex_array_(nullptr, shared_mem_deleter<boost::interprocess::interprocess_mutex>(segment_)),
      size_nbytes_(nullptr, shared_mem_deleter<std::atomic<uint64_t>>(segment_)),
      capacity_nbytes_(nullptr, shared_mem_deleter<uint64_t>(segment_)),
      capacity_(nullptr, shared_mem_deleter<uint32_t>(segment_)),
      list_capacity_(nullptr, shared_mem_deleter<uint32_t>(segment_)),
      list_padding_(nullptr, shared_mem_deleter<uint32_t>(segment_)),
      mutex_pool_capacity_(nullptr, shared_mem_deleter<uint32_t>(segment_)),
      stat_hit_(nullptr, shared_mem_deleter<std::atomic<uint64_t>>(segment_)),
      stat_miss_(nullptr, shared_mem_deleter<std::atomic<uint64_t>>(segment_)),
      stat_is_recorded_(nullptr, shared_mem_deleter<bool>(segment_)),
      list_head_(nullptr, shared_mem_deleter<std::atomic<uint32_t>>(segment_)),
      list_tail_(nullptr, shared_mem_deleter<std::atomic<uint32_t>>(segment_))
{
    const uint64_t& memory_capacity = config.memory_capacity;
    const uint32_t& capacity = config.capacity;
    const uint32_t& mutex_pool_capacity = config.mutex_pool_capacity;
    const bool& record_stat = config.record_stat;

    if (device_type != cucim::io::DeviceType::kCPU)
    {
        throw std::runtime_error(
            fmt::format("[Error] SharedMemoryImageCache doesn't support other memory type other than CPU memory!\n"));
    }

    try
    {
        // mutex_array_.reset(segment_->find_or_construct_it<boost::interprocess::interprocess_mutex>(
        //     "cucim-mutex")[mutex_pool_capacity]());
        mutex_array_ =
            segment_->construct_it<boost::interprocess::interprocess_mutex>("cucim-mutex")[mutex_pool_capacity]();

        size_nbytes_.reset(segment_->find_or_construct<std::atomic<uint64_t>>("size_nbytes_")(0)); /// size of cache
                                                                                                   /// memory used
        capacity_nbytes_.reset(
            segment_->find_or_construct<uint64_t>("capacity_nbytes_")(kOneMiB * memory_capacity)); /// size of
                                                                                                   /// cache
                                                                                                   /// memory
                                                                                                   /// allocated

        capacity_.reset(segment_->find_or_construct<uint32_t>("capacity_")(capacity)); /// capacity
                                                                                       /// of hashmap
        list_capacity_.reset(
            segment_->find_or_construct<uint32_t>("list_capacity_")(capacity + config.list_padding)); /// capacity
                                                                                                      /// of
                                                                                                      /// list

        list_padding_.reset(segment_->find_or_construct<uint32_t>("list_padding_")(config.list_padding)); /// gap
                                                                                                          /// between
                                                                                                          /// head and
                                                                                                          /// tail

        mutex_pool_capacity_.reset(segment_->find_or_construct<uint32_t>("mutex_pool_capacity_")(mutex_pool_capacity));

        stat_hit_.reset(segment_->find_or_construct<std::atomic<uint64_t>>("stat_hit_")(0)); /// cache hit count
        stat_miss_.reset(segment_->find_or_construct<std::atomic<uint64_t>>("stat_miss_")(0)); /// cache miss mcount
        stat_is_recorded_.reset(segment_->find_or_construct<bool>("stat_is_recorded_")(record_stat)); /// whether if
                                                                                                      /// cache stat is
                                                                                                      /// recorded or
                                                                                                      /// not

        list_head_.reset(segment_->find_or_construct<std::atomic<uint32_t>>("list_head_")(0)); /// head
        list_tail_.reset(segment_->find_or_construct<std::atomic<uint32_t>>("list_tail_")(0)); /// tail

        list_ = boost::interprocess::make_managed_shared_ptr(
            segment_->find_or_construct<QueueType>("cucim-list")(
                *list_capacity_, ValueAllocator(segment_->get_segment_manager())),
            *segment_);

        hashmap_ = boost::interprocess::make_managed_shared_ptr(
            segment_->find_or_construct<ImageCacheType>("cucim-hashmap")(
                calc_hashmap_capacity(capacity), MapKeyHasher(), MakKeyEqual(),
                ImageCacheAllocator(segment_->get_segment_manager())),
            *segment_);
    }
    catch (const boost::interprocess::bad_alloc& e)
    {
        throw std::runtime_error(fmt::format(
            "[Error] Couldn't allocate shared memory (size: {}). Please increase the cache memory capacity.\n",
            memory_capacity));
    }
};

SharedMemoryImageCache::~SharedMemoryImageCache()
{
    {
        // Destroy objects that uses the shared memory object(segment_)
        hashmap_.reset();
        list_.reset();
        segment_->destroy<boost::interprocess::interprocess_mutex>("cucim-mutex");
        mutex_array_ = nullptr;
        // mutex_array_.reset();

        // Destroy the shared memory object
        segment_.reset();
    }

    bool succeed = remove_shmem();
    if (!succeed)
    {
        fmt::print(stderr, "[Warning] Couldn't delete the shared memory object '{}'.",
                   cucim::CuImage::get_config()->shm_name());
    }
}

const char* SharedMemoryImageCache::type_str() const
{
    return "shared_memory";
}

std::shared_ptr<ImageCacheKey> SharedMemoryImageCache::create_key(uint64_t file_hash, uint64_t index)
{
    auto key = boost::interprocess::make_managed_shared_ptr(
        segment_->find_or_construct<ImageCacheKey>(boost::interprocess::anonymous_instance)(file_hash, index), *segment_);

    return std::shared_ptr<ImageCacheKey>(key.get().get(), null_deleter<decltype(key)>(key));
}
std::shared_ptr<ImageCacheValue> SharedMemoryImageCache::create_value(void* data,
                                                                      uint64_t size,
                                                                      const cucim::io::DeviceType device_type)
{
    auto value = boost::interprocess::make_managed_shared_ptr(
        segment_->find_or_construct<SharedMemoryImageCacheValue>(boost::interprocess::anonymous_instance)(
            data, size, &*segment_, device_type),
        *segment_);

    return std::shared_ptr<ImageCacheValue>(value.get().get(), null_deleter<decltype(value)>(value));
}

void* SharedMemoryImageCache::allocate(std::size_t n)
{
    // TODO: handling OOM exception
    void* temp = nullptr;
    try
    {
        // fmt::print(stderr, "## pid: {} memory_size: {}, memory_capacity: {}, free_memory: {}\n", getpid(),
        //            memory_size(), memory_capacity(), free_memory());
        // fmt::print(
        //     stderr, "## pid: {} size_nbytes: {}, capacity_nbytes: {}\n", getpid(), *size_nbytes_, *capacity_nbytes_);
        // fmt::print(stderr, "## pid: {}, {} hit:{} miss:{} total:{} | {}/{}  hash size:{}\n", getpid(),
        //            segment_->get_free_memory(), *stat_hit_, *stat_miss_, *stat_hit_ + *stat_miss_, size(),
        //            *list_capacity_, hashmap_->size());

        temp = segment_->allocate(n);
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error(fmt::format(
            "[Error] Couldn't allocate shared memory (size: {}). Please increase the cache memory capacity.\n", n));
    }

    return temp;
}

void SharedMemoryImageCache::lock(uint64_t index)
{
    // fmt::print(stderr, "# {}: {} {} [{}]-   lock\n",
    // std::chrono::high_resolution_clock::now().time_since_epoch().count(),
    //            getpid(), index, index % *mutex_pool_capacity_);
    mutex_array_[index % *mutex_pool_capacity_].lock();
}

void SharedMemoryImageCache::unlock(uint64_t index)
{
    // fmt::print(stderr, "# {}: {} {} [{}]-   unlock\n",
    //            std::chrono::high_resolution_clock::now().time_since_epoch().count(), getpid(), index,
    //            index % *mutex_pool_capacity_);
    mutex_array_[index % *mutex_pool_capacity_].unlock();
}

void* SharedMemoryImageCache::mutex(uint64_t index)
{
    return &mutex_array_[index % *mutex_pool_capacity_];
}

bool SharedMemoryImageCache::insert(std::shared_ptr<ImageCacheKey>& key, std::shared_ptr<ImageCacheValue>& value)
{
    if (value->size > *capacity_nbytes_ || *capacity_ < 1)
    {
        return false;
    }

    while (is_list_full() || is_memory_full(value->size))
    {
        remove_front();
    }

    auto key_impl = std::get_deleter<null_deleter<deleter_type<ImageCacheKey>>>(key)->get();
    auto value_impl = std::get_deleter<null_deleter<deleter_type<SharedMemoryImageCacheValue>>>(value)->get();
    auto item = boost::interprocess::make_managed_shared_ptr(
        segment_->find_or_construct<ImageCacheItemDetail>(boost::interprocess::anonymous_instance)(key_impl, value_impl),
        *segment_);

    bool succeed = hashmap_->insert(key_impl, item);
    if (succeed)
    {
        push_back(item);
    }
    return succeed;
}

void SharedMemoryImageCache::remove_front()
{
    while (true)
    {
        uint32_t head = (*list_head_).load(std::memory_order_relaxed);
        uint32_t tail = (*list_tail_).load(std::memory_order_relaxed);
        if (head != tail)
        {
            // Remove front by increasing head
            if ((*list_head_)
                    .compare_exchange_weak(
                        head, (head + 1) % (*list_capacity_), std::memory_order_release, std::memory_order_relaxed))
            {
                auto& head_item = (*list_)[head];
                if (head_item) // it is possible that head_item is nullptr
                {
                    (*size_nbytes_).fetch_sub(head_item->value->size, std::memory_order_relaxed);
                    hashmap_->erase(head_item->key);
                    (*list_)[head].reset(); // decrease refcount
                    break;
                }
            }
        }
        else
        {
            break; // already empty
        }
    }
}

uint32_t SharedMemoryImageCache::size() const
{
    uint32_t head = list_head_->load(std::memory_order_relaxed);
    uint32_t tail = list_tail_->load(std::memory_order_relaxed);

    return (tail + *list_capacity_ - head) % *list_capacity_;
}

uint64_t SharedMemoryImageCache::memory_size() const
{
    return size_nbytes_->load(std::memory_order_relaxed);
}

uint32_t SharedMemoryImageCache::capacity() const
{
    return *capacity_;
}

uint64_t SharedMemoryImageCache::memory_capacity() const
{
    // Return segment's size instead of the logical capacity.
    return segment_->get_size();
    // return *capacity_nbytes_;
}

uint64_t SharedMemoryImageCache::free_memory() const
{
    // Return segment's free memory instead of the logical free memory.
    return segment_->get_free_memory();
    // return *capacity_nbytes_ - size_nbytes_->load(std::memory_order_relaxed);
}

void SharedMemoryImageCache::record(bool value)
{
    config_.record_stat = value;

    stat_hit_->store(0, std::memory_order_relaxed);
    stat_miss_->store(0, std::memory_order_relaxed);
    *stat_is_recorded_ = value;
}

bool SharedMemoryImageCache::record() const
{
    return *stat_is_recorded_;
}

uint64_t SharedMemoryImageCache::hit_count() const
{
    return stat_hit_->load(std::memory_order_relaxed);
}
uint64_t SharedMemoryImageCache::miss_count() const
{
    return stat_miss_->load(std::memory_order_relaxed);
}

void SharedMemoryImageCache::reserve(const ImageCacheConfig& config)
{
    uint64_t new_memory_capacity_nbytes = kOneMiB * config.memory_capacity;
    uint32_t new_capacity = config.capacity;

    if ((*capacity_nbytes_) < new_memory_capacity_nbytes)
    {
        (*capacity_nbytes_) = new_memory_capacity_nbytes;
    }

    if ((*capacity_) < new_capacity)
    {
        config_.capacity = config.capacity;
        config_.memory_capacity = config.memory_capacity;

        uint32_t old_list_capacity = (*list_capacity_);

        (*capacity_) = new_capacity;
        (*list_capacity_) = new_capacity + (*list_padding_);

        list_->reserve(*list_capacity_);
        list_->resize(*list_capacity_);
        hashmap_->reserve(new_capacity);

        // Move items in the vector
        uint32_t head = (*list_head_).load(std::memory_order_relaxed);
        uint32_t tail = (*list_tail_).load(std::memory_order_relaxed);
        if (tail < head)
        {
            head = 0;
            uint32_t new_head = old_list_capacity;

            while (head != tail)
            {
                (*list_)[new_head] = (*list_)[head];
                (*list_)[head].reset();

                head = (head + 1) % old_list_capacity;
                new_head = (new_head + 1) % (*list_capacity_);
            }
            // Set new tail
            (*list_tail_).store(new_head, std::memory_order_relaxed);
        }
    }
}

std::shared_ptr<ImageCacheValue> SharedMemoryImageCache::find(const std::shared_ptr<ImageCacheKey>& key)
{
    MapValue::type item;
    auto key_impl = std::get_deleter<null_deleter<deleter_type<ImageCacheKey>>>(key)->get();
    const bool found = hashmap_->find(key_impl, item);
    if (*stat_is_recorded_)
    {
        if (found)
        {
            (*stat_hit_).fetch_add(1, std::memory_order_relaxed);
            return std::shared_ptr<ImageCacheValue>(item->value.get().get(), null_deleter<decltype(item)>(item));
        }
        else
        {
            (*stat_miss_).fetch_add(1, std::memory_order_relaxed);
        }
    }
    else
    {
        if (found)
        {
            return std::shared_ptr<ImageCacheValue>(item->value.get().get(), null_deleter<decltype(item)>(item));
        }
    }
    return std::shared_ptr<ImageCacheValue>();
}

bool SharedMemoryImageCache::is_list_full() const
{
    if (size() >= *capacity_)
    {
        return true;
    }
    return false;
}

bool SharedMemoryImageCache::is_memory_full(uint64_t additional_size) const
{
    if (size_nbytes_->load(std::memory_order_relaxed) + additional_size > *capacity_nbytes_)
    {
        return true;
    }
    else
    {
        return false;
    }
}

void SharedMemoryImageCache::push_back(cache_item_type<ImageCacheItemDetail>& item)
{
    uint32_t tail = (*list_tail_).load(std::memory_order_relaxed);
    while (true)
    {
        // Push back by increasing tail
        if ((*list_tail_)
                .compare_exchange_weak(
                    tail, (tail + 1) % (*list_capacity_), std::memory_order_release, std::memory_order_relaxed))
        {
            (*list_)[tail] = item;
            (*size_nbytes_).fetch_add(item->value->size, std::memory_order_relaxed);
            break;
        }

        tail = (*list_tail_).load(std::memory_order_relaxed);
    }
}

bool SharedMemoryImageCache::erase(const std::shared_ptr<ImageCacheKey>& key)
{
    auto key_impl = std::get_deleter<null_deleter<deleter_type<ImageCacheKey>>>(key)->get();
    const bool succeed = hashmap_->erase(key_impl);
    return succeed;
}


bool SharedMemoryImageCache::remove_shmem()
{
    cucim::config::Config* config = cucim::CuImage::get_config();
    if (config)
    {
        std::string shm_name = config->shm_name();
        return boost::interprocess::shared_memory_object::remove(shm_name.c_str());
    }
    return false;
}

uint32_t SharedMemoryImageCache::calc_hashmap_capacity(uint32_t capacity)
{
    return std::max((1U << 16) * 4, capacity * 4);
}

std::unique_ptr<boost::interprocess::managed_shared_memory> SharedMemoryImageCache::create_segment(
    const ImageCacheConfig& config)
{
    // Remove the existing shared memory object.
    remove_shmem();

    auto segment = std::make_unique<boost::interprocess::managed_shared_memory>(
        boost::interprocess::open_or_create, cucim::CuImage::get_config()->shm_name().c_str(), calc_segment_size(config));
    return segment;
}

} // namespace cucim::cache
