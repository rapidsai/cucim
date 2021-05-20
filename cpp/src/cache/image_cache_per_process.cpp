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

#include "image_cache_per_process.h"

#include "cucim/memory/memory_manager.h"

#include <fmt/format.h>

namespace std
{

size_t hash<std::shared_ptr<cucim::cache::ImageCacheKey>>::operator()(
    const std::shared_ptr<cucim::cache::ImageCacheKey>& s) const
{
    std::size_t h1 = std::hash<uint64_t>{}(s->file_hash);
    std::size_t h2 = std::hash<uint64_t>{}(s->location_hash);
    return h1 ^ (h2 << 1); // or use boost::hash_combine
}

bool equal_to<std::shared_ptr<cucim::cache::ImageCacheKey>>::operator()(
    const std::shared_ptr<cucim::cache::ImageCacheKey>& lhs, const std::shared_ptr<cucim::cache::ImageCacheKey>& rhs) const
{
    return lhs->location_hash == rhs->location_hash;
}

} // namespace std
namespace cucim::cache
{

struct PerProcessImageCacheItem
{
    PerProcessImageCacheItem(std::shared_ptr<ImageCacheKey>& key, std::shared_ptr<ImageCacheValue>& value)
        : key(key), value(value)
    {
    }

    std::shared_ptr<ImageCacheKey> key;
    std::shared_ptr<ImageCacheValue> value;
};

PerProcessImageCacheValue::PerProcessImageCacheValue(void* data, uint64_t size, void* user_obj)
    : ImageCacheValue(data, size, user_obj){};
PerProcessImageCacheValue::~PerProcessImageCacheValue()
{
    if (data)
    {
        cucim_free(data);
        data = nullptr;
    }
};

PerProcessImageCache::PerProcessImageCache(const ImageCacheConfig& config)
    : ImageCache(config, CacheType::kPerProcess),
      mutex_array_(config.mutex_pool_capacity),
      capacity_nbytes_(kOneMiB * config.memory_capacity),
      capacity_(config.capacity),
      list_capacity_(config.capacity + config.list_padding),
      list_padding_(config.list_padding),
      mutex_pool_capacity_(config.mutex_pool_capacity),
      stat_is_recorded_(config.record_stat),
      list_(config.capacity + config.list_padding),
      hashmap_(config.capacity){};

PerProcessImageCache::~PerProcessImageCache()
{
}

const char* PerProcessImageCache::type_str() const
{
    return "per_process";
}

std::shared_ptr<ImageCacheKey> PerProcessImageCache::create_key(uint64_t file_hash, uint64_t index)
{
    return std::make_shared<ImageCacheKey>(file_hash, index);
}
std::shared_ptr<ImageCacheValue> PerProcessImageCache::create_value(void* data, uint64_t size)
{
    return std::make_shared<PerProcessImageCacheValue>(data, size);
}

void* PerProcessImageCache::allocate(std::size_t n)
{
    return cucim_malloc(n);
}

void PerProcessImageCache::lock(uint64_t index)
{
    mutex_array_[index % mutex_pool_capacity_].lock();
}

void PerProcessImageCache::unlock(uint64_t index)
{
    mutex_array_[index % mutex_pool_capacity_].unlock();
}

bool PerProcessImageCache::insert(std::shared_ptr<ImageCacheKey>& key, std::shared_ptr<ImageCacheValue>& value)
{
    if (value->size > capacity_nbytes_ || capacity_ < 1)
    {
        return false;
    }

    while (is_list_full() || is_memory_full(value->size))
    {
        remove_front();
    }

    auto item = std::make_shared<PerProcessImageCacheItem>(key, value);
    bool succeed = hashmap_.insert(key, item);

    if (succeed)
    {
        push_back(item);
    }
    else
    {
        fmt::print(stderr, "{} existing list_[] = {}\n", std::hash<std::thread::id>{}(std::this_thread::get_id()), (uint64_t)item->key->location_hash);
    }
    return succeed;
}

uint32_t PerProcessImageCache::size() const
{
    uint32_t head = list_head_.load(std::memory_order_relaxed);
    uint32_t tail = list_tail_.load(std::memory_order_relaxed);

    return (tail + list_capacity_ - head) % list_capacity_;
}

uint64_t PerProcessImageCache::memory_size() const
{
    return size_nbytes_.load(std::memory_order_relaxed);
}

uint32_t PerProcessImageCache::capacity() const
{
    return capacity_;
}

uint64_t PerProcessImageCache::memory_capacity() const
{
    return capacity_nbytes_;
}

uint64_t PerProcessImageCache::free_memory() const
{
    return capacity_nbytes_ - size_nbytes_.load(std::memory_order_relaxed);
}

void PerProcessImageCache::record(bool value)
{
    config_.record_stat = value;

    stat_hit_.store(0, std::memory_order_relaxed);
    stat_miss_.store(0, std::memory_order_relaxed);
    stat_is_recorded_ = value;
}

bool PerProcessImageCache::record() const
{
    return stat_is_recorded_;
}

uint64_t PerProcessImageCache::hit_count() const
{
    return stat_hit_.load(std::memory_order_relaxed);
}
uint64_t PerProcessImageCache::miss_count() const
{
    return stat_miss_.load(std::memory_order_relaxed);
}

void PerProcessImageCache::reserve(const ImageCacheConfig& config)
{
    uint32_t new_capacity = config.capacity;
    uint64_t new_memory_capacity_nbytes = kOneMiB * config.memory_capacity;

    if (capacity_ < new_capacity)
    {
        config_.capacity = config.capacity;
        config_.memory_capacity = config.memory_capacity;

        uint32_t old_list_capacity = list_capacity_;

        capacity_ = new_capacity;
        list_capacity_ = new_capacity + list_padding_;

        list_.reserve(list_capacity_);
        list_.resize(list_capacity_);
        hashmap_.reserve(new_capacity);

        // Move items in the vector
        uint32_t head = list_head_.load(std::memory_order_relaxed);
        uint32_t tail = list_tail_.load(std::memory_order_relaxed);
        if (tail < head)
        {
            head = 0;
            uint32_t new_head = old_list_capacity;

            while (head != tail)
            {
                list_[new_head] = list_[head];
                list_[head].reset();

                head = (head + 1) % old_list_capacity;
                new_head = (new_head + 1) % list_capacity_;
            }
            // Set new tail
            list_tail_.store(new_head, std::memory_order_relaxed);
        }
    }

    if (capacity_nbytes_ < new_memory_capacity_nbytes)
    {
        capacity_nbytes_ = new_memory_capacity_nbytes;
    }
}

std::shared_ptr<ImageCacheValue> PerProcessImageCache::find(const std::shared_ptr<ImageCacheKey>& key)
{
    std::shared_ptr<PerProcessImageCacheItem> item;
    const bool found = hashmap_.find(key, item);
    if(stat_is_recorded_)
    {
        if (found)
        {
            stat_hit_.fetch_add(1, std::memory_order_relaxed);
            return item->value;
        }
        else
        {
            stat_miss_.fetch_add(1, std::memory_order_relaxed);
        }
    }
    else
    {
        if (found)
        {
            return item->value;
        }
    }
    return std::shared_ptr<ImageCacheValue>();
}

bool PerProcessImageCache::is_list_full() const
{
    if (size() >= capacity_)
    {
        return true;
    }
    return false;
}

bool PerProcessImageCache::is_memory_full(uint64_t additional_size) const
{
    if (size_nbytes_.load(std::memory_order_relaxed) + additional_size > capacity_nbytes_)
    {
        return true;
    }
    else
    {
        return false;
    }
}

void PerProcessImageCache::remove_front()
{
    while (true)
    {
        uint32_t head = list_head_.load(std::memory_order_relaxed);
        uint32_t tail = list_tail_.load(std::memory_order_relaxed);
        if (head != tail)
        {
            // Remove front by increasing head
            if (list_head_.compare_exchange_weak(
                    head, (head + 1) % list_capacity_, std::memory_order_release, std::memory_order_relaxed))
            {
                // fmt::print(stderr, "{} remove list_[{:05}]\n", std::hash<std::thread::id>{}(std::this_thread::get_id()), head); //[print_list]
                std::shared_ptr<PerProcessImageCacheItem> head_item = list_[head];
                // if (head_item) // it is possible that head_item is nullptr.
                // {
                size_nbytes_.fetch_sub(head_item->value->size, std::memory_order_relaxed);
                hashmap_.erase(head_item->key);
                list_[head].reset(); // decrease refcount
                break;
                    // }
            }
        }
        else
        {
            break; // already empty
        }
    }
}

void PerProcessImageCache::push_back(std::shared_ptr<PerProcessImageCacheItem>& item)
{
    uint32_t tail = list_tail_.load(std::memory_order_relaxed);
    while (true)
    {
        // Push back by increasing tail
        if (list_tail_.compare_exchange_weak(
                tail, (tail + 1) % list_capacity_, std::memory_order_release, std::memory_order_relaxed))
        {
            // fmt::print(stderr, "{} list_[{:05}]={}\n", std::hash<std::thread::id>{}(std::this_thread::get_id()), tail, (uint64_t)item->key->location_hash); // [print_list]
            list_[tail] = item;
            size_nbytes_.fetch_add(item->value->size, std::memory_order_relaxed);
            break;
        }

        tail = list_tail_.load(std::memory_order_relaxed);
    }
}

bool PerProcessImageCache::erase(const std::shared_ptr<ImageCacheKey>& key)
{
    const bool succeed = hashmap_.erase(key);
    return succeed;
}

} // namespace cucim::cache
