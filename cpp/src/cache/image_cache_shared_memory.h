/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#ifndef CUCIM_CACHE_IMAGE_CACHE_SHARED_MEMORY_H
#define CUCIM_CACHE_IMAGE_CACHE_SHARED_MEMORY_H

#include "cucim/cache/image_cache.h"

#include <boost/container_hash/hash.hpp>
#include <boost/interprocess/smart_ptr/unique_ptr.hpp>
#include <boost/interprocess/smart_ptr/shared_ptr.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <libcuckoo/cuckoohash_map.hh>

#include <atomic>
#include <type_traits>
#include <scoped_allocator>


namespace cucim::cache
{

// Forward declarations
struct ImageCacheItemDetail;

struct SharedMemoryImageCacheValue : public ImageCacheValue
{
    SharedMemoryImageCacheValue(void* data,
                                uint64_t size,
                                void* user_obj = nullptr,
                                const cucim::io::DeviceType device_type = cucim::io::DeviceType::kCPU);
    ~SharedMemoryImageCacheValue() override;
};

template <class T>
struct shared_mem_deleter
{
    shared_mem_deleter(std::unique_ptr<boost::interprocess::managed_shared_memory>& segment);
    void operator()(T* p);

private:
    std::unique_ptr<boost::interprocess::managed_shared_memory>& seg_;
};

template <class T>
using boost_unique_ptr = std::unique_ptr<T, shared_mem_deleter<T>>;


template <class T>
using boost_shared_ptr = boost::interprocess::shared_ptr<
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


using MapKey = boost::interprocess::managed_shared_ptr<ImageCacheKey, boost::interprocess::managed_shared_memory>;

using MapValue =
    boost::interprocess::managed_shared_ptr<ImageCacheItemDetail, boost::interprocess::managed_shared_memory>;

using KeyValuePair = std::pair<MapKey, MapValue>;
using ImageCacheAllocator =
    boost::interprocess::allocator<KeyValuePair, boost::interprocess::managed_shared_memory::segment_manager>;

using ValueAllocator = std::scoped_allocator_adaptor<
    boost::interprocess::allocator<MapValue::type, boost::interprocess::managed_shared_memory::segment_manager>>;

using MapKeyHasher = boost::hash<MapKey>;
using MakKeyEqual = std::equal_to<MapKey>;
using ImageCacheType =
    libcuckoo::cuckoohash_map<MapKey::type, MapValue::type, boost::hash<MapKey>, std::equal_to<MapKey>, ImageCacheAllocator>;
using QueueType = std::vector<MapValue::type, ValueAllocator>;

template <class T>
using cache_item_type = boost::interprocess::shared_ptr<
    T,
    boost::interprocess::allocator<
        void,
        boost::interprocess::segment_manager<
            char,
            boost::interprocess::rbtree_best_fit<boost::interprocess::mutex_family,
                                                 boost::interprocess::offset_ptr<void, std::ptrdiff_t, uintptr_t, 0UL>,
                                                 0UL>,
            boost::interprocess::iset_index>>,
    boost::interprocess::deleter<
        T,
        boost::interprocess::segment_manager<
            char,
            boost::interprocess::rbtree_best_fit<boost::interprocess::mutex_family,
                                                 boost::interprocess::offset_ptr<void, std::ptrdiff_t, uintptr_t, 0UL>,
                                                 0UL>,
            boost::interprocess::iset_index>>>;

/**
 * @brief Image Cache for loading tiles.
 *
 * FIFO is used for cache replacement policy here.
 *
 */

class SharedMemoryImageCache : public ImageCache
{
public:
    SharedMemoryImageCache(const ImageCacheConfig& config,
                           const cucim::io::DeviceType device_type = cucim::io::DeviceType::kCPU);
    ~SharedMemoryImageCache();

    const char* type_str() const override;

    std::shared_ptr<ImageCacheKey> create_key(uint64_t file_hash, uint64_t index) override;
    std::shared_ptr<ImageCacheValue> create_value(
        void* data, uint64_t size, const cucim::io::DeviceType device_type = cucim::io::DeviceType::kCPU) override;

    void* allocate(std::size_t n) override;
    void lock(uint64_t index) override;
    void unlock(uint64_t index) override;
    void* mutex(uint64_t index) override;

    bool insert(std::shared_ptr<ImageCacheKey>& key, std::shared_ptr<ImageCacheValue>& value) override;
    void remove_front() override;

    uint32_t size() const override;
    uint64_t memory_size() const override;

    uint32_t capacity() const override;
    uint64_t memory_capacity() const override;
    uint64_t free_memory() const override;

    void record(bool value) override;
    bool record() const override;

    uint64_t hit_count() const override;
    uint64_t miss_count() const override;

    void reserve(const ImageCacheConfig& config) override;

    std::shared_ptr<ImageCacheValue> find(const std::shared_ptr<ImageCacheKey>& key) override;

private:
    bool is_list_full() const;
    bool is_memory_full(uint64_t additional_size = 0) const;
    void push_back(cache_item_type<ImageCacheItemDetail>& item);
    bool erase(const std::shared_ptr<ImageCacheKey>& key);

    std::shared_ptr<ImageCacheItemDetail> create_cache_item(std::shared_ptr<ImageCacheKey>& key,
                                                            std::shared_ptr<ImageCacheValue>& value);

    static bool remove_shmem();

    uint32_t calc_hashmap_capacity(uint32_t capacity);
    std::unique_ptr<boost::interprocess::managed_shared_memory> create_segment(const ImageCacheConfig& config);

    std::unique_ptr<boost::interprocess::managed_shared_memory> segment_;

    // boost_unique_ptr<boost::interprocess::interprocess_mutex> mutex_array_;
    boost::interprocess::interprocess_mutex* mutex_array_ = nullptr;
    boost_unique_ptr<std::atomic<uint64_t>> size_nbytes_; /// size of cache;
                                                          /// memory used
    boost_unique_ptr<uint64_t> capacity_nbytes_; /// size of cache memory allocated
    boost_unique_ptr<uint32_t> capacity_; /// capacity of hashmap
    boost_unique_ptr<uint32_t> list_capacity_; /// capacity of list
    boost_unique_ptr<uint32_t> list_padding_; /// gap between head and tail
    boost_unique_ptr<uint32_t> mutex_pool_capacity_; /// capacity of mutex pool

    boost_unique_ptr<std::atomic<uint64_t>> stat_hit_; /// cache hit count
    boost_unique_ptr<std::atomic<uint64_t>> stat_miss_; /// cache miss mcount
    boost_unique_ptr<bool> stat_is_recorded_; /// whether if cache stat is recorded or not

    boost_unique_ptr<std::atomic<uint32_t>> list_head_; /// head
    boost_unique_ptr<std::atomic<uint32_t>> list_tail_; /// tail

    boost_shared_ptr<QueueType> list_;
    boost_shared_ptr<ImageCacheType> hashmap_;
};

} // namespace cucim::cache

#endif // CUCIM_CACHE_IMAGE_CACHE_SHARED_MEMORY_H
