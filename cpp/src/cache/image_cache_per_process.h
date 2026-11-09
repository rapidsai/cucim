/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CUCIM_CACHE_IMAGE_CACHE_PER_PROCESS_H
#define CUCIM_CACHE_IMAGE_CACHE_PER_PROCESS_H

#include "cucim/cache/image_cache.h"
#include "cucim/memory/device_resources.h"

#include <libcuckoo/cuckoohash_map.hh>
#include <memory>
#include <array>

namespace std
{

template <>
struct hash<std::shared_ptr<cucim::cache::ImageCacheKey>>
{
    size_t operator()(const std::shared_ptr<cucim::cache::ImageCacheKey>& s) const;
};

template <>
struct equal_to<std::shared_ptr<cucim::cache::ImageCacheKey>>
{
    bool operator()(const std::shared_ptr<cucim::cache::ImageCacheKey>& lhs,
                    const std::shared_ptr<cucim::cache::ImageCacheKey>& rhs) const;
};

} // namespace std

namespace cucim::cache
{

// Forward declarations
struct PerProcessImageCacheItem;

struct PerProcessImageCacheValue : public ImageCacheValue
{
    /**
     * @param device_resources Resource that allocated *data*, when
     *        *device_type* is kCUDA. Held by value rather than by reference
     *        because a cached value can outlive the cache that produced it
     *        (an entry evicted while a reader still holds it is destroyed
     *        last), and freeing through a dangling resource would be worse.
     *        Holding it by value does not duplicate the allocator: copies of
     *        an owning handle share the one resource.
     */
    PerProcessImageCacheValue(void* data,
                              uint64_t size,
                              void* user_obj = nullptr,
                              const cucim::io::DeviceType device_type = cucim::io::DeviceType::kCPU,
                              cucim::memory::DeviceResources device_resources = {});
    ~PerProcessImageCacheValue() override;

private:
    /// Must be the resource that allocated `data`; see the constructor.
    cucim::memory::DeviceResources device_resources_;
};


/**
 * @brief Image Cache for loading tiles.
 *
 * FIFO is used for cache replacement policy here.
 *
 */

class PerProcessImageCache : public ImageCache
{
public:
    /**
     * @param device_resources Where device tiles are allocated from. Fixed for
     *        the lifetime of the cache: allocate() and create_value() are
     *        separate calls, so a resource that could change between them
     *        could record the wrong allocator as an entry's deallocator.
     */
    PerProcessImageCache(const ImageCacheConfig& config,
                         const cucim::io::DeviceType device_type = cucim::io::DeviceType::kCPU,
                         cucim::memory::DeviceResources device_resources = {});
    ~PerProcessImageCache();

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
    void push_back(std::shared_ptr<PerProcessImageCacheItem>& item);
    bool erase(const std::shared_ptr<ImageCacheKey>& key);

    std::vector<std::mutex> mutex_array_;

    /// Where device tiles come from. Set at construction, never reassigned;
    /// see the constructor for why.
    cucim::memory::DeviceResources device_resources_;

    std::atomic<uint64_t> size_nbytes_ = 0; /// size of cache memory used
    uint64_t capacity_nbytes_ = 0; /// size of cache memory allocated
    uint32_t capacity_ = 0; /// capacity of hashmap
    uint32_t list_capacity_ = 0; /// capacity of list
    uint32_t list_padding_ = 0; /// gap between head and tail
    uint32_t mutex_pool_capacity_ = 0; /// capacity of mutex pool

    std::atomic<uint64_t> stat_hit_ = 0; /// cache hit count
    std::atomic<uint64_t> stat_miss_ = 0; /// cache miss mcount
    bool stat_is_recorded_ = false; /// whether if cache stat is recorded or not

    std::atomic<uint32_t> list_head_ = 0; /// head
    std::atomic<uint32_t> list_tail_ = 0; /// tail

    std::vector<std::shared_ptr<PerProcessImageCacheItem>> list_; /// circular list using vector
    libcuckoo::cuckoohash_map<std::shared_ptr<ImageCacheKey>, std::shared_ptr<PerProcessImageCacheItem>> hashmap_; /// hashmap
                                                                                                                   /// using
                                                                                                                   /// libcuckoo
};

} // namespace cucim::cache

#endif // CUCIM_CACHE_IMAGE_CACHE_PER_PROCESS_H
