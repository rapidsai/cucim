/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CUCIM_CACHE_IMAGE_CACHE_H
#define CUCIM_CACHE_IMAGE_CACHE_H

#include "cucim/core/framework.h"

#include "cucim/cache/cache_type.h"
#include "cucim/cache/image_cache_config.h"

#include <memory>
#include <atomic>
#include <functional>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

namespace cucim::cache
{

struct EXPORT_VISIBLE ImageCacheKey
{
    ImageCacheKey(uint64_t file_hash, uint64_t index);

    uint64_t file_hash = 0; /// st_dev + st_ino + st_mtime + ifd_index
    uint64_t location_hash = 0; ///  tile_index or (x , y)
};

struct EXPORT_VISIBLE ImageCacheValue
{
    ImageCacheValue(void* data,
                    uint64_t size,
                    void* user_obj = nullptr,
                    const cucim::io::DeviceType device_type = cucim::io::DeviceType::kCPU);
    virtual ~ImageCacheValue(){};

    operator bool() const;

    void* data = nullptr;
    uint64_t size = 0;
    void* user_obj = nullptr;
    cucim::io::DeviceType device_type = cucim::io::DeviceType::kCPU;
};

/**
 * @brief Image Cache for loading tiles.
 *
 * FIFO is used for cache replacement policy here.
 *
 */

class EXPORT_VISIBLE ImageCache : public std::enable_shared_from_this<ImageCache>
{
public:
    ImageCache(const ImageCacheConfig& config,
               CacheType type = CacheType::kNoCache,
               const cucim::io::DeviceType device_type = cucim::io::DeviceType::kCPU);
    virtual ~ImageCache(){};

    virtual CacheType type() const;
    virtual const char* type_str() const;
    virtual cucim::io::DeviceType device_type() const;
    virtual ImageCacheConfig& config();
    virtual ImageCacheConfig get_config() const;

    /**
     * @brief Create a key object for the image cache.
     *
     * @param file_hash An hash value used for a specific file.
     * @param index An hash value to locate a specific index/coordination in the image.
     * @return std::shared_ptr<ImageCacheKey> A shared pointer containing %ImageCacheKey.
     */
    virtual std::shared_ptr<ImageCacheKey> create_key(uint64_t file_hash, uint64_t index) = 0;
    virtual std::shared_ptr<ImageCacheValue> create_value(
        void* data, uint64_t size, const cucim::io::DeviceType device_type = cucim::io::DeviceType::kCPU) = 0;

    virtual void* allocate(std::size_t n) = 0;

    virtual void lock(uint64_t index) = 0;
    virtual void unlock(uint64_t index) = 0;
    virtual void* mutex(uint64_t index) = 0;

    virtual bool insert(std::shared_ptr<ImageCacheKey>& key, std::shared_ptr<ImageCacheValue>& value) = 0;
    virtual void remove_front() = 0;

    virtual uint32_t size() const = 0;
    virtual uint64_t memory_size() const = 0;

    virtual uint32_t capacity() const = 0;
    virtual uint64_t memory_capacity() const = 0;
    virtual uint64_t free_memory() const = 0;

    /**
     * @brief Record cache stat.
     *
     * Stat values would be reset with this method.
     *
     * @param value Whether if cache stat would be recorded or not
     */
    virtual void record(bool value) = 0;

    /**
     * @brief Return whether if cache stat is recorded or not
     *
     * @return true if cache stat is recorded. false otherwise
     */
    virtual bool record() const = 0;

    virtual uint64_t hit_count() const = 0;
    virtual uint64_t miss_count() const = 0;

    /**
     * @brief Attempt to preallocate enough memory for specified number of elements and memory size.
     *
     * This method is not thread-safe.
     *
     * @param capacity Number of elements required
     * @param memory_capacity Size of memory required in bytes
     */
    virtual void reserve(const ImageCacheConfig& config) = 0;

    virtual std::shared_ptr<ImageCacheValue> find(const std::shared_ptr<ImageCacheKey>& key) = 0;

protected:
    CacheType type_ = CacheType::kNoCache;
    cucim::io::DeviceType device_type_ = cucim::io::DeviceType::kCPU;
    ImageCacheConfig config_;
};

} // namespace cucim::cache

#endif // CUCIM_CACHE_IMAGE_CACHE_H
