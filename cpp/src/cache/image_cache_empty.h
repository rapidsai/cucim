/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CUCIM_CACHE_IMAGE_CACHE_EMPTY_H
#define CUCIM_CACHE_IMAGE_CACHE_EMPTY_H

#include "cucim/cache/image_cache.h"

namespace cucim::cache
{

/**
 * @brief Image Cache for loading tiles.
 *
 * FIFO is used for cache replacement policy here.
 *
 */

class EmptyImageCache : public ImageCache
{
public:
    EmptyImageCache(const ImageCacheConfig& config);

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
    ImageCacheConfig config_;
};

} // namespace cucim::cache

#endif // CUCIM_CACHE_IMAGE_CACHE_EMPTY_H
