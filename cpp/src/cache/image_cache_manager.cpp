/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cucim/cache/image_cache_manager.h"

#include "image_cache_empty.h"
#include "image_cache_per_process.h"
#include "image_cache_shared_memory.h"
#include "cucim/cuimage.h"
#include "cucim/profiler/nvtx3.h"

#include <cstdlib>
#include <fmt/format.h>


namespace cucim::cache
{

uint32_t preferred_memory_capacity(const std::vector<uint64_t>& image_size,
                                   const std::vector<uint32_t>& tile_size,
                                   const std::vector<uint32_t>& patch_size,
                                   uint32_t bytes_per_pixel)
{
    // https://godbolt.org/z/eMf946oE7 for test

    if (image_size.size() != 2 || tile_size.size() != 2 || patch_size.size() != 2)
    {
        throw std::invalid_argument(
            fmt::format("Please specify arguments with correct size (image_size:{}, tile_size:{}, patch_size:{})!",
                        image_size.size(), tile_size.size(), patch_size.size()));
    }
    // Number of tiles (x-axis)
    uint32_t tile_accross_count = (image_size[0] + (tile_size[0] - 1)) / tile_size[0];

    // The maximal number of tiles (y-axis) overapped with the given patch
    uint32_t patch_down_count =
        std::min(image_size[1] + (tile_size[1] - 1), static_cast<uint64_t>(patch_size[1] + (tile_size[1] - 1))) /
            tile_size[1] +
        1;

    // (tile_accross_count) x (tile width) x (tile_height) x (patch_down_count) x (bytes per pixel)
    uint64_t bytes_needed =
        (static_cast<uint64_t>(tile_accross_count) * tile_size[0] * tile_size[1] * patch_down_count * bytes_per_pixel);
    uint32_t result = bytes_needed / kOneMiB;

    return (bytes_needed % kOneMiB == 0) ? result : result + 1;
}

ImageCacheManager::ImageCacheManager()
    : cache_(create_cache()), device_cache_config_(cucim::CuImage::get_config()->cache())
{
}

ImageCache& ImageCacheManager::cache() const
{
    return *cache_;
}

std::shared_ptr<ImageCache>& ImageCacheManager::ensure_device_cache_locked() const
{
    if (!device_cache_)
    {
        ImageCacheConfig config = device_cache_config_;
        // Device pointers cannot be stored in a Boost.Interprocess segment, so
        // there is no shared-memory form of a device cache.  Downgrading beats
        // throwing here: the caller asked for GPU output, not for a particular
        // cache backend, and a per-process device cache still serves them.
        if (config.type == CacheType::kSharedMemory)
        {
            config.type = CacheType::kPerProcess;
        }
        device_cache_ = create_cache(config, cucim::io::DeviceType::kCUDA);
    }
    return device_cache_;
}

ImageCache& ImageCacheManager::device_cache() const
{
    std::lock_guard<std::mutex> guard(device_cache_mutex_);
    return *ensure_device_cache_locked();
}

std::shared_ptr<cucim::cache::ImageCache> ImageCacheManager::get_device_cache() const
{
    std::lock_guard<std::mutex> guard(device_cache_mutex_);
    return ensure_device_cache_locked();
}

std::shared_ptr<cucim::cache::ImageCache> ImageCacheManager::cache(const ImageCacheConfig& config)
{
    cache_ = create_cache(config);

    {
        std::lock_guard<std::mutex> guard(device_cache_mutex_);
        device_cache_config_ = config;
        // Drop the device cache so the new configuration takes effect on next
        // use.  This also releases whatever device memory it was holding, which
        // is what a caller reconfiguring the cache expects.
        device_cache_.reset();
    }

    return cache_;
}

std::shared_ptr<cucim::cache::ImageCache> ImageCacheManager::get_cache() const
{
    return cache_;
}

void ImageCacheManager::reserve(uint32_t new_memory_capacity)
{
    ImageCacheConfig cache_config;
    cache_config.memory_capacity = new_memory_capacity;
    cache_config.capacity = calc_default_cache_capacity(kOneMiB * new_memory_capacity);

    cache_->reserve(cache_config);
    reserve_device_cache(cache_config);
}

void ImageCacheManager::reserve(uint32_t new_memory_capacity, uint32_t new_capacity)
{
    ImageCacheConfig cache_config;
    cache_config.memory_capacity = new_memory_capacity;
    cache_config.capacity = new_capacity;

    cache_->reserve(cache_config);
    reserve_device_cache(cache_config);
}

void ImageCacheManager::reserve_device_cache(const ImageCacheConfig& cache_config)
{
    std::lock_guard<std::mutex> guard(device_cache_mutex_);

    // Track the new limits either way, so a device cache built later starts out
    // with them rather than with the capacity that was configured at startup.
    device_cache_config_.memory_capacity = cache_config.memory_capacity;
    device_cache_config_.capacity = cache_config.capacity;

    // Only grow an existing cache; do not bring one into being here, or merely
    // reserving capacity would allocate device memory for a process that never
    // asks for GPU output.
    if (device_cache_)
    {
        device_cache_->reserve(cache_config);
    }
}

std::unique_ptr<ImageCache> ImageCacheManager::create_cache(const ImageCacheConfig& cache_config,
                                                            const cucim::io::DeviceType device_type)
{
    PROF_SCOPED_RANGE(PROF_EVENT(image_cache_create_cache));
    switch (cache_config.type)
    {
    case CacheType::kNoCache:
        return std::make_unique<EmptyImageCache>(cache_config);
    case CacheType::kPerProcess:
        return std::make_unique<PerProcessImageCache>(cache_config, device_type);
    case CacheType::kSharedMemory:
        return std::make_unique<SharedMemoryImageCache>(cache_config, device_type);
    default:
        return std::make_unique<EmptyImageCache>(cache_config);
    }
}

std::unique_ptr<ImageCache> ImageCacheManager::create_cache() const
{
    ImageCacheConfig& cache_config = cucim::CuImage::get_config()->cache();

    return create_cache(cache_config);
}

} // namespace cucim::cache
