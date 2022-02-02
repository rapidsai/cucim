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

ImageCacheManager::ImageCacheManager() : cache_(create_cache())
{
}

ImageCache& ImageCacheManager::cache() const
{
    return *cache_;
}

std::shared_ptr<cucim::cache::ImageCache> ImageCacheManager::cache(const ImageCacheConfig& config)
{
    cache_ = create_cache(config);
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
}

void ImageCacheManager::reserve(uint32_t new_memory_capacity, uint32_t new_capacity)
{
    ImageCacheConfig cache_config;
    cache_config.memory_capacity = new_memory_capacity;
    cache_config.capacity = new_capacity;

    cache_->reserve(cache_config);
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
