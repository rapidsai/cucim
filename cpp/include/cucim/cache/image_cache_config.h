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

#ifndef CUCIM_CACHE_IMAGE_CACHE_CONFIG_H
#define CUCIM_CACHE_IMAGE_CACHE_CONFIG_H

#include "cucim/core/framework.h"

#include "cucim/cache/cache_type.h"

namespace cucim::cache
{

constexpr uint64_t kOneMiB = 1024UL * 1024;
constexpr std::string_view kDefaultCacheTypeStr = "nocache";
constexpr CacheType kDefaultCacheType = cucim::cache::CacheType::kNoCache;
constexpr uint64_t kDefaultCacheMemoryCapacity = 1024UL;
constexpr uint32_t kDefaultCacheMutexPoolCapacity = 11117;
constexpr uint32_t kDefaultCacheListPadding = 10000;
constexpr uint32_t kDefaultCacheExtraSharedMemorySize = 100;
constexpr bool kDefaultCacheRecordStat = false;
// Assume that user uses memory block whose size is least 256 x 256 x 3 bytes.
constexpr uint32_t calc_default_cache_capacity(uint64_t memory_capacity_in_bytes)
{
    return memory_capacity_in_bytes / (256UL * 256 * 3);
}

struct EXPORT_VISIBLE ImageCacheConfig
{
    void load_config(void* json_obj);

    CacheType type = CacheType::kNoCache;
    uint32_t memory_capacity = kDefaultCacheMemoryCapacity;
    uint32_t capacity = calc_default_cache_capacity(kOneMiB * kDefaultCacheMemoryCapacity);
    uint32_t mutex_pool_capacity = kDefaultCacheMutexPoolCapacity;
    uint32_t list_padding = kDefaultCacheListPadding;
    uint32_t extra_shared_memory_size = kDefaultCacheExtraSharedMemorySize;
    bool record_stat = kDefaultCacheRecordStat;
};

} // namespace cucim::cache

#endif // CUCIM_CACHE_IMAGE_CACHE_CONFIG_H
