/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
/**
 * @brief Mutex Pool size
 *
 * >>> from functools import reduce
 * >>> def calc(pool_size, thread_size):
 * >>>     a = reduce(lambda x,y: x*y, range(pool_size, pool_size - thread_size, -1))
 * >>>     print(1 - (a / (pool_size**thread_size)))
 *
 * >>> calc(100003, 128)
 * 0.07809410393222294
 * >>> calc(100003, 256)
 * 0.2786772006302005
 *
 * See https://godbolt.org/z/Tvx8179xK
 * Creating a pool of 100000 mutexes takes only about 4 MB which is not big.
 * I believe that making the mutex size biggger enough helps to the reduce the thread contention.
 * For systems with more than 256 threads, the pool size should be larger.
 * Choose a prime number for the pool size (https://primes.utm.edu/lists/small/100000.txt).
 */
constexpr uint32_t kDefaultCacheMutexPoolCapacity = 100003;
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
    void load_config(const void* json_obj);

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
