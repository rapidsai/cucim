/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CUCIM_CACHE_IMAGE_CACHE_MANAGER_H
#define CUCIM_CACHE_IMAGE_CACHE_MANAGER_H

#include "cucim/core/framework.h"

#include "cucim/cache/image_cache.h"
#include "cucim/io/device_type.h"

#include <mutex>

namespace cucim::cache
{

constexpr uint32_t kDefaultTileSize = 256;
constexpr uint32_t kDefaultPatchSize = 256;

uint32_t EXPORT_VISIBLE preferred_memory_capacity(const std::vector<uint64_t>& image_size,
                                                  const std::vector<uint32_t>& tile_size,
                                                  const std::vector<uint32_t>& patch_size,
                                                  uint32_t bytes_per_pixel = 3);

class EXPORT_VISIBLE ImageCacheManager
{
public:
    ImageCacheManager();

    ImageCache& cache() const;

    /**
     * Device-resident companion to cache().
     *
     * Readers that produce GPU output cache decoded tiles here instead of in
     * the host cache, so a warm read can be assembled with device-to-device
     * copies rather than staged through host memory.  Keeping it separate from
     * cache() rather than making a single cache device-aware means host and
     * device readers do not evict each other's tiles, and a process that never
     * asks for GPU output never allocates device memory: this cache is built
     * on first use.
     *
     * Both caches are built from the same configuration, so each may grow to
     * the configured memory_capacity independently.  A workload that mixes CPU
     * and GPU output can therefore hold that capacity twice over, once in host
     * and once in device memory.
     *
     * A kSharedMemory configuration has no device equivalent, since device
     * pointers cannot live in a Boost.Interprocess segment; such a
     * configuration is downgraded to kPerProcess for this cache only.
     */
    ImageCache& device_cache() const;

    std::shared_ptr<ImageCache> cache(const ImageCacheConfig& config);
    std::shared_ptr<ImageCache> get_cache() const;
    std::shared_ptr<ImageCache> get_device_cache() const;
    void reserve(uint32_t new_memory_capacity);
    void reserve(uint32_t new_memory_capacity, uint32_t new_capacity);

    static std::unique_ptr<ImageCache> create_cache(const ImageCacheConfig& cache_config,
                                                    const cucim::io::DeviceType device_type = cucim::io::DeviceType::kCPU);

private:
    std::unique_ptr<ImageCache> create_cache() const;
    void reserve_device_cache(const ImageCacheConfig& cache_config);
    // Caller must hold device_cache_mutex_.
    std::shared_ptr<ImageCache>& ensure_device_cache_locked() const;

    std::shared_ptr<ImageCache> cache_;

    // Built on first device_cache() call and rebuilt whenever the cache
    // configuration changes, so GPU readers follow the same settings the user
    // applied through cache(config).
    ImageCacheConfig device_cache_config_;
    mutable std::shared_ptr<ImageCache> device_cache_;
    mutable std::mutex device_cache_mutex_;
};

} // namespace cucim::cache

#endif // CUCIM_CACHE_IMAGE_CACHE_MANAGER_H
