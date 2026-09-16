/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cucim/cache/image_cache.h>
#include <cucim/cuimage.h>
#include <cucim/filesystem/cufile_driver.h>
#include <cucim/profiler/profiler.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cucim::python
{

std::string library_version();

struct DeviceData
{
    int16_t type;
    int16_t index;
    std::string name;
};

struct DTypeData
{
    uint8_t code;
    uint8_t bits;
    uint16_t lanes;
};

struct ResolutionData
{
    uint16_t level_count;
    std::vector<std::vector<int64_t>> level_dimensions;
    std::vector<float> level_downsamples;
    std::vector<std::vector<uint32_t>> level_tile_sizes;
};

struct ArrayInterfaceData
{
    bool has_tensor;
    bool has_data;
    uintptr_t data;
    int device_type;
    std::string typestr;
    std::vector<int64_t> shape;
};

struct CacheConfigData
{
    int type;
    std::string type_name;
    uint32_t memory_capacity;
    uint32_t capacity;
    uint32_t mutex_pool_capacity;
    uint32_t list_padding;
    uint32_t extra_shared_memory_size;
    bool record_stat;
};

class ImageIterator
{
public:
    explicit ImageIterator(std::shared_ptr<cucim::CuImage> image, bool ending = false);

    int64_t index();
    uint64_t size();
    std::shared_ptr<cucim::CuImage> next();

private:
    cucim::CuImageIterator<cucim::CuImage> iterator_;
};

std::string get_plugin_root();
void set_plugin_root(const std::string& path);

DeviceData parse_device(const std::string& name);

std::shared_ptr<cucim::CuImage> make_image(const std::string& path);
std::string image_path(const std::shared_ptr<cucim::CuImage>& image);
bool image_is_loaded(const std::shared_ptr<cucim::CuImage>& image);
DeviceData image_device(const std::shared_ptr<cucim::CuImage>& image);
std::string image_raw_metadata(const std::shared_ptr<cucim::CuImage>& image);
std::string image_metadata(const std::shared_ptr<cucim::CuImage>& image);
uint16_t image_ndim(const std::shared_ptr<cucim::CuImage>& image);
std::string image_dims(const std::shared_ptr<cucim::CuImage>& image);
std::vector<int64_t> image_shape(const std::shared_ptr<cucim::CuImage>& image);
std::vector<int64_t> image_size(const std::shared_ptr<cucim::CuImage>& image, const std::string& dim_order);
DTypeData image_dtype(const std::shared_ptr<cucim::CuImage>& image);
std::string image_typestr(const std::shared_ptr<cucim::CuImage>& image);
std::vector<std::string> image_channel_names(const std::shared_ptr<cucim::CuImage>& image);
std::vector<float> image_spacing(const std::shared_ptr<cucim::CuImage>& image, const std::string& dim_order);
std::vector<std::string> image_spacing_units(const std::shared_ptr<cucim::CuImage>& image,
                                             const std::string& dim_order);
std::vector<float> image_origin(const std::shared_ptr<cucim::CuImage>& image);
std::vector<std::vector<float>> image_direction(const std::shared_ptr<cucim::CuImage>& image);
std::string image_coord_sys(const std::shared_ptr<cucim::CuImage>& image);
ResolutionData image_resolutions(const std::shared_ptr<cucim::CuImage>& image);
std::vector<std::string> image_associated_images(const std::shared_ptr<cucim::CuImage>& image);
std::shared_ptr<cucim::CuImage> image_associated_image(const std::shared_ptr<cucim::CuImage>& image,
                                                      const std::string& name,
                                                      const std::string& device);
std::shared_ptr<cucim::CuImage> image_read_region(const std::shared_ptr<cucim::CuImage>& image,
                                                 std::vector<int64_t> location,
                                                 std::vector<int64_t> size,
                                                 int16_t level,
                                                 uint32_t num_workers,
                                                 uint32_t batch_size,
                                                 bool drop_last,
                                                 uint32_t prefetch_factor,
                                                 bool shuffle,
                                                 uint64_t seed,
                                                 const std::vector<int>& dim_chars,
                                                 const std::vector<int64_t>& dim_values,
                                                 const std::string& device);
bool image_has_multiple_batches(const std::shared_ptr<cucim::CuImage>& image, uint32_t batch_size);
ArrayInterfaceData image_array_interface(const std::shared_ptr<cucim::CuImage>& image);
void image_save(const std::shared_ptr<cucim::CuImage>& image, const std::string& file_path);
void image_close(const std::shared_ptr<cucim::CuImage>& image);
bool image_bool(const std::shared_ptr<cucim::CuImage>& image);
bool image_is_trace_enabled();

CacheConfigData cache_default_config();
int cache_type_from_name(const std::string& name);
std::shared_ptr<cucim::cache::ImageCache> cache_default();
std::shared_ptr<cucim::cache::ImageCache> cache_create(const CacheConfigData& config);
CacheConfigData cache_config(const std::shared_ptr<cucim::cache::ImageCache>& cache);
int cache_type(const std::shared_ptr<cucim::cache::ImageCache>& cache);
uint32_t cache_size(const std::shared_ptr<cucim::cache::ImageCache>& cache);
uint64_t cache_memory_size(const std::shared_ptr<cucim::cache::ImageCache>& cache);
uint32_t cache_capacity(const std::shared_ptr<cucim::cache::ImageCache>& cache);
uint64_t cache_memory_capacity(const std::shared_ptr<cucim::cache::ImageCache>& cache);
uint64_t cache_free_memory(const std::shared_ptr<cucim::cache::ImageCache>& cache);
bool cache_record_get(const std::shared_ptr<cucim::cache::ImageCache>& cache);
void cache_record_set(const std::shared_ptr<cucim::cache::ImageCache>& cache, bool value);
uint64_t cache_hit_count(const std::shared_ptr<cucim::cache::ImageCache>& cache);
uint64_t cache_miss_count(const std::shared_ptr<cucim::cache::ImageCache>& cache);
void cache_reserve(const std::shared_ptr<cucim::cache::ImageCache>& cache,
                   uint32_t memory_capacity,
                   bool has_capacity,
                   uint32_t capacity);
uint32_t preferred_memory_capacity_for_image(const std::shared_ptr<cucim::CuImage>& image,
                                             const std::vector<uint32_t>& patch_size,
                                             uint32_t bytes_per_pixel);
uint32_t preferred_memory_capacity_explicit(const std::vector<uint64_t>& image_size,
                                            const std::vector<uint32_t>& tile_size,
                                            const std::vector<uint32_t>& patch_size,
                                            uint32_t bytes_per_pixel);

std::shared_ptr<cucim::profiler::Profiler> profiler_default();
std::shared_ptr<cucim::profiler::Profiler> profiler_create(bool trace);
bool profiler_trace_get(const std::shared_ptr<cucim::profiler::Profiler>& profiler);
void profiler_trace_set(const std::shared_ptr<cucim::profiler::Profiler>& profiler, bool value);

std::shared_ptr<cucim::filesystem::CuFileDriver> file_driver_from_fd(int fd,
                                                                   bool no_gds,
                                                                   bool use_mmap,
                                                                   const std::string& file_path);
std::shared_ptr<cucim::filesystem::CuFileDriver> filesystem_open(const std::string& file_path,
                                                                const std::string& flags,
                                                                uint32_t mode);
bool filesystem_is_gds_available();
bool filesystem_close(const std::shared_ptr<cucim::filesystem::CuFileDriver>& file);
bool filesystem_discard_page_cache(const std::string& file_path);
std::string file_driver_path(const std::shared_ptr<cucim::filesystem::CuFileDriver>& file);
int64_t file_driver_pread(const std::shared_ptr<cucim::filesystem::CuFileDriver>& file,
                          uintptr_t buffer,
                          uint64_t count,
                          int64_t file_offset,
                          int64_t buffer_offset);
int64_t file_driver_pwrite(const std::shared_ptr<cucim::filesystem::CuFileDriver>& file,
                           uintptr_t buffer,
                           uint64_t count,
                           int64_t file_offset,
                           int64_t buffer_offset);

} // namespace cucim::python
