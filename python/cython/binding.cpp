/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "binding.hpp"

#include <cucim/cache/image_cache_manager.h>
#include <cucim/config/config.h>
#include <cucim/loader/thread_batch_data_loader.h>
#include <cucim/memory/dlpack.h>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace cucim::python
{
namespace
{

#define CUCIM_STRINGIFY_IMPL(value) #value
#define CUCIM_STRINGIFY(value) CUCIM_STRINGIFY_IMPL(value)

CacheConfigData to_cache_config_data(const cache::ImageCacheConfig& config)
{
    return { static_cast<int>(config.type),
             std::string(cache::lookup_cache_type_str(config.type)),
             config.memory_capacity,
             config.capacity,
             config.mutex_pool_capacity,
             config.list_padding,
             config.extra_shared_memory_size,
             config.record_stat };
}

cache::ImageCacheConfig to_cache_config(const CacheConfigData& data)
{
    cache::ImageCacheConfig config = CuImage::get_config()->cache();
    config.type = static_cast<cache::CacheType>(data.type);
    config.memory_capacity = data.memory_capacity;
    config.capacity = data.capacity;
    config.mutex_pool_capacity = data.mutex_pool_capacity;
    config.list_padding = data.list_padding;
    config.extra_shared_memory_size = data.extra_shared_memory_size;
    config.record_stat = data.record_stat;
    return config;
}

} // namespace

std::string library_version()
{
#ifdef CUCIM_VERSION
    return CUCIM_STRINGIFY(CUCIM_VERSION);
#else
    return {};
#endif
}

ImageIterator::ImageIterator(std::shared_ptr<CuImage> image, bool ending) : iterator_(std::move(image), ending) {}

int64_t ImageIterator::index()
{
    return iterator_.index();
}

uint64_t ImageIterator::size()
{
    return iterator_.size();
}

std::shared_ptr<CuImage> ImageIterator::next()
{
    if (iterator_.index() == iterator_.size())
    {
        throw std::out_of_range("CuImage iterator is exhausted");
    }
    ++iterator_;
    return *iterator_;
}

std::string get_plugin_root()
{
    return CuImage::get_framework()->get_plugin_root();
}

void set_plugin_root(const std::string& path)
{
    CuImage::get_framework()->set_plugin_root(path.c_str());
}

DeviceData parse_device(const std::string& name)
{
    io::Device device(name);
    return { static_cast<int16_t>(device.type()), device.index(), static_cast<std::string>(device) };
}

std::shared_ptr<CuImage> make_image(const std::string& path)
{
    return std::make_shared<CuImage>(path);
}

std::string image_path(const std::shared_ptr<CuImage>& image)
{
    return image->path();
}

bool image_is_loaded(const std::shared_ptr<CuImage>& image)
{
    return image->is_loaded();
}

DeviceData image_device(const std::shared_ptr<CuImage>& image)
{
    auto device = image->device();
    return { static_cast<int16_t>(device.type()), device.index(), static_cast<std::string>(device) };
}

std::string image_raw_metadata(const std::shared_ptr<CuImage>& image)
{
    return image->raw_metadata();
}

std::string image_metadata(const std::shared_ptr<CuImage>& image)
{
    return image->metadata();
}

uint16_t image_ndim(const std::shared_ptr<CuImage>& image)
{
    return image->ndim();
}

std::string image_dims(const std::shared_ptr<CuImage>& image)
{
    return image->dims();
}

std::vector<int64_t> image_shape(const std::shared_ptr<CuImage>& image)
{
    return image->shape();
}

std::vector<int64_t> image_size(const std::shared_ptr<CuImage>& image, const std::string& dim_order)
{
    return image->size(dim_order);
}

DTypeData image_dtype(const std::shared_ptr<CuImage>& image)
{
    auto dtype = image->dtype();
    return { dtype.code, dtype.bits, dtype.lanes };
}

std::string image_typestr(const std::shared_ptr<CuImage>& image)
{
    return image->typestr();
}

std::vector<std::string> image_channel_names(const std::shared_ptr<CuImage>& image)
{
    return image->channel_names();
}

std::vector<float> image_spacing(const std::shared_ptr<CuImage>& image, const std::string& dim_order)
{
    return image->spacing(dim_order);
}

std::vector<std::string> image_spacing_units(const std::shared_ptr<CuImage>& image, const std::string& dim_order)
{
    return image->spacing_units(dim_order);
}

std::vector<float> image_origin(const std::shared_ptr<CuImage>& image)
{
    auto origin = image->origin();
    return { origin.begin(), origin.end() };
}

std::vector<std::vector<float>> image_direction(const std::shared_ptr<CuImage>& image)
{
    auto direction = image->direction();
    return { { direction[0].begin(), direction[0].end() },
             { direction[1].begin(), direction[1].end() },
             { direction[2].begin(), direction[2].end() } };
}

std::string image_coord_sys(const std::shared_ptr<CuImage>& image)
{
    return image->coord_sys();
}

ResolutionData image_resolutions(const std::shared_ptr<CuImage>& image)
{
    auto resolutions = image->resolutions();
    ResolutionData result;
    result.level_count = resolutions.level_count();
    result.level_downsamples = resolutions.level_downsamples();
    result.level_dimensions.reserve(result.level_count);
    result.level_tile_sizes.reserve(result.level_count);
    for (uint16_t level = 0; level < result.level_count; ++level)
    {
        result.level_dimensions.emplace_back(resolutions.level_dimension(level));
        result.level_tile_sizes.emplace_back(resolutions.level_tile_size(level));
    }
    return result;
}

std::vector<std::string> image_associated_images(const std::shared_ptr<CuImage>& image)
{
    auto associated = image->associated_images();
    return { associated.begin(), associated.end() };
}

std::shared_ptr<CuImage> image_associated_image(const std::shared_ptr<CuImage>& image,
                                               const std::string& name,
                                               const std::string& device)
{
    return std::make_shared<CuImage>(image->associated_image(name, io::Device(device)));
}

std::shared_ptr<CuImage> image_read_region(const std::shared_ptr<CuImage>& image,
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
                                          const std::string& device)
{
    if (!size.empty() && size.size() != 2)
    {
        throw std::runtime_error("size (patch size) should be 2!");
    }
    if (dim_chars.size() != dim_values.size())
    {
        throw std::invalid_argument("dimension names and values must have the same length");
    }

    std::vector<std::pair<char, int64_t>> dimensions;
    dimensions.reserve(dim_chars.size());
    for (std::size_t index = 0; index < dim_chars.size(); ++index)
    {
        dimensions.emplace_back(static_cast<char>(dim_chars[index]), dim_values[index]);
    }
    DimIndices indices(dimensions);

    return std::make_shared<CuImage>(image->read_region(std::move(location),
                                                       std::move(size),
                                                       level,
                                                       num_workers,
                                                       batch_size,
                                                       drop_last,
                                                       prefetch_factor,
                                                       shuffle,
                                                       seed,
                                                       indices,
                                                       io::Device(device),
                                                       nullptr,
                                                       ""));
}

bool image_has_multiple_batches(const std::shared_ptr<CuImage>& image, uint32_t batch_size)
{
    auto loader = image->loader();
    return batch_size > 1 || (loader && loader->size() > 1);
}

ArrayInterfaceData image_array_interface(const std::shared_ptr<CuImage>& image)
{
    auto loader = image->loader();
    memory::DLTContainer container = image->container();
    DLTensor* tensor = static_cast<DLTensor*>(container);
    if (!tensor)
    {
        return { false, false, 0, 0, "", {} };
    }
    if (loader)
    {
        tensor->data = loader->data();
    }
    return { true,
             tensor->data != nullptr,
             reinterpret_cast<uintptr_t>(tensor->data),
             static_cast<int>(tensor->device.device_type),
             tensor->data ? container.numpy_dtype() : "",
             image->shape() };
}

void image_save(const std::shared_ptr<CuImage>& image, const std::string& file_path)
{
    image->save(file_path);
}

void image_close(const std::shared_ptr<CuImage>& image)
{
    image->close();
}

bool image_bool(const std::shared_ptr<CuImage>& image)
{
    return static_cast<bool>(*image);
}

bool image_is_trace_enabled()
{
    return CuImage::is_trace_enabled();
}

CacheConfigData cache_default_config()
{
    return to_cache_config_data(CuImage::get_config()->cache());
}

int cache_type_from_name(const std::string& name)
{
    return static_cast<int>(cache::lookup_cache_type(name));
}

std::shared_ptr<cache::ImageCache> cache_default()
{
    return CuImage::cache();
}

std::shared_ptr<cache::ImageCache> cache_create(const CacheConfigData& data)
{
    auto config = to_cache_config(data);
    return CuImage::cache(config);
}

CacheConfigData cache_config(const std::shared_ptr<cache::ImageCache>& image_cache)
{
    return to_cache_config_data(image_cache->get_config());
}

int cache_type(const std::shared_ptr<cache::ImageCache>& image_cache)
{
    return static_cast<int>(image_cache->type());
}

uint32_t cache_size(const std::shared_ptr<cache::ImageCache>& image_cache)
{
    return image_cache->size();
}

uint64_t cache_memory_size(const std::shared_ptr<cache::ImageCache>& image_cache)
{
    return image_cache->memory_size();
}

uint32_t cache_capacity(const std::shared_ptr<cache::ImageCache>& image_cache)
{
    return image_cache->capacity();
}

uint64_t cache_memory_capacity(const std::shared_ptr<cache::ImageCache>& image_cache)
{
    return image_cache->memory_capacity();
}

uint64_t cache_free_memory(const std::shared_ptr<cache::ImageCache>& image_cache)
{
    return image_cache->free_memory();
}

bool cache_record_get(const std::shared_ptr<cache::ImageCache>& image_cache)
{
    return image_cache->record();
}

void cache_record_set(const std::shared_ptr<cache::ImageCache>& image_cache, bool value)
{
    image_cache->record(value);
}

uint64_t cache_hit_count(const std::shared_ptr<cache::ImageCache>& image_cache)
{
    return image_cache->hit_count();
}

uint64_t cache_miss_count(const std::shared_ptr<cache::ImageCache>& image_cache)
{
    return image_cache->miss_count();
}

void cache_reserve(const std::shared_ptr<cache::ImageCache>& image_cache,
                   uint32_t memory_capacity,
                   bool has_capacity,
                   uint32_t capacity)
{
    auto config = CuImage::get_config()->cache();
    config.memory_capacity = memory_capacity;
    config.capacity = has_capacity ? capacity : cache::calc_default_cache_capacity(cache::kOneMiB * memory_capacity);
    image_cache->reserve(config);
}

uint32_t preferred_memory_capacity_for_image(const std::shared_ptr<CuImage>& image,
                                             const std::vector<uint32_t>& patch_size,
                                             uint32_t bytes_per_pixel)
{
    std::vector<uint64_t> image_size;
    auto signed_image_size = image->size("XY");
    image_size.insert(image_size.end(), signed_image_size.begin(), signed_image_size.end());

    auto tile_size = image->resolutions().level_tile_size(0);
    auto dimensions = image->dims();
    std::size_t pivot = std::max(dimensions.rfind('X'), dimensions.rfind('Y'));
    if (pivot == std::string::npos)
    {
        bytes_per_pixel = 3;
    }
    else if (pivot < dimensions.size())
    {
        auto trailing_size = image->size(&dimensions.c_str()[pivot + 1]);
        int64_t item_count = 1;
        for (auto value : trailing_size)
        {
            item_count *= value;
        }
        bytes_per_pixel = (image->dtype().bits * item_count + 7) / 8;
    }

    auto actual_patch_size = patch_size;
    if (actual_patch_size.size() != 2)
    {
        actual_patch_size = { cache::kDefaultPatchSize, cache::kDefaultPatchSize };
    }
    return cache::preferred_memory_capacity(image_size, tile_size, actual_patch_size, bytes_per_pixel);
}

uint32_t preferred_memory_capacity_explicit(const std::vector<uint64_t>& image_size,
                                            const std::vector<uint32_t>& tile_size,
                                            const std::vector<uint32_t>& patch_size,
                                            uint32_t bytes_per_pixel)
{
    auto actual_tile_size = tile_size;
    auto actual_patch_size = patch_size;
    if (image_size.size() != 2)
    {
        throw std::invalid_argument(
            "Please specify 'image_size' parameter (e.g., 'image_size=(100000, 100000)')!");
    }
    if (actual_tile_size.size() != 2)
    {
        actual_tile_size = { cache::kDefaultTileSize, cache::kDefaultTileSize };
    }
    if (actual_patch_size.size() != 2)
    {
        actual_patch_size = { cache::kDefaultPatchSize, cache::kDefaultPatchSize };
    }
    return cache::preferred_memory_capacity(image_size, actual_tile_size, actual_patch_size, bytes_per_pixel);
}

std::shared_ptr<profiler::Profiler> profiler_default()
{
    return CuImage::profiler();
}

std::shared_ptr<profiler::Profiler> profiler_create(bool trace)
{
    auto config = CuImage::get_config()->profiler();
    config.trace = trace;
    return CuImage::profiler(config);
}

bool profiler_trace_get(const std::shared_ptr<profiler::Profiler>& image_profiler)
{
    return image_profiler->trace();
}

void profiler_trace_set(const std::shared_ptr<profiler::Profiler>& image_profiler, bool value)
{
    image_profiler->trace(value);
}

std::shared_ptr<filesystem::CuFileDriver> file_driver_from_fd(int fd,
                                                             bool no_gds,
                                                             bool use_mmap,
                                                             const std::string& file_path)
{
    return std::make_shared<filesystem::CuFileDriver>(fd, no_gds, use_mmap, file_path.c_str());
}

std::shared_ptr<filesystem::CuFileDriver> filesystem_open(const std::string& file_path,
                                                         const std::string& flags,
                                                         uint32_t mode)
{
    return filesystem::open(file_path.c_str(), flags.c_str(), mode);
}

bool filesystem_is_gds_available()
{
    return filesystem::is_gds_available();
}

bool filesystem_close(const std::shared_ptr<filesystem::CuFileDriver>& file)
{
    return filesystem::close(file);
}

bool filesystem_discard_page_cache(const std::string& file_path)
{
    return filesystem::discard_page_cache(file_path.c_str());
}

std::string file_driver_path(const std::shared_ptr<filesystem::CuFileDriver>& file)
{
    return file->path();
}

int64_t file_driver_pread(const std::shared_ptr<filesystem::CuFileDriver>& file,
                         uintptr_t buffer,
                         uint64_t count,
                         int64_t file_offset,
                         int64_t buffer_offset)
{
    return file->pread(reinterpret_cast<void*>(buffer), count, file_offset, buffer_offset);
}

int64_t file_driver_pwrite(const std::shared_ptr<filesystem::CuFileDriver>& file,
                          uintptr_t buffer,
                          uint64_t count,
                          int64_t file_offset,
                          int64_t buffer_offset)
{
    return file->pwrite(reinterpret_cast<const void*>(buffer), count, file_offset, buffer_offset);
}

} // namespace cucim::python
