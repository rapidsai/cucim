/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cache_py.h"
#include "cache_pydoc.h"

#include <fmt/format.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <cucim/cache/image_cache.h>
#include <cucim/cuimage.h>

#include "../compat.h"
#include "image_cache_pydoc.h"

using namespace nanobind::literals;

namespace cucim::cache
{

void init_cache(nb::module_ cache)
{
    bind_enum<CacheType>(cache, "CacheType") //
        .value("NoCache", CacheType::kNoCache) //
        .value("PerProcess", CacheType::kPerProcess) //
        .value("SharedMemory", CacheType::kSharedMemory);

    nb::class_<ImageCache>(cache, "ImageCache", nb::is_weak_referenceable())
        .def_prop_ro("type", &ImageCache::type, doc::ImageCache::doc_type, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("config", &py_config, doc::ImageCache::doc_config)
        .def_prop_ro("size", &ImageCache::size, doc::ImageCache::doc_size, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("memory_size", &ImageCache::memory_size, doc::ImageCache::doc_memory_size,
                     nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro(
            "capacity", &ImageCache::capacity, doc::ImageCache::doc_capacity, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("memory_capacity", &ImageCache::memory_capacity, doc::ImageCache::doc_memory_capacity,
                     nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("free_memory", &ImageCache::free_memory, doc::ImageCache::doc_free_memory,
                     nb::call_guard<nb::gil_scoped_release>())
        .def("record", &py_record, doc::ImageCache::doc_record, //
             "value"_a = nb::none())
        .def_prop_ro("hit_count", &ImageCache::hit_count, doc::ImageCache::doc_hit_count,
                     nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("miss_count", &ImageCache::miss_count, doc::ImageCache::doc_miss_count,
                     nb::call_guard<nb::gil_scoped_release>())
        .def("reserve", &py_image_cache_reserve, doc::ImageCache::doc_reserve, //
             "memory_capacity"_a, "kwargs"_a);

    cache.def("preferred_memory_capacity", &py_preferred_memory_capacity, doc::doc_preferred_memory_capacity, //
              "img"_a = nb::none(), //
              "image_size"_a = nb::none(), //
              "tile_size"_a = nb::none(), //
              "patch_size"_a = nb::none(), //
              "bytes_per_pixel"_a = 3);
}

bool py_record(ImageCache& cache, nb::handle value)
{
    if (value.is_none())
    {
        return cache.record();
    }
    else if (nb::isinstance<nb::bool_>(value))
    {
        bool v = nb::cast<bool>(value);
        cache.record(v);
        return v;
    }
    else
    {
        throw std::invalid_argument(fmt::format("Only 'NoneType' or 'bool' is available for the argument"));
    }
}

nb::dict py_config(ImageCache& cache)
{
    ImageCacheConfig& config = cache.config();
    const std::string_view type_str = lookup_cache_type_str(config.type);

    nb::dict result;
    result["type"] = nb::str(type_str.data(), type_str.size());
    result["memory_capacity"] = config.memory_capacity;
    result["capacity"] = config.capacity;
    result["mutex_pool_capacity"] = config.mutex_pool_capacity;
    result["list_padding"] = config.list_padding;
    result["extra_shared_memory_size"] = config.extra_shared_memory_size;
    result["record_stat"] = config.record_stat;
    return result;
}

void py_image_cache_reserve(ImageCache& cache, uint32_t memory_capacity, nb::kwargs kwargs)
{
    cucim::cache::ImageCacheConfig config = cucim::CuImage::get_config()->cache();
    config.memory_capacity = memory_capacity;

    if (kwargs.contains("capacity"))
    {
        config.capacity = cast_or_runtime_error<uint32_t>(kwargs["capacity"]);
    }
    else
    {
        // Update capacity depends on memory_capacity.
        config.capacity = calc_default_cache_capacity(kOneMiB * memory_capacity);
    }

    nb::gil_scoped_release release;
    cache.reserve(config);
}

uint32_t py_preferred_memory_capacity(nb::handle img,
                                      const std::optional<std::vector<uint64_t>>& image_size,
                                      const std::optional<std::vector<uint32_t>>& tile_size,
                                      const std::optional<std::vector<uint32_t>>& patch_size,
                                      uint32_t bytes_per_pixel)
{
    std::vector<uint64_t> param_image;
    std::vector<uint32_t> param_tile;
    std::vector<uint32_t> param_patch;

    if (!img.is_none())
    {
        const CuImage& cuimg = *cast_or_runtime_error<cucim::CuImage*>(img);
        std::vector<int64_t> image_size_vec = cuimg.size("XY");
        param_image.insert(param_image.end(), image_size_vec.begin(), image_size_vec.end());
        std::vector<uint32_t> tile_size_vec = cuimg.resolutions().level_tile_size(0);
        param_tile.insert(param_tile.end(), tile_size_vec.begin(), tile_size_vec.end());

        // Calculate pixel size in bytes
        // (For example, if axes == "YXC" or "YXS", calculate [bytes per pixel] * [# items inside dims after 'X'] )
        std::string dims = cuimg.dims();
        std::size_t pivot = std::max(dims.rfind('X'), dims.rfind('Y'));
        if (pivot == std::string::npos)
        {
            bytes_per_pixel = 3;
        }
        else
        {
            if (pivot < dims.size())
            {
                std::vector<int64_t> size_vec = cuimg.size(&dims.c_str()[pivot + 1]);
                int64_t item_count = 1;
                for (auto size : size_vec)
                {
                    item_count *= size;
                }
                bytes_per_pixel = (cuimg.dtype().bits * item_count + 7) / 8;
            }
        }
    }
    else
    {
        if (!image_size || image_size->size() != 2)
        {
            throw std::invalid_argument(
                fmt::format("Please specify 'image_size' parameter (e.g., 'image_size=(100000, 100000)')!"));
        }
        if (!tile_size || tile_size->size() != 2)
        {
            param_tile = { kDefaultTileSize, kDefaultTileSize };
        }
    }

    if (!patch_size || patch_size->size() != 2)
    {
        param_patch = { kDefaultPatchSize, kDefaultPatchSize };
    }

    return preferred_memory_capacity(!param_image.empty() ? param_image : image_size.value(), //
                                     !param_tile.empty() ? param_tile : tile_size.value(), //
                                     !param_patch.empty() ? param_patch : patch_size.value(), //
                                     bytes_per_pixel);
}

} // namespace cucim::cache
