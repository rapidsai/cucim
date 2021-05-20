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

#include "cache_py.h"
#include "cache_pydoc.h"

#include <pybind11/stl.h>

#include <cucim/cache/image_cache.h>
#include <cucim/cuimage.h>

#include "image_cache_py.h"
#include "image_cache_pydoc.h"

using namespace pybind11::literals;
namespace py = pybind11;

namespace cucim::cache
{

void init_cache(py::module& cache)
{
    py::enum_<CacheType>(cache, "CacheType") //
        .value("NoCache", CacheType::kNoCache) //
        .value("PerProcess", CacheType::kPerProcess) //
        .value("SharedMemory", CacheType::kSharedMemory);

    py::class_<ImageCache, std::shared_ptr<ImageCache>>(cache, "ImageCache")
        .def_property(
            "type", &ImageCache::type, nullptr, doc::ImageCache::doc_type, py::call_guard<py::gil_scoped_release>())
        .def_property("config", &py_config, nullptr, doc::ImageCache::doc_type, py::call_guard<py::gil_scoped_release>())
        .def_property(
            "size", &ImageCache::size, nullptr, doc::ImageCache::doc_size, py::call_guard<py::gil_scoped_release>())
        .def_property("memory_size", &ImageCache::memory_size, nullptr, doc::ImageCache::doc_memory_size,
                      py::call_guard<py::gil_scoped_release>())
        .def_property("capacity", &ImageCache::capacity, nullptr, doc::ImageCache::doc_capacity,
                      py::call_guard<py::gil_scoped_release>())
        .def_property("memory_capacity", &ImageCache::memory_capacity, nullptr, doc::ImageCache::doc_memory_capacity,
                      py::call_guard<py::gil_scoped_release>())
        .def_property("free_memory", &ImageCache::free_memory, nullptr, doc::ImageCache::doc_free_memory,
                      py::call_guard<py::gil_scoped_release>())
        .def("record", &py_record, doc::ImageCache::doc_record, py::call_guard<py::gil_scoped_release>(), //
             py::arg("value") = py::none())
        .def_property("hit_count", &ImageCache::hit_count, nullptr, doc::ImageCache::doc_hit_count,
                      py::call_guard<py::gil_scoped_release>())
        .def_property("miss_count", &ImageCache::miss_count, nullptr, doc::ImageCache::doc_miss_count,
                      py::call_guard<py::gil_scoped_release>())
        .def("reserve", &py_image_cache_reserve, doc::ImageCache::doc_reserve, py::call_guard<py::gil_scoped_release>(), //
             py::arg("memory_capacity"));

    cache.def("preferred_memory_capacity", &py_preferred_memory_capacity, doc::doc_preferred_memory_capacity, //
              py::arg("img") = py::none(), //
              py::arg("image_size") = std::nullopt, //
              py::arg("tile_size") = std::nullopt, //
              py::arg("patch_size") = std::nullopt, //
              py::arg("bytes_per_pixel") = 3, //
              py::call_guard<py::gil_scoped_release>());
}

bool py_record(ImageCache& cache, py::object value)
{
    if (value.is_none())
    {
        return cache.record();
    }
    else if (py::isinstance<py::bool_>(value))
    {
        py::bool_ v = value.cast<py::bool_>();
        cache.record(v);
        return v;
    }
    else
    {
        throw std::invalid_argument(fmt::format("Only 'NoneType' or 'bool' is available for the argument"));
    }
}

py::dict py_config(ImageCache& cache)
{
    ImageCacheConfig& config = cache.config();

    return py::dict{
        "type"_a = pybind11::str(std::string(lookup_cache_type_str(config.type))), //
        "memory_capacity"_a = pybind11::int_(config.memory_capacity), //
        "capacity"_a = pybind11::int_(config.capacity), //
        "mutex_pool_capacity"_a = pybind11::int_(config.mutex_pool_capacity), //
        "list_padding"_a = pybind11::int_(config.list_padding), //
        "extra_shared_memory_size"_a = pybind11::int_(config.extra_shared_memory_size), //
        "record_stat"_a = pybind11::bool_(config.record_stat) //
    };
}

void py_image_cache_reserve(ImageCache& cache, uint32_t memory_capacity, py::kwargs kwargs)
{
    cucim::cache::ImageCacheConfig config = cucim::CuImage::get_config()->cache();
    config.memory_capacity = memory_capacity;

    if (kwargs.contains("capacity"))
    {
        config.capacity = py::cast<uint32_t>(kwargs["capacity"]);
    }
    else
    {
        // Update capacity depends on memory_capacity.
        config.capacity = calc_default_cache_capacity(kOneMiB * memory_capacity);
    }

    cache.reserve(config);
}

py::int_ py_preferred_memory_capacity(const py::object& img,
                                      const std::optional<const std::vector<uint32_t>>& image_size,
                                      const std::optional<const std::vector<uint32_t>>& tile_size,
                                      const std::optional<const std::vector<uint32_t>>& patch_size,
                                      uint32_t bytes_per_pixel)
{
    std::vector<uint32_t> param_image;
    std::vector<uint32_t> param_tile;
    std::vector<uint32_t> param_patch;

    if (!img.is_none())
    {
        const CuImage& cuimg = *img.cast<cucim::CuImage*>();
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