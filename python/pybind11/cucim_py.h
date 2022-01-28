/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#ifndef PYCUCIM_CUIMAGE_PY_H
#define PYCUCIM_CUIMAGE_PY_H

#include <vector>

#include <nlohmann/json.hpp>
#include <pybind11_json/pybind11_json.hpp>

using json = nlohmann::json;

namespace cucim
{

// Forward declarations
class CuImage;
template <typename DataType = CuImage>
class CuImageIterator;
namespace io
{
class Device;
}
namespace cache
{
class ImageCache;
}
namespace profiler
{
class Profiler;
}

std::string get_plugin_root();
void set_plugin_root(std::string path);

/**
 * Converts an object with std::vector type to one with pybind11::tuple type.
 *
 * The code is derived from `make_tuple()` method in pybind11/cast.h which is under BSD-3-Clause License.
 * Please see LICENSE-3rdparty.md for the detail.
 * (https://github.com/pybind/pybind11/blob/993495c96c869c5d3f3266c3ed3b1b8439340fd2/include/pybind11/cast.h#L1817)
 *
 * @tparam PT Python type
 * @tparam T Vector type
 * @param vec A vector object to convert
 * @return An object of pybind11::tuple type to which `vec` is converted
 */
template<typename PT, typename T>
pybind11::tuple vector2pytuple(const std::vector<T>& vec);

std::shared_ptr<cucim::cache::ImageCache> py_cache(const py::object& ctype, const py::kwargs& kwargs);
std::shared_ptr<cucim::profiler::Profiler> py_profiler(const py::kwargs& kwargs);
bool py_is_trace_enabled(py::object /* self */);

json py_metadata(const CuImage& cuimg);
py::dict py_resolutions(const CuImage& cuimg);
py::object py_read_region(const CuImage& cuimg,
                          const py::iterable& location,
                          std::vector<int64_t>&& size,
                          int16_t level,
                          uint32_t num_workers,
                          uint32_t batch_size,
                          bool drop_last,
                          uint32_t prefetch_factor,
                          bool shuffle,
                          uint64_t seed,
                          const io::Device& device,
                          const py::object& buf,
                          const std::string& shm_name,
                          const py::kwargs& kwargs);
py::object py_associated_image(const CuImage& cuimg, const std::string& name, const io::Device& device);

py::object py_cuimage_iterator_next(CuImageIterator<CuImage>& it);

void _set_array_interface(const py::object& cuimg_obj);
} // namespace cucim

#endif // PYCUCIM_CUIMAGE_PY_H
