/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <pybind11_json/pybind11_json.hpp>
#include <nlohmann/json.hpp>
#include <vector>

#include <cucim/cuimage.h>

using json = nlohmann::json;

namespace cucim
{

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

json py_metadata(const CuImage& cuimg);
py::dict py_resolutions(const CuImage& cuimg);
CuImage py_read_region(CuImage& cuimg,
                    std::vector<int64_t> location,
                    std::vector<int64_t> size,
                    int16_t level,
                    io::Device device,
                    py::object buf,
                    const std::string& shm_name,
                    py::kwargs kwargs);
py::dict get_array_interface(const CuImage& cuimg);
} // namespace cucim

#endif // PYCUCIM_CUIMAGE_PY_H
