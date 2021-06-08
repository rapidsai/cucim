/*
 * Copyright (c) 2021, NVIDIA CORPORATcacheN.
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

#ifndef PYCUCIM_CACHE_INIT_H
#define PYCUCIM_CACHE_INIT_H

#include <optional>
#include <vector>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace cucim::cache
{

// Forward declaration
class ImageCache;

void init_cache(py::module& m);


bool py_record(ImageCache& cache, py::object value);

py::dict py_config(ImageCache& cache);

void py_image_cache_reserve(ImageCache& cache, uint32_t memory_capacity, py::kwargs kwargs);

py::int_ py_preferred_memory_capacity(const py::object& img,
                                      const std::optional<const std::vector<uint32_t>>& image_size,
                                      const std::optional<const std::vector<uint32_t>>& tile_size,
                                      const std::optional<const std::vector<uint32_t>>& patch_size,
                                      uint32_t bytes_per_pixel);

} // namespace cucim::cache


#endif // PYCUCIM_CACHE_INIT_H
