/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

uint32_t py_preferred_memory_capacity(const py::object& img,
                                      const std::optional<const std::vector<uint64_t>>& image_size,
                                      const std::optional<const std::vector<uint32_t>>& tile_size,
                                      const std::optional<const std::vector<uint32_t>>& patch_size,
                                      uint32_t bytes_per_pixel);

} // namespace cucim::cache


#endif // PYCUCIM_CACHE_INIT_H
