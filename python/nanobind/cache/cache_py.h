/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYCUCIM_CACHE_INIT_H
#define PYCUCIM_CACHE_INIT_H

#include <optional>
#include <vector>

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace cucim::cache
{

// Forward declaration
class ImageCache;

void init_cache(nb::module_ m);


bool py_record(ImageCache& cache, nb::handle value);

nb::dict py_config(ImageCache& cache);

void py_image_cache_reserve(ImageCache& cache, uint32_t memory_capacity, nb::kwargs kwargs);

uint32_t py_preferred_memory_capacity(nb::handle img,
                                      const std::optional<std::vector<uint64_t>>& image_size,
                                      const std::optional<std::vector<uint32_t>>& tile_size,
                                      const std::optional<std::vector<uint32_t>>& patch_size,
                                      uint32_t bytes_per_pixel);

} // namespace cucim::cache


#endif // PYCUCIM_CACHE_INIT_H
