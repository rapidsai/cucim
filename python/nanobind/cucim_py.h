/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYCUCIM_CUIMAGE_PY_H
#define PYCUCIM_CUIMAGE_PY_H

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>

#include "compat.h"

namespace nb = nanobind;

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
void set_plugin_root(const std::filesystem::path& path);

std::shared_ptr<cucim::cache::ImageCache> py_cache(nb::handle type, nb::kwargs kwargs);
std::shared_ptr<cucim::profiler::Profiler> py_profiler(nb::kwargs kwargs);

nb::object py_metadata(const CuImage& cuimg);
nb::dict py_resolutions(const CuImage& cuimg);
nb::object py_read_region(const CuImage& cuimg,
                          nb::iterable location,
                          std::vector<int64_t> size,
                          int16_t level,
                          uint32_t num_workers,
                          uint32_t batch_size,
                          Bool drop_last,
                          uint32_t prefetch_factor,
                          Bool shuffle,
                          uint64_t seed,
                          const io::Device& device,
                          nb::handle buf,
                          const std::string& shm_name,
                          nb::kwargs kwargs);
nb::object py_associated_image(const CuImage& cuimg, const std::string& name, const io::Device& device);

nb::object py_cuimage_iterator_next(CuImageIterator<CuImage>& it);

void _set_array_interface(nb::handle cuimg_obj);
} // namespace cucim

#endif // PYCUCIM_CUIMAGE_PY_H
