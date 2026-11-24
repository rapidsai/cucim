/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "segmented_sort.h"

#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nanobind::literals;

namespace
{

template <typename T>
void def_segmented_sort(nb::module_& m, const char* temp_storage_bytes_name, const char* ranges_name)
{
    m.def(temp_storage_bytes_name, &cucim::skimage::segmented_sort_keys_temp_storage_bytes<T>, "num_items"_a,
          "num_segments"_a,
          "Return the size in bytes of the temporary device storage required by the matching "
          "segmented_sort_keys functions.");
    m.def(ranges_name, &cucim::skimage::segmented_sort_keys_ranges<T>, "keys_in"_a, "keys_out"_a, "begin_offsets"_a,
          "end_offsets"_a, "num_items"_a, "num_segments"_a, "temp_storage"_a, "temp_storage_bytes"_a, "stream_ptr"_a = 0,
          "Sort selected key ranges using cub::DeviceSegmentedSort.\n\n"
          "This private binding accepts raw device pointers. The caller must provide non-overlapping buffers "
          "on the current device: num_items keys of the named dtype in each key buffer, num_segments "
          "uint64 offsets in each offset buffer, and the queried number of workspace bytes. Each range must "
          "lie within the key buffers. All buffers must remain valid until the work on stream_ptr completes.");
}

} // namespace

NB_MODULE(_cucim_skimage_cpp_ext, m)
{
    m.doc() = "Optional compiled helpers for cucim.skimage.";
    def_segmented_sort<std::uint8_t>(
        m, "segmented_sort_keys_temp_storage_bytes_uint8", "segmented_sort_keys_ranges_uint8");
    def_segmented_sort<std::uint16_t>(
        m, "segmented_sort_keys_temp_storage_bytes_uint16", "segmented_sort_keys_ranges_uint16");
    def_segmented_sort<float>(m, "segmented_sort_keys_temp_storage_bytes_float32", "segmented_sort_keys_ranges_float32");
    def_segmented_sort<double>(m, "segmented_sort_keys_temp_storage_bytes_float64", "segmented_sort_keys_ranges_float64");
}
