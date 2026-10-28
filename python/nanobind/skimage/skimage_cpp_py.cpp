/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "segmented_radix_sort.h"

#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nanobind::literals;

namespace
{

template <typename T>
void def_segmented_radix_sort(nb::module_& m,
                              const char* temp_storage_bytes_name,
                              const char* sort_name,
                              const char* ranges_name)
{
    m.def(temp_storage_bytes_name, &cucim::skimage::segmented_radix_sort_keys_temp_storage_bytes<T>, "num_items"_a,
          "num_segments"_a,
          "Return the size in bytes of the temporary device storage required by the matching "
          "segmented_radix_sort_keys functions.");
    m.def(sort_name, &cucim::skimage::segmented_radix_sort_keys<T>, "keys_in"_a, "keys_out"_a, "segment_offsets"_a,
          "num_items"_a, "num_segments"_a, "temp_storage"_a, "temp_storage_bytes"_a, "stream_ptr"_a = 0,
          "Sort keys independently within each segment using cub::DeviceSegmentedRadixSort.\n\n"
          "The sort is enqueued asynchronously on ``stream_ptr``. ``temp_storage`` must provide at least the "
          "number of bytes reported by the matching temp_storage_bytes function.");
    m.def(ranges_name, &cucim::skimage::segmented_radix_sort_keys_ranges<T>, "keys_in"_a, "keys_out"_a, "begin_offsets"_a,
          "end_offsets"_a, "num_items"_a, "num_segments"_a, "temp_storage"_a, "temp_storage_bytes"_a, "stream_ptr"_a = 0,
          "Sort selected key ranges using cub::DeviceSegmentedRadixSort.\n\n"
          "The sort is enqueued asynchronously on ``stream_ptr``. ``temp_storage`` must provide at least the "
          "number of bytes reported by the matching temp_storage_bytes function.");
}

} // namespace

NB_MODULE(_cucim_skimage_cpp_ext, m)
{
    m.doc() = "Optional compiled helpers for cucim.skimage.";
    def_segmented_radix_sort<std::uint8_t>(m, "segmented_radix_sort_keys_temp_storage_bytes_uint8",
                                           "segmented_radix_sort_keys_uint8", "segmented_radix_sort_keys_ranges_uint8");
    def_segmented_radix_sort<std::uint16_t>(m, "segmented_radix_sort_keys_temp_storage_bytes_uint16",
                                            "segmented_radix_sort_keys_uint16",
                                            "segmented_radix_sort_keys_ranges_uint16");
    def_segmented_radix_sort<float>(m, "segmented_radix_sort_keys_temp_storage_bytes_float32",
                                    "segmented_radix_sort_keys_float32", "segmented_radix_sort_keys_ranges_float32");
    def_segmented_radix_sort<double>(m, "segmented_radix_sort_keys_temp_storage_bytes_float64",
                                     "segmented_radix_sort_keys_float64", "segmented_radix_sort_keys_ranges_float64");
}
