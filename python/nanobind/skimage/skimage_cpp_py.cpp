/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <nanobind/nanobind.h>

#include <cstddef>
#include <cstdint>

using namespace nanobind::literals;

namespace cucim::skimage
{

void segmented_radix_sort_keys_float32(std::uintptr_t keys_in,
                                       std::uintptr_t keys_out,
                                       std::uintptr_t segment_offsets,
                                       std::int64_t num_items,
                                       std::int64_t num_segments,
                                       std::uintptr_t stream_ptr);
void segmented_radix_sort_keys_ranges_float32(std::uintptr_t keys_in,
                                              std::uintptr_t keys_out,
                                              std::uintptr_t begin_offsets,
                                              std::uintptr_t end_offsets,
                                              std::int64_t num_items,
                                              std::int64_t num_segments,
                                              std::uintptr_t stream_ptr);

void segmented_radix_sort_keys_float64(std::uintptr_t keys_in,
                                       std::uintptr_t keys_out,
                                       std::uintptr_t segment_offsets,
                                       std::int64_t num_items,
                                       std::int64_t num_segments,
                                       std::uintptr_t stream_ptr);
void segmented_radix_sort_keys_ranges_float64(std::uintptr_t keys_in,
                                              std::uintptr_t keys_out,
                                              std::uintptr_t begin_offsets,
                                              std::uintptr_t end_offsets,
                                              std::int64_t num_items,
                                              std::int64_t num_segments,
                                              std::uintptr_t stream_ptr);

} // namespace cucim::skimage

NB_MODULE(_cucim_skimage_cpp_ext, m)
{
    m.doc() = "Optional compiled helpers for cucim.skimage.";
    m.def("segmented_radix_sort_keys_float32", &cucim::skimage::segmented_radix_sort_keys_float32,
          "keys_in"_a,
          "keys_out"_a,
          "segment_offsets"_a,
          "num_items"_a,
          "num_segments"_a,
          "stream_ptr"_a = 0,
          "Sort float32 keys independently within each segment using cub::DeviceSegmentedRadixSort.");
    m.def("segmented_radix_sort_keys_float64", &cucim::skimage::segmented_radix_sort_keys_float64,
          "keys_in"_a,
          "keys_out"_a,
          "segment_offsets"_a,
          "num_items"_a,
          "num_segments"_a,
          "stream_ptr"_a = 0,
          "Sort float64 keys independently within each segment using cub::DeviceSegmentedRadixSort.");
    m.def("segmented_radix_sort_keys_ranges_float32", &cucim::skimage::segmented_radix_sort_keys_ranges_float32,
          "keys_in"_a,
          "keys_out"_a,
          "begin_offsets"_a,
          "end_offsets"_a,
          "num_items"_a,
          "num_segments"_a,
          "stream_ptr"_a = 0,
          "Sort selected float32 key ranges using cub::DeviceSegmentedRadixSort.");
    m.def("segmented_radix_sort_keys_ranges_float64", &cucim::skimage::segmented_radix_sort_keys_ranges_float64,
          "keys_in"_a,
          "keys_out"_a,
          "begin_offsets"_a,
          "end_offsets"_a,
          "num_items"_a,
          "num_segments"_a,
          "stream_ptr"_a = 0,
          "Sort selected float64 key ranges using cub::DeviceSegmentedRadixSort.");
}
