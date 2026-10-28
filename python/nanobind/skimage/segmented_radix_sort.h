/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYCUCIM_SKIMAGE_SEGMENTED_RADIX_SORT_H
#define PYCUCIM_SKIMAGE_SEGMENTED_RADIX_SORT_H

#include <cstddef>
#include <cstdint>

namespace cucim::skimage
{

// These templates are explicitly instantiated in segmented_radix_sort.cu for
// std::uint8_t, std::uint16_t, float and double keys.

// Return the size in bytes of the temporary device storage that
// cub::DeviceSegmentedRadixSort::SortKeys requires for the given problem size.
template <typename T>
std::size_t segmented_radix_sort_keys_temp_storage_bytes(std::int64_t num_items, std::int64_t num_segments);

// Sort keys within the segments [segment_offsets[i], segment_offsets[i + 1]).
// The sort is enqueued on stream_ptr without synchronizing. temp_storage must
// point to at least segmented_radix_sort_keys_temp_storage_bytes<T>() bytes of
// device memory that stays valid until the sort completes.
template <typename T>
void segmented_radix_sort_keys(std::uintptr_t keys_in,
                               std::uintptr_t keys_out,
                               std::uintptr_t segment_offsets,
                               std::int64_t num_items,
                               std::int64_t num_segments,
                               std::uintptr_t temp_storage,
                               std::size_t temp_storage_bytes,
                               std::uintptr_t stream_ptr);

// Same as segmented_radix_sort_keys, but with independent begin/end offsets
// so that only selected ranges are sorted.
template <typename T>
void segmented_radix_sort_keys_ranges(std::uintptr_t keys_in,
                                      std::uintptr_t keys_out,
                                      std::uintptr_t begin_offsets,
                                      std::uintptr_t end_offsets,
                                      std::int64_t num_items,
                                      std::int64_t num_segments,
                                      std::uintptr_t temp_storage,
                                      std::size_t temp_storage_bytes,
                                      std::uintptr_t stream_ptr);

} // namespace cucim::skimage

#endif // PYCUCIM_SKIMAGE_SEGMENTED_RADIX_SORT_H
