/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYCUCIM_SKIMAGE_SEGMENTED_SORT_H
#define PYCUCIM_SKIMAGE_SEGMENTED_SORT_H

#include <cstddef>
#include <cstdint>

namespace cucim::skimage
{

// These templates are explicitly instantiated in segmented_sort.cu for
// std::uint8_t, std::uint16_t, float and double keys.

// Return the size in bytes of the temporary device storage that
// cub::DeviceSegmentedSort::SortKeys requires for the given problem size.
template <typename T>
std::size_t segmented_sort_keys_temp_storage_bytes(std::int64_t num_items, std::int64_t num_segments);

// Sort selected [begin, end) ranges asynchronously on stream_ptr. The caller
// supplies non-overlapping device buffers on the current device: num_items
// keys of type T in each key buffer, num_segments uint64_t offsets in each
// offset buffer, and the queried number of bytes of temporary storage. The
// ranges must be within [0, num_items) and the buffers must remain valid until
// work on the stream completes.
template <typename T>
void segmented_sort_keys_ranges(std::uintptr_t keys_in,
                                std::uintptr_t keys_out,
                                std::uintptr_t begin_offsets,
                                std::uintptr_t end_offsets,
                                std::int64_t num_items,
                                std::int64_t num_segments,
                                std::uintptr_t temp_storage,
                                std::size_t temp_storage_bytes,
                                std::uintptr_t stream_ptr);

} // namespace cucim::skimage

#endif // PYCUCIM_SKIMAGE_SEGMENTED_SORT_H
