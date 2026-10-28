/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "segmented_radix_sort.h"

#include <cub/device/device_segmented_radix_sort.cuh>
#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace cucim::skimage
{

namespace
{

using SegmentOffset = unsigned long long;

void check_cuda(cudaError_t status, const char* context)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
    }
}

void check_sizes(std::int64_t num_items, std::int64_t num_segments)
{
    if (num_items < 0)
    {
        throw std::invalid_argument("num_items must be non-negative");
    }
    if (num_segments < 0)
    {
        throw std::invalid_argument("num_segments must be non-negative");
    }
}

} // namespace

template <typename T>
std::size_t segmented_radix_sort_keys_temp_storage_bytes(std::int64_t num_items, std::int64_t num_segments)
{
    check_sizes(num_items, num_segments);
    // With a null temp_storage pointer, SortKeys only computes the required
    // size on the host; none of the data pointers are dereferenced.
    std::size_t temp_storage_bytes = 0;
    check_cuda(cub::DeviceSegmentedRadixSort::SortKeys(nullptr, temp_storage_bytes, static_cast<const T*>(nullptr),
                                                       static_cast<T*>(nullptr), num_items, num_segments,
                                                       static_cast<const SegmentOffset*>(nullptr),
                                                       static_cast<const SegmentOffset*>(nullptr), 0, sizeof(T) * 8),
               "cub::DeviceSegmentedRadixSort::SortKeys size query failed");
    return temp_storage_bytes;
}

template <typename T>
void segmented_radix_sort_keys_ranges(std::uintptr_t keys_in,
                                      std::uintptr_t keys_out,
                                      std::uintptr_t begin_offsets,
                                      std::uintptr_t end_offsets,
                                      std::int64_t num_items,
                                      std::int64_t num_segments,
                                      std::uintptr_t temp_storage,
                                      std::size_t temp_storage_bytes,
                                      std::uintptr_t stream_ptr)
{
    check_sizes(num_items, num_segments);
    if (num_items == 0 || num_segments == 0)
    {
        return;
    }
    // A null temp_storage would turn the call below into a silent size query.
    if (temp_storage == 0)
    {
        throw std::invalid_argument("temp_storage must be a non-null device pointer");
    }
    check_cuda(cub::DeviceSegmentedRadixSort::SortKeys(reinterpret_cast<void*>(temp_storage), temp_storage_bytes,
                                                       reinterpret_cast<const T*>(keys_in),
                                                       reinterpret_cast<T*>(keys_out), num_items, num_segments,
                                                       reinterpret_cast<const SegmentOffset*>(begin_offsets),
                                                       reinterpret_cast<const SegmentOffset*>(end_offsets), 0,
                                                       sizeof(T) * 8, reinterpret_cast<cudaStream_t>(stream_ptr)),
               "cub::DeviceSegmentedRadixSort::SortKeys failed");
}

template <typename T>
void segmented_radix_sort_keys(std::uintptr_t keys_in,
                               std::uintptr_t keys_out,
                               std::uintptr_t segment_offsets,
                               std::int64_t num_items,
                               std::int64_t num_segments,
                               std::uintptr_t temp_storage,
                               std::size_t temp_storage_bytes,
                               std::uintptr_t stream_ptr)
{
    segmented_radix_sort_keys_ranges<T>(keys_in, keys_out, segment_offsets, segment_offsets + sizeof(SegmentOffset),
                                        num_items, num_segments, temp_storage, temp_storage_bytes, stream_ptr);
}

#define CUCIM_INSTANTIATE_SEGMENTED_RADIX_SORT(T)                                                                      \
    template std::size_t segmented_radix_sort_keys_temp_storage_bytes<T>(std::int64_t, std::int64_t);                  \
    template void segmented_radix_sort_keys<T>(std::uintptr_t, std::uintptr_t, std::uintptr_t, std::int64_t,           \
                                               std::int64_t, std::uintptr_t, std::size_t, std::uintptr_t);             \
    template void segmented_radix_sort_keys_ranges<T>(std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uintptr_t,  \
                                                      std::int64_t, std::int64_t, std::uintptr_t, std::size_t,         \
                                                      std::uintptr_t);

CUCIM_INSTANTIATE_SEGMENTED_RADIX_SORT(std::uint8_t)
CUCIM_INSTANTIATE_SEGMENTED_RADIX_SORT(std::uint16_t)
CUCIM_INSTANTIATE_SEGMENTED_RADIX_SORT(float)
CUCIM_INSTANTIATE_SEGMENTED_RADIX_SORT(double)

#undef CUCIM_INSTANTIATE_SEGMENTED_RADIX_SORT

} // namespace cucim::skimage
