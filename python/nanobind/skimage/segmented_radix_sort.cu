/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cub/device/device_segmented_radix_sort.cuh>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace cucim::skimage
{

namespace
{

void check_cuda(cudaError_t status, const char* context)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
void segmented_radix_sort_keys_ranges_impl(std::uintptr_t keys_in,
                                           std::uintptr_t keys_out,
                                           std::uintptr_t begin_offsets,
                                           std::uintptr_t end_offsets,
                                           std::int64_t num_items,
                                           std::int64_t num_segments,
                                           std::uintptr_t stream_ptr)
{
    if (num_items < 0)
    {
        throw std::invalid_argument("num_items must be non-negative");
    }
    if (num_segments < 0)
    {
        throw std::invalid_argument("num_segments must be non-negative");
    }
    const T* d_keys_in = reinterpret_cast<const T*>(keys_in);
    T* d_keys_out = reinterpret_cast<T*>(keys_out);
    const unsigned long long* d_begin_offsets = reinterpret_cast<const unsigned long long*>(begin_offsets);
    const unsigned long long* d_end_offsets = reinterpret_cast<const unsigned long long*>(end_offsets);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    void* temp_storage = nullptr;
    std::size_t temp_storage_bytes = 0;
    check_cuda(
        cub::DeviceSegmentedRadixSort::SortKeys(temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items,
                                                num_segments, d_begin_offsets, d_end_offsets, 0, sizeof(T) * 8, stream),
        "cub::DeviceSegmentedRadixSort::SortKeys size query failed");
    if (temp_storage_bytes == 0)
    {
        return;
    }
    check_cuda(cudaMalloc(&temp_storage, temp_storage_bytes), "cudaMalloc failed");
    try
    {
        check_cuda(cub::DeviceSegmentedRadixSort::SortKeys(temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
                                                           num_items, num_segments, d_begin_offsets, d_end_offsets, 0,
                                                           sizeof(T) * 8, stream),
                   "cub::DeviceSegmentedRadixSort::SortKeys failed");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed");
    }
    catch (...)
    {
        cudaFree(temp_storage);
        throw;
    }
    check_cuda(cudaFree(temp_storage), "cudaFree failed");
}

template <typename T>
void segmented_radix_sort_keys_impl(std::uintptr_t keys_in,
                                    std::uintptr_t keys_out,
                                    std::uintptr_t segment_offsets,
                                    std::int64_t num_items,
                                    std::int64_t num_segments,
                                    std::uintptr_t stream_ptr)
{
    segmented_radix_sort_keys_ranges_impl<T>(keys_in, keys_out, segment_offsets,
                                             segment_offsets + sizeof(unsigned long long), num_items, num_segments,
                                             stream_ptr);
}

} // namespace

void segmented_radix_sort_keys_uint8(std::uintptr_t keys_in,
                                     std::uintptr_t keys_out,
                                     std::uintptr_t segment_offsets,
                                     std::int64_t num_items,
                                     std::int64_t num_segments,
                                     std::uintptr_t stream_ptr)
{
    segmented_radix_sort_keys_impl<std::uint8_t>(keys_in, keys_out, segment_offsets, num_items, num_segments, stream_ptr);
}

void segmented_radix_sort_keys_ranges_uint8(std::uintptr_t keys_in,
                                            std::uintptr_t keys_out,
                                            std::uintptr_t begin_offsets,
                                            std::uintptr_t end_offsets,
                                            std::int64_t num_items,
                                            std::int64_t num_segments,
                                            std::uintptr_t stream_ptr)
{
    segmented_radix_sort_keys_ranges_impl<std::uint8_t>(
        keys_in, keys_out, begin_offsets, end_offsets, num_items, num_segments, stream_ptr);
}

void segmented_radix_sort_keys_uint16(std::uintptr_t keys_in,
                                      std::uintptr_t keys_out,
                                      std::uintptr_t segment_offsets,
                                      std::int64_t num_items,
                                      std::int64_t num_segments,
                                      std::uintptr_t stream_ptr)
{
    segmented_radix_sort_keys_impl<std::uint16_t>(
        keys_in, keys_out, segment_offsets, num_items, num_segments, stream_ptr);
}

void segmented_radix_sort_keys_ranges_uint16(std::uintptr_t keys_in,
                                             std::uintptr_t keys_out,
                                             std::uintptr_t begin_offsets,
                                             std::uintptr_t end_offsets,
                                             std::int64_t num_items,
                                             std::int64_t num_segments,
                                             std::uintptr_t stream_ptr)
{
    segmented_radix_sort_keys_ranges_impl<std::uint16_t>(
        keys_in, keys_out, begin_offsets, end_offsets, num_items, num_segments, stream_ptr);
}

void segmented_radix_sort_keys_float32(std::uintptr_t keys_in,
                                       std::uintptr_t keys_out,
                                       std::uintptr_t segment_offsets,
                                       std::int64_t num_items,
                                       std::int64_t num_segments,
                                       std::uintptr_t stream_ptr)
{
    segmented_radix_sort_keys_impl<float>(keys_in, keys_out, segment_offsets, num_items, num_segments, stream_ptr);
}

void segmented_radix_sort_keys_ranges_float32(std::uintptr_t keys_in,
                                              std::uintptr_t keys_out,
                                              std::uintptr_t begin_offsets,
                                              std::uintptr_t end_offsets,
                                              std::int64_t num_items,
                                              std::int64_t num_segments,
                                              std::uintptr_t stream_ptr)
{
    segmented_radix_sort_keys_ranges_impl<float>(
        keys_in, keys_out, begin_offsets, end_offsets, num_items, num_segments, stream_ptr);
}

void segmented_radix_sort_keys_float64(std::uintptr_t keys_in,
                                       std::uintptr_t keys_out,
                                       std::uintptr_t segment_offsets,
                                       std::int64_t num_items,
                                       std::int64_t num_segments,
                                       std::uintptr_t stream_ptr)
{
    segmented_radix_sort_keys_impl<double>(keys_in, keys_out, segment_offsets, num_items, num_segments, stream_ptr);
}

void segmented_radix_sort_keys_ranges_float64(std::uintptr_t keys_in,
                                              std::uintptr_t keys_out,
                                              std::uintptr_t begin_offsets,
                                              std::uintptr_t end_offsets,
                                              std::int64_t num_items,
                                              std::int64_t num_segments,
                                              std::uintptr_t stream_ptr)
{
    segmented_radix_sort_keys_ranges_impl<double>(
        keys_in, keys_out, begin_offsets, end_offsets, num_items, num_segments, stream_ptr);
}

} // namespace cucim::skimage
