/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cucim/memory/device_resources.h"

#include "cucim/profiler/nvtx3.h"

#include <cuda/devices>
#include <cuda/memory_pool>

#include <cuda_runtime.h>

#include <new>
#include <utility>

namespace cucim::memory
{

namespace
{

/// The default pool of whichever device is current. See
/// DeviceMemoryPoolResource for why this is resolved per call.
::cuda::device_memory_pool_ref current_device_pool()
{
    int device_id = 0;
    if (cudaGetDevice(&device_id) != cudaSuccess)
    {
        static_cast<void>(cudaGetLastError());
        throw std::bad_alloc();
    }
    return ::cuda::device_default_memory_pool(::cuda::device_ref{ device_id });
}

} // namespace

void* DeviceMemoryPoolResource::allocate_sync(std::size_t bytes, std::size_t alignment)
{
    PROF_SCOPED_RANGE(PROF_EVENT_P(cucim_malloc, bytes));

    return current_device_pool().allocate_sync(bytes, alignment);
}

void DeviceMemoryPoolResource::deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment)
{
    PROF_SCOPED_RANGE(PROF_EVENT(cucim_free));

    current_device_pool().deallocate_sync(ptr, bytes, alignment);
}

void* DeviceMemoryPoolResource::allocate(::cuda::stream_ref stream, std::size_t bytes, std::size_t alignment)
{
    PROF_SCOPED_RANGE(PROF_EVENT_P(cucim_malloc, bytes));

    return current_device_pool().allocate(stream, bytes, alignment);
}

void DeviceMemoryPoolResource::deallocate(::cuda::stream_ref stream,
                                          void* ptr,
                                          std::size_t bytes,
                                          std::size_t alignment)
{
    PROF_SCOPED_RANGE(PROF_EVENT(cucim_free));

    current_device_pool().deallocate(stream, ptr, bytes, alignment);
}

void* CudaMemoryResource::allocate_sync(std::size_t bytes, std::size_t /*alignment*/)
{
    PROF_SCOPED_RANGE(PROF_EVENT_P(cucim_malloc, bytes));

    // cudaMalloc already satisfies any alignment cuCIM asks for (256 bytes),
    // so the requested alignment needs no extra handling. A zero-byte request
    // still goes through, so that a logging or statistics resource wrapping
    // this one sees every call.
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, bytes) != cudaSuccess)
    {
        // Clear the sticky error so the next CUDA call is not blamed for this
        // failure, then report it the way a resource is expected to.
        static_cast<void>(cudaGetLastError());
        throw std::bad_alloc();
    }
    return ptr;
}

void CudaMemoryResource::deallocate_sync(void* ptr, std::size_t /*bytes*/, std::size_t /*alignment*/)
{
    PROF_SCOPED_RANGE(PROF_EVENT(cucim_free));

    static_cast<void>(cudaFree(ptr));
}

void* CudaMemoryResource::allocate(::cuda::stream_ref /*stream*/, std::size_t bytes, std::size_t alignment)
{
    // cudaMalloc synchronizes the device, so it is trivially ordered with
    // respect to any stream and the argument can be ignored.
    return allocate_sync(bytes, alignment);
}

void CudaMemoryResource::deallocate(::cuda::stream_ref /*stream*/,
                                    void* ptr,
                                    std::size_t bytes,
                                    std::size_t alignment)
{
    deallocate_sync(ptr, bytes, alignment);
}

DeviceResources::DeviceResources()
    : DeviceResources(any_device_resource{ DeviceMemoryPoolResource{} })
{
}

DeviceResources::DeviceResources(any_device_resource resource, ::cuda::stream_ref stream)
    : owned_(std::make_shared<any_device_resource>(std::move(resource))), resource_(*owned_), stream_(stream)
{
}

DeviceResources::DeviceResources(device_resource_ref resource, ::cuda::stream_ref stream)
    : owned_(nullptr), resource_(resource), stream_(stream)
{
}

void* DeviceResources::allocate(std::size_t bytes, std::size_t alignment)
{
    return resource_.allocate(stream_, bytes, alignment);
}

void* DeviceResources::allocate(::cuda::stream_ref stream, std::size_t bytes, std::size_t alignment)
{
    stream_ = stream;
    return resource_.allocate(stream_, bytes, alignment);
}

void DeviceResources::deallocate(void* ptr, std::size_t bytes, std::size_t alignment) noexcept
{
    resource_.deallocate(stream_, ptr, bytes, alignment);
}

void DeviceResources::deallocate(::cuda::stream_ref stream,
                                 void* ptr,
                                 std::size_t bytes,
                                 std::size_t alignment) noexcept
{
    stream_ = stream;
    resource_.deallocate(stream_, ptr, bytes, alignment);
}

} // namespace cucim::memory
