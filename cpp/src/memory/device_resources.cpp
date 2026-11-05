/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cucim/memory/device_resources.h"

#include "cucim/profiler/nvtx3.h"

#include <cuda_runtime.h>

#include <new>

namespace cucim::memory
{

void* CudaMemoryResource::allocate_sync(std::size_t bytes, std::size_t /*alignment*/)
{
    PROF_SCOPED_RANGE(PROF_EVENT_P(cucim_malloc, bytes));

    if (bytes == 0)
    {
        return nullptr;
    }

    // cudaMalloc already satisfies any alignment cuCIM asks for (256 bytes),
    // so the requested alignment needs no extra handling.
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

    if (ptr != nullptr)
    {
        static_cast<void>(cudaFree(ptr));
    }
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
    : resource_(CudaMemoryResource{}), stream_(::cuda::stream_ref{ cudaStream_t{} })
{
}

DeviceResources::DeviceResources(any_device_resource resource, ::cuda::stream_ref stream)
    : resource_(std::move(resource)), stream_(stream)
{
}

void* DeviceResources::allocate(std::size_t bytes)
{
    if (bytes == 0)
    {
        return nullptr;
    }
    return resource_.allocate(stream_, bytes, kDefaultAlignment);
}

void DeviceResources::deallocate(void* ptr, std::size_t bytes) noexcept
{
    if (ptr == nullptr)
    {
        return;
    }
    // Deallocation runs from destructors, where letting an exception escape
    // would terminate the process; a leak is the better failure here.
    try
    {
        resource_.deallocate(stream_, ptr, bytes, kDefaultAlignment);
    }
    catch (...)
    {
    }
}

device_resource_ref DeviceResources::resource() noexcept
{
    return device_resource_ref{ resource_ };
}

} // namespace cucim::memory
