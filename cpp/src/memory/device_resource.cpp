/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cucim/memory/device_resource.h"

#include "cucim/profiler/nvtx3.h"

#include <cuda/devices>
#include <cuda/memory_pool>

#include <cuda_runtime.h>

#include <cstdio>
#include <new>
#include <stdexcept>
#include <utility>

namespace cucim::memory
{

namespace
{

/// The default pool of whichever device is current, with that device's primary
/// context current on this thread. The pool allocates through the driver API,
/// which unlike the runtime does not create a context on first use, so the
/// null stream would otherwise be rejected with CUDA_ERROR_INVALID_CONTEXT.
::cuda::device_memory_pool_ref current_device_pool()
{
    int device_id = 0;
    if (cudaFree(nullptr) != cudaSuccess || cudaGetDevice(&device_id) != cudaSuccess)
    {
        static_cast<void>(cudaGetLastError());
        throw std::bad_alloc();
    }
    return ::cuda::device_default_memory_pool(::cuda::device_ref{ device_id });
}

/// Release a pool block through the runtime, which frees by pointer from any
/// device's pool and creates the context on a thread that never used CUDA.
void free_pool_block(void* ptr, cudaStream_t stream) noexcept
{
    const cudaError_t status = cudaFreeAsync(ptr, stream);
    if (status != cudaSuccess)
    {
        static_cast<void>(cudaGetLastError());
        std::fprintf(stderr, "[cuCIM] cudaFreeAsync failed: %s\n", cudaGetErrorString(status));
    }
}

void check_cuda_malloc_alignment(std::size_t alignment)
{
    if (alignment == 0 || alignment > ::cuda::mr::default_cuda_malloc_alignment ||
        ::cuda::mr::default_cuda_malloc_alignment % alignment != 0)
    {
        throw std::invalid_argument("CudaMemoryResource: alignment must divide 256");
    }
}

} // namespace

void* DeviceMemoryPoolResource::allocate_sync(std::size_t bytes, std::size_t alignment)
{
    PROF_SCOPED_RANGE(PROF_EVENT_P(cucim_malloc, bytes));

    return current_device_pool().allocate_sync(bytes, alignment);
}

void DeviceMemoryPoolResource::deallocate_sync(void* ptr, std::size_t /*bytes*/, std::size_t /*alignment*/) noexcept
{
    PROF_SCOPED_RANGE(PROF_EVENT(cucim_free));

    free_pool_block(ptr, kDefaultStream);
}

void* DeviceMemoryPoolResource::allocate(::cuda::stream_ref stream, std::size_t bytes, std::size_t alignment)
{
    PROF_SCOPED_RANGE(PROF_EVENT_P(cucim_malloc, bytes));

    return current_device_pool().allocate(stream, bytes, alignment);
}

void DeviceMemoryPoolResource::deallocate(::cuda::stream_ref stream,
                                          void* ptr,
                                          std::size_t /*bytes*/,
                                          std::size_t /*alignment*/) noexcept
{
    PROF_SCOPED_RANGE(PROF_EVENT(cucim_free));

    free_pool_block(ptr, stream.get());
}

void* CudaMemoryResource::allocate_sync(std::size_t bytes, std::size_t alignment)
{
    PROF_SCOPED_RANGE(PROF_EVENT_P(cucim_malloc, bytes));

    check_cuda_malloc_alignment(alignment);
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, bytes) != cudaSuccess)
    {
        static_cast<void>(cudaGetLastError());
        throw std::bad_alloc();
    }
    return ptr;
}

void CudaMemoryResource::deallocate_sync(void* ptr, std::size_t /*bytes*/, std::size_t /*alignment*/) noexcept
{
    PROF_SCOPED_RANGE(PROF_EVENT(cucim_free));

    static_cast<void>(cudaFree(ptr));
}

void* CudaMemoryResource::allocate(::cuda::stream_ref /*stream*/, std::size_t bytes, std::size_t alignment)
{
    return allocate_sync(bytes, alignment);
}

void CudaMemoryResource::deallocate(::cuda::stream_ref /*stream*/,
                                    void* ptr,
                                    std::size_t bytes,
                                    std::size_t alignment) noexcept
{
    deallocate_sync(ptr, bytes, alignment);
}

DeviceResource::DeviceResource() : DeviceResource(any_device_resource{ DeviceMemoryPoolResource{} })
{
}

DeviceResource::DeviceResource(any_device_resource resource)
    : owned_(std::make_shared<any_device_resource>(std::move(resource))), resource_(*owned_)
{
}

DeviceResource::DeviceResource(device_resource_ref resource) : owned_(nullptr), resource_(resource)
{
}

void* DeviceResource::allocate(::cuda::stream_ref stream, std::size_t bytes, std::size_t alignment)
{
    return resource_.allocate(stream, bytes, alignment);
}

void DeviceResource::deallocate(::cuda::stream_ref stream,
                                 void* ptr,
                                 std::size_t bytes,
                                 std::size_t alignment) noexcept
{
    resource_.deallocate(stream, ptr, bytes, alignment);
}

} // namespace cucim::memory
