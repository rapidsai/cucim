/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define CUCIM_EXPORTS // For exporting functions globally

#include "cucim/memory/memory_manager.h"

#include <cuda_runtime.h>
#include <fmt/format.h>

#include "cucim/io/device_type.h"
#include "cucim/memory/device_resources.h"
#include "cucim/profiler/nvtx3.h"
#include "cucim/util/cuda.h"

CUCIM_API void* cucim_malloc(size_t size)
{
    PROF_SCOPED_RANGE(PROF_EVENT_P(cucim_malloc, size));
    return malloc(size);
}

CUCIM_API void cucim_free(void* ptr)
{
    PROF_SCOPED_RANGE(PROF_EVENT(cucim_free));
    free(ptr);
}

namespace cucim::memory
{

void get_pointer_attributes(PointerAttributes& attr, const void* ptr)
{
    cudaError_t cuda_status;

    cudaPointerAttributes attributes;
    CUDA_TRY(cudaPointerGetAttributes(&attributes, ptr));
    if (cuda_status)
    {
        return;
    }

    cudaMemoryType& memory_type = attributes.type;
    switch (memory_type)
    {
    case cudaMemoryTypeUnregistered:
        attr.device = cucim::io::Device(cucim::io::DeviceType::kCPU, -1);
        attr.ptr = const_cast<void*>(ptr);
        break;
    case cudaMemoryTypeHost:
        attr.device = cucim::io::Device(cucim::io::DeviceType::kCUDAHost, attributes.device);
        attr.ptr = attributes.hostPointer;
        break;
    case cudaMemoryTypeDevice:
        attr.device = cucim::io::Device(cucim::io::DeviceType::kCUDA, attributes.device);
        attr.ptr = attributes.devicePointer;
        break;
    case cudaMemoryTypeManaged:
        attr.device = cucim::io::Device(cucim::io::DeviceType::kCUDAManaged, attributes.device);
        attr.ptr = attributes.devicePointer;
        break;
    }
}

EXPORT_VISIBLE bool move_raster_from_host(void** target,
                                          size_t size,
                                          const cucim::io::Device& dst_device,
                                          DeviceResources& resources)
{
    switch (dst_device.type())
    {
    case cucim::io::DeviceType::kCPU:
        break;
    case cucim::io::DeviceType::kCUDA: {
        cudaError_t cuda_status;
        void* host_mem = *target;
        // Throws std::bad_alloc on failure, which is what the hand-rolled
        // check below used to raise.
        void* cuda_mem = resources.allocate(size);
        // On the resource's stream rather than the default one: a
        // stream-ordered allocator only guarantees the block is usable by
        // work enqueued on the stream it was allocated on.
        const cudaStream_t stream = resources.stream().get();
        CUDA_TRY(cudaMemcpyAsync(cuda_mem, host_mem, size, cudaMemcpyHostToDevice, stream));
        if (cuda_status == cudaSuccess)
        {
            // This function is synchronous, and the source buffer is freed
            // just below, so the copy has to have read it by then. Only
            // reached when the copy itself succeeded, so a synchronization
            // failure cannot mask the earlier error.
            CUDA_TRY(cudaStreamSynchronize(stream));
        }
        if (cuda_status)
        {
            // Give the block back before unwinding, or the failed transfer
            // leaks it. The original code leaked here.
            resources.deallocate(cuda_mem, size);
            throw std::bad_alloc();
        }
        cucim_free(host_mem);
        *target = cuda_mem;
        break;
    }
    default:
        throw std::runtime_error("Unsupported device type");
    }
    return true;
}

EXPORT_VISIBLE bool move_raster_from_device(void** target,
                                            size_t size,
                                            const cucim::io::Device& dst_device,
                                            DeviceResources& resources)
{
    switch (dst_device.type())
    {
    case cucim::io::DeviceType::kCPU: {
        cudaError_t cuda_status;
        void* cuda_mem = *target;
        void* host_mem = cucim_malloc(size);
        // On the resource's stream, so the copy is ordered after whatever
        // produced the device buffer on that stream.
        const cudaStream_t stream = resources.stream().get();
        CUDA_TRY(cudaMemcpyAsync(host_mem, cuda_mem, size, cudaMemcpyDeviceToHost, stream));
        if (cuda_status == cudaSuccess)
        {
            // The host buffer is returned to the caller, so the copy has to
            // have landed before this returns. This also orders the free
            // below after the read.
            CUDA_TRY(cudaStreamSynchronize(stream));
        }
        if (cuda_status)
        {
            cucim_free(host_mem);
            throw std::bad_alloc();
        }
        resources.deallocate(cuda_mem, size);
        *target = host_mem;
        break;
    }
    case cucim::io::DeviceType::kCUDA:
        break;
    default:
        throw std::runtime_error("Unsupported device type");
    }
    return true;
}

namespace
{

// The resource-free overloads below are the long-standing entry points and are
// what the plugins call. They pin themselves to cudaMalloc/cudaFree rather than
// taking the default resource, which is now a stream-ordered pool.
//
// That is not conservatism: the device buffers these produce escape through
// ImageDataDesc and are released with a bare cudaFree (see CuImage's destructor
// and CuImage::save), which is not a valid way to release a pool allocation.
// Until those frees go through a resource too, the only correct allocator here
// is the one whose pointers cudaFree accepts.
//
// The same reasoning applies in the other direction: a buffer allocated inside
// a plugin came from a raw cudaMalloc, so returning it to a caller's pool would
// corrupt that pool.
DeviceResources cuda_malloc_resources()
{
    return DeviceResources{ any_device_resource{ CudaMemoryResource{} } };
}

} // namespace

CUCIM_API bool move_raster_from_host(void** target, size_t size, const cucim::io::Device& dst_device)
{
    DeviceResources resources = cuda_malloc_resources();
    return move_raster_from_host(target, size, dst_device, resources);
}

CUCIM_API bool move_raster_from_device(void** target, size_t size, const cucim::io::Device& dst_device)
{
    DeviceResources resources = cuda_malloc_resources();
    return move_raster_from_device(target, size, dst_device, resources);
}

} // namespace cucim::memory
