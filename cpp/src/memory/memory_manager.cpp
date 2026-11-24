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
                                          DeviceResources& resources,
                                          ::cuda::stream_ref stream)
{
    switch (dst_device.type())
    {
    case cucim::io::DeviceType::kCPU:
        break;
    case cucim::io::DeviceType::kCUDA: {
        cudaError_t cuda_status;
        void* host_mem = *target;
        void* cuda_mem = resources.allocate(stream, size);
        CUDA_TRY(cudaMemcpyAsync(cuda_mem, host_mem, size, cudaMemcpyHostToDevice, stream.get()));
        if (cuda_status == cudaSuccess)
        {
            // The source is freed below, so the copy has to have read it.
            CUDA_TRY(cudaStreamSynchronize(stream.get()));
        }
        if (cuda_status)
        {
            resources.deallocate(stream, cuda_mem, size);
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
                                            DeviceResources& resources,
                                            ::cuda::stream_ref stream,
                                            std::size_t alignment)
{
    switch (dst_device.type())
    {
    case cucim::io::DeviceType::kCPU: {
        cudaError_t cuda_status;
        void* cuda_mem = *target;
        void* host_mem = cucim_malloc(size);
        CUDA_TRY(cudaMemcpyAsync(host_mem, cuda_mem, size, cudaMemcpyDeviceToHost, stream.get()));
        if (cuda_status == cudaSuccess)
        {
            // The host buffer is returned to the caller, so the copy has to
            // have landed before this returns.
            CUDA_TRY(cudaStreamSynchronize(stream.get()));
        }
        if (cuda_status)
        {
            cucim_free(host_mem);
            throw std::bad_alloc();
        }
        resources.deallocate(stream, cuda_mem, size, alignment);
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

// The resource-free overloads below are what the plugins call. They use
// cudaMalloc/cudaFree rather than the default pool because the buffers escape
// through ImageDataDesc and are released with a bare cudaFree (CuImage's
// destructor and CuImage::save), which cannot release pool memory. Likewise a
// buffer a plugin allocated came from cudaMalloc and cannot go to a pool.
//
// Borrowed, so that a no-op move costs no allocation.
DeviceResources cuda_malloc_resources()
{
    static CudaMemoryResource backend;
    return DeviceResources{ device_resource_ref{ backend } };
}

} // namespace

CUCIM_API bool move_raster_from_host(void** target, size_t size, const cucim::io::Device& dst_device)
{
    DeviceResources resources = cuda_malloc_resources();
    return move_raster_from_host(target, size, dst_device, resources, ::cuda::stream_ref{ kDefaultStream });
}

CUCIM_API bool move_raster_from_device(void** target, size_t size, const cucim::io::Device& dst_device)
{
    DeviceResources resources = cuda_malloc_resources();
    return move_raster_from_device(target, size, dst_device, resources, ::cuda::stream_ref{ kDefaultStream });
}

} // namespace cucim::memory
