/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
        CUDA_TRY(cudaMemcpy(cuda_mem, host_mem, size, cudaMemcpyHostToDevice));
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
        CUDA_TRY(cudaMemcpy(host_mem, cuda_mem, size, cudaMemcpyDeviceToHost));
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

// The resource-free overloads are the long-standing entry points and are what
// the plugins call. They delegate to a default-constructed handle, which is
// cudaMalloc/cudaFree, so their behaviour is unchanged.
//
// This is also the only correct thing they can do: a buffer allocated inside a
// plugin came from a raw cudaMalloc, and freeing it through a caller's pool
// resource would corrupt that pool.

CUCIM_API bool move_raster_from_host(void** target, size_t size, const cucim::io::Device& dst_device)
{
    DeviceResources resources{};
    return move_raster_from_host(target, size, dst_device, resources);
}

CUCIM_API bool move_raster_from_device(void** target, size_t size, const cucim::io::Device& dst_device)
{
    DeviceResources resources{};
    return move_raster_from_device(target, size, dst_device, resources);
}

} // namespace cucim::memory
