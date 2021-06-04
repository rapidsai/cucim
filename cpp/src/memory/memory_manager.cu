/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cucim/memory/memory_manager.h"

#include <memory_resource>

#include <cuda_runtime.h>
#include <fmt/format.h>

#include "cucim/io/device_type.h"
#include "cucim/util/cuda.h"


CUCIM_API void* cucim_malloc(size_t size)
{
    return malloc(size);
}

CUCIM_API void cucim_free(void* ptr)
{
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
        attr.device = cucim::io::Device(cucim::io::DeviceType::kPinned, attributes.device);
        attr.ptr = attributes.hostPointer;
        break;
    case cudaMemoryTypeDevice:
    case cudaMemoryTypeManaged:
        attr.device = cucim::io::Device(cucim::io::DeviceType::kCUDA, attributes.device);
        attr.ptr = attributes.devicePointer;
        break;
    }
}

bool move_raster_from_host(void** target, size_t size, cucim::io::Device& dst_device)
{
    switch (dst_device.type())
    {
    case cucim::io::DeviceType::kCPU:
        break;
    case cucim::io::DeviceType::kCUDA:
        cudaError_t cuda_status;
        void* host_mem = *target;
        void* cuda_mem;
        CUDA_TRY(cudaMalloc(&cuda_mem, size));
        if (cuda_status)
        {
            throw std::bad_alloc();
        }
        CUDA_TRY(cudaMemcpy(cuda_mem, host_mem, size, cudaMemcpyHostToDevice));
        if (cuda_status)
        {
            throw std::bad_alloc();
        }
        cucim_free(host_mem);
        *target = cuda_mem;
    }
    return true;
}

bool move_raster_from_device(void** target, size_t size, cucim::io::Device& dst_device)
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
            throw std::bad_alloc();
        }
        cudaFree(cuda_mem);
        *target = host_mem;
        break;
    }
    case cucim::io::DeviceType::kCUDA:
        break;
    }
    return true;
}

} // namespace cucim::memory