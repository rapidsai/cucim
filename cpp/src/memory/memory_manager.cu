/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <fmt/format.h>
#include <cuda_runtime.h>

#include <memory_resource>

#define CUDA_TRY(stmt)                                                                                                 \
    {                                                                                                                  \
        cuda_status = stmt;                                                                                            \
        if (cudaSuccess != cuda_status)                                                                                \
        {                                                                                                              \
            fmt::print(stderr, "[Error] CUDA Runtime call {} in line {} of file {} failed with '{}' ({}).\n", #stmt,   \
                       __LINE__, __FILE__, cudaGetErrorString(cuda_status), cuda_status);                              \
        }                                                                                                              \
    }

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

} // namespace cucim::memory