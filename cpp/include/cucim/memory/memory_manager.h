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
//
#ifndef CUCIM_MEMORY_MANAGER_H
#define CUCIM_MEMORY_MANAGER_H

#include "cucim/io/device.h"

#include <cucim/macros/api_header.h>
#include <cstddef>

/**
 * Host memory allocator for exchanged data
 * @param size Number of bytes to allocate
 * @return Pointer to the allocated memory
 */
CUCIM_API void* cucim_malloc(size_t size);

/**
 * Free allocated memory by cucim_malloc
 * @param Pointer to the allocated memory
 */
CUCIM_API void cucim_free(void* ptr);

namespace cucim
{
namespace memory
{

/**
 * Pointer attributes
 */
struct PointerAttributes
{
    /**
     * The type of device
     */
    cucim::io::Device device{};

    /**
     * The address which may be dereferenced on the current device to access
     * the memory or nullptr if no such address exists.
     */
    void* ptr = nullptr;
};

/**
 * A wrapper for cudaPointerGetAttributes() in CUDA.
 *
 * Instead of cudaPointerAttributes
 *
 * @param ptr Pointer to the allocated memory
 * @return Pointer attribute information in 'PointerAttributes' struct
 */
CUCIM_API void get_pointer_attributes(PointerAttributes& attr, const void* ptr);


} // namespace memory
} // namespace cucim
#endif // CUCIM_MEMORY_MANAGER_H
