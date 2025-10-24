/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
//
#ifndef CUCIM_MEMORY_MANAGER_H
#define CUCIM_MEMORY_MANAGER_H

#include "cucim/macros/api_header.h"

#include <cstddef>

#include "cucim/io/device.h"

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

namespace cucim::memory
{

/**
 * Pointer attributes
 */
struct PointerAttributes
{
    /**
     * @brief The type of device
     */
    cucim::io::Device device{};

    /**
     * The address which may be dereferenced on the current device to access
     * the memory or nullptr if no such address exists.
     */
    void* ptr = nullptr;
};

/**
 * @brief A wrapper for cudaPointerGetAttributes() in CUDA.
 *
 * Instead of cudaPointerAttributes
 *
 * @param ptr Pointer to the allocated memory
 * @return Pointer attribute information in 'PointerAttributes' struct
 */
CUCIM_API void get_pointer_attributes(PointerAttributes& attr, const void* ptr);

/**
 * @brief Move host memory of `size` bytes to a new memory in `out_device`.
 *
 * Set the pointer of the new memory to `target` and free the host memory previously indicated by `target.
 * Do nothing if `out_device` is CPU memory.
 *
 * @param[in, out] target Pointer to the pointer of the host memory.
 * @param size Size of the host memory.
 * @param dst_device Destination device of the memory.
 * @return `true` if succeed.
 */
CUCIM_API bool move_raster_from_host(void** target, size_t size, const cucim::io::Device& dst_device);

/**
 * @brief Move device memory of `size` bytes to a new memory in `out_device`.
 *
 * Set the pointer of the new memory to `target` and free the device memory previously indicated by `target.
 * Do nothing if `out_device` is CUDA memory.
 *
 * @param[in, out] target Pointer to the pointer of the device memory.
 * @param size Size of the device memory.
 * @param dst_device Destination device of the memory.
 * @return `true` if succeed.
 */
CUCIM_API bool move_raster_from_device(void** target, size_t size, const cucim::io::Device& dst_device);

} // namespace cucim::memory
#endif // CUCIM_MEMORY_MANAGER_H
