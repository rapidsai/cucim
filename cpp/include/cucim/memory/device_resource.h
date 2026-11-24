/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cucim/core/framework.h"
#include "cucim/io/device.h"

#include <cuda/memory_resource>
#include <cuda/stream>

#include <cstddef>
#include <memory>

/**
 * Device memory allocation through a caller-supplied CCCL memory resource.
 */
namespace cucim::memory
{

using ::cuda::mr::device_accessible;

/// Alignment cuCIM requests for device allocations.
inline constexpr std::size_t kDefaultAlignment = 256;

/// Default stream, spelled out rather than left as a value-initialized handle.
inline constexpr cudaStream_t kDefaultStream = cudaStream_t{ cudaStreamDefault };

/// Stream-ordered, device-accessible resource reference.
using device_resource_ref = ::cuda::mr::resource_ref<device_accessible>;

/// Owning, type-erased, device-accessible resource.
using any_device_resource = ::cuda::mr::any_resource<device_accessible>;

/**
 * @brief The current device's default CCCL memory pool. cuCIM's default.
 *
 * Stateless: every instance refers to the same process-wide pool. The pool is
 * resolved on each call rather than at construction, so constructing one needs
 * no device.
 *
 * Every call makes the current device's primary context current on the calling
 * thread, so the first allocation needs no prior CUDA call and a buffer can be
 * released from a thread that never used CUDA.
 */
class EXPORT_VISIBLE DeviceMemoryPoolResource
{
public:
    void* allocate_sync(std::size_t bytes, std::size_t alignment);
    void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment) noexcept;

    void* allocate(::cuda::stream_ref stream, std::size_t bytes, std::size_t alignment);
    void deallocate(::cuda::stream_ref stream, void* ptr, std::size_t bytes, std::size_t alignment) noexcept;

    bool operator==(const DeviceMemoryPoolResource&) const noexcept
    {
        return true;
    }
    bool operator!=(const DeviceMemoryPoolResource&) const noexcept
    {
        return false;
    }

    friend void get_property(const DeviceMemoryPoolResource&, device_accessible) noexcept
    {
    }
};

/**
 * @brief A resource that allocates with `cudaMalloc` and frees with `cudaFree`.
 *
 * For buffers that are later released with a bare `cudaFree`, which cannot
 * release pool memory; the resource-free raster moves in memory_manager.h use
 * it for that reason. Stateless.
 *
 * The stream-ordered overloads ignore the stream: `cudaMalloc` and `cudaFree`
 * synchronize the whole device.
 *
 * An alignment that does not divide 256 is rejected with
 * std::invalid_argument, as CCCL's pools do.
 */
class EXPORT_VISIBLE CudaMemoryResource
{
public:
    void* allocate_sync(std::size_t bytes, std::size_t alignment);
    void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment) noexcept;

    void* allocate(::cuda::stream_ref stream, std::size_t bytes, std::size_t alignment);
    void deallocate(::cuda::stream_ref stream, void* ptr, std::size_t bytes, std::size_t alignment) noexcept;

    bool operator==(const CudaMemoryResource&) const noexcept
    {
        return true;
    }
    bool operator!=(const CudaMemoryResource&) const noexcept
    {
        return false;
    }

    friend void get_property(const CudaMemoryResource&, device_accessible) noexcept
    {
    }
};

/**
 * @brief Where device memory comes from.
 *
 * Passed to whatever allocates, and kept by objects that own device memory so
 * they can free it through the same resource.
 *
 * Holds no stream: every call takes the stream it is ordered on. An object
 * that owns a buffer keeps the buffer's stream beside its handle, and frees on
 * the stream the buffer was last used on.
 *
 * Copies share the resource rather than duplicating it. A handle built from a
 * `device_resource_ref` borrows, and the caller keeps that resource alive for
 * as long as any copy of the handle or any buffer from it is in use.
 */
class EXPORT_VISIBLE DeviceResource
{
public:
    /// The current device's default memory pool.
    DeviceResource();

    /// Own *resource*, sharing it with every copy of this handle.
    explicit DeviceResource(any_device_resource resource);

    /// Borrow *resource*; the caller keeps it alive.
    explicit DeviceResource(device_resource_ref resource);

    /**
     * @brief Allocate *bytes* of device memory ordered on *stream*.
     *
     * @throws Whatever the resource throws: std::bad_alloc from cuCIM's
     * resource when memory runs out, std::invalid_argument for an alignment
     * they cannot honor, and any type from a caller's resource.
     */
    void* allocate(::cuda::stream_ref stream, std::size_t bytes, std::size_t alignment = kDefaultAlignment);

    /// Free memory from allocate(), ordered on *stream*. *bytes* and
    /// *alignment* must match the allocation.
    void deallocate(::cuda::stream_ref stream,
                    void* ptr,
                    std::size_t bytes,
                    std::size_t alignment = kDefaultAlignment) noexcept;

    /// Borrowed reference, for handing to APIs that take a resource.
    device_resource_ref resource() const noexcept
    {
        return resource_;
    }

private:
    /// Null when borrowing; otherwise the resource `resource_` refers to.
    std::shared_ptr<any_device_resource> owned_;
    device_resource_ref resource_;
};

/**
 * @brief move_raster_from_host() allocating from *resource* on *stream*.
 *
 * The caller frees the device memory written to *target* through the same
 * resource. The transfer is synchronized on *stream* before returning.
 *
 * Declared here rather than in memory_manager.h, which the plugins include and
 * which would then put CCCL on all their include paths. EXPORT_VISIBLE rather
 * than CUCIM_API, whose `extern "C"` rules out overloading.
 */
EXPORT_VISIBLE bool move_raster_from_host(void** target,
                                          size_t size,
                                          const cucim::io::Device& dst_device,
                                          DeviceResource& resource,
                                          ::cuda::stream_ref stream);

/**
 * @brief move_raster_from_device() freeing through *resource* on *stream*.
 *
 * *resource* and *alignment* must be the resource and alignment that
 * allocated `*target`. Buffers from a raw `cudaMalloc`, such as those a plugin
 * allocates, go through the overload in memory_manager.h instead.
 */
EXPORT_VISIBLE bool move_raster_from_device(void** target,
                                            size_t size,
                                            const cucim::io::Device& dst_device,
                                            DeviceResource& resource,
                                            ::cuda::stream_ref stream,
                                            std::size_t alignment = kDefaultAlignment);

} // namespace cucim::memory
