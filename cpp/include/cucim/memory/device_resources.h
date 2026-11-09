/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CUCIM_MEMORY_DEVICE_RESOURCES_H
#define CUCIM_MEMORY_DEVICE_RESOURCES_H

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

/**
 * @brief Alignment cuCIM requests for device allocations.
 *
 * Image rows are addressed on 256-byte boundaries downstream.
 */
inline constexpr std::size_t kDefaultAlignment = 256;

/// Default stream, spelled out rather than left as a value-initialized handle.
inline constexpr cudaStream_t kDefaultStream = cudaStream_t{ cudaStreamDefault };

/**
 * @brief Stream-ordered, device-accessible resource reference.
 *
 * This is the type that should appear in cuCIM APIs that allocate device
 * memory. It binds to resources from either CCCL or RMM.
 */
using device_resource_ref = ::cuda::mr::resource_ref<device_accessible>;

/**
 * @brief Owning, type-erased, device-accessible resource.
 *
 * Used where cuCIM has to keep a resource alive rather than borrow one, most
 * importantly for memory handed back to the user: whatever allocated a buffer
 * has to still exist when that buffer is freed, which for an image can be long
 * after the call that produced it returned.
 */
using any_device_resource = ::cuda::mr::any_resource<device_accessible>;

/**
 * @brief The current device's default CCCL memory pool.
 *
 * Stream-ordered, so an allocation does not synchronize the whole device the
 * way `cudaMalloc` does. This is cuCIM's default.
 *
 * Stateless: the pool is owned by CCCL and shared process-wide, so every
 * instance refers to the same pool and copies are interchangeable. The pool is
 * resolved on first use rather than at construction, which keeps a
 * default-constructed DeviceResources usable on a host with no device as long
 * as nothing allocates through it.
 */
class EXPORT_VISIBLE DeviceMemoryPoolResource
{
public:
    void* allocate_sync(std::size_t bytes, std::size_t alignment);
    void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment);

    void* allocate(::cuda::stream_ref stream, std::size_t bytes, std::size_t alignment);
    void deallocate(::cuda::stream_ref stream, void* ptr, std::size_t bytes, std::size_t alignment);

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
 * What cuCIM did before memory resources existed, kept as an explicit opt-in
 * for callers that need allocation to be device-synchronizing. Stateless, so
 * it compares equal to every other instance.
 *
 * The stream-ordered overloads ignore the stream: `cudaMalloc` and `cudaFree`
 * synchronize the whole device, which makes them trivially ordered against any
 * stream but not stream-ordered.
 */
class EXPORT_VISIBLE CudaMemoryResource
{
public:
    void* allocate_sync(std::size_t bytes, std::size_t alignment);
    void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment);

    void* allocate(::cuda::stream_ref stream, std::size_t bytes, std::size_t alignment);
    void deallocate(::cuda::stream_ref stream, void* ptr, std::size_t bytes, std::size_t alignment);

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
 * @brief Where device memory comes from, and the stream that orders it.
 *
 * Deliberately not a singleton and never consulted from global scope: an
 * instance is passed to whatever needs to allocate, and carried by objects
 * that own device memory so they can free it through the same resource later.
 *
 * Copying a handle does **not** copy the resource. An owning handle keeps its
 * resource in a shared owner, so every copy allocates from and frees through
 * the one resource instance; a stateful allocator is therefore never
 * duplicated behind the caller's back. A handle built from a
 * `device_resource_ref` borrows instead, and the caller is responsible for
 * keeping that resource alive for as long as any copy of the handle, or any
 * buffer obtained through it, is still in use.
 *
 * The stream is per-handle rather than shared, so copies can be moved between
 * streams independently. The handle's stream is what orders deallocation; the
 * stream-taking methods below rebind it so that a free is always ordered after
 * the last operation the handle was used for.
 */
class EXPORT_VISIBLE DeviceResources
{
public:
    /// The current device's default memory pool, on the default stream.
    DeviceResources();

    /// Own *resource*, sharing it with every copy of this handle.
    explicit DeviceResources(any_device_resource resource,
                             ::cuda::stream_ref stream = ::cuda::stream_ref{ kDefaultStream });

    /// Borrow *resource*; the caller keeps it alive. See the class note.
    explicit DeviceResources(device_resource_ref resource,
                             ::cuda::stream_ref stream = ::cuda::stream_ref{ kDefaultStream });

    /**
     * @brief Allocate *bytes* of device memory on this handle's stream.
     *
     * @throws std::bad_alloc if the resource cannot satisfy the request.
     */
    void* allocate(std::size_t bytes, std::size_t alignment = kDefaultAlignment);

    /**
     * @brief Allocate *bytes* of device memory ordered on *stream*.
     *
     * Rebinds this handle to *stream*, so a later deallocate() through the
     * same handle is ordered after this allocation.
     */
    void* allocate(::cuda::stream_ref stream, std::size_t bytes, std::size_t alignment = kDefaultAlignment);

    /**
     * @brief Return memory obtained from allocate().
     *
     * *bytes* and *alignment* must match the original request; pool resources
     * need them to return the block to the right free list.
     */
    void deallocate(void* ptr, std::size_t bytes, std::size_t alignment = kDefaultAlignment) noexcept;

    /// deallocate() ordered on *stream*, which this handle then adopts.
    void deallocate(::cuda::stream_ref stream,
                    void* ptr,
                    std::size_t bytes,
                    std::size_t alignment = kDefaultAlignment) noexcept;

    /// Borrowed reference, for handing to APIs that take a resource.
    device_resource_ref resource() const noexcept
    {
        return resource_;
    }

    ::cuda::stream_ref stream() const noexcept
    {
        return stream_;
    }

private:
    /// Null when borrowing; otherwise the resource `resource_` refers to.
    std::shared_ptr<any_device_resource> owned_;
    device_resource_ref resource_;
    ::cuda::stream_ref stream_;
};

/**
 * @brief move_raster_from_host() allocating from *resources*.
 *
 * Same contract as the overload in memory_manager.h, except that the device
 * memory written to *target* comes from *resources*. The caller therefore has
 * to free it through the same resource, so it must keep *resources* alive at
 * least as long as the buffer.
 *
 * The transfer is issued on `resources.stream()` and synchronized before
 * returning, so the call is complete when it returns even if the resource
 * allocates asynchronously.
 *
 * Declared here rather than beside the original in memory_manager.h because
 * that header is included throughout the plugins, which would then all need
 * CCCL on their include path.
 *
 * EXPORT_VISIBLE rather than CUCIM_API: the latter adds `extern "C"`, which
 * would both rule out overloading the existing name and be wrong for a
 * function taking a C++ reference.
 */
EXPORT_VISIBLE bool move_raster_from_host(void** target,
                                          size_t size,
                                          const cucim::io::Device& dst_device,
                                          DeviceResources& resources);

/**
 * @brief move_raster_from_device() freeing through *resources*.
 *
 * *resources* must be the resource that allocated `*target`; this releases it
 * after copying to the host. Passing a different resource is undefined, which
 * is why the plain overload in memory_manager.h still exists for buffers that
 * came from raw `cudaMalloc` (notably anything allocated inside a plugin).
 */
EXPORT_VISIBLE bool move_raster_from_device(void** target,
                                            size_t size,
                                            const cucim::io::Device& dst_device,
                                            DeviceResources& resources);

} // namespace cucim::memory

#endif // CUCIM_MEMORY_DEVICE_RESOURCES_H
