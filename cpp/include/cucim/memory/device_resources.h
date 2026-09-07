/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CUCIM_MEMORY_DEVICE_RESOURCES_H
#define CUCIM_MEMORY_DEVICE_RESOURCES_H

#include "cucim/core/framework.h"
#include "cucim/io/device.h"

#include <cuda/memory_resource>
#include <cuda/stream_ref>

#include <cstddef>

/**
 * Device memory allocation through a caller-supplied CCCL memory resource.
 *
 * Note on packaging: this header exposes `cuda::mr` types, so anything
 * including it needs CCCL headers on its include path. The build links CCCL
 * through `$<BUILD_INTERFACE:deps::cccl>`; exporting it for installed consumers
 * (either by installing the headers or by depending on `cuda-cccl`) is still
 * outstanding.
 */
namespace cucim::memory
{

/**
 * @brief The set of CCCL properties cuCIM requires of a device memory resource.
 *
 * Device-accessible only. cuCIM hands the resulting pointers to CUDA kernels,
 * nvImageCodec and `__cuda_array_interface__` consumers, none of which can use
 * host-only memory.
 */
using device_accessible = ::cuda::mr::device_accessible;

/**
 * @brief Alignment cuCIM requests for device allocations.
 *
 * `cudaMalloc` already guarantees at least this much, so the default resource
 * ignores the value; it is stated explicitly because a pool resource supplied
 * by the caller will honour it, and image rows are addressed on 256-byte
 * boundaries downstream.
 */
inline constexpr std::size_t kDefaultAlignment = 256;

/**
 * @brief Stream-ordered, device-accessible resource reference.
 *
 * This is the type that should appear in cuCIM APIs that allocate device
 * memory. It binds to resources from either CCCL or RMM, so callers already
 * holding an `rmm::device_async_resource_ref` can pass it straight through.
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
 * @brief A resource that allocates with `cudaMalloc` and frees with `cudaFree`.
 *
 * This is what cuCIM did before memory resources existed, expressed as a
 * resource so that the default path stays byte-for-byte what it was and only
 * callers who supply their own resource see different behaviour.
 *
 * Stateless, so it can be stored by value and compares equal to every other
 * instance. Being stateless is also why it needs no global: there is nothing
 * to share.
 *
 * The stream-ordered overloads ignore the stream, since `cudaMalloc` and
 * `cudaFree` synchronize the whole device rather than a single stream. That
 * makes them correct but not stream-ordered, which is the reason a caller
 * wanting real asynchrony should pass a pool resource instead.
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
 * Deliberately not a singleton and never consulted from global scope. RAPIDS
 * guidance is to avoid a process-wide "current" resource, because it makes the
 * owner of a buffer impossible to determine locally and is being removed from
 * RMM; instead an instance of this is passed to whatever needs to allocate, and
 * carried by objects that own device memory so they can free it through the
 * same resource later.
 *
 * The resource is held by value in an owning `any_device_resource` rather than
 * as a bare `device_resource_ref`. A reference would be the cheaper choice but
 * would let a handle outlive what it points at, which is easy to do here since
 * handles get copied into images that may outlive the call that created them.
 *
 * Because it owns, constructing a handle *copies* the resource, and so does
 * copying a handle. For the usual case that is what you want: callers wrap a
 * reference to their allocator, such as an `rmm::device_async_resource_ref`,
 * and copying a reference still points at the one underlying pool. Wrapping a
 * stateful allocator *by value* instead would duplicate it, and the copy the
 * handle owns is then the one that serves every request.
 *
 * Default construction yields the pre-existing behaviour: `cudaMalloc` and
 * `cudaFree` on the default stream.
 */
class EXPORT_VISIBLE DeviceResources
{
public:
    /// Raw `cudaMalloc`/`cudaFree` on the default stream.
    DeviceResources();

    /// Allocate from *resource*, ordering allocations on *stream*.
    explicit DeviceResources(any_device_resource resource,
                             ::cuda::stream_ref stream = ::cuda::stream_ref{ cudaStream_t{} });

    /**
     * @brief Allocate *bytes* of device memory.
     *
     * @return pointer to device memory, or nullptr when *bytes* is 0.
     * @throws std::bad_alloc if the resource cannot satisfy the request.
     */
    void* allocate(std::size_t bytes);

    /**
     * @brief Return memory obtained from allocate().
     *
     * *bytes* must be the size originally requested; pool resources need it to
     * return the block to the right free list, unlike `cudaFree`.
     */
    void deallocate(void* ptr, std::size_t bytes) noexcept;

    /**
     * @brief Borrowed reference, for handing to APIs that take a resource.
     *
     * Non-const because a `resource_ref` allocates through the resource it
     * refers to, so CCCL will only build one from a non-const resource.
     */
    device_resource_ref resource() noexcept;

    ::cuda::stream_ref stream() const noexcept
    {
        return stream_;
    }

private:
    any_device_resource resource_;
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
