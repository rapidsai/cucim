/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cucim/memory/device_resource.h"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

using cucim::memory::any_device_resource;
using cucim::memory::CudaMemoryResource;
using cucim::memory::DeviceMemoryPoolResource;
using cucim::memory::DeviceResource;
using cucim::memory::device_resource_ref;

namespace
{

/**
 * A resource that hands out host memory and records what it was asked for.
 *
 * Host memory is deliberate: these tests are about whether cuCIM routes its
 * allocations through the resource it was given and pairs every allocation
 * with a matching deallocation, and answering that does not require a GPU.
 * Claiming `device_accessible` is a lie that nothing here dereferences.
 *
 * The counters live in a separate State object so that a test can keep
 * watching them after the resource itself has been handed to a
 * `DeviceResource`, which type-erases it.
 *
 * Each instance also carries an identity that copying does *not* preserve,
 * which is what makes it possible to tell a shared resource from a duplicated
 * one. Without that, a handle that cloned its resource would be
 * indistinguishable from one that shared it, and the tests below would pass
 * against the bug they exist to catch.
 */
class TrackingResource
{
public:
    struct State
    {
        int allocate_calls = 0;
        int deallocate_calls = 0;
        int instances_created = 0;
        std::size_t last_alignment = 0;
        /// Stream of the last stream-ordered call, allocation or release.
        cudaStream_t last_stream = nullptr;
        bool size_matched_on_free = true;
        bool saw_foreign_pointer = false;
        bool freed_through_a_different_instance = false;
        /// pointer -> (size requested, id of the instance that issued it)
        std::unordered_map<void*, std::pair<std::size_t, int>> outstanding;
    };

    explicit TrackingResource(State& state) : state_(&state), id_(++state.instances_created)
    {
    }

    /// A copy is a *different* allocator, and says so.
    TrackingResource(const TrackingResource& other) : state_(other.state_), id_(++other.state_->instances_created)
    {
    }

    /// A move is the same allocator in a new place, so it keeps its identity.
    TrackingResource(TrackingResource&& other) noexcept : state_(other.state_), id_(other.id_)
    {
    }

    TrackingResource& operator=(const TrackingResource& other)
    {
        if (this != &other)
        {
            state_ = other.state_;
            id_ = ++state_->instances_created;
        }
        return *this;
    }

    TrackingResource& operator=(TrackingResource&& other) noexcept
    {
        state_ = other.state_;
        id_ = other.id_;
        return *this;
    }

    ~TrackingResource() = default;

    void* allocate_sync(std::size_t bytes, std::size_t alignment)
    {
        ++state_->allocate_calls;
        state_->last_alignment = alignment;
        void* ptr = ::operator new(bytes);
        state_->outstanding[ptr] = { bytes, id_ };
        return ptr;
    }

    void deallocate_sync(void* ptr, std::size_t bytes, std::size_t)
    {
        ++state_->deallocate_calls;
        auto it = state_->outstanding.find(ptr);
        // A resource is only able to reclaim a block if it is handed back the
        // size it gave out, by the instance that gave it out; a pool would
        // corrupt its free lists otherwise.
        if (it != state_->outstanding.end())
        {
            state_->size_matched_on_free = state_->size_matched_on_free && (it->second.first == bytes);
            state_->freed_through_a_different_instance =
                state_->freed_through_a_different_instance || (it->second.second != id_);
            state_->outstanding.erase(it);
        }
        else
        {
            state_->saw_foreign_pointer = true;
        }
        ::operator delete(ptr);
    }

    void* allocate(::cuda::stream_ref stream, std::size_t bytes, std::size_t alignment)
    {
        state_->last_stream = stream.get();
        return allocate_sync(bytes, alignment);
    }

    void deallocate(::cuda::stream_ref stream, void* ptr, std::size_t bytes, std::size_t alignment)
    {
        state_->last_stream = stream.get();
        deallocate_sync(ptr, bytes, alignment);
    }

    bool operator==(const TrackingResource& other) const noexcept
    {
        return id_ == other.id_;
    }
    bool operator!=(const TrackingResource& other) const noexcept
    {
        return id_ != other.id_;
    }

    friend void get_property(const TrackingResource&, cucim::memory::device_accessible) noexcept
    {
    }

private:
    State* state_;
    int id_;
};

/// A resource that refuses every request, to exercise the failure path.
class FailingResource
{
public:
    void* allocate_sync(std::size_t, std::size_t)
    {
        throw std::bad_alloc();
    }
    void deallocate_sync(void*, std::size_t, std::size_t) {}
    void* allocate(::cuda::stream_ref, std::size_t, std::size_t)
    {
        throw std::bad_alloc();
    }
    void deallocate(::cuda::stream_ref, void*, std::size_t, std::size_t) {}

    bool operator==(const FailingResource&) const noexcept
    {
        return true;
    }
    bool operator!=(const FailingResource&) const noexcept
    {
        return false;
    }
    friend void get_property(const FailingResource&, cucim::memory::device_accessible) noexcept
    {
    }
};

/**
 * A resource that fails the way a CCCL pool does.
 *
 * `cuda::cuda_error` derives from `std::runtime_error`, not `std::bad_alloc`,
 * so exhaustion of the default pool is not caught by a `catch (bad_alloc&)`.
 * This stands in for it without needing a device.
 */
class PoolStyleFailingResource
{
public:
    void* allocate_sync(std::size_t, std::size_t)
    {
        throw std::runtime_error("cudaErrorMemoryAllocation");
    }
    void deallocate_sync(void*, std::size_t, std::size_t) {}
    void* allocate(::cuda::stream_ref, std::size_t, std::size_t)
    {
        throw std::runtime_error("cudaErrorMemoryAllocation");
    }
    void deallocate(::cuda::stream_ref, void*, std::size_t, std::size_t) {}

    bool operator==(const PoolStyleFailingResource&) const noexcept
    {
        return true;
    }
    bool operator!=(const PoolStyleFailingResource&) const noexcept
    {
        return false;
    }
    friend void get_property(const PoolStyleFailingResource&, cucim::memory::device_accessible) noexcept
    {
    }
};

const ::cuda::stream_ref kStream{ cucim::memory::kDefaultStream };

/// A stream handle that is never submitted to, only compared.
::cuda::stream_ref fake_stream(std::uintptr_t id)
{
    return ::cuda::stream_ref{ reinterpret_cast<cudaStream_t>(id) };
}

} // namespace

TEST_CASE("cuCIM handle satisfy the CCCL concepts", "[test_device_resource.cpp]")
{
    STATIC_REQUIRE(::cuda::mr::synchronous_resource<DeviceMemoryPoolResource>);
    STATIC_REQUIRE(::cuda::mr::resource<DeviceMemoryPoolResource>);
    STATIC_REQUIRE(::cuda::mr::synchronous_resource<CudaMemoryResource>);
    STATIC_REQUIRE(::cuda::mr::resource<CudaMemoryResource>);
    STATIC_REQUIRE(::cuda::mr::synchronous_resource<TrackingResource>);
    STATIC_REQUIRE(::cuda::mr::resource<TrackingResource>);

    STATIC_REQUIRE(::cuda::has_property<DeviceMemoryPoolResource, cucim::memory::device_accessible>);
    STATIC_REQUIRE(::cuda::has_property<CudaMemoryResource, cucim::memory::device_accessible>);
    STATIC_REQUIRE(::cuda::has_property<TrackingResource, cucim::memory::device_accessible>);
}

TEST_CASE("DeviceResource allocates from the resource it was given", "[test_device_resource.cpp]")
{
    TrackingResource::State state;
    DeviceResource handle{ any_device_resource{ TrackingResource{ state } } };

    SECTION("allocation and deallocation reach the supplied resource")
    {
        constexpr std::size_t nbytes = 4096;
        void* ptr = handle.allocate(kStream, nbytes);
        REQUIRE(ptr != nullptr);
        REQUIRE(state.allocate_calls == 1);
        REQUIRE(state.last_alignment == cucim::memory::kDefaultAlignment);

        handle.deallocate(kStream, ptr, nbytes);
        REQUIRE(state.deallocate_calls == 1);
        REQUIRE(state.outstanding.empty());
    }

    SECTION("an explicit alignment is passed through")
    {
        constexpr std::size_t nbytes = 128;
        constexpr std::size_t alignment = 4096;
        void* ptr = handle.allocate(kStream, nbytes, alignment);
        REQUIRE(state.last_alignment == alignment);

        handle.deallocate(kStream, ptr, nbytes, alignment);
        REQUIRE(state.outstanding.empty());
        REQUIRE_FALSE(state.saw_foreign_pointer);
    }

    SECTION("a zero-byte request is still the resource's decision")
    {
        void* ptr = handle.allocate(kStream, 0);
        REQUIRE(state.allocate_calls == 1);

        handle.deallocate(kStream, ptr, 0);
        REQUIRE(state.deallocate_calls == 1);
        REQUIRE(state.outstanding.empty());
    }
}

TEST_CASE("DeviceResource pairs allocations with sized deallocations", "[test_device_resource.cpp]")
{
    TrackingResource::State state;
    DeviceResource handle{ any_device_resource{ TrackingResource{ state } } };

    std::vector<std::pair<void*, std::size_t>> blocks;
    for (std::size_t bytes : { std::size_t{ 512 }, std::size_t{ 4096 }, std::size_t{ 1u << 20 } })
    {
        void* ptr = handle.allocate(kStream, bytes);
        REQUIRE(ptr != nullptr);
        blocks.emplace_back(ptr, bytes);
    }
    REQUIRE(state.allocate_calls == 3);

    for (auto& [ptr, bytes] : blocks)
    {
        handle.deallocate(kStream, ptr, bytes);
    }

    REQUIRE(state.deallocate_calls == 3);
    REQUIRE(state.outstanding.empty());
    REQUIRE(state.size_matched_on_free);
    REQUIRE_FALSE(state.saw_foreign_pointer);
}

TEST_CASE("DeviceResource propagates allocation failure", "[test_device_resource.cpp]")
{
    SECTION("a resource that throws bad_alloc")
    {
        DeviceResource handle{ any_device_resource{ FailingResource{} } };
        REQUIRE_THROWS_AS(handle.allocate(kStream, 1024), std::bad_alloc);
    }

    SECTION("a pool-style resource, which does not throw bad_alloc")
    {
        DeviceResource handle{ any_device_resource{ PoolStyleFailingResource{} } };
        REQUIRE_THROWS_AS(handle.allocate(kStream, 1024), std::exception);
        REQUIRE_THROWS_AS(handle.allocate(kStream, 1024), std::runtime_error);
    }
}

TEST_CASE("Copying DeviceResource shares one resource", "[test_device_resource.cpp]")
{
    // Cached values hold a copy of the handle and can outlive the cache, so a
    // copy must free through the instance that allocated.
    TrackingResource::State state;

    SECTION("a copy frees through the instance that allocated")
    {
        DeviceResource original{ any_device_resource{ TrackingResource{ state } } };
        // Copied before allocating, so this cannot collapse into a move.
        DeviceResource copy = original;
        REQUIRE(copy.resource() == original.resource());

        constexpr std::size_t nbytes = 2048;
        void* ptr = original.allocate(kStream, nbytes);
        REQUIRE(ptr != nullptr);
        copy.deallocate(kStream, ptr, nbytes);

        REQUIRE(state.allocate_calls == 1);
        REQUIRE(state.deallocate_calls == 1);
        REQUIRE_FALSE(state.freed_through_a_different_instance);
        REQUIRE_FALSE(state.saw_foreign_pointer);
        REQUIRE(state.outstanding.empty());
    }

    SECTION("copy assignment shares too")
    {
        DeviceResource original{ any_device_resource{ TrackingResource{ state } } };
        DeviceResource other{ any_device_resource{ TrackingResource{ state } } };
        other = original;
        REQUIRE(other.resource() == original.resource());
    }

    SECTION("a copy outlives the original")
    {
        constexpr std::size_t nbytes = 2048;
        void* ptr = nullptr;
        DeviceResource copy{ any_device_resource{ TrackingResource{ state } } };
        {
            DeviceResource original{ any_device_resource{ TrackingResource{ state } } };
            copy = original;
            ptr = original.allocate(kStream, nbytes);
        }
        // The resource the original owned is still alive, held by the copy.
        copy.deallocate(kStream, ptr, nbytes);

        REQUIRE_FALSE(state.freed_through_a_different_instance);
        REQUIRE_FALSE(state.saw_foreign_pointer);
        REQUIRE(state.outstanding.empty());
    }
}

TEST_CASE("DeviceResource can borrow a resource it does not own", "[test_device_resource.cpp]")
{
    TrackingResource::State state;
    TrackingResource resource{ state };
    DeviceResource handle{ device_resource_ref{ resource } };

    constexpr std::size_t nbytes = 256;
    void* ptr = handle.allocate(kStream, nbytes);
    REQUIRE(ptr != nullptr);
    REQUIRE(state.allocate_calls == 1);

    // A copy of a borrowing handle borrows the same resource.
    DeviceResource copy = handle;
    REQUIRE(copy.resource() == handle.resource());
    copy.deallocate(kStream, ptr, nbytes);

    REQUIRE(state.deallocate_calls == 1);
    REQUIRE_FALSE(state.freed_through_a_different_instance);
    REQUIRE_FALSE(state.saw_foreign_pointer);
    REQUIRE(state.outstanding.empty());
}

TEST_CASE("DeviceResource orders each call on the stream it is given", "[test_device_resource.cpp]")
{
    // The handle holds no stream, so allocating a second buffer on B cannot
    // move the release of a buffer still in use on A.
    TrackingResource::State state;
    DeviceResource handle{ any_device_resource{ TrackingResource{ state } } };
    const auto a = fake_stream(0x1000);
    const auto b = fake_stream(0x3000);

    void* on_a = handle.allocate(a, 512);
    REQUIRE(state.last_stream == a.get());
    void* on_b = handle.allocate(b, 512);
    REQUIRE(state.last_stream == b.get());

    handle.deallocate(a, on_a, 512);
    REQUIRE(state.last_stream == a.get());
    handle.deallocate(b, on_b, 512);
    REQUIRE(state.last_stream == b.get());
    REQUIRE(state.outstanding.empty());
}

TEST_CASE("CudaMemoryResource rejects alignments cudaMalloc cannot honor", "[test_device_resource.cpp]")
{
    // Rejected before cudaMalloc is reached, so no device is needed.
    CudaMemoryResource resource;
    for (std::size_t alignment : { std::size_t{ 0 }, std::size_t{ 3 }, std::size_t{ 512 }, std::size_t{ 4096 } })
    {
        CAPTURE(alignment);
        REQUIRE_THROWS_AS(resource.allocate_sync(16, alignment), std::invalid_argument);
        REQUIRE_THROWS_AS(resource.allocate(fake_stream(0x1000), 16, alignment), std::invalid_argument);
    }
}
