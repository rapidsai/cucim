/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cucim/memory/device_resources.h"

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
using cucim::memory::DeviceResources;
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
 * `DeviceResources`, which type-erases it.
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

    void* allocate(::cuda::stream_ref, std::size_t bytes, std::size_t alignment)
    {
        return allocate_sync(bytes, alignment);
    }

    void deallocate(::cuda::stream_ref, void* ptr, std::size_t bytes, std::size_t alignment)
    {
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

/// A stream handle that is never submitted to, only compared.
::cuda::stream_ref fake_stream(std::uintptr_t id)
{
    return ::cuda::stream_ref{ reinterpret_cast<cudaStream_t>(id) };
}

} // namespace

// The whole point of adopting CCCL rather than inventing another allocator
// interface is that a resource written elsewhere - by RMM, by CCCL, or by a
// user - drops in. If cuCIM's own resources stopped satisfying the concepts,
// none of them would be interchangeable any more.
TEST_CASE("cuCIM resources satisfy the CCCL concepts", "[test_device_resources.cpp]")
{
    STATIC_REQUIRE(::cuda::mr::synchronous_resource<DeviceMemoryPoolResource>);
    STATIC_REQUIRE(::cuda::mr::resource<DeviceMemoryPoolResource>);
    STATIC_REQUIRE(::cuda::mr::synchronous_resource<CudaMemoryResource>);
    STATIC_REQUIRE(::cuda::mr::resource<CudaMemoryResource>);
    STATIC_REQUIRE(::cuda::mr::synchronous_resource<TrackingResource>);
    STATIC_REQUIRE(::cuda::mr::resource<TrackingResource>);

    // device_accessible is what cuCIM requires of a resource, since the
    // pointers reach CUDA kernels and __cuda_array_interface__ consumers.
    STATIC_REQUIRE(::cuda::has_property<DeviceMemoryPoolResource, cucim::memory::device_accessible>);
    STATIC_REQUIRE(::cuda::has_property<CudaMemoryResource, cucim::memory::device_accessible>);
    STATIC_REQUIRE(::cuda::has_property<TrackingResource, cucim::memory::device_accessible>);
}

TEST_CASE("DeviceResources allocates from the resource it was given", "[test_device_resources.cpp]")
{
    TrackingResource::State state;
    DeviceResources resources{ any_device_resource{ TrackingResource{ state } } };

    SECTION("allocation and deallocation reach the supplied resource")
    {
        constexpr std::size_t nbytes = 4096;
        void* ptr = resources.allocate(nbytes);
        REQUIRE(ptr != nullptr);
        REQUIRE(state.allocate_calls == 1);
        REQUIRE(state.last_alignment == cucim::memory::kDefaultAlignment);

        resources.deallocate(ptr, nbytes);
        REQUIRE(state.deallocate_calls == 1);
        REQUIRE(state.outstanding.empty());
    }

    SECTION("an explicit alignment is passed through")
    {
        constexpr std::size_t nbytes = 128;
        constexpr std::size_t alignment = 4096;
        void* ptr = resources.allocate(nbytes, alignment);
        REQUIRE(state.last_alignment == alignment);

        resources.deallocate(ptr, nbytes, alignment);
        REQUIRE(state.outstanding.empty());
        REQUIRE_FALSE(state.saw_foreign_pointer);
    }

    SECTION("a zero-byte request is still the resource's decision")
    {
        // Not filtered out here: a resource is entitled to define what an
        // empty allocation means, and short-circuiting would hand back a
        // pointer the resource never issued and would then be asked to free.
        void* ptr = resources.allocate(0);
        REQUIRE(state.allocate_calls == 1);

        resources.deallocate(ptr, 0);
        REQUIRE(state.deallocate_calls == 1);
        REQUIRE(state.outstanding.empty());
    }
}

// Every allocation must come back to the same resource carrying the size it
// was given out with. A pool resource needs that size to return the block to
// the right free list, which is the reason deallocate() takes one at all.
TEST_CASE("DeviceResources pairs allocations with sized deallocations", "[test_device_resources.cpp]")
{
    TrackingResource::State state;
    DeviceResources resources{ any_device_resource{ TrackingResource{ state } } };

    std::vector<std::pair<void*, std::size_t>> blocks;
    for (std::size_t bytes : { std::size_t{ 512 }, std::size_t{ 4096 }, std::size_t{ 1u << 20 } })
    {
        void* ptr = resources.allocate(bytes);
        REQUIRE(ptr != nullptr);
        blocks.emplace_back(ptr, bytes);
    }
    REQUIRE(state.allocate_calls == 3);

    for (auto& [ptr, bytes] : blocks)
    {
        resources.deallocate(ptr, bytes);
    }

    REQUIRE(state.deallocate_calls == 3);
    REQUIRE(state.outstanding.empty());
    REQUIRE(state.size_matched_on_free);
    REQUIRE_FALSE(state.saw_foreign_pointer);
}

TEST_CASE("DeviceResources propagates allocation failure", "[test_device_resources.cpp]")
{
    // Callers such as the tile cache rely on this being an exception rather
    // than a null return, so they can decide whether to fall back.
    SECTION("a resource that throws bad_alloc")
    {
        DeviceResources resources{ any_device_resource{ FailingResource{} } };
        REQUIRE_THROWS_AS(resources.allocate(1024), std::bad_alloc);
    }

    SECTION("a pool-style resource, which does not throw bad_alloc")
    {
        // The reason callers have to catch std::exception: nothing requires a
        // resource to signal exhaustion with bad_alloc, and the default pool
        // does not.
        DeviceResources resources{ any_device_resource{ PoolStyleFailingResource{} } };
        REQUIRE_THROWS_AS(resources.allocate(1024), std::exception);
        REQUIRE_THROWS_AS(resources.allocate(1024), std::runtime_error);
    }
}

TEST_CASE("Copying DeviceResources shares one resource", "[test_device_resources.cpp]")
{
    // Cached tile values hold a copy of the handle precisely so that they can
    // outlive the cache that allocated them. Copying must therefore share the
    // original resource rather than duplicate it: a copy that carried its own
    // clone of a stateful allocator would free blocks through an allocator
    // that never issued them.
    TrackingResource::State state;

    SECTION("a copy frees through the instance that allocated")
    {
        DeviceResources original{ any_device_resource{ TrackingResource{ state } } };
        // Copied before allocating, so this cannot collapse into a move.
        DeviceResources copy = original;
        REQUIRE(copy.resource() == original.resource());

        constexpr std::size_t nbytes = 2048;
        void* ptr = original.allocate(nbytes);
        REQUIRE(ptr != nullptr);
        copy.deallocate(ptr, nbytes);

        REQUIRE(state.allocate_calls == 1);
        REQUIRE(state.deallocate_calls == 1);
        REQUIRE_FALSE(state.freed_through_a_different_instance);
        REQUIRE_FALSE(state.saw_foreign_pointer);
        REQUIRE(state.outstanding.empty());
    }

    SECTION("copy assignment shares too")
    {
        DeviceResources original{ any_device_resource{ TrackingResource{ state } } };
        DeviceResources other{ any_device_resource{ TrackingResource{ state } } };
        other = original;
        REQUIRE(other.resource() == original.resource());
    }

    SECTION("a copy outlives the original")
    {
        constexpr std::size_t nbytes = 2048;
        void* ptr = nullptr;
        DeviceResources copy{ any_device_resource{ TrackingResource{ state } } };
        {
            DeviceResources original{ any_device_resource{ TrackingResource{ state } } };
            copy = original;
            ptr = original.allocate(nbytes);
        }
        // The resource the original owned is still alive, held by the copy.
        copy.deallocate(ptr, nbytes);

        REQUIRE_FALSE(state.freed_through_a_different_instance);
        REQUIRE_FALSE(state.saw_foreign_pointer);
        REQUIRE(state.outstanding.empty());
    }
}

TEST_CASE("DeviceResources can borrow a resource it does not own", "[test_device_resources.cpp]")
{
    // The path for callers that already hold a resource, such as an
    // rmm::device_async_resource_ref: cuCIM allocates from it without taking
    // a copy, and the caller keeps it alive.
    TrackingResource::State state;
    TrackingResource resource{ state };
    DeviceResources resources{ device_resource_ref{ resource } };

    constexpr std::size_t nbytes = 256;
    void* ptr = resources.allocate(nbytes);
    REQUIRE(ptr != nullptr);
    REQUIRE(state.allocate_calls == 1);

    // A copy of a borrowing handle borrows the same resource.
    DeviceResources copy = resources;
    REQUIRE(copy.resource() == resources.resource());
    copy.deallocate(ptr, nbytes);

    REQUIRE(state.deallocate_calls == 1);
    REQUIRE_FALSE(state.freed_through_a_different_instance);
    REQUIRE_FALSE(state.saw_foreign_pointer);
    REQUIRE(state.outstanding.empty());
}

TEST_CASE("DeviceResources tracks the stream its memory is ordered on", "[test_device_resources.cpp]")
{
    TrackingResource::State state;
    DeviceResources resources{ any_device_resource{ TrackingResource{ state } } };
    REQUIRE(resources.stream() == ::cuda::stream_ref{ cucim::memory::kDefaultStream });

    // Allocating on a stream rebinds the handle, so that a later deallocate()
    // through the same handle is ordered after the work that used the block
    // rather than against whatever stream the handle started on.
    const auto other = fake_stream(0x1000);
    void* ptr = resources.allocate(other, 512);
    REQUIRE(resources.stream() == other);

    resources.deallocate(ptr, 512);
    REQUIRE(state.outstanding.empty());

    SECTION("a handle can be constructed on a specific stream")
    {
        const auto initial = fake_stream(0x2000);
        DeviceResources on_stream{ any_device_resource{ TrackingResource{ state } }, initial };
        REQUIRE(on_stream.stream() == initial);
    }
}
