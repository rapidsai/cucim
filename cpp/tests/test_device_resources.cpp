/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cucim/memory/device_resources.h"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <new>
#include <unordered_map>
#include <utility>
#include <vector>

using cucim::memory::any_device_resource;
using cucim::memory::CudaMemoryResource;
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
 */
class TrackingResource
{
public:
    void* allocate_sync(std::size_t bytes, std::size_t alignment)
    {
        ++allocate_calls;
        last_alignment = alignment;
        void* ptr = ::operator new(bytes);
        outstanding[ptr] = bytes;
        return ptr;
    }

    void deallocate_sync(void* ptr, std::size_t bytes, std::size_t)
    {
        ++deallocate_calls;
        auto it = outstanding.find(ptr);
        // A resource is only able to reclaim a block if it is told the size it
        // handed out; a pool would corrupt its free lists otherwise.
        if (it != outstanding.end())
        {
            size_matched_on_free = size_matched_on_free && (it->second == bytes);
            outstanding.erase(it);
        }
        else
        {
            saw_foreign_pointer = true;
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
        return this == &other;
    }
    bool operator!=(const TrackingResource& other) const noexcept
    {
        return this != &other;
    }

    friend void get_property(const TrackingResource&, cucim::memory::device_accessible) noexcept
    {
    }

    int allocate_calls = 0;
    int deallocate_calls = 0;
    std::size_t last_alignment = 0;
    bool size_matched_on_free = true;
    bool saw_foreign_pointer = false;
    std::unordered_map<void*, std::size_t> outstanding;
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
 * Reach the resource that *resources* actually allocates from.
 *
 * `any_device_resource` stores the resource by value and DeviceResources takes
 * it by value in turn, so the object handed to the constructor is not the one
 * serving requests. Anything inspecting a stateful resource has to go through
 * the handle, which is what this does. The returned pointer refers into the
 * handle, so it stays valid for as long as *resources* does.
 */
TrackingResource* tracker_of(DeviceResources& resources)
{
    auto ref = resources.resource();
    return ::cuda::mr::resource_cast<TrackingResource>(&ref);
}

} // namespace

// The whole point of adopting CCCL rather than inventing another allocator
// interface is that a resource written elsewhere - by RMM, by CCCL, or by a
// user - drops in. If cuCIM's own resources stopped satisfying the concepts,
// none of them would be interchangeable any more.
TEST_CASE("cuCIM resources satisfy the CCCL concepts", "[test_device_resources.cpp]")
{
    STATIC_REQUIRE(::cuda::mr::synchronous_resource<CudaMemoryResource>);
    STATIC_REQUIRE(::cuda::mr::resource<CudaMemoryResource>);
    STATIC_REQUIRE(::cuda::mr::synchronous_resource<TrackingResource>);
    STATIC_REQUIRE(::cuda::mr::resource<TrackingResource>);

    // device_accessible is what cuCIM requires of a resource, since the
    // pointers reach CUDA kernels and __cuda_array_interface__ consumers.
    STATIC_REQUIRE(::cuda::has_property<CudaMemoryResource, cucim::memory::device_accessible>);
    STATIC_REQUIRE(::cuda::has_property<TrackingResource, cucim::memory::device_accessible>);
}

TEST_CASE("DeviceResources allocates from the resource it was given", "[test_device_resources.cpp]")
{
    SECTION("allocation and deallocation reach the supplied resource")
    {
        DeviceResources resources{ any_device_resource{ TrackingResource{} } };
        auto* tracker = tracker_of(resources);
        REQUIRE(tracker != nullptr);

        constexpr std::size_t nbytes = 4096;
        void* ptr = resources.allocate(nbytes);
        REQUIRE(ptr != nullptr);
        REQUIRE(tracker->allocate_calls == 1);
        REQUIRE(tracker->last_alignment == cucim::memory::kDefaultAlignment);

        resources.deallocate(ptr, nbytes);
        REQUIRE(tracker->deallocate_calls == 1);
        REQUIRE(tracker->outstanding.empty());
    }

    SECTION("a zero-byte request never reaches the resource")
    {
        // Callers ask for empty regions, and cudaMalloc(0) is not something
        // every resource tolerates, so this is filtered out before dispatch.
        DeviceResources resources{ any_device_resource{ TrackingResource{} } };
        auto* tracker = tracker_of(resources);

        REQUIRE(resources.allocate(0) == nullptr);
        REQUIRE(tracker->allocate_calls == 0);
    }

    SECTION("deallocating nullptr is a no-op")
    {
        DeviceResources resources{ any_device_resource{ TrackingResource{} } };
        auto* tracker = tracker_of(resources);

        resources.deallocate(nullptr, 0);
        REQUIRE(tracker->deallocate_calls == 0);
    }
}

// Every allocation must come back to the same resource carrying the size it
// was given out with. A pool resource needs that size to return the block to
// the right free list, which is the reason deallocate() takes one at all.
TEST_CASE("DeviceResources pairs allocations with sized deallocations", "[test_device_resources.cpp]")
{
    DeviceResources resources{ any_device_resource{ TrackingResource{} } };
    auto* tracker = tracker_of(resources);
    REQUIRE(tracker != nullptr);

    std::vector<std::pair<void*, std::size_t>> blocks;
    for (std::size_t bytes : { std::size_t{ 512 }, std::size_t{ 4096 }, std::size_t{ 1u << 20 } })
    {
        void* ptr = resources.allocate(bytes);
        REQUIRE(ptr != nullptr);
        blocks.emplace_back(ptr, bytes);
    }
    REQUIRE(tracker->allocate_calls == 3);

    for (auto& [ptr, bytes] : blocks)
    {
        resources.deallocate(ptr, bytes);
    }

    REQUIRE(tracker->deallocate_calls == 3);
    REQUIRE(tracker->outstanding.empty());
    REQUIRE(tracker->size_matched_on_free);
    REQUIRE_FALSE(tracker->saw_foreign_pointer);
}

TEST_CASE("DeviceResources propagates allocation failure as bad_alloc", "[test_device_resources.cpp]")
{
    DeviceResources resources{ any_device_resource{ FailingResource{} } };

    // Callers such as the tile cache rely on this being an exception rather
    // than a null return, so they can decide whether to fall back.
    REQUIRE_THROWS_AS(resources.allocate(1024), std::bad_alloc);
}

TEST_CASE("DeviceResources is copyable and keeps its resource alive", "[test_device_resources.cpp]")
{
    // Cached tile values hold a copy of the handle precisely so that they can
    // outlive the cache that allocated them, so a copy has to remain usable
    // after the original is gone.
    void* ptr = nullptr;
    constexpr std::size_t nbytes = 2048;

    DeviceResources copy = [&] {
        DeviceResources original{ any_device_resource{ TrackingResource{} } };
        ptr = original.allocate(nbytes);
        return original;
    }();

    REQUIRE(ptr != nullptr);

    // Freeing through the copy must reach the same resource the original
    // allocated from; an owning handle is what makes this well defined. If the
    // handle held a bare resource_ref instead, this would dangle.
    copy.deallocate(ptr, nbytes);

    auto* tracker = tracker_of(copy);
    REQUIRE(tracker != nullptr);
    REQUIRE(tracker->allocate_calls == 1);
    REQUIRE(tracker->deallocate_calls == 1);
    REQUIRE_FALSE(tracker->saw_foreign_pointer);
    REQUIRE(tracker->outstanding.empty());
}

TEST_CASE("DeviceResources exposes a borrowable resource_ref", "[test_device_resources.cpp]")
{
    // This is the type cuCIM APIs should take, and it is what lets an
    // rmm::device_async_resource_ref be passed straight through.
    // Uses a tracking resource so the test needs no GPU.
    DeviceResources resources{ any_device_resource{ TrackingResource{} } };
    device_resource_ref ref = resources.resource();

    void* ptr = ref.allocate(resources.stream(), 256, cucim::memory::kDefaultAlignment);
    REQUIRE(ptr != nullptr);
    ref.deallocate(resources.stream(), ptr, 256, cucim::memory::kDefaultAlignment);
}
