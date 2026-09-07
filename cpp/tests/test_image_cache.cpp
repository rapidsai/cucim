/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cucim/cache/image_cache.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <set>

using cucim::cache::ImageCacheKey;

// The lock hash selects which mutex of the cache's pool guards an entry, so it
// is deliberately not an identity.  What callers depend on is that it is a pure
// function of the key: every lock(), unlock() and mutex() call for one entry has
// to land on the same mutex, and deriving the hash from the key is what makes
// that true regardless of where in the read path the call happens.
TEST_CASE("ImageCacheKey::lock_hash is a pure function of the key", "[test_image_cache.cpp]")
{
    const uint64_t file_hash = 0x123456789abcdefULL;

    SECTION("repeated calls agree")
    {
        ImageCacheKey key{ file_hash, 42 };
        REQUIRE(key.lock_hash() == key.lock_hash());
    }

    SECTION("separately constructed but equal keys agree")
    {
        // This is the property the read path relies on: a key built during the
        // lookup and a key built later for the insert must lock the same mutex.
        ImageCacheKey first{ file_hash, 42 };
        ImageCacheKey second{ file_hash, 42 };
        REQUIRE(first.lock_hash() == second.lock_hash());
    }

    SECTION("matches what the IFD readers computed inline")
    {
        // Pins the value rather than just the behaviour, so that the existing
        // readers keep mapping each tile onto the mutex they mapped it to
        // before this method existed.
        const uint64_t index = 42;
        ImageCacheKey key{ file_hash, index };
        REQUIRE(key.lock_hash() == (file_hash ^ (index | (index << 32))));
    }
}

TEST_CASE("ImageCacheKey::lock_hash separates neighbouring tiles", "[test_image_cache.cpp]")
{
    SECTION("consecutive tile indices occupy distinct pool slots")
    {
        // Tile assembly walks a contiguous run of tile indices, so if neighbours
        // collided in the mutex pool they would serialize against each other for
        // no reason.  Checked modulo the pool size, which is how the pool is
        // actually indexed.
        constexpr uint64_t pool_capacity = 64;
        const uint64_t file_hash = 0xfeedfacecafebeefULL;

        std::set<uint64_t> slots;
        for (uint64_t index = 0; index < pool_capacity; ++index)
        {
            slots.insert(ImageCacheKey{ file_hash, index }.lock_hash() % pool_capacity);
        }
        REQUIRE(slots.size() == pool_capacity);
    }

    SECTION("the same tile index in different IFDs does not share a slot")
    {
        // Tiles from different IFDs are independent entries; they should not
        // contend just because they sit at the same grid position.
        constexpr uint64_t index = 7;
        ImageCacheKey first{ 0x1111111111111111ULL, index };
        ImageCacheKey second{ 0x2222222222222222ULL, index };
        REQUIRE(first.lock_hash() != second.lock_hash());
    }
}
