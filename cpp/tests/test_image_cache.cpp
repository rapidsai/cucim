/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cucim/cache/image_cache_manager.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>

namespace
{

std::shared_ptr<cucim::cache::ImageCacheValue> create_cache_value(cucim::cache::ImageCache& cache, uint8_t byte)
{
    auto* data = static_cast<uint8_t*>(cache.allocate(1));
    REQUIRE(data != nullptr);
    data[0] = byte;
    return cache.create_value(data, 1, cucim::io::DeviceType::kCPU);
}

} // namespace

TEST_CASE("Per-process image cache should distinguish file hash for matching tile locations", "[test_image_cache.cpp]")
{
    cucim::cache::ImageCacheConfig config{};
    config.type = cucim::cache::CacheType::kPerProcess;
    config.capacity = 4;
    config.memory_capacity = 1;
    config.list_padding = 1;

    auto cache = cucim::cache::ImageCacheManager::create_cache(config, cucim::io::DeviceType::kCPU);

    auto first_key = cache->create_key(0x101, 42);
    auto second_key = cache->create_key(0x202, 42);
    auto first_value = create_cache_value(*cache, 0x11);
    auto second_value = create_cache_value(*cache, 0x22);

    REQUIRE(cache->insert(first_key, first_value));
    REQUIRE(cache->insert(second_key, second_value));

    auto found_first = cache->find(first_key);
    auto found_second = cache->find(second_key);

    REQUIRE(found_first);
    REQUIRE(found_second);
    REQUIRE(found_first != found_second);
    REQUIRE(static_cast<uint8_t*>(found_first->data)[0] == 0x11);
    REQUIRE(static_cast<uint8_t*>(found_second->data)[0] == 0x22);
}
