/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "config.h"
#include "cuslide/tiff/tiff.h"

#include <cucim/codec/hash_function.h>
#include <cucim/cuimage.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cuda_runtime.h>

#include <chrono>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace
{

struct RestoreCacheConfig
{
    cucim::cache::ImageCacheConfig config = cucim::CuImage::cache()->get_config();

    ~RestoreCacheConfig()
    {
        cucim::CuImage::cache(config);
    }
};

// Exercise IFD::read directly so the test always uses cuslide2, independently
// of which plugin the process selects for the public CuImage API.
std::vector<uint8_t> read_device_pixels(cuslide::tiff::TIFF& tiff,
                                        cuslide::tiff::IFD& ifd,
                                        const cucim::io::format::ImageReaderRegionRequestDesc& request)
{
    struct Raster
    {
        cucim::io::format::ImageDataDesc data{};
        ~Raster()
        {
            cudaFree(data.container.data);
            cucim_free(data.container.shape);
        }
    } raster;
    cucim::io::format::ImageMetadata metadata{};
    metadata.level_count(1).level_downsamples({ 1.0 }).level_ndim(3);
    if (!ifd.read(&tiff, &metadata.desc(), &request, &raster.data))
    {
        throw std::runtime_error("GPU region read failed");
    }
    const size_t size = request.size[0] * request.size[1] * ifd.samples_per_pixel() * ifd.bits_per_sample() / 8;
    std::vector<uint8_t> pixels(size);
    if (cudaMemcpy(pixels.data(), raster.data.container.data, size, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        throw std::runtime_error("GPU region copy failed");
    }
    return pixels;
}

} // namespace

TEST_CASE("GPU reads retain their cache across reconfiguration", "[device_cache]")
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
    {
        SKIP("A CUDA device is required");
    }

    const auto replacement_type = GENERATE(cucim::cache::CacheType::kNoCache, cucim::cache::CacheType::kPerProcess);
    const uint64_t paused_tile = GENERATE(0, 1);
    const bool keep_explicit_handle = GENERATE(false, true);
    CAPTURE(replacement_type, paused_tile, keep_explicit_handle);

    RestoreCacheConfig restore;
    cucim::cache::ImageCacheConfig config;
    config.type = cucim::cache::CacheType::kPerProcess;
    config.memory_capacity = 64;
    config.capacity = 4;
    // Adjacent tiles use different mutexes, so tile 1 can be held while tile 0
    // completes. Holding tile 0 instead pauses before the first lookup.
    config.mutex_pool_capacity = 2;
    config.record_stat = true;
    cucim::CuImage::cache(config);

    auto tiff = cuslide::tiff::TIFF::open(g_config.get_input_path());
    tiff->construct_ifds();
    auto ifd = tiff->ifd(0);
    REQUIRE(ifd->tile_width() > 0);
    REQUIRE(ifd->tile_height() > 0);
    REQUIRE(ifd->width() >= 2 * ifd->tile_width());
    REQUIRE(ifd->height() >= ifd->tile_height());

    int64_t location[] = { 0, 0 };
    int64_t size[] = { 2 * static_cast<int64_t>(ifd->tile_width()), ifd->tile_height() };
    cucim::io::format::ImageReaderRegionRequestDesc request{};
    request.location = location;
    request.location_len = 1;
    request.size = size;
    request.device = const_cast<char*>("cuda");

    const auto expected = read_device_pixels(*tiff, *ifd, request);
    auto old_cache = cucim::CuImage::device_cache();
    REQUIRE(old_cache->size() == 2);
    const uint64_t ifd_hash = tiff->file_handle()->hash_value ^ cucim::codec::splitmix64(0);
    auto key = old_cache->create_key(ifd_hash, 0);
    auto value = old_cache->find(key);
    REQUIRE(value);
    REQUIRE(value->device_type == cucim::io::DeviceType::kCUDA);
    std::weak_ptr<cucim::cache::ImageCacheValue> old_tile = value;
    value.reset();
    old_cache->record(true);
    std::weak_ptr<cucim::cache::ImageCache> old_owner = old_cache;

    const uint64_t lock_hash = ifd_hash ^ (paused_tile | (paused_tile << 32));
    // Destruction order matters on failure: unlock before waiting for the
    // reader's future, and keep old_cache alive until that reader has joined.
    std::future<std::vector<uint8_t>> reader;
    std::unique_lock<std::mutex> tile_lock(*static_cast<std::mutex*>(old_cache->mutex(lock_hash)));
    reader = std::async(std::launch::async, [&] { return read_device_pixels(*tiff, *ifd, request); });

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while ((old_owner.use_count() < 3 || old_cache->hit_count() != paused_tile) &&
           std::chrono::steady_clock::now() < deadline)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // Manager + test + reader must all own the selected cache. In particular,
    // the test's handle must not mask a borrowed reference in IFD::read.
    const bool reader_owns_cache = old_owner.use_count() >= 3 && old_cache->hit_count() == paused_tile;
    if (!reader_owns_cache)
    {
        tile_lock.unlock();
        reader.get();
        FAIL("GPU reader did not retain its cache before the paused lookup");
    }

    config.type = replacement_type;
    cucim::CuImage::cache(config);
    auto replacement = cucim::CuImage::device_cache();
    REQUIRE(replacement != old_cache);
    REQUIRE(replacement->type() == replacement_type);

    std::shared_ptr<cucim::cache::ImageCache> explicit_handle;
    if (keep_explicit_handle)
    {
        explicit_handle = old_cache;
    }
    old_cache.reset();
    REQUIRE_FALSE(old_owner.expired());
    REQUIRE_FALSE(old_tile.expired());
    tile_lock.unlock();

    REQUIRE(reader.get() == expected);
    REQUIRE(replacement->hit_count() == 0);
    REQUIRE(replacement->miss_count() == 0);
    REQUIRE(old_owner.expired() == !keep_explicit_handle);
    REQUIRE(old_tile.expired() == !keep_explicit_handle);
    explicit_handle.reset();
    REQUIRE(old_owner.expired());
    REQUIRE(old_tile.expired());

    // Later reads use the replacement, while the paused read used only the old
    // cache. Device tile allocations are released with the final old owner.
    REQUIRE(read_device_pixels(*tiff, *ifd, request) == expected);
    REQUIRE(replacement->hit_count() == 0);
    REQUIRE(replacement->miss_count() == (replacement_type == cucim::cache::CacheType::kPerProcess ? 2 : 0));
}
