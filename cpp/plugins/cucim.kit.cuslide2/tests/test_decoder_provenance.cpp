/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuslide/nvimgcodec/nvimgcodec_decoder.h"
#include "cuslide/tiff/tiff.h"
#include "config.h"

#include <catch2/catch_test_macros.hpp>
#include <cucim/memory/memory_manager.h>
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <cstdio>
#include <fcntl.h>
#include <vector>

using cuslide2::nvimgcodec::BackendProbe;
using cuslide2::nvimgcodec::decode_provenance_json;
using cuslide2::nvimgcodec::DecodeBackend;
using cuslide2::nvimgcodec::DecodeProvenance;
using cuslide2::nvimgcodec::select_decode_backend;

TEST_CASE("decoder backend selection distinguishes execution classes", "[decoder_provenance]")
{
    SECTION("GPU")
    {
        REQUIRE(select_decode_backend({ { DecodeBackend::kGpuOnly, 1, true } }) == DecodeBackend::kGpuOnly);
    }
    SECTION("hybrid CPU and GPU")
    {
        REQUIRE(select_decode_backend({
                    { DecodeBackend::kGpuOnly, 0x13, true },
                    { DecodeBackend::kHybridCpuGpu, 1, true },
                }) == DecodeBackend::kHybridCpuGpu);
    }
    SECTION("CPU fallback")
    {
        REQUIRE(select_decode_backend({
                    { DecodeBackend::kGpuOnly, 0x13, true },
                    { DecodeBackend::kHybridCpuGpu, 0x13, true },
                    { DecodeBackend::kCpuOnly, 1, true },
                }) == DecodeBackend::kCpuOnly);
    }
    SECTION("unavailable codec")
    {
        REQUIRE(select_decode_backend({
                    { DecodeBackend::kGpuOnly, 0xb, true },
                    { DecodeBackend::kHybridCpuGpu, 0xb, true },
                    { DecodeBackend::kCpuOnly, 0xb, true },
                }) == DecodeBackend::kUnknown);
    }
}

TEST_CASE("provenance JSON preserves unknown backend and decoder failure", "[decoder_provenance]")
{
    DecodeProvenance unknown;
    unknown.codec = "jpeg";
    unknown.success = true;
    unknown.output_memory = "host";
    unknown.cuda_stream = "none";
    unknown.execution_detail = "generic_dispatch_backend_not_exposed";

    DecodeProvenance failure;
    failure.codec = "jpeg2000";
    failure.backend = DecodeBackend::kHybridCpuGpu;
    failure.processing_status = 0xb;
    failure.failure_reason = "no_available_decoder_backend";
    failure.output_memory = "cuda";
    failure.cuda_stream = "dedicated";

    const auto document = nlohmann::json::parse(decode_provenance_json({ unknown, failure }));
    const auto& samples = document.at("decoder_provenance").at("samples");

    REQUIRE(samples.at(0).at("backend") == "unknown");
    REQUIRE(samples.at(0).at("execution") == "unknown");
    REQUIRE(samples.at(0).at("execution_detail") == "generic_dispatch_backend_not_exposed");
    REQUIRE(samples.at(0).at("failure_reason").is_null());
    REQUIRE(samples.at(0).at("plugin_id").is_null());
    REQUIRE(samples.at(0).at("plugin_id_status") == "not_exposed_by_nvimagecodec_0_9");

    REQUIRE(samples.at(1).at("execution") == "decoder_failure");
    REQUIRE(samples.at(1).at("processing_status") == "codec_unsupported");
    REQUIRE(samples.at(1).at("failure_reason") == "no_available_decoder_backend");
}

TEST_CASE("TIFF codec names are reported without claiming an implementation", "[decoder_provenance]")
{
    for (const char* codec : { "jpeg", "jpeg2000", "lzw", "uncompressed" })
    {
        DecodeProvenance provenance;
        provenance.codec = codec;
        provenance.backend = DecodeBackend::kCpuOnly;
        provenance.processing_status = 1;
        provenance.success = true;
        provenance.output_memory = "host";
        provenance.cuda_stream = "none";

        const auto document = nlohmann::json::parse(decode_provenance_json({ provenance }));
        const auto& sample = document.at("decoder_provenance").at("samples").at(0);
        REQUIRE(sample.at("codec") == codec);
        REQUIRE(sample.at("execution") == "cpu_fallback");
        REQUIRE(sample.at("plugin_id").is_null());
    }
}

TEST_CASE("TIFF reads attach decoder provenance", "[decoder_provenance][integration]")
{
    if (g_config.test_file.empty())
    {
        SKIP("pass --test_file to exercise a public TIFF fixture");
    }

    const std::string input_path = g_config.get_input_path();
    auto tif = std::make_shared<cuslide::tiff::TIFF>(input_path.c_str(), O_RDONLY);
    tif->construct_ifds();

    cucim::io::format::ImageMetadata metadata{};
    metadata.level_count(1).level_downsamples({ 1.0 }).level_ndim(3);

    int64_t location[2] = { 0, 0 };
    int64_t size[2] = { 64, 64 };
    cucim::io::format::ImageReaderRegionRequestDesc request{};
    request.location = location;
    request.size = size;
    request.device = const_cast<char*>("cpu");

    cucim::io::format::ImageDataDesc image_data{};
    cucim::io::format::ImageMetadata read_metadata{};
    REQUIRE(tif->read(&metadata.desc(), &request, &image_data, &read_metadata.desc()));
    REQUIRE(read_metadata.desc().json_data != nullptr);

    const auto document = nlohmann::json::parse(read_metadata.desc().json_data);
    if (std::getenv("CUCIM_PRINT_DECODER_PROVENANCE"))
    {
        std::printf("decoder_provenance=%s\n", read_metadata.desc().json_data);
    }
    const auto& report = document.at("decoder_provenance");
    REQUIRE(report.at("samples").empty() == false);
    const auto& sample = report.at("samples").at(0);
    REQUIRE(sample.at("codec").is_string());
    if (const char* expected_codec = std::getenv("CUCIM_TEST_CODEC"))
    {
        REQUIRE(sample.at("codec") == expected_codec);
    }
    if (const char* expected_execution = std::getenv("CUCIM_TEST_EXECUTION"))
    {
        REQUIRE(sample.at("execution") == expected_execution);
    }
    REQUIRE(sample.at("backend").is_string());
    REQUIRE(sample.at("execution").is_string());
    REQUIRE(sample.at("plugin_id").is_null());
    REQUIRE(sample.at("boundaries").at("output_memory") == "host");

    tif->close();
    cucim_free(image_data.container.data);
    // TIFF's production lifetime is owned by its CuCIMFileHandle callback.
    // This test owns TIFF through shared_ptr, so disarm that callback before
    // shared_ptr destroys the same object.
    tif->file_handle()->client_data = nullptr;
}
