/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuslide/nvimgcodec/nvimgcodec_tiff_parser.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#ifdef CUCIM_HAS_NVIMGCODEC

#    include <nvimgcodec.h>

#    include <cstdint>
#    include <string>

using cuslide2::nvimgcodec::format_nvimgcodec_version;
using cuslide2::nvimgcodec::is_supported_nvimgcodec_version;

// cuCIM checks the nvImageCodec version at run time because the build cannot:
// the library is dlopen'd by default and the SONAMEs searched end with
// libnvimgcodec.so.0 and libnvimgcodec.so, which resolve to whichever major-0
// build is installed. These tests pin the range that check enforces.
//
// The range must stay in step with the pin in dependencies.yaml and with the
// one cmake/deps/nvimgcodec.cmake enforces on headers. These tests do not read
// those files; detecting packaging-pin drift requires a separate check.
TEST_CASE("supported nvImageCodec runtime range is 0.10.x", "[test_nvimgcodec_version.cpp]")
{
    SECTION("versions below 0.10.0 are rejected")
    {
        // 0.9 is not interchangeable with 0.10: nvimgcodecExecutionParams_t
        // gained the plugin-selection fields in 0.10 and the library now
        // rejects an execution-parameter struct_size that does not match its
        // own header, so a 0.10 build cannot drive a 0.9 library or vice
        // versa.
        REQUIRE_FALSE(is_supported_nvimgcodec_version(NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 9, 0)));
        REQUIRE_FALSE(is_supported_nvimgcodec_version(NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 9, 99)));
        REQUIRE_FALSE(is_supported_nvimgcodec_version(NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 8, 0)));
    }

    SECTION("0.10.x is accepted")
    {
        REQUIRE(is_supported_nvimgcodec_version(NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 10, 0)));
        REQUIRE(is_supported_nvimgcodec_version(NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 10, 1)));
        REQUIRE(is_supported_nvimgcodec_version(NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 10, 99)));
    }

    SECTION("0.11.0 and later are rejected")
    {
        REQUIRE_FALSE(is_supported_nvimgcodec_version(NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 11, 0)));
        REQUIRE_FALSE(is_supported_nvimgcodec_version(NVIMGCODEC_MAKE_SEMANTIC_VERSION(1, 0, 0)));
        REQUIRE_FALSE(is_supported_nvimgcodec_version(NVIMGCODEC_MAKE_SEMANTIC_VERSION(2, 3, 4)));
    }

    SECTION("the headers cuCIM is built against are themselves in range")
    {
        // If this fails, the vendored nvimgcodec_version.h was bumped out of
        // the supported range without the range being updated, which would
        // reject every install including the intended one.
        REQUIRE(is_supported_nvimgcodec_version(NVIMGCODEC_VER));
    }
}

// The version compared at run time is produced by the loaded library, not by
// these headers, so a library older than 0.10 reports a value packed as
// major * 1000 + minor * 100 + patch. 0.10 widened that to
// major * 10000 + minor * 100 + patch to stop a two-digit minor overflowing
// into the major. Every released 0.x packs identically under both, which is
// what lets the check reject an old library by version rather than by
// misreading it as something in range.
TEST_CASE("packed nvImageCodec versions survive the 0.10 encoding change", "[test_nvimgcodec_version.cpp]")
{
    constexpr auto narrow_packing = [](uint32_t major, uint32_t minor, uint32_t patch) {
        return (major * 1000u) + (minor * 100u) + patch;
    };

    SECTION("released 0.x versions pack to the same value under both encodings")
    {
        REQUIRE(narrow_packing(0, 8, 0) == NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 8, 0));
        REQUIRE(narrow_packing(0, 9, 0) == NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 9, 0));
        REQUIRE(narrow_packing(0, 9, 99) == NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 9, 99));
    }

    SECTION("the widened encoding orders versions and no longer collides")
    {
        REQUIRE(NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 9, 99) < NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 10, 0));
        REQUIRE(NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 10, 0) < NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 11, 0));
        REQUIRE(NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 11, 0) < NVIMGCODEC_MAKE_SEMANTIC_VERSION(1, 0, 0));

        // The narrow encoding reported 0.10.0 as 1.0.0; the widened one does
        // not, which is why diagnostics can now name an exact version.
        REQUIRE(narrow_packing(0, 10, 0) == narrow_packing(1, 0, 0));
        REQUIRE(NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 10, 0) != NVIMGCODEC_MAKE_SEMANTIC_VERSION(1, 0, 0));
    }
}

TEST_CASE("nvImageCodec version diagnostics name an exact version", "[test_nvimgcodec_version.cpp]")
{
    SECTION("versions inside the supported range")
    {
        REQUIRE(format_nvimgcodec_version(NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 10, 0)) == "0.10.0");
        REQUIRE(format_nvimgcodec_version(NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 10, 99)) == "0.10.99");
    }

    SECTION("versions outside it, which are the ones a user has to act on")
    {
        REQUIRE(format_nvimgcodec_version(NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 8, 0)) == "0.8.0");
        REQUIRE(format_nvimgcodec_version(NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 9, 0)) == "0.9.0");
        REQUIRE(format_nvimgcodec_version(NVIMGCODEC_MAKE_SEMANTIC_VERSION(1, 0, 0)) == "1.0.0");
        REQUIRE(format_nvimgcodec_version(NVIMGCODEC_MAKE_SEMANTIC_VERSION(2, 3, 4)) == "2.3.4");
    }

    SECTION("anything the diagnostic can name outside the range is still rejected")
    {
        const uint32_t packed_version = GENERATE(0u, 800u, 900u, 1100u, 10000u, 20304u);
        REQUIRE_FALSE(is_supported_nvimgcodec_version(packed_version));
    }
}

#endif // CUCIM_HAS_NVIMGCODEC
