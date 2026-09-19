/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
TEST_CASE("supported nvImageCodec runtime range is 0.9.x", "[test_nvimgcodec_version.cpp]")
{
    SECTION("versions below 0.9.0 are rejected")
    {
        // 0.8 is the case that prompted this check: it was still listed as a
        // SONAME candidate after the pin had moved to 0.9, and the removal of
        // limit_images in 0.9 means the two are not interchangeable.
        REQUIRE_FALSE(is_supported_nvimgcodec_version(MAKE_SEMANTIC_VERSION(0, 8, 0)));
        REQUIRE_FALSE(is_supported_nvimgcodec_version(MAKE_SEMANTIC_VERSION(0, 8, 99)));
        REQUIRE_FALSE(is_supported_nvimgcodec_version(MAKE_SEMANTIC_VERSION(0, 7, 0)));
    }

    SECTION("0.9.x is accepted")
    {
        REQUIRE(is_supported_nvimgcodec_version(MAKE_SEMANTIC_VERSION(0, 9, 0)));
        REQUIRE(is_supported_nvimgcodec_version(MAKE_SEMANTIC_VERSION(0, 9, 1)));
        REQUIRE(is_supported_nvimgcodec_version(MAKE_SEMANTIC_VERSION(0, 9, 99)));
    }

    SECTION("0.10.0 and later are rejected")
    {
        REQUIRE_FALSE(is_supported_nvimgcodec_version(MAKE_SEMANTIC_VERSION(0, 10, 0)));
        REQUIRE_FALSE(is_supported_nvimgcodec_version(MAKE_SEMANTIC_VERSION(1, 0, 0)));
        REQUIRE_FALSE(is_supported_nvimgcodec_version(MAKE_SEMANTIC_VERSION(2, 3, 4)));
    }

    SECTION("the headers cuCIM is built against are themselves in range")
    {
        // If this fails, the vendored nvimgcodec_version.h was bumped out of
        // the supported range without the range being updated, which would
        // reject every install including the intended one.
        REQUIRE(is_supported_nvimgcodec_version(NVIMGCODEC_VER));
    }
}

// The packed encoding is major * 1000 + minor * 100 + patch, so the minor
// component only has room for one digit. 0.10.0 and 1.0.0 collide at 1000.
// That is worth pinning: it is why the range is compared as packed integers
// rather than by unpacking major/minor, and why diagnostics report ambiguous
// out-of-range values without assigning an exact semantic version.
TEST_CASE("packed nvImageCodec versions preserve ordering", "[test_nvimgcodec_version.cpp]")
{
    REQUIRE(MAKE_SEMANTIC_VERSION(0, 8, 0) < MAKE_SEMANTIC_VERSION(0, 9, 0));
    REQUIRE(MAKE_SEMANTIC_VERSION(0, 9, 0) < MAKE_SEMANTIC_VERSION(0, 9, 1));
    REQUIRE(MAKE_SEMANTIC_VERSION(0, 9, 99) < MAKE_SEMANTIC_VERSION(0, 10, 0));

    // Documents the collision rather than endorsing it; both are out of range,
    // so the check stays correct either way.
    REQUIRE(MAKE_SEMANTIC_VERSION(0, 10, 0) == MAKE_SEMANTIC_VERSION(1, 0, 0));
}

TEST_CASE("nvImageCodec version diagnostics preserve ambiguous packed values", "[test_nvimgcodec_version.cpp]")
{
    SECTION("unambiguous versions retain their readable form")
    {
        REQUIRE(format_nvimgcodec_version(MAKE_SEMANTIC_VERSION(0, 8, 0)) == "0.8.0");
        REQUIRE(format_nvimgcodec_version(MAKE_SEMANTIC_VERSION(0, 9, 0)) == "0.9.0");
        REQUIRE(format_nvimgcodec_version(MAKE_SEMANTIC_VERSION(0, 9, 99)) == "0.9.99");
    }

    SECTION("ambiguous versions retain the raw value instead of a guessed version")
    {
        const uint32_t packed_version = GENERATE(1000, 1001, 2304, 10000);
        const std::string expected =
            "packed version " + std::to_string(packed_version) + " (semantic version is ambiguous with the 0.9 headers)";
        REQUIRE(format_nvimgcodec_version(packed_version) == expected);
        REQUIRE_FALSE(is_supported_nvimgcodec_version(packed_version));
    }
}

#endif // CUCIM_HAS_NVIMGCODEC
