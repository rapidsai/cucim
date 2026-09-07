/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuslide/nvimgcodec/nvimgcodec_tiff_parser.h"

#include <catch2/catch_test_macros.hpp>

#ifdef CUCIM_HAS_NVIMGCODEC

#include <nvimgcodec.h>

#include <cstdint>

using cuslide2::nvimgcodec::is_supported_nvimgcodec_version;

// cuCIM checks the nvImageCodec version at run time because the build cannot:
// the library is dlopen'd by default and the SONAMEs searched end with
// libnvimgcodec.so.0 and libnvimgcodec.so, which resolve to whichever major-0
// build is installed. These tests pin the range that check enforces.
//
// The range must stay in step with the pin in dependencies.yaml and with the
// one cmake/deps/nvimgcodec.cmake enforces on headers. Nothing derives one from
// the other, so this is what catches them drifting apart.
TEST_CASE("supported nvImageCodec range matches the packaging pin", "[test_nvimgcodec_version.cpp]")
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
// rather than by unpacking major/minor, and why a diagnostic can render an
// out-of-range 0.10.0 as "1.0.0".
TEST_CASE("packed nvImageCodec versions preserve ordering", "[test_nvimgcodec_version.cpp]")
{
    REQUIRE(MAKE_SEMANTIC_VERSION(0, 8, 0) < MAKE_SEMANTIC_VERSION(0, 9, 0));
    REQUIRE(MAKE_SEMANTIC_VERSION(0, 9, 0) < MAKE_SEMANTIC_VERSION(0, 9, 1));
    REQUIRE(MAKE_SEMANTIC_VERSION(0, 9, 99) < MAKE_SEMANTIC_VERSION(0, 10, 0));

    // Documents the collision rather than endorsing it; both are out of range,
    // so the check stays correct either way.
    REQUIRE(MAKE_SEMANTIC_VERSION(0, 10, 0) == MAKE_SEMANTIC_VERSION(1, 0, 0));
}

#endif // CUCIM_HAS_NVIMGCODEC
