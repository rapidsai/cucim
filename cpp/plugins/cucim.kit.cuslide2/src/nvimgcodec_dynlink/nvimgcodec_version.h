/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// clang-format off
#ifndef NVIMGCODEC_VERSION_H__
#define NVIMGCODEC_VERSION_H__

#define NVIMGCODEC_VER_MAJOR 0
#define NVIMGCODEC_VER_MINOR 10
#define NVIMGCODEC_VER_PATCH 0
#define NVIMGCODEC_VER_BUILD

// Two decimal digits are reserved for the minor and two for the patch component,
// so packed values are unique and ordered only while both stay below 100.
#define NVIMGCODEC_MAKE_SEMANTIC_VERSION(major, minor, patch) (((major) * 10000u) + ((minor) * 100u) + (patch))
#define NVIMGCODEC_MAJOR_FROM_SEMVER(ver) ((ver) / 10000u)
#define NVIMGCODEC_MINOR_FROM_SEMVER(ver) (((ver) % 10000u) / 100u)
#define NVIMGCODEC_PATCH_FROM_SEMVER(ver) ((ver) % 100u)
#define NVIMGCODEC_STREAM_VER(ver) \
    NVIMGCODEC_MAJOR_FROM_SEMVER(ver) << "." << NVIMGCODEC_MINOR_FROM_SEMVER(ver) << "." << NVIMGCODEC_PATCH_FROM_SEMVER(ver)

#define NVIMGCODEC_VER NVIMGCODEC_MAKE_SEMANTIC_VERSION(NVIMGCODEC_VER_MAJOR, NVIMGCODEC_VER_MINOR, NVIMGCODEC_VER_PATCH)

#endif // NVIMGCODEC_VERSION
