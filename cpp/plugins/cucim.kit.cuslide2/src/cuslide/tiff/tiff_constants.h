/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CUSLIDE_TIFF_CONSTANTS_H
#define CUSLIDE_TIFF_CONSTANTS_H

#include <cstdint>

/**
 * TIFF constants extracted from libtiff headers.
 * These are standard TIFF specification values that don't change.
 * We define them here so we don't need libtiff headers.
 */

namespace cuslide::tiff {

// TIFF Tags
constexpr uint32_t TIFFTAG_SOFTWARE = 305;
constexpr uint32_t TIFFTAG_MODEL = 272;
constexpr uint32_t TIFFTAG_IMAGEDESCRIPTION = 270;
constexpr uint32_t TIFFTAG_RESOLUTIONUNIT = 296;
constexpr uint32_t TIFFTAG_XRESOLUTION = 282;
constexpr uint32_t TIFFTAG_YRESOLUTION = 283;
constexpr uint32_t TIFFTAG_PREDICTOR = 317;
constexpr uint32_t TIFFTAG_JPEGTABLES = 347;

// TIFF Compression Types
constexpr uint16_t COMPRESSION_NONE = 1;
constexpr uint16_t COMPRESSION_LZW = 5;
constexpr uint16_t COMPRESSION_JPEG = 7;
constexpr uint16_t COMPRESSION_DEFLATE = 8;
constexpr uint16_t COMPRESSION_ADOBE_DEFLATE = 32946;

// Aperio JPEG2000 compression (vendor-specific)
constexpr uint16_t COMPRESSION_APERIO_JP2K_YCBCR = 33003;
constexpr uint16_t COMPRESSION_APERIO_JP2K_RGB = 33005;

// TIFF Photometric Interpretation
constexpr uint16_t PHOTOMETRIC_RGB = 2;
constexpr uint16_t PHOTOMETRIC_YCBCR = 6;

// TIFF Planar Configuration
constexpr uint16_t PLANARCONFIG_CONTIG = 1;
constexpr uint16_t PLANARCONFIG_SEPARATE = 2;

// TIFF Flags
constexpr uint32_t TIFF_ISTILED = 0x00000004;

} // namespace cuslide::tiff

#endif // CUSLIDE_TIFF_CONSTANTS_H
