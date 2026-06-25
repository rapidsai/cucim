/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef CUSLIDE_LIBOPENJPEG_H
#define CUSLIDE_LIBOPENJPEG_H

#include <cucim/io/device.h>

namespace cuslide::jpeg2k
{
constexpr uint32_t kAperioJpeg2kYCbCr = 33003; // Jpeg 2000 with YCbCr format, possibly with a chroma subsampling of
                                               // 4:2:2
constexpr uint32_t kAperioJpeg2kRGB = 33005; // Jpeg 2000 with RGB format

enum class ColorSpace : uint8_t
{
    kUnspecified = 0, // not specified in the codestream
    kRGB = 1, // sRGB
    kGRAY = 2, // grayscale
    kSYCC = 3, // YUV
    kEYCC = 4, // e-YCC
    kCMYK = 5 // CMYK
};

bool decode_libopenjpeg(int fd,
                        unsigned char* jpeg_buf,
                        uint64_t offset,
                        uint64_t size,
                        uint8_t** dest,
                        uint64_t dest_nbytes,
                        const cucim::io::Device& out_device,
                        ColorSpace color_space);

} // namespace cuslide::jpeg2k

#endif // CUSLIDE_LIBOPENJPEG_H
