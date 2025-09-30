/*
 * Apache License, Version 2.0
 * Copyright 2021 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
