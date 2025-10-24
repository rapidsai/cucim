/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef CUSLIDE_LIBJPEG_TURBO_H
#define CUSLIDE_LIBJPEG_TURBO_H

#include <cucim/io/device.h>

namespace cuslide::jpeg
{

bool decode_libjpeg(int fd,
                    unsigned char* jpeg_buf,
                    uint64_t offset,
                    uint64_t size,
                    const void* jpegtable_data,
                    uint32_t jpegtable_count,
                    uint8_t** dest,
                    const cucim::io::Device& out_device,
                    int jpeg_color_space = 0 /* 0: JCS_UNKNOWN, 2: JCS_RGB, 3: JCS_YCbCr */);

/**
 * Reads jpeg header tables.
 *
 * TIFF file's TIFFTAG_JPEGTABLES tag has the information about JPEG Quantization table.
 * This method is for reading the information.
 * If Quantization table information is not interpreted, the following error message can occurs:
 *
 *     Quantization table 0x00 was not defined
 *
 * @param handle A pointer to tjinstance
 * @param jpeg_buf jpeg buffer data
 * @param jpeg_size jpeg buffer size
 * @return true if it succeeds
 */
bool read_jpeg_header_tables(const void* handle, const void* jpeg_buf, unsigned long jpeg_size);

int jpeg_decode_buffer(const void* handle,
                       const unsigned char* jpegBuf,
                       unsigned long jpegSize,
                       unsigned char* dstBuf,
                       int width,
                       int pitch,
                       int height,
                       int pixelFormat,
                       int flags,
                       int jpegColorSpace = 0 /* 0: JCS_UNKNOWN, 2: JCS_RGB, 3: JCS_YCbCr */);

bool get_dimension(const void* image_buf, uint64_t offset, uint64_t size, int* out_width, int* out_height);

} // namespace cuslide::jpeg


#endif // CUSLIDE_LIBJPEG_TURBO_H
