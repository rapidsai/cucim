/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef CUSLIDE_LIBNVJPEG_H
#define CUSLIDE_LIBNVJPEG_H

#include <cucim/io/device.h>

namespace cuslide::jpeg
{

EXPORT_VISIBLE bool decode_libnvjpeg(int fd,
                                     const unsigned char* jpeg_buf,
                                     uint64_t offset,
                                     uint64_t size,
                                     const void* jpegtable_data,
                                     uint32_t jpegtable_count,
                                     uint8_t** dest,
                                     const cucim::io::Device& out_device);

} // namespace cuslide::jpeg

#endif // CUSLIDE_LIBNVJPEG_H
