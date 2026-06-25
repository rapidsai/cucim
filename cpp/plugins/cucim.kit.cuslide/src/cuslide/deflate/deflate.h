/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef CUSLIDE_DEFLATE_H
#define CUSLIDE_DEFLATE_H

#include <cucim/io/device.h>

namespace cuslide::deflate
{

bool decode_deflate(int fd,
                    unsigned char* deflate_buf,
                    uint64_t offset,
                    uint64_t size,
                    uint8_t** dest,
                    uint64_t dest_nbytes,
                    const cucim::io::Device& out_device);
}
#endif // CUSLIDE_DEFLATE_H
