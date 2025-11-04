/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef CUSLIDE_LZW_H
#define CUSLIDE_LZW_H

#include <cucim/io/device.h>

#include "lzw_libtiff.h"

namespace cuslide::lzw
{

bool decode_lzw(int fd,
                unsigned char* raw_buf,
                uint64_t offset,
                uint64_t size,
                uint8_t** dest,
                uint64_t dest_nbytes,
                const cucim::io::Device& out_device);
}
#endif // CUSLIDE_LZW_H
