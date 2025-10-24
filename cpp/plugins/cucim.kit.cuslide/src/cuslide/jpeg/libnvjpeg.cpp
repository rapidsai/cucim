/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#include "libnvjpeg.h"

#include <cstring>
#include <unistd.h>

namespace cuslide::jpeg
{


#define THROW(action, message)                                                                                         \
    {                                                                                                                  \
        printf("ERROR in line %d while %s:\n%s\n", __LINE__, action, message);                                         \
        retval = -1;                                                                                                   \
        goto bailout;                                                                                                  \
    }


bool decode_libnvjpeg(const int fd,
                      const unsigned char* jpeg_buf,
                      const uint64_t offset,
                      const uint64_t size,
                      const void* jpegtable_data,
                      const uint32_t jpegtable_count,
                      uint8_t** dest,
                      const cucim::io::Device& out_device)
{
    (void)out_device;
    (void)fd;
    (void)jpeg_buf;
    (void)offset;
    (void)size;
    (void)jpegtable_data;
    (void)jpegtable_count;
    (void)dest;
    (void)out_device;

    return true;
}

} // namespace cuslide::jpeg
