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

/**
 * Code below is using libdeflate library which is under MIT license
 * Please see LICENSE-3rdparty.md for the detail.
 */

#include "deflate.h"

#include <stdexcept>
#include <unistd.h>
#include "libdeflate.h"

namespace cuslide::deflate
{

bool decode_deflate(int fd,
                    unsigned char* deflate_buf,
                    uint64_t offset,
                    uint64_t size,
                    uint8_t** dest,
                    uint64_t dest_nbytes,
                    const cucim::io::Device& out_device)
{
    (void)out_device;
    struct libdeflate_decompressor* d;

    d = libdeflate_alloc_decompressor();

    if (d == nullptr)
    {
        throw std::runtime_error("Unable to allocate decompressor for libdeflate!");
    }

    if (deflate_buf == nullptr)
    {
        if ((deflate_buf = (unsigned char*)malloc(size)) == nullptr)
        {
            throw std::runtime_error("Unable to allocate buffer for libdeflate!");
        }

        if (pread(fd, deflate_buf, size, offset) < 1)
        {
            throw std::runtime_error("Unable to read file for libdeflate!");
        }
    }
    else
    {
        fd = -1;
        deflate_buf += offset;
    }

    size_t out_size;
    libdeflate_zlib_decompress(d, deflate_buf, size /*in_nbytes*/, *dest, dest_nbytes /*out_nbytes_avail*/, &out_size);

    if (fd != -1)
    {
        free(deflate_buf);
    }

    libdeflate_free_decompressor(d);
    return true;
}

} // namespace cuslide::deflate
