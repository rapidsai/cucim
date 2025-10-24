/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Code below is using libdeflate library which is under MIT license
 * Please see LICENSE-3rdparty.md for the detail.
 */

#include "deflate.h"

#include <stdexcept>
#include <unistd.h>

#include <cucim/memory/memory_manager.h>
#include <cucim/profiler/nvtx3.h>

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

    if (dest == nullptr)
    {
        throw std::runtime_error("'dest' shouldn't be nullptr in decode_deflate()");
    }

    // Allocate memory only when dest is not null
    if (*dest == nullptr)
    {
        if ((*dest = (unsigned char*)cucim_malloc(dest_nbytes)) == nullptr)
        {
            throw std::runtime_error("Unable to allocate uncompressed image buffer");
        }
    }

    {
        PROF_SCOPED_RANGE(PROF_EVENT(libdeflate_alloc_decompressor));
        d = libdeflate_alloc_decompressor();
    }

    if (d == nullptr)
    {
        throw std::runtime_error("Unable to allocate decompressor for libdeflate!");
    }

    if (deflate_buf == nullptr)
    {
        if ((deflate_buf = (unsigned char*)cucim_malloc(size)) == nullptr)
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
    {
        PROF_SCOPED_RANGE(PROF_EVENT(libdeflate_zlib_decompress));
        libdeflate_zlib_decompress(
            d, deflate_buf, size /*in_nbytes*/, *dest, dest_nbytes /*out_nbytes_avail*/, &out_size);
    }

    if (fd != -1)
    {
        cucim_free(deflate_buf);
    }

    {
        PROF_SCOPED_RANGE(PROF_EVENT(libdeflate_free_decompressor));
        libdeflate_free_decompressor(d);
    }
    return true;
}

} // namespace cuslide::deflate
