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

#include <fmt/format.h>
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

        ssize_t bytes_read = pread(fd, deflate_buf, size, offset);
        if (bytes_read < 0 || static_cast<uint64_t>(bytes_read) != size)
        {
            cucim_free(deflate_buf);
            throw std::runtime_error(
                fmt::format("Short read for deflate data: expected {} bytes, got {}", size, bytes_read));
        }
    }
    else
    {
        fd = -1;
        deflate_buf += offset;
    }

    size_t out_size;
    enum libdeflate_result decompress_result;
    {
        PROF_SCOPED_RANGE(PROF_EVENT(libdeflate_zlib_decompress));
        decompress_result = libdeflate_zlib_decompress(
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

    if (decompress_result != LIBDEFLATE_SUCCESS)
    {
        const char* reason = "unknown error";
        switch (decompress_result)
        {
        case LIBDEFLATE_BAD_DATA:
            reason = "corrupt or invalid compressed data";
            break;
        case LIBDEFLATE_SHORT_OUTPUT:
            reason = "decompressed size is less than expected";
            break;
        case LIBDEFLATE_INSUFFICIENT_SPACE:
            reason = "output buffer too small for decompressed data";
            break;
        default:
            break;
        }
        throw std::runtime_error(
            fmt::format("Deflate decompression failed: {}", reason));
    }

    return true;
}

} // namespace cuslide::deflate
