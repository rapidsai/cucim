/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * LZW compression:
 *   https://www.adobe.io/content/dam/udp/en/open/standards/tiff/TIFF6.pdf
 **/

#include "lzw.h"

#include <cstring>
#include <stdexcept>
#include <unistd.h>

#include <cucim/memory/memory_manager.h>
#include <cucim/profiler/nvtx3.h>


namespace cuslide::lzw
{

bool decode_lzw(int fd,
                unsigned char* lzw_buf,
                uint64_t offset,
                uint64_t size,
                uint8_t** dest,
                uint64_t dest_nbytes,
                const cucim::io::Device& out_device)
{
    (void)out_device;

    if (dest == nullptr)
    {
        throw std::runtime_error("'dest' shouldn't be nullptr in decode_lzw()");
    }

    // Allocate memory only when dest is not null
    if (*dest == nullptr)
    {
        if ((*dest = (unsigned char*)cucim_malloc(dest_nbytes)) == nullptr)
        {
            throw std::runtime_error("Unable to allocate uncompressed image buffer");
        }
    }

    if (lzw_buf == nullptr)
    {
        if ((lzw_buf = (unsigned char*)cucim_malloc(size)) == nullptr)
        {
            throw std::runtime_error("Unable to allocate buffer for lzw data!");
        }

        if (pread(fd, lzw_buf, size, offset) < 1)
        {
            throw std::runtime_error("Unable to read file for lzw data!");
        }
    }
    else
    {
        fd = -1;
        lzw_buf += offset;
    }

    TIFF tif;
    tif.tif_rawdata = tif.tif_rawcp = lzw_buf;
    tif.tif_rawcc = size;

    if (TIFFInitLZW(&tif) == 0)
    {
        return false;
    }
    if (tif.tif_predecode(&tif, 0 /* unused */) == 0)
    {
        goto bad;
    }
    if (tif.tif_decodestrip(&tif, *dest, dest_nbytes, 0 /* unused */) == 0)
    {
        goto bad;
    }
    tif.tif_cleanup(&tif);

    if (fd != -1)
    {
        cucim_free(lzw_buf);
    }

    return true;
bad:
    if (fd != -1)
    {
        cucim_free(lzw_buf);
    }
    tif.tif_cleanup(&tif);
    return false;
}

} // namespace cuslide::lzw
