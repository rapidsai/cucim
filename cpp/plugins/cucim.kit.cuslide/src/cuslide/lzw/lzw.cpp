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
 * LZW compression:
 *   https://www.adobe.io/content/dam/udp/en/open/standards/tiff/TIFF6.pdf
 **/

#include "lzw.h"

#include <cstring>
#include <stdexcept>
#include <unistd.h>


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
        if ((*dest = (unsigned char*)malloc(dest_nbytes)) == nullptr)
        {
            throw std::runtime_error("Unable to allocate uncompressed image buffer");
        }
    }

    if (lzw_buf == nullptr)
    {
        if ((lzw_buf = (unsigned char*)malloc(size)) == nullptr)
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
        free(lzw_buf);
    }

    return true;
bad:
    if (fd != -1)
    {
        free(lzw_buf);
    }
    tif.tif_cleanup(&tif);
    return false;
}

} // namespace cuslide::lzw
