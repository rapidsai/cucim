/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Code below is using libdeflate library which is under MIT license
 * Please see LICENSE-3rdparty.md for the detail.
 */

#include "raw.h"

#include <cstring>
#include <stdexcept>
#include <unistd.h>

#include <fmt/format.h>
#include <cucim/memory/memory_manager.h>
#include <cucim/profiler/nvtx3.h>


namespace cuslide::raw
{

bool decode_raw(int fd,
                unsigned char* raw_buf,
                uint64_t offset,
                uint64_t size,
                uint8_t** dest,
                uint64_t dest_nbytes,
                const cucim::io::Device& out_device)
{
    (void)out_device;

    if (dest == nullptr)
    {
        throw std::runtime_error("'dest' shouldn't be nullptr in decode_raw()");
    }

    // Allocate memory only when dest is not null
    if (*dest == nullptr)
    {
        if ((*dest = (unsigned char*)cucim_malloc(dest_nbytes)) == nullptr)
        {
            throw std::runtime_error("Unable to allocate uncompressed image buffer");
        }
    }

    if (raw_buf == nullptr)
    {
        if ((raw_buf = (unsigned char*)cucim_malloc(size)) == nullptr)
        {
            throw std::runtime_error("Unable to allocate buffer for raw data!");
        }

        ssize_t bytes_read = pread(fd, raw_buf, size, offset);
        if (bytes_read < 0 || static_cast<uint64_t>(bytes_read) != size)
        {
            cucim_free(raw_buf);
            throw std::runtime_error(
                fmt::format("Short read for raw data: expected {} bytes, got {}", size, bytes_read));
        }
    }
    else
    {
        fd = -1;
        raw_buf += offset;
    }

    if (size < dest_nbytes)
    {
        if (fd != -1)
        {
            cucim_free(raw_buf);
        }
        throw std::runtime_error(
            fmt::format("Raw tile data size ({}) is smaller than expected destination buffer ({})", size, dest_nbytes));
    }

    memcpy(*dest, raw_buf, dest_nbytes);

    if (fd != -1)
    {
        cucim_free(raw_buf);
    }

    return true;
}

} // namespace cuslide::raw
