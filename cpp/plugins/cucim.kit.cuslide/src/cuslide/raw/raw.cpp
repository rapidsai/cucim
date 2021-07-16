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

#include "raw.h"

#include <cstring>
#include <stdexcept>
#include <unistd.h>


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

    if (raw_buf == nullptr)
    {
        if ((raw_buf = (unsigned char*)malloc(size)) == nullptr)
        {
            throw std::runtime_error("Unable to allocate buffer for raw data!");
        }

        if (pread(fd, raw_buf, size, offset) < 1)
        {
            throw std::runtime_error("Unable to read file for raw data!");
        }
    }
    else
    {
        fd = -1;
        raw_buf += offset;
    }

    memcpy(*dest, raw_buf, dest_nbytes);

    if (fd != -1)
    {
        free(raw_buf);
    }

    return true;
}

} // namespace cuslide::raw
