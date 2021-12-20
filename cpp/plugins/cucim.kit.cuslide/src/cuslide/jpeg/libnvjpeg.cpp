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


bool decode_libnvjpeg(int fd,
                      unsigned char* jpeg_buf,
                      uint64_t offset,
                      uint64_t size,
                      const void* jpegtable_data,
                      uint32_t jpegtable_count,
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
