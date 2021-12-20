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
#ifndef CUSLIDE_LIBNVJPEG_H
#define CUSLIDE_LIBNVJPEG_H

#include <cucim/io/device.h>

namespace cuslide::jpeg
{

EXPORT_VISIBLE bool decode_libnvjpeg(int fd,
                                     unsigned char* jpeg_buf,
                                     uint64_t offset,
                                     uint64_t size,
                                     const void* jpegtable_data,
                                     uint32_t jpegtable_count,
                                     uint8_t** dest,
                                     const cucim::io::Device& out_device);

} // namespace cuslide::jpeg

#endif // CUSLIDE_LIBNVJPEG_H
