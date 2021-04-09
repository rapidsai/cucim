/*
 * Apache License, Version 2.0
 * Copyright 2020 NVIDIA Corporation
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
#ifndef CUSLIDE_LIBJPEG_TURBO_H
#define CUSLIDE_LIBJPEG_TURBO_H

#include <cucim/io/device.h>

namespace cuslide::jpeg
{

bool decode_libjpeg(int fd,
                    unsigned char* jpeg_buf,
                    uint64_t offset,
                    uint64_t size,
                    const void* jpegtable_data,
                    uint32_t jpegtable_count,
                    uint8_t** dest,
                    const cucim::io::Device& out_device);

/**
 * Reads jpeg header tables.
 *
 * TIFF file's TIFFTAG_JPEGTABLES tag has the information about JPEG Quantization table.
 * This method is for reading the information.
 * If Quantization table information is not interpreted, the following error message can occurs:
 *
 *     Quantization table 0x00 was not defined
 *
 * @param handle A pointer to tjinstance
 * @param jpeg_buf jpeg buffer data
 * @param jpeg_size jpeg buffer size
 * @return true if it succeeds
 */
bool read_jpeg_header_tables(const void* handle, const void* jpeg_buf, unsigned long jpeg_size);

bool get_dimension(const void* image_buf, uint64_t offset, uint64_t size, int* out_width, int* out_height);

} // namespace cuslide::jpeg


#endif // CUSLIDE_LIBJPEG_TURBO_H
