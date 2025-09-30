/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#ifndef CUSLIDE_TYPES_H
#define CUSLIDE_TYPES_H

#include <cstdint>

#include <cucim/codec/methods.h>

namespace cuslide::tiff
{

using ifd_offset_t = uint64_t;

enum class TiffType : uint32_t
{
    Generic = 0,
    Philips = 1,
    Aperio = 2,
};

enum class AssociatedImageBufferType : uint8_t
{
    IFD = 0,
    IFD_IMAGE_DESC = 1,
    FILE_OFFSET = 2,
    BUFFER_POINTER = 3,
    OWNED_BUFFER_POINTER = 4,
};

struct AssociatedImageBufferDesc
{
    AssociatedImageBufferType type; /// 0: IFD index, 1: IFD index + image description offset&size (base64-encoded text)
                                    /// 2: file offset + size, 3: buffer pointer (owned by others) + size
                                    /// 4: allocated (owned) buffer pointer (so need to free after use) + size
    cucim::codec::CompressionMethod compression;
    union
    {
        ifd_offset_t ifd_index;
        struct
        {
            ifd_offset_t desc_ifd_index;
            uint64_t desc_offset;
            uint64_t desc_size;
        };
        struct
        {
            uint64_t file_offset;
            uint64_t file_size;
        };
        struct
        {
            void* buf_ptr;
            uint64_t buf_size;
        };
        struct
        {
            void* owned_ptr;
            uint64_t owned_size;
        };
    };
};


} // namespace cuslide::tiff

#endif // CUSLIDE_TYPES_H
