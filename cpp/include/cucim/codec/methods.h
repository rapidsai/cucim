/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#ifndef CUCIM_METHODS_H
#define CUCIM_METHODS_H

#include "cucim/macros/defines.h"

namespace cucim::codec
{

/// Compression method (Followed https://www.awaresystems.be/imaging/tiff/tifftags/compression.html)
enum class CompressionMethod : uint16_t
{
    NONE = 1,
    JPEG = 7,
};

} // namespace cucim::codec

#endif // CUCIM_METHODS_H
