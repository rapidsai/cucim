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

#ifndef CUCIM_LOADER_TILE_INFO_H
#define CUCIM_LOADER_TILE_INFO_H

#include "cucim/macros/api_header.h"

#include <cstdint>

namespace cucim::loader
{

struct EXPORT_VISIBLE TileInfo
{
    int64_t location_index = 0; // patch #
    int64_t index = 0; // tile #
    uint64_t offset = 0;
    uint64_t size = 0;
};

} // namespace cucim::loader

#endif // CUCIM_LOADER_TILE_INFO_H
