/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
