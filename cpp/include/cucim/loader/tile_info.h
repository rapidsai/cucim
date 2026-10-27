/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

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
