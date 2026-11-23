/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

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
