/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
