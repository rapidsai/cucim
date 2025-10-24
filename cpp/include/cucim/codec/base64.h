/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef CUCIM_BASE64_H
#define CUCIM_BASE64_H

#include "cucim/macros/defines.h"

namespace cucim::codec::base64
{
EXPORT_VISIBLE bool encode(const char* src, int src_count, char** out_dst, int* out_count);
EXPORT_VISIBLE bool decode(const char* src, int src_count, char** out_dst, int* out_count);
}
#endif // CUCIM_BASE64_H
