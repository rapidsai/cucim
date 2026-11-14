/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "cucim/macros/defines.h"

namespace cucim::codec::base64
{
EXPORT_VISIBLE bool encode(const char* src, int src_count, char** out_dst, int* out_count);
EXPORT_VISIBLE bool decode(const char* src, int src_count, char** out_dst, int* out_count);
}
