/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
//
#ifndef CUCIM_UTIL_PLATFORM_H
#define CUCIM_UTIL_PLATFORM_H

#include "cucim/macros/api_header.h"

/**
 * @brief Platform-specific macros and functions.
 */
namespace cucim::util
{

EXPORT_VISIBLE bool is_in_wsl();

} // namespace cucim::util

#endif // CUCIM_UTIL_PLATFORM_H
