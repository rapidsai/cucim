/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
//
#ifndef CUCIM_UTIL_FILE_H
#define CUCIM_UTIL_FILE_H

#include "cucim/core/framework.h"

/**
 * @brief Utility methods that need to be refactored later.
 */
namespace cucim::util
{

EXPORT_VISIBLE bool file_exists(const char* path);

} // namespace cucim::util

#endif // CUCIM_UTIL_FILE_H
