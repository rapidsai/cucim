/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//
#pragma once

#include "cucim/macros/api_header.h"

/**
 * @brief Platform-specific macros and functions.
 */
namespace cucim::util
{

EXPORT_VISIBLE bool is_in_wsl();

} // namespace cucim::util
