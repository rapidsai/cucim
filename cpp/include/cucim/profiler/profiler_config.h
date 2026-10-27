/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cucim/core/framework.h"

#include <string>
#include <vector>

namespace cucim::profiler
{

constexpr bool kDefaultProfilerTrace = false;

struct EXPORT_VISIBLE ProfilerConfig
{
    void load_config(const void* json_obj);

    bool trace = kDefaultProfilerTrace;
};

} // namespace cucim::profiler
