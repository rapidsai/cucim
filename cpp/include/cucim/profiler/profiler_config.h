/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CUCIM_PROFILER_PROFILER_CONFIG_H
#define CUCIM_PROFILER_PROFILER_CONFIG_H

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

#endif // CUCIM_PROFILER_PROFILER_CONFIG_H
