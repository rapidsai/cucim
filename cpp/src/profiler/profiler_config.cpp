/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cucim/profiler/profiler_config.h"

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace cucim::profiler
{

void ProfilerConfig::load_config(const void* json_obj)
{
    const json& profiler_config = *(static_cast<const json*>(json_obj));

    if (profiler_config.contains("trace") && profiler_config["trace"].is_boolean())
    {
        trace = profiler_config.value("trace", kDefaultProfilerTrace);
    }
}

} // namespace cucim::profiler
