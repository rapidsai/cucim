/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cucim/profiler/profiler_config.h"

#include <fmt/format.h>
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
