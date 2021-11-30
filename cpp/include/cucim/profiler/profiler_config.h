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
