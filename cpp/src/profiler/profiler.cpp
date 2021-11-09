/*
 * Apache License, Version 2.0
 * Copyright 2021 NVIDIA Corporation
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

#include "cucim/profiler/profiler.h"

#include "cucim/cuimage.h"

namespace cucim::profiler
{

Profiler::Profiler(ProfilerConfig& config) : config_(config){};

ProfilerConfig& Profiler::config()
{
    return config_;
}

ProfilerConfig Profiler::get_config() const
{
    return config_;
}

void Profiler::trace(bool value)
{
    config_.trace = value;
}

/**
 * @brief Return whether if trace is enabled or not
 *
 * @return true if profiler is enabled. false otherwise
 */
bool Profiler::trace() const
{
    return config_.trace;
}

} // namespace cucim::profiler
