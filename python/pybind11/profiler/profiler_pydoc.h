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
#ifndef PYCUCIM_PROFILER_PROFILER_PYDOC_H
#define PYCUCIM_PROFILER_PROFILER_PYDOC_H

#include "../macros.h"

namespace cucim::profiler::doc::Profiler
{

PYDOC(config, R"doc(
Returns the dictionary of configuration.
)doc")

// void trace(bool value) = 0;
// bool trace() const = 0;
PYDOC(trace, R"doc(
Traces method executions with NVTX.
)doc")

} // namespace cucim::profiler::doc::Profiler

#endif // PYCUCIM_PROFILER_PROFILER_PYDOC_H
