/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
