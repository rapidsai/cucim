/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

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
