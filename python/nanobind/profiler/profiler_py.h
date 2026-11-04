/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYCUCIM_PROFILER_INIT_H
#define PYCUCIM_PROFILER_INIT_H

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace cucim::profiler
{

// Forward declaration
class Profiler;

void init_profiler(nb::module_ m);

bool py_trace(Profiler& profiler, nb::handle value);

nb::dict py_config(Profiler& profiler);

} // namespace cucim::profiler

#endif // PYCUCIM_PROFILER_INIT_H
