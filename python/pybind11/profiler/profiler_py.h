/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYCUCIM_PROFILER_INIT_H
#define PYCUCIM_PROFILER_INIT_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace cucim::profiler
{

// Forward declaration
class Profiler;

void init_profiler(py::module& m);

bool py_trace(Profiler& profiler, py::object value);

py::dict py_config(Profiler& profiler);

} // namespace cucim::profiler

#endif // PYCUCIM_PROFILER_INIT_H
