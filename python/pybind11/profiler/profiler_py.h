/*
 * Copyright (c) 2021, NVIDIA CORPORATcacheN.
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
