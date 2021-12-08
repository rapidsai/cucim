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

#include "profiler_py.h"
#include "profiler_pydoc.h"

#include <cucim/profiler/profiler.h>
#include <cucim/cuimage.h>

using namespace pybind11::literals;
namespace py = pybind11;

namespace cucim::profiler
{

void init_profiler(py::module& profiler)
{
    py::class_<Profiler, std::shared_ptr<Profiler>>(profiler, "Profiler")
        .def_property("config", &py_config, nullptr, doc::Profiler::doc_config, py::call_guard<py::gil_scoped_release>())
        .def("trace", &py_trace, doc::Profiler::doc_trace, py::call_guard<py::gil_scoped_release>(), //
             py::arg("value") = py::none() //
        );
}

bool py_trace(Profiler& profiler, py::object value)
{
    if (value.is_none())
    {
        return profiler.trace();
    }
    else if (py::isinstance<py::bool_>(value))
    {
        py::bool_ v = value.cast<py::bool_>();
        profiler.trace(v);
        return v;
    }
    else
    {
        throw std::invalid_argument(fmt::format("Only 'NoneType' or 'bool' is available for the argument"));
    }
}

py::dict py_config(Profiler& profiler)
{
    ProfilerConfig& config = profiler.config();

    return py::dict{
        "trace"_a = pybind11::bool_(config.trace) //
    };
}


} // namespace cucim::profiler
