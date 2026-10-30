/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "profiler_py.h"
#include "profiler_pydoc.h"

#include <fmt/format.h>

#include <cucim/cuimage.h>
#include <cucim/profiler/profiler.h>

using namespace nanobind::literals;

namespace cucim::profiler
{

void init_profiler(nb::module_ profiler)
{
    nb::class_<Profiler>(profiler, "Profiler", nb::is_weak_referenceable())
        .def_prop_ro("config", &py_config, doc::Profiler::doc_config)
        .def("trace", &py_trace, doc::Profiler::doc_trace, //
             "value"_a = nb::none() //
        );
}

bool py_trace(Profiler& profiler, nb::handle value)
{
    if (value.is_none())
    {
        return profiler.trace();
    }
    else if (nb::isinstance<nb::bool_>(value))
    {
        bool v = nb::cast<bool>(value);
        profiler.trace(v);
        return v;
    }
    else
    {
        throw std::invalid_argument(fmt::format("Only 'NoneType' or 'bool' is available for the argument"));
    }
}

nb::dict py_config(Profiler& profiler)
{
    ProfilerConfig& config = profiler.config();

    nb::dict result;
    result["trace"] = config.trace;
    return result;
}


} // namespace cucim::profiler
