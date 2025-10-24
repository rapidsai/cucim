/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef PYCUCIM_MEMORY_INIT_H
#define PYCUCIM_MEMORY_INIT_H

#include <pybind11/pybind11.h>

#include <cucim/io/device.h>

namespace py = pybind11;

namespace cucim::memory
{

void init_memory(py::module& m);

void get_memory_info(const py::object& buf_obj,
                     void** out_buf,
                     cucim::io::Device* out_device = nullptr,
                     size_t* out_memory_size = 0,
                     bool* out_readonly = nullptr);

} // namespace cucim::memory


#endif // PYCUCIM_MEMORY_INIT_H
