/*
 * SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYCUCIM_IO_INIT_H
#define PYCUCIM_IO_INIT_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace cucim::io
{

void init_io(py::module& m);

}


#endif // PYCUCIM_IO_INIT_H
