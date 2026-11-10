/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYCUCIM_IO_INIT_H
#define PYCUCIM_IO_INIT_H

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace cucim::io
{

void init_io(nb::module_ m);

}


#endif // PYCUCIM_IO_INIT_H
