/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef PYCUCIM_CUFILE_PY_H
#define PYCUCIM_CUFILE_PY_H

#include "cucim/filesystem/cufile_driver.h"

#include <cstdio>
#include <pybind11/pytypes.h>

namespace py = pybind11;

namespace cucim::filesystem
{
// Note: there would be name conflict with pread/pwrite in cufile_driver.h so prefixed 'fd_'.
ssize_t fd_pread(const CuFileDriver& fd, const py::object& buf, size_t count, off_t file_offset, off_t buf_offset = 0);
ssize_t fd_pwrite(CuFileDriver& fd, const py::object& buf, size_t count, off_t file_offset, off_t buf_offset = 0);
} // namespace cucim::filesystem

#endif // PYCUCIM_CUFILE_PY_H
