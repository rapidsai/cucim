/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
