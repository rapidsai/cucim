/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef PYCUCIM_FILESYSTEM_INIT_H
#define PYCUCIM_FILESYSTEM_INIT_H

#include <memory>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace cucim::filesystem
{

// Forward declaration
class CuFileDriver;

void init_filesystem(py::module& m);

std::shared_ptr<CuFileDriver> py_open(const char* file_path, const char* flags, mode_t mode);
ssize_t py_pread(const std::shared_ptr<CuFileDriver>& fd, py::object buf, size_t count, off_t file_offset, off_t buf_offset = 0);
ssize_t py_pwrite(const std::shared_ptr<CuFileDriver>& fd, py::object buf, size_t count, off_t file_offset, off_t buf_offset = 0);
// bool py_close(const std::shared_ptr<CuFileDriver>& fd);
// bool py_discard_page_cache(const char* file_path);
}

#endif // PYCUCIM_FILESYSTEM_INIT_H
