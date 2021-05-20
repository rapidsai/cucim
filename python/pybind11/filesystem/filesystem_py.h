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
