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

#include "cufile_py.h"
#include "cufile_pydoc.h"

#include <pybind11/pybind11.h>

#include <cucim/filesystem/cufile_driver.h>

#include "../memory/memory_py.h"

namespace py = pybind11;

namespace cucim::filesystem
{

ssize_t fd_pread(const CuFileDriver& fd, const py::object& obj, size_t count, off_t file_offset, off_t buf_offset)
{
    void* buf = nullptr;
    size_t memory_size = 0;
    bool readonly = false;

    cucim::memory::get_memory_info(obj, &buf, nullptr, &memory_size, &readonly);

    if (buf == nullptr)
    {
        throw std::runtime_error("Cannot Recognize the array object!");
    }
    if (readonly)
    {
        throw std::runtime_error("The buffer is readonly so cannot be used for pread!");
    }
    if (memory_size && count > memory_size) {
        throw std::runtime_error(fmt::format("[Error] 'count' ({}) is larger than the size of the array object ({})!", count, memory_size));
    }

    py::call_guard<py::gil_scoped_release>();
    return fd.pread(buf, count, file_offset, buf_offset);
}
ssize_t fd_pwrite(CuFileDriver& fd, const py::object& obj, size_t count, off_t file_offset, off_t buf_offset)
{
    void* buf = nullptr;
    size_t memory_size = 0;

    cucim::memory::get_memory_info(obj, &buf, nullptr, &memory_size, nullptr);

    if (buf == nullptr)
    {
        throw std::runtime_error("Cannot Recognize the array object!");
    }
    if (memory_size && count > memory_size) {
        throw std::runtime_error(fmt::format("[Error] 'count' ({}) is larger than the size of the array object ({})!", count, memory_size));
    }

    py::call_guard<py::gil_scoped_release>();
    return fd.pwrite(buf, count, file_offset, buf_offset);
}
} // namespace cucim::filesystem