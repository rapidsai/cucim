/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
