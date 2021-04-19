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

#include "init.h"
#include "filesystem_pydoc.h"
#include "cufile_py.h"
#include "cufile_pydoc.h"

#include <pybind11/pybind11.h>
#include <cucim/filesystem/cufile_driver.h>

namespace py = pybind11;

namespace cucim::filesystem
{

void init_filesystem(py::module& fs)
{
    py::enum_<FileHandleType>(fs, "FileHandleType") //
        .value("Unknown", FileHandleType::kUnknown) //
        .value("Posix", FileHandleType::kPosix) //
        .value("PosixODirect", FileHandleType::kPosixODirect) //
        .value("MemoryMapped", FileHandleType::kMemoryMapped) //
        .value("GPUDirect", FileHandleType::kGPUDirect);

    py::class_<CuFileDriver, std::shared_ptr<CuFileDriver>>(fs, "CuFileDriver")
        .def(py::init<int, bool, bool, const char*>(), doc::CuFileDriver::doc_CuFileDriver, //
             py::arg("fd"), //
             py::arg("no_gds") = false, //
             py::arg("use_mmap") = false, //
             py::arg("file_path") = py::str(""), //
             py::call_guard<py::gil_scoped_release>())
        .def("pread", &fd_pread, doc::CuFileDriver::doc_pread, // Do not release GIL as it would access properties of
                                                               // python object
             py::arg("buf"), //
             py::arg("count"), //
             py::arg("file_offset"), //
             py::arg("buf_offset") = 0) //
        .def("pwrite", &fd_pwrite, doc::CuFileDriver::doc_pwrite, // Do not release GIL as it would access properties of
                                                                  // python object
             py::arg("buf"), //
             py::arg("count"), //
             py::arg("file_offset"), //
             py::arg("buf_offset") = 0) //
        .def("close", &CuFileDriver::close, doc::CuFileDriver::doc_close, py::call_guard<py::gil_scoped_release>()) //
        .def("__repr__", [](const CuFileDriver& fd) {
            return fmt::format("<cucim.clara.filesystem.CuFileDriver path:{}>", fd.path());
        });

    fs.def("is_gds_available", &is_gds_available, doc::doc_is_gds_available, py::call_guard<py::gil_scoped_release>())
        .def("open", &py_open, doc::doc_open,
             py::arg("file_path"), //
             py::arg("flags"), //
             py::arg("mode") = 0644, //
             py::call_guard<py::gil_scoped_release>())
        .def("pread", &py_pread, doc::doc_pread, // Do not release GIL as it would access properties of python object
             py::arg("fd"), //
             py::arg("buf"), //
             py::arg("count"), //
             py::arg("file_offset"), //
             py::arg("buf_offset") = 0) //
        .def("pwrite", &py_pwrite, doc::doc_pwrite, // Do not release GIL as it would access properties of python object
             py::arg("fd"), //
             py::arg("buf"), //
             py::arg("count"), //
             py::arg("file_offset"), //
             py::arg("buf_offset") = 0) //
        .def("close", &close, doc::doc_close, py::call_guard<py::gil_scoped_release>())
        .def("discard_page_cache", &discard_page_cache, doc::doc_discard_page_cache,
             py::arg("file_path"), //
             py::call_guard<py::gil_scoped_release>());
}

std::shared_ptr<CuFileDriver> py_open(const char* file_path, const char* flags, mode_t mode)
{
    return open(file_path, flags, mode);
}
ssize_t py_pread(const std::shared_ptr<CuFileDriver>& fd, py::object buf, size_t count, off_t file_offset, off_t buf_offset)
{
    if (fd != nullptr)
    {
        return fd_pread(*fd, buf, count, file_offset, buf_offset);
    }
    else
    {
        fmt::print(stderr, "fd (CuFileDriver) is None!");
        return -1;
    }
}
ssize_t py_pwrite(const std::shared_ptr<CuFileDriver>& fd, py::object buf, size_t count, off_t file_offset, off_t buf_offset)
{
    if (fd != nullptr)
    {
        return fd_pwrite(*fd, buf, count, file_offset, buf_offset);
    }
    else
    {
        fmt::print(stderr, "fd (CuFileDriver) is None!");
        return -1;
    }
}

} // namespace cucim::filesystem