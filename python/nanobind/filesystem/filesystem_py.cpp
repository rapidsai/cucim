/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "filesystem_py.h"
#include "filesystem_pydoc.h"

#include <fmt/format.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <cucim/filesystem/cufile_driver.h>

#include "../compat.h"
#include "cufile_py.h"
#include "cufile_pydoc.h"

using namespace nanobind::literals;

namespace cucim::filesystem
{

void init_filesystem(nb::module_ fs)
{
    bind_enum<FileHandleType>(fs, "FileHandleType") //
        .value("Unknown", FileHandleType::kUnknown) //
        .value("Posix", FileHandleType::kPosix) //
        .value("PosixODirect", FileHandleType::kPosixODirect) //
        .value("MemoryMapped", FileHandleType::kMemoryMapped) //
        .value("GPUDirect", FileHandleType::kGPUDirect);

    nb::class_<CuFileDriver>(fs, "CuFileDriver", nb::is_weak_referenceable())
        .def(
            "__init__",
            [](CuFileDriver* self, int fd, Bool no_gds, Bool use_mmap, const std::filesystem::path& file_path) {
                nb::gil_scoped_release release;
                new (self) CuFileDriver(fd, no_gds, use_mmap, file_path.c_str());
            },
            "fd"_a, //
            "no_gds"_a.none() = false, //
            "use_mmap"_a.none() = false, //
            "file_path"_a = "", //
            doc::CuFileDriver::doc_CuFileDriver)
        .def("pread", &fd_pread, doc::CuFileDriver::doc_pread, // Do not release GIL as it would access properties of
                                                               // python object
             "buf"_a, //
             "count"_a, //
             "file_offset"_a, //
             "buf_offset"_a = 0) //
        .def("pwrite", &fd_pwrite, doc::CuFileDriver::doc_pwrite, // Do not release GIL as it would access properties of
                                                                  // python object
             "buf"_a, //
             "count"_a, //
             "file_offset"_a, //
             "buf_offset"_a = 0) //
        .def("close", &CuFileDriver::close, doc::CuFileDriver::doc_close, nb::call_guard<nb::gil_scoped_release>()) //
        .def("__enter__",
             [](nb::handle self) { //
                 return self; //
             })
        .def(
            "__exit__",
            [](CuFileDriver& fd, nb::handle /* type */, nb::handle /* value */, nb::handle /* traceback */) { //
                nb::gil_scoped_release release;
                fd.close(); //
            },
            "type"_a.none(), "value"_a.none(), "traceback"_a.none())
        .def("__repr__", [](const CuFileDriver& fd) {
            return fmt::format("<cucim.clara.filesystem.CuFileDriver path:{}>", fd.path());
        });

    fs.def("is_gds_available", &is_gds_available, doc::doc_is_gds_available, nb::call_guard<nb::gil_scoped_release>())
        .def("open", &py_open, doc::doc_open,
             "file_path"_a, //
             "flags"_a, //
             "mode"_a = 0644, //
             nb::call_guard<nb::gil_scoped_release>())
        .def("pread", &py_pread, doc::doc_pread, // Do not release GIL as it would access properties of python object
             "fd"_a.none(), //
             "buf"_a, //
             "count"_a, //
             "file_offset"_a, //
             "buf_offset"_a = 0) //
        .def("pwrite", &py_pwrite, doc::doc_pwrite, // Do not release GIL as it would access properties of python object
             "fd"_a.none(), //
             "buf"_a, //
             "count"_a, //
             "file_offset"_a, //
             "buf_offset"_a = 0) //
        .def("close", &close, doc::doc_close, nb::call_guard<nb::gil_scoped_release>())
        .def(
            "discard_page_cache",
            [](const std::filesystem::path& file_path) { return discard_page_cache(file_path.c_str()); },
            doc::doc_discard_page_cache,
            "file_path"_a, //
            nb::call_guard<nb::gil_scoped_release>());
}

std::shared_ptr<CuFileDriver> py_open(const std::filesystem::path& file_path, const std::string& flags, mode_t mode)
{
    return open(file_path.c_str(), flags.c_str(), mode);
}
ssize_t py_pread(const std::shared_ptr<CuFileDriver>& fd, nb::handle buf, size_t count, off_t file_offset, off_t buf_offset)
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
ssize_t py_pwrite(const std::shared_ptr<CuFileDriver>& fd, nb::handle buf, size_t count, off_t file_offset, off_t buf_offset)
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
