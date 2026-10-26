/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef PYCUCIM_FILESYSTEM_INIT_H
#define PYCUCIM_FILESYSTEM_INIT_H

#include <memory>
#include <string>

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace cucim::filesystem
{

// Forward declaration
class CuFileDriver;

void init_filesystem(nb::module_ m);

std::shared_ptr<CuFileDriver> py_open(const std::string& file_path, const std::string& flags, mode_t mode);
ssize_t py_pread(
    const std::shared_ptr<CuFileDriver>& fd, nb::handle buf, size_t count, off_t file_offset, off_t buf_offset = 0);
ssize_t py_pwrite(
    const std::shared_ptr<CuFileDriver>& fd, nb::handle buf, size_t count, off_t file_offset, off_t buf_offset = 0);
} // namespace cucim::filesystem

#endif // PYCUCIM_FILESYSTEM_INIT_H
