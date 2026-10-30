/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef PYCUCIM_MEMORY_INIT_H
#define PYCUCIM_MEMORY_INIT_H

#include <nanobind/nanobind.h>

#include <cucim/io/device.h>

namespace nb = nanobind;

namespace cucim::memory
{

void get_memory_info(nb::handle buf_obj,
                     void** out_buf,
                     cucim::io::Device* out_device = nullptr,
                     size_t* out_memory_size = 0,
                     bool* out_readonly = nullptr);

} // namespace cucim::memory


#endif // PYCUCIM_MEMORY_INIT_H
