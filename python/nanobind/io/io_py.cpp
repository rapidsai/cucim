/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io_py.h"
#include "device_pydoc.h"

#include <nanobind/stl/string.h>

#include <cucim/io/device.h>

namespace cucim::io
{
void init_io(nb::module_ io)
{
    nb::enum_<DeviceType>(io, "DeviceType") //
        .value("CPU", DeviceType::kCPU) //
        .value("CUDA", DeviceType::kCUDA) //
        .value("CUDAHost", DeviceType::kCUDAHost) //
        .value("CUDAManaged", DeviceType::kCUDAManaged) //
        .value("CPUShared", DeviceType::kCPUShared) //
        .value("CUDAShared", DeviceType::kCUDAShared);

    nb::class_<Device>(io, "Device", nb::is_weak_referenceable()) //
        .def(nb::init_implicit<const std::string&>(), doc::Device::doc_Device) //
        .def_static("parse_type", &Device::parse_type, doc::Device::doc_parse_type) //
        .def_prop_ro("type", &Device::type, doc::Device::doc_type) //
        .def_prop_ro("index", &Device::index, doc::Device::doc_index)
        .def("__repr__", [](const Device& device) { return std::string(device); });
}
} // namespace cucim::io
