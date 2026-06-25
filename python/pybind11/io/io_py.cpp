/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io_pydoc.h"
#include "device_pydoc.h"

#include <pybind11/pybind11.h>

#include <cucim/io/device.h>

namespace py = pybind11;

namespace cucim::io
{
void init_io(py::module& io)
{
    py::enum_<DeviceType>(io, "DeviceType") //
        .value("CPU", DeviceType::kCPU) //
        .value("CUDA", DeviceType::kCUDA) //
        .value("CUDAHost", DeviceType::kCUDAHost) //
        .value("CUDAManaged", DeviceType::kCUDAManaged) //
        .value("CPUShared", DeviceType::kCPUShared) //
        .value("CUDAShared", DeviceType::kCUDAShared);

    py::class_<Device>(io, "Device") //
        .def(py::init<const std::string&>(), doc::Device::doc_Device) //
        .def_static("parse_type", &Device::parse_type, doc::Device::doc_parse_type) //
        .def_property("type", &Device::type, nullptr, doc::Device::doc_type) //
        .def_property("index", &Device::index, nullptr, doc::Device::doc_index)
        .def("__repr__", [](const Device& device) { return std::string(device); });
}
} // namespace cucim::io
