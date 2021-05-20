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
        .value("Pinned", DeviceType::kPinned) //
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