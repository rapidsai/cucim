/*
 * SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYCUCIM_DEVICE_PYDOC_H
#define PYCUCIM_DEVICE_PYDOC_H

#include "../macros.h"

namespace cucim::io::doc::Device
{

// explicit Device();

// explicit Device(const std::string& device_name);
PYDOC(Device, R"doc(
Constructor for `Device`.
)doc")

// Device(const char* device_name);
// Device(DeviceType type, DeviceIndex index);
// Device(DeviceType type, DeviceIndex index, const std::string& param);

// static DeviceType parse_type(const std::string& device_name);
PYDOC(parse_type, R"doc(
Create `DeviceType` object from the device name string.
)doc")

// explicit operator std::string() const;

// DeviceType type() const;
PYDOC(type, R"doc(
Device type.
)doc")

// DeviceIndex index() const;
PYDOC(index, R"doc(
Device index.
)doc")

} // namespace cucim::io::doc::Device

#endif // PYCUCIM_DEVICE_PYDOC_H
