/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
