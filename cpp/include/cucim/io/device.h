/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CUCIM_IO_DEVICE_H
#define CUCIM_IO_DEVICE_H

#include "cucim/macros/api_header.h"

#include <cstdint>
#include <string>

#include "device_type.h"

namespace cucim::io
{

using DeviceIndex = int16_t;

// Make the following public libraries visible (default visibility) as this header's implementation is in device.cpp
// and provided by cucim library.
// Without that, a plugin module cannot find the definition of those methods when Device class is used in the plugin
// module.
class EXPORT_VISIBLE Device
{
public:
    explicit Device();
    Device(const Device& device);
    Device& operator=(const Device& device) = default;
    explicit Device(const std::string& device_name);
    Device(const char* device_name);
    Device(DeviceType type, DeviceIndex index);
    Device(DeviceType type, DeviceIndex index, const std::string& param);

    static DeviceType parse_type(const std::string& device_name);
    explicit operator std::string() const;

    DeviceType type() const;
    DeviceIndex index() const;
    const std::string& shm_name() const;

    void set_values(DeviceType type, DeviceIndex index = -1, const std::string& param = "");

private:
    DeviceType type_ = DeviceType::kCPU;
    DeviceIndex index_ = -1;
    std::string shm_name_; /// used for shared memory name

    bool validate_device();
};

} // namespace cucim::io


#endif // CUCIM_IO_DEVICE_H
