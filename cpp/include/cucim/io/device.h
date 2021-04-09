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

#ifndef CUCIM_DEVICE_H
#define CUCIM_DEVICE_H

#include <cucim/macros/api_header.h>
#include <cstdint>
#include <string>

namespace cucim::io
{

using DeviceIndex = int16_t;

/**
 * Value for each device type follows https://github.com/dmlc/dlpack/blob/v0.3/include/dlpack/dlpack.h
 * Naming convention follows PyTorch (torch/include/c10/core/DeviceType.h)
 */
enum class DeviceType : int16_t
{
    kCPU = 1,
    kCUDA = 2,
    kPinned = 3,

    kCPUShared = 101, /// custom type for CPU-shared memory
    kCUDAShared = 102, /// custom type for GPU-shared memory
};

// Make the following public libraries visible (default visibility) as this header's implementation is in device.cpp
// and provided by cucim library.
// Without that, a plugin module cannot find the definition of those methods when Device class is used in the plugin
// module.
class EXPORT_VISIBLE Device
{
public:
    explicit Device();
    Device(const Device& device);
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


#endif // CUCIM_DEVICE_H
