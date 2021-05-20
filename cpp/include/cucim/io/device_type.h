/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#ifndef CUCIM_IO_DEVICE_TYPE_H
#define CUCIM_IO_DEVICE_TYPE_H

#include "cucim/macros/api_header.h"

#include <array>
#include <cstdint>
#include <string_view>


namespace cucim::io
{

using DeviceIndex = int16_t;

constexpr std::size_t kDeviceTypeCount = 5;
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

struct DeviceTypeMap
{
    std::array<std::pair<std::string_view, DeviceType>, kDeviceTypeCount> data;

    [[nodiscard]] constexpr DeviceType at(const std::string_view& key) const;
};

EXPORT_VISIBLE DeviceType lookup_device_type(const std::string_view sv);

struct DeviceTypeStrMap
{
    std::array<std::pair<DeviceType, std::string_view>, kDeviceTypeCount> data;

    [[nodiscard]] constexpr std::string_view at(const DeviceType& key) const;
};

EXPORT_VISIBLE std::string_view lookup_device_type_str(const DeviceType type);


} // namespace cucim::io


#endif // CUCIM_IO_DEVICE_TYPE_H
