/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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

constexpr std::size_t kDeviceTypeCount = 6;
/**
 * Value for each device type follows https://github.com/dmlc/dlpack/blob/v0.6/include/dlpack/dlpack.h
 * Naming convention follows PyTorch (torch/include/c10/core/DeviceType.h)
 */
enum class DeviceType : int16_t
{
    kCPU = 1,
    kCUDA = 2,
    kCUDAHost = 3,
    kCUDAManaged = 13,

    kCPUShared = 101, /// custom type for CPU-shared memory
    kCUDAShared = 102, /// custom type for GPU-shared memory
};

// Using constexpr map (https://www.youtube.com/watch?v=INn3xa4pMfg)
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
