/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cucim/io/device_type.h"
#include "cucim/cpp20/find_if.h"


namespace cucim::io
{

using namespace std::literals::string_view_literals;

constexpr DeviceType DeviceTypeMap::at(const std::string_view& key) const
{
    const auto itr = cucim::cpp20::find_if(begin(data), end(data), [&key](const auto& v) { return v.first == key; });

    if (itr != end(data))
    {
        return itr->second;
    }
    else
    {
        return DeviceType::kCPU;
    }
}

constexpr std::string_view DeviceTypeStrMap::at(const DeviceType& key) const
{
    const auto itr = cucim::cpp20::find_if(begin(data), end(data), [&key](const auto& v) { return v.first == key; });

    if (itr != end(data))
    {
        return itr->second;
    }
    else
    {
        return "cpu"sv;
    }
}

static constexpr std::array<std::pair<std::string_view, DeviceType>, kDeviceTypeCount> device_type_values{
    { { "cpu"sv, DeviceType::kCPU },
      { "cuda"sv, DeviceType::kCUDA },
      { "cuda_host"sv, DeviceType::kCUDAHost },
      { "cuda_managed"sv, DeviceType::kCUDAManaged },
      { "cpu_shared"sv, DeviceType::kCPUShared },
      { "cuda_shared"sv, DeviceType::kCUDAShared } }
};


DeviceType lookup_device_type(const std::string_view sv)
{
    static constexpr auto map = DeviceTypeMap{ { device_type_values } };
    return map.at(sv);
}

static constexpr std::array<std::pair<DeviceType, std::string_view>, kDeviceTypeCount> device_type_str_values{
    { { DeviceType::kCPU, "cpu"sv },
      { DeviceType::kCUDA, "cuda"sv },
      { DeviceType::kCUDAHost, "cuda_host"sv },
      { DeviceType::kCUDAManaged, "cuda_managed"sv },
      { DeviceType::kCPUShared, "cpu_shared"sv },
      { DeviceType::kCUDAShared, "cuda_shared"sv } }
};

std::string_view lookup_device_type_str(const DeviceType key)
{
    static constexpr auto map = DeviceTypeStrMap{ { device_type_str_values } };
    return map.at(key);
}

} // namespace cucim::io
