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

#include "cucim/io/device_type.h"

#include <algorithm>


namespace cucim::io
{

// https://en.cppreference.com/w/cpp/algorithm/find
#if __cplusplus < 202002L
template <class InputIt, class UnaryPredicate>
constexpr InputIt myfind_if(InputIt first, InputIt last, UnaryPredicate p)
{
    for (; first != last; ++first)
    {
        if (p(*first))
        {
            return first;
        }
    }
    return last;
}
#else
template <class InputIt, class UnaryPredicate>
constexpr InputIt myfind_if(InputIt first, InputIt last, UnaryPredicate p)
{
    return std::find_if(first, last, p);
}
#endif

using namespace std::literals::string_view_literals;

constexpr DeviceType DeviceTypeMap::at(const std::string_view& key) const
{
    const auto itr = myfind_if(begin(data), end(data), [&key](const auto& v) { return v.first == key; });

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
    const auto itr = myfind_if(begin(data), end(data), [&key](const auto& v) { return v.first == key; });

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
      { "pinned"sv, DeviceType::kPinned },
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
      { DeviceType::kPinned, "pinned"sv },
      { DeviceType::kCPUShared, "cpu_shared"sv },
      { DeviceType::kCUDAShared, "cuda_shared"sv } }
};

std::string_view lookup_device_type_str(const DeviceType key)
{
    static constexpr auto map = DeviceTypeStrMap{ { device_type_str_values } };
    return map.at(key);
}

} // namespace cucim::io