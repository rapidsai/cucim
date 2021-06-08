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

#include "cucim/io/device.h"

#include <regex>
#include <string>
#include <string_view>

#include <fmt/format.h>

#include "cucim/macros/defines.h"


namespace cucim::io
{

Device::Device()
{
    // TODO: consider default case (how to handle -1 index?)
}

Device::Device(const Device& device) : type_(device.type_), index_(device.index_), shm_name_(device.shm_name_)
{
}

Device::Device(const std::string& device_name)
{
    // 'cuda', 'cuda:0', 'cpu[shm0]', 'cuda:0[cuda_shm0]'
    static const std::regex name_regex("([a-z]+)(?::(0|[1-9]\\d*))?(?:\\[([a-zA-Z0-9_\\-][a-zA-Z0-9_\\-\\.]*)\\])?");

    std::smatch match;
    if (std::regex_match(device_name, match, name_regex))
    {
        type_ = parse_type(match[1].str());
        if (match[2].matched)
        {
            index_ = std::stoi(match[2].str());
        }
        if (match[3].matched)
        {
            shm_name_ = match[3].str();
        }
    }
    else
    {
        CUCIM_ERROR("Device name doesn't match!");
    }

    validate_device();
}
Device::Device(const char* device_name) : Device::Device(std::string(device_name))
{
}

Device::Device(DeviceType type, DeviceIndex index)
{
    type_ = type;
    index_ = index;
    validate_device();
}

Device::Device(DeviceType type, DeviceIndex index, const std::string& param)
{
    type_ = type;
    index_ = index;
    shm_name_ = param;
    validate_device();
}

DeviceType Device::parse_type(const std::string& device_name)
{
    return lookup_device_type(device_name);
}
Device::operator std::string() const
{
    std::string_view device_type_str = lookup_device_type_str(type_);

    if (index_ == -1 && shm_name_.empty())
    {
        return fmt::format("{}", device_type_str);
    }
    else if (index_ != -1 && shm_name_.empty())
    {
        return fmt::format("{}:{}", device_type_str, index_);
    }
    else
    {
        return fmt::format("{}:{}[{}]", device_type_str, index_, shm_name_);
    }
}

DeviceType Device::type() const
{
    return type_;
};
DeviceIndex Device::index() const
{
    return index_;
}
const std::string& Device::shm_name() const
{
    return shm_name_;
}

void Device::set_values(DeviceType type, DeviceIndex index, const std::string& param)
{
    type_ = type;
    index_ = index;
    shm_name_ = param;
}

bool Device::validate_device()
{
    // TODO: implement this
    return true;
}

} // namespace cucim::io