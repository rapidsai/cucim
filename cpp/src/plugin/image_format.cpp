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


#include "cucim/io/format/image_format.h"
#include "cucim/plugin/image_format.h"

#include <fmt/format.h>


namespace cucim::plugin
{

bool ImageFormat::add_interfaces(const cucim::io::format::IImageFormat* image_formats)
{
    if (image_formats && image_formats->format_count > 0)
    {
        for (size_t i = 0; i < image_formats->format_count; ++i)
        {
            cucim::io::format::ImageFormatDesc* format = &(image_formats->formats[i]);
            image_formats_.push_back(format);
        }
    }
    else
    {
        return false;
    }
    return true;
}

cucim::io::format::ImageFormatDesc* ImageFormat::detect_image_format(const cucim::filesystem::Path& path)
{
    for (auto& format : image_formats_)
    {
        if (format->image_checker.is_valid(path.c_str(), nullptr, 0))
        {
            return format;
        }
    }
    throw std::invalid_argument(fmt::format("Cannot find a plugin to handle '{}'!", path));
}

} // namespace cucim::plugin
