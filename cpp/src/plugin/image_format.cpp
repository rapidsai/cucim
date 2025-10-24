/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


#include "cucim/io/format/image_format.h"
#include "cucim/plugin/image_format.h"
#include "cucim/profiler/nvtx3.h"

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
    PROF_SCOPED_RANGE(PROF_EVENT(cucim_plugin_detect_image_format));
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
