/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cucim/filesystem/file_path.h"
#include "cucim/io/format/image_format.h"


namespace cucim::plugin
{

class ImageFormat
{
public:
    ImageFormat() = default;
    ~ImageFormat() = default;

    bool add_interfaces(const cucim::io::format::IImageFormat* image_formats);
    cucim::io::format::ImageFormatDesc* detect_image_format(const filesystem::Path& path);

    operator bool() const
    {
        return !image_formats_.empty();
    }

private:
    std::vector<cucim::io::format::ImageFormatDesc*> image_formats_;
};

} // namespace cucim::plugin
