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

#ifndef CUCIM_PLUGIN_IMAGE_FORMAT_H
#define CUCIM_PLUGIN_IMAGE_FORMAT_H

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

#endif // CUCIM_PLUGIN_IMAGE_FORMAT_H
