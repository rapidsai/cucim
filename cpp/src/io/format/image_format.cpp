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

#include "cucim/macros/defines.h"
#include "cucim/io/format/image_format.h"
#include "cucim/memory/memory_manager.h"

#include <fmt/format.h>


namespace cucim::io::format
{

ImageMetadata::ImageMetadata()
{
    desc_.handle = this;
}

void* ImageMetadata::allocate(size_t size)
{
    return res_.allocate(size);
}

std::pmr::monotonic_buffer_resource& ImageMetadata::get_resource()
{
    return res_;
}

ImageMetadataDesc& ImageMetadata::desc()
{
    return desc_;
}

ImageMetadata& ImageMetadata::ndim(uint16_t ndim)
{
    desc_.ndim = ndim;
    return *this;
}

ImageMetadata& ImageMetadata::dims(std::string_view&& dims)
{
    dims_ = dims;
    desc_.dims = dims_.data();
    return *this;
}

ImageMetadata& ImageMetadata::shape(std::pmr::vector<int64_t>&& shape)
{
    shape_ = shape;
    desc_.shape = const_cast<int64_t*>(shape_.data());
    return *this;
}

ImageMetadata& ImageMetadata::dtype(const DLDataType& dtype)
{
    desc_.dtype = dtype;
    return *this;
}

ImageMetadata& ImageMetadata::channel_names(std::pmr::vector<std::string_view>&& channel_names)
{
    const int channel_len = channel_names.size();
    channel_names_ = channel_names;

    desc_.channel_names = static_cast<char**>(allocate(channel_len * sizeof(char*)));
    for (int i = 0; i < channel_len; ++i)
    {
        desc_.channel_names[i] = const_cast<char*>(channel_names_[i].data());
    }
    return *this;
}

ImageMetadata& ImageMetadata::spacing(std::pmr::vector<float>&& spacing)
{
    spacing_ = spacing;
    desc_.spacing = const_cast<float*>(spacing_.data());
    return *this;
}

ImageMetadata& ImageMetadata::spacing_units(std::pmr::vector<std::string_view>&& spacing_units)
{
    const int ndim = spacing_units.size();
    spacing_units_ = spacing_units;

    desc_.spacing_units = static_cast<char**>(allocate(ndim * sizeof(char*)));
    for (int i = 0; i < ndim; ++i)
    {
        desc_.spacing_units[i] = const_cast<char*>(spacing_units_[i].data());
    }
    return *this;
}

ImageMetadata& ImageMetadata::origin(std::pmr::vector<float>&& origin)
{
    origin_ = origin;
    desc_.origin = const_cast<float*>(origin_.data());
    return *this;
}

ImageMetadata& ImageMetadata::direction(std::pmr::vector<float>&& direction)
{
    direction_ = direction;
    desc_.direction = const_cast<float*>(direction_.data());
    return *this;
}

ImageMetadata& ImageMetadata::coord_sys(std::string_view&& coord_sys)
{
    coord_sys_ = coord_sys;
    desc_.coord_sys = coord_sys_.data();
    return *this;
}

ImageMetadata& ImageMetadata::level_count(uint16_t level_count)
{
    desc_.resolution_info.level_count = level_count;
    return *this;
}

ImageMetadata& ImageMetadata::level_ndim(uint16_t level_ndim)
{
    desc_.resolution_info.level_ndim = level_ndim;
    return *this;
}

ImageMetadata& ImageMetadata::level_dimensions(std::pmr::vector<int64_t>&& level_dimensions)
{
    level_dimensions_ = level_dimensions;
    desc_.resolution_info.level_dimensions = const_cast<int64_t*>(level_dimensions_.data());
    return *this;
}

ImageMetadata& ImageMetadata::level_downsamples(std::pmr::vector<float>&& level_downsamples)
{
    level_downsamples_ = level_downsamples;
    desc_.resolution_info.level_downsamples = const_cast<float*>(level_downsamples_.data());
    return *this;
}

ImageMetadata& ImageMetadata::level_tile_sizes(std::pmr::vector<uint32_t>&& level_tile_sizes)
{
    level_tile_sizes_ = level_tile_sizes;
    desc_.resolution_info.level_tile_sizes = const_cast<uint32_t*>(level_tile_sizes_.data());
    return *this;
}

ImageMetadata& ImageMetadata::image_count(uint16_t image_count)
{
    desc_.associated_image_info.image_count = image_count;
    return *this;
}

ImageMetadata& ImageMetadata::image_names(std::pmr::vector<std::string_view>&& image_names)
{
    const int image_size = image_names.size();
    image_names_ = image_names;

    desc_.associated_image_info.image_names = static_cast<char**>(allocate(image_size * sizeof(char*)));
    for (int i = 0; i < image_size; ++i)
    {
        desc_.associated_image_info.image_names[i] = const_cast<char*>(image_names_[i].data());
    }
    return *this;
}

ImageMetadata& ImageMetadata::raw_data(const std::string_view& raw_data)
{
    desc_.raw_data = raw_data.data();
    return *this;
}

ImageMetadata& ImageMetadata::json_data(const std::string_view& json_data)
{
    desc_.json_data = const_cast<char*>(json_data.data());
    return *this;
}

ImageMetadata::~ImageMetadata()
{
    // Memory for json_data needs to be manually released if image_metadata_->json_data is not ""
    // This logic may be already executed(@CuImage::~CuImage()) if this object is part of CuImage object.
    if (desc_.json_data && *desc_.json_data != '\0')
    {
        cucim_free(desc_.json_data);
        desc_.json_data = nullptr;
    }
    desc_.handle = nullptr;
}

} // namespace cucim::io::format
