/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#define CUCIM_EXPORTS

#include "cuslide.h"

#include "cucim/core/framework.h"
#include "cucim/core/plugin_util.h"
#include "cucim/io/format/image_format.h"
#include "tiff/tiff.h"

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <array>
#include <cassert>
#include <cstring>
#include <fcntl.h>
#include <memory>

using json = nlohmann::json;

const struct cucim::PluginImplDesc kPluginImpl = {
    "cucim.kit.cuslide", // name
    { 0, 1, 0 }, // version
    "dev", // build
    "clara team", // author
    "cuslide", // description
    "cuslide plugin", // long_description
    "Apache-2.0", // license
    "https://github.com/rapidsai/cucim", // url
    "linux", // platforms,
    cucim::PluginHotReload::kDisabled, // hot_reload
};

// Using CARB_PLUGIN_IMPL_MINIMAL instead of CARB_PLUGIN_IMPL
// This minimal macro doesn't define global variables for logging, profiler, crash reporting,
// and also doesn't call for the client registration for those systems
CUCIM_PLUGIN_IMPL_MINIMAL(kPluginImpl, cucim::io::format::IImageFormat)
CUCIM_PLUGIN_IMPL_NO_DEPS()


static void set_enabled(bool val)
{
    (void)val;
}

static bool is_enabled()
{
    return true;
}

static const char* get_format_name()
{
    return "Generic TIFF";
}

static bool CUCIM_ABI checker_is_valid(const char* file_name, const char* buf) // TODO: need buffer size parameter
{
    // TODO implement this
    (void)file_name;
    (void)buf;
    return true;
}

static CuCIMFileHandle CUCIM_ABI parser_open(const char* file_path)
{
    auto tif = new cuslide::tiff::TIFF(file_path, O_RDONLY);
    tif->construct_ifds();
    return tif->file_handle();
}

static bool CUCIM_ABI parser_parse(CuCIMFileHandle* handle, cucim::io::format::ImageMetadataDesc* out_metadata_desc)
{
    if (!out_metadata_desc || !out_metadata_desc->handle)
    {
        throw std::runtime_error("out_metadata_desc shouldn't be nullptr!");
    }
    cucim::io::format::ImageMetadata& out_metadata =
        *reinterpret_cast<cucim::io::format::ImageMetadata*>(out_metadata_desc->handle);

    auto tif = static_cast<cuslide::tiff::TIFF*>(handle->client_data);

    std::vector<size_t> main_ifd_list;

    size_t ifd_count = tif->ifd_count();
    size_t level_count = tif->level_count();
    for (size_t i = 0; i < ifd_count; i++)
    {
        const std::shared_ptr<cuslide::tiff::IFD>& ifd = tif->ifd(i);

        //        const char* char_ptr = ifd->model().c_str();
        //        uint32_t width = ifd->width();
        //        uint32_t height = ifd->height();
        //        uint32_t bits_per_sample = ifd->bits_per_sample();
        //        uint32_t samples_per_pixel = ifd->samples_per_pixel();
        uint64_t subfile_type = ifd->subfile_type();
        //        printf("image_description:\n%s\n", ifd->image_description().c_str());
        //        printf("model=%s, width=%u, height=%u, model=%p bits_per_sample:%u, samples_per_pixel=%u, %lu \n",
        //        char_ptr,
        //               width, height, char_ptr, bits_per_sample, samples_per_pixel, subfile_type);
        if (subfile_type == 0)
        {
            main_ifd_list.push_back(i);
        }
    }

    // Assume that the image has only one main (high resolution) image.
    if (main_ifd_list.size() != 1)
    {
        throw std::runtime_error(
            fmt::format("This format has more than one image with Subfile Type 0 so cannot be loaded!"));
    }

    // Explicitly forbid loading SVS format (#17)
    if (tif->ifd(0)->image_description().rfind("Aperio", 0) == 0)
    {
        throw std::runtime_error(
            fmt::format("cuCIM doesn't support Aperio SVS for now (https://github.com/rapidsai/cucim/issues/17)."));
    }

    //
    // Metadata Setup
    //

    // Note: int-> uint16_t due to type differences between ImageMetadataDesc.ndim and DLTensor.ndim
    const uint16_t ndim = 3;
    auto& resource = out_metadata.get_resource();

    std::string_view dims{ "YXC" };

    const auto& level0_ifd = tif->level_ifd(0);
    std::pmr::vector<int64_t> shape(
        { level0_ifd->height(), level0_ifd->width(), level0_ifd->samples_per_pixel() }, &resource);

    DLDataType dtype{ kDLUInt, 8, 1 };

    // TODO: Fill correct values for cucim::io::format::ImageMetadataDesc
    // TODO: Do not assume channel names as 'RGB'
    std::pmr::vector<std::string_view> channel_names(
        { std::string_view{ "R" }, std::string_view{ "G" }, std::string_view{ "B" } }, &resource);

    // TODO: Set correct spacing value
    std::pmr::vector<float> spacing(&resource);
    spacing.reserve(ndim);
    spacing.insert(spacing.end(), ndim, 1.0);

    // TODO: Set correct spacing units
    std::pmr::vector<std::string_view> spacing_units(&resource);
    spacing_units.reserve(ndim);
    spacing_units.emplace_back(std::string_view{ "micrometer" });
    spacing_units.emplace_back(std::string_view{ "micrometer" });
    spacing_units.emplace_back(std::string_view{ "color" });

    std::pmr::vector<float> origin({ 0.0, 0.0, 0.0 }, &resource);
    // Direction cosines (size is always 3x3)
    // clang-format off
    std::pmr::vector<float> direction({ 1.0, 0.0, 0.0,
                                        0.0, 1.0, 0.0,
                                        0.0, 0.0, 1.0}, &resource);
    // clang-format on

    // The coordinate frame in which the direction cosines are measured (either 'LPS'(ITK/DICOM) or 'RAS'(NIfTI/3D
    // Slicer))
    std::string_view coord_sys{ "LPS" };

    const uint16_t level_ndim = 2;
    std::pmr::vector<int64_t> level_dimensions(&resource);
    level_dimensions.reserve(level_count * 2);
    for (size_t i = 0; i < level_count; ++i)
    {
        const auto& level_ifd = tif->level_ifd(i);
        level_dimensions.emplace_back(level_ifd->width());
        level_dimensions.emplace_back(level_ifd->height());
    }

    std::pmr::vector<float> level_downsamples(&resource);
    float orig_width = static_cast<float>(shape[1]);
    float orig_height = static_cast<float>(shape[0]);
    for (size_t i = 0; i < level_count; ++i)
    {
        const auto& level_ifd = tif->level_ifd(i);
        level_downsamples.emplace_back(((orig_width / level_ifd->width()) + (orig_height / level_ifd->height())) / 2);
    }

    std::pmr::vector<uint32_t> level_tile_sizes(&resource);
    level_tile_sizes.reserve(level_count * 2);
    for (size_t i = 0; i < level_count; ++i)
    {
        const auto& level_ifd = tif->level_ifd(i);
        level_tile_sizes.emplace_back(level_ifd->tile_width());
        level_tile_sizes.emplace_back(level_ifd->tile_height());
    }

    const size_t associated_image_count = tif->associated_image_count();
    std::pmr::vector<std::string_view> associated_image_names(&resource);
    for (const auto& associated_image : tif->associated_images())
    {
        associated_image_names.emplace_back(std::string_view{ associated_image.first.c_str() });
    }

    auto& image_description = level0_ifd->image_description();
    std::string_view raw_data{ image_description.empty() ? "" : image_description.c_str() };

    // Dynamically allocate memory for json_data (need to be freed manually);
    const std::string& json_str = tif->metadata();
    char* json_data_ptr = static_cast<char*>(cucim_malloc(json_str.size() + 1));
    memcpy(json_data_ptr, json_str.data(), json_str.size() + 1);
    std::string_view json_data{ json_data_ptr, json_str.size() };

    out_metadata.ndim(ndim);
    out_metadata.dims(dims);
    out_metadata.shape(shape);
    out_metadata.dtype(dtype);
    out_metadata.channel_names(channel_names);
    out_metadata.spacing(spacing);
    out_metadata.spacing_units(spacing_units);
    out_metadata.origin(origin);
    out_metadata.direction(direction);
    out_metadata.coord_sys(coord_sys);
    out_metadata.level_count(level_count);
    out_metadata.level_ndim(level_ndim);
    out_metadata.level_dimensions(level_dimensions);
    out_metadata.level_downsamples(level_downsamples);
    out_metadata.level_tile_sizes(level_tile_sizes);
    out_metadata.image_count(associated_image_count);
    out_metadata.image_names(associated_image_names);
    out_metadata.raw_data(raw_data);
    out_metadata.json_data(json_data);

    return true;
}

static bool CUCIM_ABI parser_close(CuCIMFileHandle* handle)
{
    auto tif = static_cast<cuslide::tiff::TIFF*>(handle->client_data);
    delete tif;
    handle->client_data = nullptr;
    return true;
}

static bool CUCIM_ABI reader_read(const CuCIMFileHandle* handle,
                                  const cucim::io::format::ImageMetadataDesc* metadata,
                                  const cucim::io::format::ImageReaderRegionRequestDesc* request,
                                  cucim::io::format::ImageDataDesc* out_image_data,
                                  cucim::io::format::ImageMetadataDesc* out_metadata = nullptr)
{
    auto tif = static_cast<cuslide::tiff::TIFF*>(handle->client_data);
    bool result = tif->read(metadata, request, out_image_data, out_metadata);

    return result;
}

static bool CUCIM_ABI writer_write(const CuCIMFileHandle* handle,
                                   const cucim::io::format::ImageMetadataDesc* metadata,
                                   const cucim::io::format::ImageDataDesc* image_data)
{
    (void)handle;
    (void)metadata;
    (void)image_data;

    return true;
}

void fill_interface(cucim::io::format::IImageFormat& iface)
{
    static cucim::io::format::ImageCheckerDesc image_checker = { 0, 80, checker_is_valid };
    static cucim::io::format::ImageParserDesc image_parser = { parser_open, parser_parse, parser_close };

    static cucim::io::format::ImageReaderDesc image_reader = { reader_read };
    static cucim::io::format::ImageWriterDesc image_writer = { writer_write };

    // clang-format off
    static cucim::io::format::ImageFormatDesc image_format_desc = {
        set_enabled,
        is_enabled,
        get_format_name,
        image_checker,
        image_parser,
        image_reader,
        image_writer
    };
    // clang-format on

    // clang-format off
    iface =
    {
        &image_format_desc,
        1
    };
    // clang-format on
}
