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
#define CUCIM_EXPORTS

#include "cumed.h"

#include <fcntl.h>
#include <filesystem>

#include <fmt/format.h>

#include <cucim/core/framework.h>
#include <cucim/core/plugin_util.h>
#include <cucim/filesystem/file_path.h>
#include <cucim/io/format/image_format.h>
#include <cucim/memory/memory_manager.h>


const struct cucim::PluginImplDesc kPluginImpl = {
    "cucim.kit.cumed", // name
    { 0, 1, 0 }, // version
    "dev", // build
    "clara team", // author
    "cumed", // description
    "cumed plugin", // long_description
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
    return "MetaIO";
}

static bool CUCIM_ABI checker_is_valid(const char* file_name, const char* buf, size_t size)
{
    (void)buf;
    (void)size;
    auto file = std::filesystem::path(file_name);
    auto extension = file.extension().string();
    if (extension.compare(".mhd") == 0)
    {
        return true;
    }
    return false;
}

static CuCIMFileHandle_share CUCIM_ABI parser_open(const char* file_path_)
{
    const cucim::filesystem::Path& file_path = file_path_;

    int mode = O_RDONLY;
    // Copy file path (Allocated memory would be freed at close() method.)
    char* file_path_cstr = static_cast<char*>(malloc(file_path.size() + 1));
    (void)file_path_cstr;
    memcpy(file_path_cstr, file_path.c_str(), file_path.size());
    file_path_cstr[file_path.size()] = '\0';

    int fd = ::open(file_path_cstr, mode, 0666);
    if (fd == -1)
    {
        cucim_free(file_path_cstr);
        throw std::invalid_argument(fmt::format("Cannot open {}!", file_path));
    }

    auto file_handle = std::make_shared<CuCIMFileHandle>(fd, nullptr, FileHandleType::kPosix, file_path_cstr, nullptr);
    CuCIMFileHandle_share handle = new std::shared_ptr<CuCIMFileHandle>(std::move(file_handle));

    return handle;
}

static bool CUCIM_ABI parser_parse(CuCIMFileHandle_ptr handle, cucim::io::format::ImageMetadataDesc* out_metadata_desc)
{
    (void)handle;
    if (!out_metadata_desc || !out_metadata_desc->handle)
    {
        throw std::runtime_error("out_metadata_desc shouldn't be nullptr!");
    }
    cucim::io::format::ImageMetadata& out_metadata =
        *reinterpret_cast<cucim::io::format::ImageMetadata*>(out_metadata_desc->handle);


    //
    // Metadata Setup
    //

    // Note: int-> uint16_t due to type differences between ImageMetadataDesc.ndim and DLTensor.ndim
    const uint16_t ndim = 3;
    auto& resource = out_metadata.get_resource();

    std::string_view dims{ "YXC" };

    std::pmr::vector<int64_t> shape({ 256, 256, 3 }, &resource);

    DLDataType dtype{ kDLUInt, 8, 1 };

    // Assume RGB
    std::pmr::vector<std::string_view> channel_names(
        { std::string_view{ "R" }, std::string_view{ "G" }, std::string_view{ "B" } }, &resource);

    std::pmr::vector<float> spacing(&resource);
    spacing.reserve(ndim);
    spacing.insert(spacing.end(), ndim, 1.0);

    std::pmr::vector<std::string_view> spacing_units(&resource);
    spacing_units.reserve(ndim);
    spacing_units.emplace_back(std::string_view{ "pixel" });
    spacing_units.emplace_back(std::string_view{ "pixel" });
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

    size_t level_count = 1;
    const uint16_t level_ndim = 2; // {'X', 'Y'}
    std::pmr::vector<int64_t> level_dimensions(&resource);
    level_dimensions.reserve(level_count * 2);
    for (size_t i = 0; i < level_count; ++i)
    {
        level_dimensions.emplace_back(256);
        level_dimensions.emplace_back(256);
    }

    std::pmr::vector<float> level_downsamples(&resource);
    for (size_t i = 0; i < level_count; ++i)
    {
        level_downsamples.emplace_back(1.0);
    }

    std::pmr::vector<uint32_t> level_tile_sizes(&resource);
    level_tile_sizes.reserve(level_count * 2);
    for (size_t i = 0; i < level_count; ++i)
    {
        level_tile_sizes.emplace_back(256);
        level_tile_sizes.emplace_back(256);
    }

    const size_t associated_image_count = 0;
    std::pmr::vector<std::string_view> associated_image_names(&resource);

    std::string_view raw_data{ "" };

    // Dynamically allocate memory for json_data (need to be freed manually);
    const std::string& json_str = std::string{};
    char* json_data_ptr = static_cast<char*>(cucim_malloc(json_str.size() + 1));
    memcpy(json_data_ptr, json_str.data(), json_str.size() + 1);
    std::string_view json_data{ json_data_ptr, json_str.size() };

    out_metadata.ndim(ndim);
    out_metadata.dims(std::move(dims));
    out_metadata.shape(std::move(shape));
    out_metadata.dtype(dtype);
    out_metadata.channel_names(std::move(channel_names));
    out_metadata.spacing(std::move(spacing));
    out_metadata.spacing_units(std::move(spacing_units));
    out_metadata.origin(std::move(origin));
    out_metadata.direction(std::move(direction));
    out_metadata.coord_sys(std::move(coord_sys));
    out_metadata.level_count(level_count);
    out_metadata.level_ndim(level_ndim);
    out_metadata.level_dimensions(std::move(level_dimensions));
    out_metadata.level_downsamples(std::move(level_downsamples));
    out_metadata.level_tile_sizes(std::move(level_tile_sizes));
    out_metadata.image_count(associated_image_count);
    out_metadata.image_names(std::move(associated_image_names));
    out_metadata.raw_data(raw_data);
    out_metadata.json_data(json_data);

    return true;
}

static bool CUCIM_ABI parser_close(CuCIMFileHandle_ptr handle_ptr)
{
    CuCIMFileHandle* handle = reinterpret_cast<CuCIMFileHandle*>(handle_ptr);

    if (handle->client_data)
    {
        // TODO: comment out and reinterpret_cast when needed.
        // delete reinterpret_cast<xx*>(handle->client_data);
        handle->client_data = nullptr;
    }

    return true;
}

static bool CUCIM_ABI reader_read(const CuCIMFileHandle_ptr handle_ptr,
                                  const cucim::io::format::ImageMetadataDesc* metadata,
                                  const cucim::io::format::ImageReaderRegionRequestDesc* request,
                                  cucim::io::format::ImageDataDesc* out_image_data,
                                  cucim::io::format::ImageMetadataDesc* out_metadata_desc = nullptr)
{
    CuCIMFileHandle* handle = reinterpret_cast<CuCIMFileHandle*>(handle_ptr);
    (void)handle;
    (void)metadata;

    std::string device_name(request->device);
    if (request->shm_name)
    {
        device_name = device_name + fmt::format("[{}]", request->shm_name); // TODO: check performance
    }
    cucim::io::Device out_device(device_name);

    uint8_t* raster = nullptr;
    uint32_t width = 256;
    uint32_t height = 256;
    uint32_t samples_per_pixel = 3;
    size_t raster_size = width * height * samples_per_pixel;

    // Raw metadata for the associated image
    const char* raw_data_ptr = nullptr;
    size_t raw_data_len = 0;
    // Json metadata for the associated image
    char* json_data_ptr = nullptr;

    // Populate image data
    const uint16_t ndim = 3;

    int64_t* container_shape = static_cast<int64_t*>(cucim_malloc(sizeof(int64_t) * ndim));
    container_shape[0] = height;
    container_shape[1] = width;
    container_shape[2] = 3; // hard-coded for 'C'

    // Copy the raster memory and free it if needed.
    cucim::memory::move_raster_from_host((void**)&raster, raster_size, out_device);

    auto& out_image_container = out_image_data->container;
    out_image_container.data = raster;
    out_image_container.device = DLDevice{ static_cast<DLDeviceType>(out_device.type()), out_device.index() };
    out_image_container.ndim = ndim;
    out_image_container.dtype = { kDLUInt, 8, 1 };
    out_image_container.shape = container_shape;
    out_image_container.strides = nullptr; // Tensor is compact and row-majored
    out_image_container.byte_offset = 0;

    auto& shm_name = out_device.shm_name();
    size_t shm_name_len = shm_name.size();
    if (shm_name_len != 0)
    {
        out_image_data->shm_name = static_cast<char*>(cucim_malloc(shm_name_len + 1));
        memcpy(out_image_data->shm_name, shm_name.c_str(), shm_name_len + 1);
    }
    else
    {
        out_image_data->shm_name = nullptr;
    }

    // Populate metadata
    if (out_metadata_desc && out_metadata_desc->handle)
    {
        cucim::io::format::ImageMetadata& out_metadata =
            *reinterpret_cast<cucim::io::format::ImageMetadata*>(out_metadata_desc->handle);
        auto& resource = out_metadata.get_resource();

        std::string_view dims{ "YXC" };

        std::pmr::vector<int64_t> shape(&resource);
        shape.reserve(ndim);
        shape.insert(shape.end(), &container_shape[0], &container_shape[ndim]);

        DLDataType dtype{ kDLUInt, 8, 1 };

        // TODO: Do not assume channel names as 'RGB'
        std::pmr::vector<std::string_view> channel_names(
            { std::string_view{ "R" }, std::string_view{ "G" }, std::string_view{ "B" } }, &resource);


        // We don't know physical pixel size for associated image so fill it with default value 1
        std::pmr::vector<float> spacing(&resource);
        spacing.reserve(ndim);
        spacing.insert(spacing.end(), ndim, 1.0);

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

        // Manually set resolution dimensions to 2
        const uint16_t level_ndim = 2;
        std::pmr::vector<int64_t> level_dimensions(&resource);
        level_dimensions.reserve(level_ndim * 1); // it has only one size
        level_dimensions.emplace_back(shape[1]); // width
        level_dimensions.emplace_back(shape[0]); // height

        std::pmr::vector<float> level_downsamples(&resource);
        level_downsamples.reserve(1);
        level_downsamples.emplace_back(1.0);

        std::pmr::vector<uint32_t> level_tile_sizes(&resource);
        level_tile_sizes.reserve(level_ndim * 1); // it has only one size
        level_tile_sizes.emplace_back(shape[1]); // tile_width
        level_tile_sizes.emplace_back(shape[0]); // tile_height

        // Empty associated images
        const size_t associated_image_count = 0;
        std::pmr::vector<std::string_view> associated_image_names(&resource);

        std::string_view raw_data{ raw_data_ptr ? raw_data_ptr : "", raw_data_len };
        std::string_view json_data{ json_data_ptr ? json_data_ptr : "" };

        out_metadata.ndim(ndim);
        out_metadata.dims(std::move(dims));
        out_metadata.shape(std::move(shape));
        out_metadata.dtype(dtype);
        out_metadata.channel_names(std::move(channel_names));
        out_metadata.spacing(std::move(spacing));
        out_metadata.spacing_units(std::move(spacing_units));
        out_metadata.origin(std::move(origin));
        out_metadata.direction(std::move(direction));
        out_metadata.coord_sys(std::move(coord_sys));
        out_metadata.level_count(1);
        out_metadata.level_ndim(2);
        out_metadata.level_dimensions(std::move(level_dimensions));
        out_metadata.level_downsamples(std::move(level_downsamples));
        out_metadata.level_tile_sizes(std::move(level_tile_sizes));
        out_metadata.image_count(associated_image_count);
        out_metadata.image_names(std::move(associated_image_names));
        out_metadata.raw_data(raw_data);
        out_metadata.json_data(json_data);
    }

    return true;
}

static bool CUCIM_ABI writer_write(const CuCIMFileHandle_ptr handle_ptr,
                                   const cucim::io::format::ImageMetadataDesc* metadata,
                                   const cucim::io::format::ImageDataDesc* image_data)
{
    CuCIMFileHandle* handle = reinterpret_cast<CuCIMFileHandle*>(handle_ptr);
    (void)handle;
    (void)metadata;
    (void)image_data;

    return true;
}

void fill_interface(cucim::io::format::IImageFormat& iface)
{
    static cucim::io::format::ImageCheckerDesc image_checker = { 0, 0, checker_is_valid };
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
