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

#include "cucim/cuimage.h"

#include "cucim/util/file.h"

#include <fmt/format.h>

#include <iostream>
#include <fstream>
#include <cstring>

#define XSTR(x) STR(x)
#define STR(x) #x

namespace cucim
{

DimIndices::DimIndices(const char* dims)
{
    if (!dims)
    {
        return;
    }
    // TODO: check illegal characters

    int index = 0;
    for (const char* ptr = dims; *ptr != 0; ++ptr, ++index)
    {
        char dim_char = toupper(*ptr);
        dim_indices_.indices[dim_char - 'A'] = index;
    }
}
DimIndices::DimIndices(std::vector<std::pair<char, int64_t>> init_list)
{
    // TODO: check illegal characters
    for (auto& object : init_list)
    {
        char dim_char = toupper(object.first);
        dim_indices_.indices[dim_char - 'A'] = object.second;
    }
}
int64_t DimIndices::index(char dim_char) const
{
    dim_char = toupper(dim_char);
    return dim_indices_.indices[dim_char - 'A'];
}

ResolutionInfo::ResolutionInfo(io::format::ResolutionInfoDesc desc)
{
    level_count_ = desc.level_count;
    level_ndim_ = desc.level_ndim;

    level_dimensions_.insert(
        level_dimensions_.end(), &desc.level_dimensions[0], &desc.level_dimensions[level_count_ * level_ndim_]);
    level_downsamples_.insert(
        level_downsamples_.end(), &desc.level_downsamples[0], &desc.level_downsamples[level_count_]);
    level_tile_sizes_.insert(
        level_tile_sizes_.end(), &desc.level_tile_sizes[0], &desc.level_tile_sizes[level_count_ * level_ndim_]);
}
uint16_t ResolutionInfo::level_count() const
{
    return level_count_;
}
const std::vector<int64_t>& ResolutionInfo::level_dimensions() const
{
    return level_dimensions_;
}
std::vector<int64_t> ResolutionInfo::level_dimension(uint16_t level) const
{
    if (level >= level_count_)
    {
        throw std::invalid_argument(fmt::format("'level' should be less than {}", level_count_));
    }
    std::vector<int64_t> result;
    auto start_index = level_dimensions_.begin() + (level * level_ndim_);
    result.insert(result.end(), start_index, start_index + level_ndim_);
    return result;
}
const std::vector<float>& ResolutionInfo::level_downsamples() const
{
    return level_downsamples_;
}
float ResolutionInfo::level_downsample(uint16_t level) const
{
    if (level >= level_count_)
    {
        throw std::invalid_argument(fmt::format("'level' should be less than {}", level_count_));
    }
    return level_downsamples_.at(level);
}
const std::vector<uint32_t>& ResolutionInfo::level_tile_sizes() const
{
    return level_tile_sizes_;
}
std::vector<uint32_t> ResolutionInfo::level_tile_size(uint16_t level) const
{
    if (level >= level_count_)
    {
        throw std::invalid_argument(fmt::format("'level' should be less than {}", level_count_));
    }
    std::vector<uint32_t> result;
    auto start_index = level_tile_sizes_.begin() + (level * level_ndim_);
    result.insert(result.end(), start_index, start_index + level_ndim_);
    return result;
}

DetectedFormat detect_format(filesystem::Path path)
{
    // TODO: implement this
    (void)path;
    return { "Generic TIFF", { "cucim.kit.cuslide" } };
}


Framework* CuImage::framework_ = cucim::acquire_framework("cucim");
std::unique_ptr<config::Config> CuImage::config_ = std::make_unique<config::Config>();
std::unique_ptr<cache::ImageCacheManager> CuImage::cache_manager_ = std::make_unique<cache::ImageCacheManager>();


CuImage::CuImage(const filesystem::Path& path)
{
    //    printf("[cuCIM] CuImage::CuImage(filesystem::Path path)\n");
    ensure_init();
    (void)path;

    // TODO: need to detect available format for the file path
    file_handle_ = image_formats_->formats[0].image_parser.open(path.c_str());
    //    printf("[GB] file_handle: %s\n", file_handle_.path);
    //    fmt::print("[GB] CuImage path char: '{}'\n", file_handle_.path[0]);


    io::format::ImageMetadata& image_metadata = *(new io::format::ImageMetadata{});
    image_metadata_ = &image_metadata.desc();
    is_loaded_ = image_formats_->formats[0].image_parser.parse(&file_handle_, image_metadata_);
    dim_indices_ = DimIndices(image_metadata_->dims);

    auto& associated_image_info = image_metadata_->associated_image_info;
    uint16_t image_count = associated_image_info.image_count;
    if (image_count != associated_images_.size())
    {
        for (int i = 0; i < image_count; ++i)
        {
            associated_images_.emplace(associated_image_info.image_names[i]);
        }
    }
}
CuImage::CuImage(const filesystem::Path& path, const std::string& plugin_name)
{
    // TODO: implement this
    (void)path;
    (void)plugin_name;
}

// CuImage::CuImage(const CuImage& cuimg) : std::enable_shared_from_this<CuImage>()
//{
//    printf("[cuCIM] CuImage::CuImage(const CuImage& cuimg)\n");
//    (void)cuimg;
//
//}

CuImage::CuImage(CuImage&& cuimg) : std::enable_shared_from_this<CuImage>()
{
    // printf("[cuCIM] CuImage::CuImage(CuImage&& cuimg) %s\n", cuimg.file_handle_.path);
    (void)cuimg;
    std::swap(file_handle_, cuimg.file_handle_);
    std::swap(image_formats_, cuimg.image_formats_);
    std::swap(image_metadata_, cuimg.image_metadata_);
    std::swap(image_data_, cuimg.image_data_);
    std::swap(is_loaded_, cuimg.is_loaded_);
    std::swap(dim_indices_, cuimg.dim_indices_);
    cuimg.associated_images_.swap(associated_images_);
}

CuImage::CuImage(const CuImage* cuimg,
                 io::format::ImageMetadataDesc* image_metadata,
                 cucim::io::format::ImageDataDesc* image_data)
    : std::enable_shared_from_this<CuImage>()
{
    //    printf(
    //        "[cuCIM] CuImage::CuImage(CuImage* cuimg, io::format::ImageMetadataDesc* image_metadata,
    //        cucim::io::format::ImageDataDesc* image_data)\n");

    //    file_handle_ = cuimg->file_handle_; ==> Don't do this. it will cause a double free.
    image_formats_ = cuimg->image_formats_;
    image_metadata_ = image_metadata;
    image_data_ = image_data;
    is_loaded_ = true;
    if (image_metadata)
    {
        dim_indices_ = DimIndices(image_metadata->dims);
    }

    auto& associated_image_info = image_metadata_->associated_image_info;
    uint16_t image_count = associated_image_info.image_count;
    if (image_count != associated_images_.size())
    {
        for (int i = 0; i < image_count; ++i)
        {
            associated_images_.emplace(associated_image_info.image_names[i]);
        }
    }
}

CuImage::CuImage() : std::enable_shared_from_this<CuImage>()
{
    file_handle_.path = const_cast<char*>("<null>");
}

CuImage::~CuImage()
{
    //    printf("[cuCIM] CuImage::~CuImage()\n");
    if (file_handle_.client_data)
    {
        image_formats_->formats[0].image_parser.close(&file_handle_);
    }
    image_formats_ = nullptr; // memory release is handled by the framework
    if (image_metadata_)
    {
        // Memory for json_data needs to be manually released if image_metadata_->json_data is not ""
        if (image_metadata_->json_data && *image_metadata_->json_data != '\0')
        {
            cucim_free(image_metadata_->json_data);
            image_metadata_->json_data = nullptr;
        }
        // Delete object (cucim::io::format::ImageMetadata) that embeds image_metadata_
        if (image_metadata_->handle)
        {
            // Keep original handle pointer before clearing it and delete the class object.
            void* handle_ptr = image_metadata_->handle;
            image_metadata_->handle = nullptr;
            delete static_cast<cucim::io::format::ImageMetadata*>(handle_ptr);
        }
        image_metadata_ = nullptr;
    }
    if (image_data_)
    {
        if (image_data_->container.data)
        {
            cucim_free(image_data_->container.data);
            image_data_->container.data = nullptr;
        }
        if (image_data_->container.shape)
        {
            cucim_free(image_data_->container.shape);
            image_data_->container.shape = nullptr;
        }
        if (image_data_->container.strides)
        {
            cucim_free(image_data_->container.strides);
            image_data_->container.strides = nullptr;
        }
        cucim_free(image_data_);
        image_data_ = nullptr;
    }
}

Framework* CuImage::get_framework()
{
    return framework_;
}

config::Config* CuImage::get_config()
{
    return config_.get();
}

cache::ImageCacheManager& CuImage::cache_manager()
{
    return *cache_manager_;
}

std::shared_ptr<cache::ImageCache> CuImage::cache()
{
    return cache_manager_->get_cache();
}

std::shared_ptr<cache::ImageCache> CuImage::cache(cache::ImageCacheConfig& config)
{
    return cache_manager_->cache(config);
}

filesystem::Path CuImage::path() const
{
    return file_handle_.path == nullptr ? "" : file_handle_.path;
}
bool CuImage::is_loaded() const
{
    return is_loaded_;
}
io::Device CuImage::device() const
{
    return io::Device("cpu");
}
Metadata CuImage::raw_metadata() const
{
    if (image_metadata_ && image_metadata_->raw_data)
    {
        return Metadata(image_metadata_->raw_data);
    }
    return Metadata{};
}
Metadata CuImage::metadata() const
{
    if (image_metadata_)
    {
        return Metadata(image_metadata_->json_data);
    }
    return Metadata{};
}
uint16_t CuImage::ndim() const
{
    return image_metadata_->ndim;
}
std::string CuImage::dims() const
{
    if (image_metadata_)
    {
        return image_metadata_->dims;
    }
    return std::string{};
}
Shape CuImage::shape() const
{
    std::vector<int64_t> result_shape;
    if (image_metadata_)
    {
        uint16_t ndim = image_metadata_->ndim;
        result_shape.reserve(ndim);
        for (int i = 0; i < ndim; ++i)
        {
            result_shape.push_back(image_metadata_->shape[i]);
        }
    }

    return result_shape;
}
std::vector<int64_t> CuImage::size(std::string dim_order) const
{
    std::vector<int64_t> result_size;
    if (image_metadata_)
    {
        if (dim_order.empty())
        {
            dim_order = std::string(image_metadata_->dims);
        }

        result_size.reserve(dim_order.size());
        for (const char& c : dim_order)
        {
            auto index = dim_indices_.index(c);
            if (index != -1)
            {
                result_size.push_back(image_metadata_->shape[index]);
            }
        }
    }
    return result_size;
}
DLDataType CuImage::dtype() const
{
    // TODO: support string conversion like Device class
    return DLDataType({ DLDataTypeCode::kDLUInt, 8, 1 });
}
std::vector<std::string> CuImage::channel_names() const
{
    std::vector<std::string> channel_names;
    if (image_metadata_)
    {
        auto channel_index = dim_indices_.index('C');
        if (channel_index != -1)
        {
            int channel_size = image_metadata_->shape[channel_index];
            channel_names.reserve(channel_size);
            for (int i = 0; i < channel_size; ++i)
            {
                channel_names.emplace_back(std::string(image_metadata_->channel_names[i]));
            }
        }
    }
    return channel_names;
}
std::vector<float> CuImage::spacing(std::string dim_order) const
{
    std::vector<float> result_spacing;
    result_spacing.reserve(dim_order.size());
    if (image_metadata_)
    {
        if (dim_order.empty())
        {
            dim_order = std::string(image_metadata_->dims);
            result_spacing.reserve(dim_order.size());
        }

        for (const char& c : dim_order)
        {
            auto index = dim_indices_.index(c);
            if (index != -1)
            {
                result_spacing.push_back(image_metadata_->spacing[index]);
            }
            else
            {
                result_spacing.push_back(1.0);
            }
        }
    }
    else
    {
        for (const char& c : dim_order)
        {
            (void)c;
            result_spacing.push_back(1.0);
        }
    }
    return result_spacing;
}

std::vector<std::string> CuImage::spacing_units(std::string dim_order) const
{
    std::vector<std::string> result_spacing_units;
    result_spacing_units.reserve(dim_order.size());
    if (image_metadata_)
    {
        if (dim_order.empty())
        {
            dim_order = std::string(image_metadata_->dims);
            result_spacing_units.reserve(dim_order.size());
        }

        for (const char& c : dim_order)
        {
            auto index = dim_indices_.index(c);
            if (index != -1)
            {
                result_spacing_units.emplace_back(std::string(image_metadata_->spacing_units[index]));
            }
            else
            {
                result_spacing_units.emplace_back(std::string(""));
            }
        }
    }
    else
    {
        for (const char& c : dim_order)
        {
            (void)c;
            result_spacing_units.emplace_back(std::string(""));
        }
    }

    return result_spacing_units;
}

std::array<float, 3> CuImage::origin() const
{
    std::array<float, 3> result_origin;
    if (image_metadata_->origin)
    {
        std::memcpy(result_origin.data(), image_metadata_->origin, sizeof(float) * 3);
    }
    return std::array<float, 3>{ 0., 0., 0. };
}

std::array<std::array<float, 3>, 3> CuImage::direction() const
{
    std::array<std::array<float, 3>, 3> result_direction;
    if (image_metadata_->direction)
    {
        std::memcpy(result_direction.data(), image_metadata_->direction, sizeof(float) * 9);
        return result_direction;
    }
    else
    {
        result_direction = { { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } } };
    }
    return result_direction;
}

std::string CuImage::coord_sys() const
{
    if (image_metadata_->coord_sys)
    {
        return std::string(image_metadata_->coord_sys);
    }
    return std::string("LPS");
}

ResolutionInfo CuImage::resolutions() const
{
    if (image_metadata_)
    {
        return ResolutionInfo(image_metadata_->resolution_info);
    }
    return ResolutionInfo(io::format::ResolutionInfoDesc{});
}

memory::DLTContainer CuImage::container() const
{
    if (image_data_)
    {
        return memory::DLTContainer(&image_data_->container);
    }
    else
    {
        return memory::DLTContainer(nullptr);
    }
}

CuImage CuImage::read_region(std::vector<int64_t>&& location,
                             std::vector<int64_t>&& size,
                             uint16_t level,
                             const DimIndices& region_dim_indices,
                             const io::Device& device,
                             DLTensor* buf,
                             const std::string& shm_name) const
{
    (void)region_dim_indices;
    (void)device;
    (void)buf;
    (void)shm_name;

    // If location is not specified, location would be (0, 0) if Z=0. Otherwise, location would be (0, 0, 0)
    if (location.empty())
    {
        location.emplace_back(0);
        location.emplace_back(0);
    }
    // If `size` is not specified, size would be (width, height) of the image at the specified `level`.
    if (size.empty())
    {
        const ResolutionInfo& res_info = resolutions();
        const auto level_count = res_info.level_count();
        if (level_count == 0)
        {
            throw std::runtime_error("[Error] No available resolutions in the image!");
        }
        const auto& level_dimension = res_info.level_dimension(level);
        size.insert(size.end(), level_dimension.begin(), level_dimension.end());
    }

    cucim::io::format::ImageReaderRegionRequestDesc request{};
    int64_t request_location[2] = { location[0], location[1] };
    request.location = request_location;
    request.level = level;
    int64_t request_size[2] = { size[0], size[1] };
    request.size = request_size;
    request.device = const_cast<char*>("cpu");

    //    cucim::io::format::ImageDataDesc image_data{};

    cucim::io::format::ImageDataDesc* image_data =
        static_cast<cucim::io::format::ImageDataDesc*>(cucim_malloc(sizeof(cucim::io::format::ImageDataDesc)));
    memset(image_data, 0, sizeof(cucim::io::format::ImageDataDesc));
    try
    {
        // Read region from internal file if image_data_ is nullptr
        if (image_data_ == nullptr)
        {
            if (!image_formats_->formats[0].image_reader.read(
                    &file_handle_, image_metadata_, &request, image_data, nullptr /*out_metadata*/))
            {
                cucim_free(image_data);
                throw std::runtime_error("[Error] Failed to read image!");
            }
        }
        else // Read region by cropping image
        {
            crop_image(image_metadata_, &request, image_data);
        }
    }
    catch (std::invalid_argument& e)
    {
        cucim_free(image_data);
        throw e;
    }

    //
    // Metadata Setup
    //

    // TODO: fill correct metadata information

    io::format::ImageMetadata& out_metadata = *(new io::format::ImageMetadata{});
    DLTensor& image_container = image_data->container;

    // Note: int-> uint16_t due to type differences between ImageMetadataDesc.ndim and DLTensor.ndim
    const uint16_t ndim = image_container.ndim;
    auto& resource = out_metadata.get_resource();

    std::string_view dims{ "YXC" };

    // Information from image_data
    std::pmr::vector<int64_t> shape(&resource);
    shape.reserve(ndim);
    shape.insert(shape.end(), &image_container.shape[0], &image_container.shape[ndim]);

    DLDataType& dtype = image_container.dtype;

    // TODO: Do not assume channel names as 'RGB' or 'RGBA'
    uint8_t n_ch = image_container.shape[2];
    std::pmr::vector<std::string_view> channel_names(&resource);
    channel_names.reserve(n_ch);
    if (n_ch == 3)
    {
        // std::pmr::vector<std::string_view> channel_names(
        //     { std::string_view{ "R" }, std::string_view{ "G" }, std::string_view{ "B" } }, &resource);
        channel_names.emplace_back(std::string_view{ "R" });
        channel_names.emplace_back(std::string_view{ "G" });
        channel_names.emplace_back(std::string_view{ "B" });
    }
    else
    {
        channel_names.emplace_back(std::string_view{ "R" });
        channel_names.emplace_back(std::string_view{ "G" });
        channel_names.emplace_back(std::string_view{ "B" });
        channel_names.emplace_back(std::string_view{ "A" });
    }


    std::pmr::vector<float> spacing(&resource);
    spacing.reserve(ndim);
    float* image_spacing = image_metadata_->spacing;
    spacing.insert(spacing.end(), &image_spacing[0], &image_spacing[ndim]);

    std::pmr::vector<std::string_view> spacing_units(&resource);
    spacing_units.reserve(ndim);
    for (int i = 0; i < ndim; i++)
    {
        int64_t dim_char = dim_indices_.index(dims[i]);

        const char* str_ptr = image_metadata_->spacing_units[dim_char];
        size_t str_len = strlen(image_metadata_->spacing_units[dim_char]);

        char* spacing_unit = static_cast<char*>(resource.allocate(str_len + 1));
        memcpy(spacing_unit, str_ptr, str_len);
        spacing_unit[str_len] = '\0';
        // std::pmr::string spacing_unit{ image_metadata_->spacing_units[dim_char], &resource };

        spacing_units.emplace_back(std::string_view{ spacing_unit });
    }

    std::pmr::vector<float> origin(&resource);
    origin.reserve(3);
    float* image_origin = image_metadata_->origin;
    origin.insert(origin.end(), &image_origin[0], &image_origin[3]);

    // Direction cosines (size is always 3x3)
    std::pmr::vector<float> direction(&resource);
    direction.reserve(3);
    float* image_direction = image_metadata_->direction;
    direction.insert(direction.end(), &image_direction[0], &image_direction[3 * 3]);

    // The coordinate frame in which the direction cosines are measured (either 'LPS'(ITK/DICOM) or 'RAS'(NIfTI/3D
    // Slicer))

    std::string_view coord_sys{ "" };
    const char* coord_sys_ptr = image_metadata_->coord_sys;
    if (coord_sys_ptr)
    {
        size_t coord_sys_len = strlen(coord_sys_ptr);
        char* coord_sys_str = static_cast<char*>(resource.allocate(coord_sys_len + 1));
        memcpy(coord_sys_str, coord_sys_ptr, coord_sys_len);
        coord_sys_str[coord_sys_len] = '\0';
        coord_sys = std::string_view{ coord_sys_str };
    }
    // std::pmr::string coord_sys_str{ image_metadata_->coord_sys ? image_metadata_->coord_sys : "", &resource };
    // std::string_view coord_sys{ coord_sys_str };

    // Manually set resolution dimensions to 2
    const uint16_t level_ndim = 2;
    std::pmr::vector<int64_t> level_dimensions(&resource);
    level_dimensions.reserve(level_ndim * 1); // it has only one size
    level_dimensions.insert(level_dimensions.end(), &size[0], &size[level_ndim]);

    std::pmr::vector<float> level_downsamples(&resource);
    level_downsamples.reserve(1);
    level_downsamples.emplace_back(1.0);

    std::pmr::vector<uint32_t> level_tile_sizes(&resource);
    level_tile_sizes.reserve(level_ndim * 1); // it has only one size
    level_tile_sizes.insert(level_tile_sizes.end(), &size[0], &size[level_ndim]); // same with level_dimension

    // Empty associated images
    const size_t associated_image_count = 0;
    std::pmr::vector<std::string_view> associated_image_names(&resource);

    // Partial image doesn't include raw metadata
    std::string_view raw_data{ "" };
    // Partial image doesn't include json metadata
    std::string_view json_data{ "" };

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
    out_metadata.level_count(1);
    out_metadata.level_ndim(2);
    out_metadata.level_dimensions(level_dimensions);
    out_metadata.level_downsamples(level_downsamples);
    out_metadata.level_tile_sizes(level_tile_sizes);
    out_metadata.image_count(associated_image_count);
    out_metadata.image_names(associated_image_names);
    out_metadata.raw_data(raw_data);
    out_metadata.json_data(json_data);

    return CuImage(this, &out_metadata.desc(), image_data);
}

std::set<std::string> CuImage::associated_images() const
{
    return associated_images_;
}

CuImage CuImage::associated_image(const std::string& name) const
{
    auto it = associated_images_.find(name);
    if (it != associated_images_.end())
    {
        io::format::ImageReaderRegionRequestDesc request{};
        request.associated_image_name = const_cast<char*>(name.c_str());
        request.device = const_cast<char*>("cpu");

        io::format::ImageDataDesc* out_image_data =
            static_cast<cucim::io::format::ImageDataDesc*>(cucim_malloc(sizeof(cucim::io::format::ImageDataDesc)));

        io::format::ImageMetadata& out_metadata = *(new io::format::ImageMetadata{});

        if (!image_formats_->formats[0].image_reader.read(
                &file_handle_, image_metadata_, &request, out_image_data, &out_metadata.desc()))
        {
            cucim_free(out_image_data);
            delete &out_metadata;
            throw std::runtime_error("[Error] Failed to read image!");
        }

        return CuImage(this, &out_metadata.desc(), out_image_data);
    }
    return CuImage{};
}

void CuImage::save(std::string file_path) const
{
    // Save ppm file for now.
    if (image_data_)
    {
        std::fstream fs(file_path, std::fstream::out | std::fstream::binary);

        if (fs.bad())
        {
            CUCIM_ERROR("Opening file failed!");
        }
        fs << "P6\n";
        auto image_size = size("XY");
        auto width = image_size[0];
        auto height = image_size[1];
        fs << width << "\n" << height << "\n" << 0xff << "\n";

        uint8_t* data = static_cast<uint8_t*>(image_data_->container.data);
        size_t data_size = width * height * 3;
        for (unsigned int i = 0; (i < data_size) && fs.good(); ++i)
        {
            fs << data[i];
        }
        fs.flush();
        if (fs.bad())
        {
            CUCIM_ERROR("Writing data failed!");
        }
        fs.close();
    }
}
void CuImage::ensure_init()
{
    ScopedLock g(mutex_);

    if (!framework_)
    {
        CUCIM_ERROR("Framework is not initialized!");
    }
    if (!image_formats_)
    {
        auto plugin_root = framework_->get_plugin_root();
        // TODO: Here 'LINUX' path separator is used. Need to make it generalize once filesystem library is
        // available.
        std::string plugin_file_path = (plugin_root && *plugin_root != 0) ?
                                           fmt::format("{}/cucim.kit.cuslide@{}.{}.{}.so", plugin_root,
                                                       CUCIM_VERSION_MAJOR, CUCIM_VERSION_MINOR, CUCIM_VERSION_PATCH) :
                                           fmt::format("cucim.kit.cuslide@{}.{}.{}.so", CUCIM_VERSION_MAJOR,
                                                       CUCIM_VERSION_MINOR, CUCIM_VERSION_PATCH);
        if (!cucim::util::file_exists(plugin_file_path.c_str()))
        {
            plugin_file_path = fmt::format("cucim.kit.cuslide@" XSTR(CUCIM_VERSION) ".so");
        }
        image_formats_ =
            framework_->acquire_interface_from_library<cucim::io::format::IImageFormat>(plugin_file_path.c_str());
        if (image_formats_ == nullptr)
        {
            throw std::runtime_error(
                fmt::format("Dependent library 'cucim.kit.cuslide@" XSTR(CUCIM_VERSION) ".so' cannot be loaded!"));
        }
    }
}

bool CuImage::crop_image(io::format::ImageMetadataDesc* metadata,
                         io::format::ImageReaderRegionRequestDesc* request,
                         io::format::ImageDataDesc* out_image_data) const
{
    // TODO: assume length of location/size to 2.
    constexpr int32_t ndims = 2;

    if (request->level >= metadata->resolution_info.level_count)
    {
        throw std::invalid_argument(fmt::format("Invalid level ({}) in the request! (Should be < {})", request->level,
                                                metadata->resolution_info.level_count));
    }

    auto original_img_width = image_metadata_->shape[dim_indices_.index('X')];
    auto original_img_height = image_metadata_->shape[dim_indices_.index('Y')];
    // TODO: consider other cases where samples_per_pixel is not same with # of channels
    //       (we cannot use `ifd->samples_per_pixel()` here)
    uint32_t samples_per_pixel = static_cast<uint32_t>(image_metadata_->shape[dim_indices_.index('C')]);

    for (int32_t i = 0; i < ndims; ++i)
    {
        if (request->location[i] < 0)
        {
            throw std::invalid_argument(
                fmt::format("Invalid location ({}) in the request! (Should be >= 0)", request->location[i]));
        }
        if (request->size[i] <= 0)
        {
            throw std::invalid_argument(
                fmt::format("Invalid size ({}) in the request! (Should be > 0)", request->size[i]));
        }
    }
    if (request->location[0] + request->size[0] > original_img_width)
    {
        throw std::invalid_argument(
            fmt::format("Invalid location/size (it exceeds the image width {})", original_img_width));
    }
    if (request->location[1] + request->size[1] > original_img_height)
    {
        throw std::invalid_argument(
            fmt::format("Invalid location/size (it exceeds the image height {})", original_img_height));
    }

    int64_t sx = request->location[0];
    int64_t sy = request->location[1];
    int64_t w = request->size[0];
    int64_t h = request->size[1];

    uint64_t ex = sx + w - 1;
    uint64_t ey = sy + h - 1;

    uint8_t* src_ptr = static_cast<uint8_t*>(image_data_->container.data);

    void* raster = cucim_malloc(w * h * samples_per_pixel); // RGB image
    auto dest_ptr = static_cast<uint8_t*>(raster);
    int64_t dest_stride_x_bytes = w * samples_per_pixel;

    int64_t src_stride_x = original_img_width;
    int64_t src_stride_x_bytes = original_img_width * samples_per_pixel;

    int64_t start_offset = (sx + (sy * src_stride_x)) * samples_per_pixel;
    int64_t end_offset = (ex + (ey * src_stride_x)) * samples_per_pixel;

    for (int64_t src_offset = start_offset; src_offset <= end_offset; src_offset += src_stride_x_bytes)
    {
        memcpy(dest_ptr, src_ptr + src_offset, dest_stride_x_bytes);
        dest_ptr += dest_stride_x_bytes;
    }

    out_image_data->container.data = raster;
    out_image_data->container.ctx = DLContext{ static_cast<DLDeviceType>(cucim::io::DeviceType::kCPU), 0 };
    out_image_data->container.ndim = metadata->ndim;
    out_image_data->container.dtype = metadata->dtype;
    out_image_data->container.strides = nullptr; // Tensor is compact and row-majored
    out_image_data->container.byte_offset = 0;
    // Set correct shape
    out_image_data->container.shape = static_cast<int64_t*>(cucim_malloc(sizeof(int64_t) * metadata->ndim));
    memcpy(out_image_data->container.shape, metadata->shape, sizeof(int64_t) * metadata->ndim);
    out_image_data->container.shape[0] = h;
    out_image_data->container.shape[1] = w;

    return true;
}

} // namespace cucim