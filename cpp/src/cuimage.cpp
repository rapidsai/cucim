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

#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>

#if CUCIM_SUPPORT_CUDA
#    include <cuda_runtime.h>
#endif
#include <fmt/format.h>

#include "cucim/profiler/nvtx3.h"
#include "cucim/util/cuda.h"
#include "cucim/util/file.h"


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
std::shared_ptr<profiler::Profiler> CuImage::profiler_ = std::make_shared<profiler::Profiler>(config_->profiler());
std::unique_ptr<cache::ImageCacheManager> CuImage::cache_manager_ = std::make_unique<cache::ImageCacheManager>();
std::unique_ptr<plugin::ImageFormat> CuImage::image_format_plugins_ = std::make_unique<plugin::ImageFormat>();


CuImage::CuImage(const filesystem::Path& path)
{
    PROF_SCOPED_RANGE(PROF_EVENT_P(cuimage_cuimage, 1));

    ensure_init();
    image_format_ = image_format_plugins_->detect_image_format(path);

    // TODO: need to detect available format for the file path
    {
        PROF_SCOPED_RANGE(PROF_EVENT(cuimage_cuimage_open));
        std::shared_ptr<CuCIMFileHandle>* file_handle_shared =
            reinterpret_cast<std::shared_ptr<CuCIMFileHandle>*>(image_format_->image_parser.open(path.c_str()));
        file_handle_ = *file_handle_shared;
        delete file_handle_shared;

        // Set deleter to close the file handle
        file_handle_->set_deleter(image_format_->image_parser.close);
    }

    io::format::ImageMetadata& image_metadata = *(new io::format::ImageMetadata{});
    image_metadata_ = &image_metadata.desc();
    is_loaded_ = image_format_->image_parser.parse(file_handle_.get(), image_metadata_);
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
    PROF_SCOPED_RANGE(PROF_EVENT_P(cuimage_cuimage, 2));
    // TODO: implement this
    (void)path;
    (void)plugin_name;
}

CuImage::CuImage(CuImage&& cuimg) : std::enable_shared_from_this<CuImage>()
{
    PROF_SCOPED_RANGE(PROF_EVENT_P(cuimage_cuimage, 3));

    std::swap(file_handle_, cuimg.file_handle_);
    std::swap(image_format_, cuimg.image_format_);
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
    PROF_SCOPED_RANGE(PROF_EVENT_P(cuimage_cuimage, 4));

    file_handle_ = cuimg->file_handle_;
    image_format_ = cuimg->image_format_;
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
    PROF_SCOPED_RANGE(PROF_EVENT_P(cuimage_cuimage, 5));
    file_handle_ = std::make_shared<CuCIMFileHandle>();
    file_handle_->path = const_cast<char*>("");
}

CuImage::~CuImage()
{
    PROF_SCOPED_RANGE(PROF_EVENT(cuimage__cuimage));

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
            DLContext& ctx = image_data_->container.ctx;
            auto device_type = static_cast<io::DeviceType>(ctx.device_type);
            switch (device_type)
            {
            case io::DeviceType::kCPU:
                if (image_data_->loader)
                {
                    delete[] reinterpret_cast<uint8_t*>(image_data_->container.data);
                }
                else
                {
                    cucim_free(image_data_->container.data);
                }
                image_data_->container.data = nullptr;
                break;
            case io::DeviceType::kCUDA:
                cudaError_t cuda_status;
                CUDA_TRY(cudaFree(image_data_->container.data));
                image_data_->container.data = nullptr;
                if (cuda_status)
                {
                    fmt::print(stderr, "[Error] Cannot free memory!");
                }
                break;
            case io::DeviceType::kPinned:
            case io::DeviceType::kCPUShared:
            case io::DeviceType::kCUDAShared:
                fmt::print(stderr, "Device type {} is not supported!", device_type);
                break;
            }
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
        if (image_data_->shm_name)
        {
            cucim_free(image_data_->shm_name);
            image_data_->shm_name = nullptr;
        }
        if (image_data_->loader)
        {
            auto loader = reinterpret_cast<cucim::loader::ThreadBatchDataLoader*>(image_data_->loader);
            delete loader;

            image_data_->loader = nullptr;
        }

        cucim_free(image_data_);
        image_data_ = nullptr;
    }

    close(); // close file handle (NOTE:: close the file handle after loader is deleted)
    image_format_ = nullptr; // memory release is handled by the framework
}

Framework* CuImage::get_framework()
{
    return framework_;
}

config::Config* CuImage::get_config()
{
    return config_.get();
}

std::shared_ptr<profiler::Profiler> CuImage::profiler()
{
    return profiler_;
}

std::shared_ptr<profiler::Profiler> CuImage::profiler(profiler::ProfilerConfig& config)
{
    profiler_->trace(config.trace);
    return profiler_;
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

bool CuImage::is_trace_enabled()
{
    return profiler_->trace();
}

filesystem::Path CuImage::path() const
{
    return file_handle_->path == nullptr ? "" : file_handle_->path;
}
bool CuImage::is_loaded() const
{
    return is_loaded_;
}
io::Device CuImage::device() const
{
    if (image_data_)
    {
        DLContext& ctx = image_data_->container.ctx;
        auto device_type = static_cast<io::DeviceType>(ctx.device_type);
        auto device_id = static_cast<io::DeviceIndex>(ctx.device_id);
        std::string shm_name = image_data_->shm_name == nullptr ? "" : image_data_->shm_name;
        return io::Device(device_type, device_id, shm_name);
    }
    else
    {
        return io::Device("cpu");
    }
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
        return memory::DLTContainer(&image_data_->container, image_data_->shm_name);
    }
    else
    {
        return memory::DLTContainer(nullptr);
    }
}

loader::ThreadBatchDataLoader* CuImage::loader() const
{
    if (image_data_)
    {
        return reinterpret_cast<loader::ThreadBatchDataLoader*>(image_data_->loader);
    }
    else
    {
        return nullptr;
    }
}

CuImage CuImage::read_region(std::vector<int64_t>&& location,
                             std::vector<int64_t>&& size,
                             uint16_t level,
                             uint32_t num_workers,
                             uint32_t batch_size,
                             bool drop_last,
                             uint32_t prefetch_factor,
                             bool shuffle,
                             uint64_t seed,
                             const DimIndices& region_dim_indices,
                             const io::Device& device,
                             DLTensor* buf,
                             const std::string& shm_name) const
{
    PROF_SCOPED_RANGE(PROF_EVENT(cuimage_read_region));
    (void)region_dim_indices;
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

    // The number of locations should be the multiplication of the number of dimensions in the size.
    if (location.size() % size.size() != 0)
    {
        throw std::runtime_error(
            "[Error] The number of locations should be the multiplication of the number of dimensions in the size!");
    }

    // Make sure the batch size is not zero.
    if (batch_size == 0)
    {
        batch_size = 1;
    }

    uint32_t size_ndim = size.size();
    uint64_t location_len = location.size() / size_ndim;
    std::string device_name = std::string(device);
    cucim::io::format::ImageReaderRegionRequestDesc request{};

    if (location_len > 1 || batch_size > 1 || num_workers > 0)
    {
        // ::Note:: Here, to pass vector data to C interface, we move data in the original vector to the vector in heap
        // memory and create a unique pointer with 'new'. The data is transferred to ThreadBatchDataLoader class members
        // (locations_ and size_) for automatic deletion on exit.
        auto location_ptr = new std::vector<int64_t>();
        location_ptr->swap(location);
        auto location_unique = reinterpret_cast<void*>(new std::unique_ptr<std::vector<int64_t>>(location_ptr));

        auto size_ptr = new std::vector<int64_t>();
        size_ptr->swap(size);
        auto size_unique = reinterpret_cast<void*>(new std::unique_ptr<std::vector<int64_t>>(size_ptr));

        request.location = location_ptr->data();
        request.location_unique = location_unique;
        request.size = size_ptr->data();
        request.size_unique = size_unique;
    }
    else
    {
        request.location = location.data();
        request.size = size.data();
    }
    request.location_len = location_len;
    request.size_ndim = size_ndim;
    request.level = level;
    request.num_workers = num_workers;
    request.batch_size = batch_size;
    request.drop_last = drop_last;
    request.prefetch_factor = prefetch_factor;
    request.shuffle = shuffle;
    request.seed = seed;
    request.device = device_name.data();

    auto image_data = std::unique_ptr<io::format::ImageDataDesc, decltype(cucim_free)*>(
        reinterpret_cast<io::format::ImageDataDesc*>(cucim_malloc(sizeof(io::format::ImageDataDesc))), cucim_free);
    memset(image_data.get(), 0, sizeof(io::format::ImageDataDesc));

    try
    {
        // Read region from internal file if image_data_ is nullptr
        if (image_data_ == nullptr)
        {
            if (!file_handle_) // file_handle_ is not opened
            {
                throw std::runtime_error("[Error] The image file is closed!");
            }
            if (!image_format_->image_reader.read(
                    file_handle_.get(), image_metadata_, &request, image_data.get(), nullptr /*out_metadata*/))
            {
                throw std::runtime_error("[Error] Failed to read image!");
            }
        }
        else // Read region by cropping image
        {
            const char* dims_str = image_metadata_->dims;
            if (strncmp("YXC", dims_str, 4) != 0)
            {
                throw std::runtime_error(fmt::format("[Error] The image is not in YXC format! ({})", dims_str));
            }
            if (image_data_->container.data == nullptr)
            {
                throw std::runtime_error(
                    "[Error] The image data is nullptr! It is possible that the object is iterator and the image data "
                    "is not loaded yet! Please advance the iterator first!");
            }
            crop_image(request, *image_data);
        }
    }
    catch (std::invalid_argument& e)
    {
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
    if (batch_size > 1)
    {
        dims = { "NYXC" };
    }

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

    int index = 0;
    if (ndim == 4)
    {
        index = 1;
        // The first dimension is for 'batch' ('N')
        spacing_units.emplace_back(std::string_view{ "batch" });
    }
    for (; index < ndim; ++index)
    {
        int64_t dim_char = dim_indices_.index(dims[index]);

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
    level_dimensions.insert(level_dimensions.end(), request.location, &request.location[request.location_len]);

    std::pmr::vector<float> level_downsamples(&resource);
    level_downsamples.reserve(1);
    level_downsamples.emplace_back(1.0);

    std::pmr::vector<uint32_t> level_tile_sizes(&resource);
    level_tile_sizes.reserve(level_ndim * 1); // it has only one size
    level_tile_sizes.insert(
        level_tile_sizes.end(), request.location, &request.location[request.location_len]); // same with level_dimension

    // Empty associated images
    const size_t associated_image_count = 0;
    std::pmr::vector<std::string_view> associated_image_names(&resource);

    // Partial image doesn't include raw metadata
    std::string_view raw_data{ "" };
    // Partial image doesn't include json metadata
    std::string_view json_data{ "" };

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

    return CuImage(this, &out_metadata.desc(), image_data.release());
}

std::set<std::string> CuImage::associated_images() const
{
    return associated_images_;
}

CuImage CuImage::associated_image(const std::string& name, const io::Device& device) const
{
    PROF_SCOPED_RANGE(PROF_EVENT(cuimage_associated_image));
    if (file_handle_->fd < 0) // file_handle_ is not opened
    {
        throw std::runtime_error("[Error] The image file is closed!");
    }
    auto it = associated_images_.find(name);
    if (it != associated_images_.end())
    {
        io::format::ImageReaderRegionRequestDesc request{};
        request.associated_image_name = const_cast<char*>(name.c_str());
        std::string device_name = std::string(device);
        request.device = device_name.data();

        auto out_image_data = std::unique_ptr<io::format::ImageDataDesc, decltype(cucim_free)*>(
            reinterpret_cast<io::format::ImageDataDesc*>(cucim_malloc(sizeof(io::format::ImageDataDesc))), cucim_free);
        memset(out_image_data.get(), 0, sizeof(io::format::ImageDataDesc));

        io::format::ImageMetadata& out_metadata = *(new io::format::ImageMetadata{});

        if (!image_format_->image_reader.read(
                file_handle_.get(), image_metadata_, &request, out_image_data.get(), &out_metadata.desc()))
        {
            throw std::runtime_error("[Error] Failed to read image!");
        }

        return CuImage(this, &out_metadata.desc(), out_image_data.release());
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
        uint8_t* raster = nullptr;
        size_t raster_size = width * height * 3;

        const cucim::io::Device& in_device = device();
        if (in_device.type() == cucim::io::DeviceType::kCUDA)
        {
            cudaError_t cuda_status;
            raster = static_cast<uint8_t*>(cucim_malloc(raster_size));
            CUDA_TRY(cudaMemcpy(raster, data, raster_size, cudaMemcpyDeviceToHost));
            if (cuda_status)
            {
                cucim_free(raster);
                throw std::runtime_error("Error during cudaMemcpy!");
            }
            data = raster;
        }

        for (unsigned int i = 0; (i < raster_size) && fs.good(); ++i)
        {
            fs << data[i];
        }
        fs.flush();
        if (fs.bad())
        {
            if (in_device.type() == cucim::io::DeviceType::kCUDA)
            {
                cucim_free(raster);
            }
            CUCIM_ERROR("Writing data failed!");
        }
        fs.close();
    }
}

void CuImage::close()
{
    file_handle_ = nullptr;
}

void CuImage::ensure_init()
{
    PROF_SCOPED_RANGE(PROF_EVENT(cuimage_ensure_init));
    ScopedLock g(mutex_);

    if (!framework_)
    {
        CUCIM_ERROR("Framework is not initialized!");
    }
    if (!(*image_format_plugins_))
    {
        image_format_plugins_ = std::make_unique<cucim::plugin::ImageFormat>();

        const std::vector<std::string>& plugin_names = get_config()->plugin().plugin_names;

        const char* plugin_root = framework_->get_plugin_root();
        for (auto& plugin_name : plugin_names)
        {
            PROF_SCOPED_RANGE(PROF_EVENT(cuimage_ensure_init_plugin_iter));
            // TODO: Here 'LINUX' path separator is used. Need to make it generalize once filesystem library is
            // available.
            std::string plugin_file_path = (plugin_root && *plugin_root != 0) ?
                                               fmt::format("{}/{}", plugin_root, plugin_name) :
                                               fmt::format("{}", plugin_name);
            if (!cucim::util::file_exists(plugin_file_path.c_str()))
            {
                plugin_file_path = fmt::format("{}", plugin_name);
            }

            const auto& image_formats =
                framework_->acquire_interface_from_library<cucim::io::format::IImageFormat>(plugin_file_path.c_str());

            image_format_plugins_->add_interfaces(image_formats);

            if (image_formats == nullptr)
            {
                throw std::runtime_error(fmt::format("Dependent library '{}' cannot be loaded!", plugin_file_path));
            }
        }
    }
}

bool CuImage::crop_image(const io::format::ImageReaderRegionRequestDesc& request,
                         io::format::ImageDataDesc& out_image_data) const
{
    PROF_SCOPED_RANGE(PROF_EVENT(cuimage_crop_image));
    const int32_t ndim = request.size_ndim;

    if (request.level >= image_metadata_->resolution_info.level_count)
    {
        throw std::invalid_argument(fmt::format("Invalid level ({}) in the request! (Should be < {})", request.level,
                                                image_metadata_->resolution_info.level_count));
    }

    const cucim::io::Device& in_device = device();

    auto original_img_width = image_metadata_->shape[dim_indices_.index('X')];
    auto original_img_height = image_metadata_->shape[dim_indices_.index('Y')];
    // TODO: consider other cases where samples_per_pixel is not same with # of channels
    //       (we cannot use `ifd->samples_per_pixel()` here)
    uint32_t samples_per_pixel = static_cast<uint32_t>(image_metadata_->shape[dim_indices_.index('C')]);

    for (int32_t i = 0; i < ndim; ++i)
    {
        if (request.location[i] < 0)
        {
            throw std::invalid_argument(
                fmt::format("Invalid location ({}) in the request! (Should be >= 0)", request.location[i]));
        }
        if (request.size[i] <= 0)
        {
            throw std::invalid_argument(fmt::format("Invalid size ({}) in the request! (Should be > 0)", request.size[i]));
        }
    }
    if (request.location[0] + request.size[0] > original_img_width)
    {
        throw std::invalid_argument(
            fmt::format("Invalid location/size (it exceeds the image width {})", original_img_width));
    }
    if (request.location[1] + request.size[1] > original_img_height)
    {
        throw std::invalid_argument(
            fmt::format("Invalid location/size (it exceeds the image height {})", original_img_height));
    }

    std::string device_name(request.device);

    if (request.shm_name)
    {
        device_name = device_name + fmt::format("[{}]", request.shm_name); // TODO: check performance
    }
    cucim::io::Device out_device(device_name);

    int64_t sx = request.location[0];
    int64_t sy = request.location[1];
    int64_t w = request.size[0];
    int64_t h = request.size[1];

    uint64_t ex = sx + w - 1;
    uint64_t ey = sy + h - 1;

    uint8_t* src_ptr = static_cast<uint8_t*>(image_data_->container.data);
    size_t raster_size = w * h * samples_per_pixel;

    void* raster = nullptr;
    int64_t dest_stride_x_bytes = w * samples_per_pixel;

    int64_t src_stride_x = original_img_width;
    int64_t src_stride_x_bytes = original_img_width * samples_per_pixel;

    int64_t start_offset = (sx + (sy * src_stride_x)) * samples_per_pixel;
    int64_t end_offset = (ex + (ey * src_stride_x)) * samples_per_pixel;

    switch (in_device.type())
    {
    case cucim::io::DeviceType::kCPU: {
        raster = cucim_malloc(raster_size);
        auto dest_ptr = static_cast<uint8_t*>(raster);
        for (int64_t src_offset = start_offset; src_offset <= end_offset; src_offset += src_stride_x_bytes)
        {
            memcpy(dest_ptr, src_ptr + src_offset, dest_stride_x_bytes);
            dest_ptr += dest_stride_x_bytes;
        }
        // Copy the raster memory and free it if needed.
        cucim::memory::move_raster_from_host((void**)&raster, raster_size, out_device);
        break;
    }
    case cucim::io::DeviceType::kCUDA: {
        cudaError_t cuda_status;

        if (out_device.type() == cucim::io::DeviceType::kCPU)
        {
            // cuda -> host at bulk then host -> host per row is faster than cuda-> cuda per row, then cuda->host at
            // bulk.
            uint8_t* copied_src_ptr = static_cast<uint8_t*>(cucim_malloc(src_stride_x_bytes * h));
            CUDA_TRY(cudaMemcpy(copied_src_ptr, src_ptr + start_offset, src_stride_x_bytes * h, cudaMemcpyDeviceToHost));
            if (cuda_status)
            {
                cucim_free(copied_src_ptr);
                throw std::runtime_error("Error during cudaMemcpy!");
            }

            end_offset -= start_offset;
            start_offset = 0;

            raster = cucim_malloc(raster_size);
            auto dest_ptr = static_cast<uint8_t*>(raster);
            for (int64_t src_offset = start_offset; src_offset <= end_offset; src_offset += src_stride_x_bytes)
            {
                memcpy(dest_ptr, copied_src_ptr + src_offset, dest_stride_x_bytes);
                dest_ptr += dest_stride_x_bytes;
            }
            cucim_free(copied_src_ptr);
        }
        else
        {
            CUDA_TRY(cudaMalloc(&raster, raster_size));
            if (cuda_status)
            {
                throw std::bad_alloc();
            }
            auto dest_ptr = static_cast<uint8_t*>(raster);
            CUDA_TRY(cudaMemcpy2D(dest_ptr, dest_stride_x_bytes, src_ptr + start_offset, src_stride_x_bytes,
                                  dest_stride_x_bytes, h, cudaMemcpyDeviceToDevice));
            if (cuda_status)
            {
                throw std::runtime_error("Error during cudaMemcpy2D!");
            }
            // Copy the raster memory and free it if needed.
            cucim::memory::move_raster_from_device((void**)&raster, raster_size, out_device);
        }
        break;
    }
    case cucim::io::DeviceType::kPinned:
    case cucim::io::DeviceType::kCPUShared:
    case cucim::io::DeviceType::kCUDAShared:
        throw std::runtime_error(fmt::format("Device type {} not supported!", in_device.type()));
        break;
    }

    auto& out_image_container = out_image_data.container;
    out_image_container.data = raster;
    out_image_container.ctx = DLContext{ static_cast<DLDeviceType>(out_device.type()), out_device.index() };
    out_image_container.ndim = image_metadata_->ndim;
    out_image_container.dtype = image_metadata_->dtype;
    out_image_container.strides = nullptr; // Tensor is compact and row-majored
    out_image_container.byte_offset = 0;
    // Set correct shape
    out_image_container.shape = static_cast<int64_t*>(cucim_malloc(sizeof(int64_t) * image_metadata_->ndim));
    memcpy(out_image_container.shape, image_metadata_->shape, sizeof(int64_t) * image_metadata_->ndim);
    out_image_container.shape[0] = h;
    out_image_container.shape[1] = w;

    auto& shm_name = out_device.shm_name();
    size_t shm_name_len = shm_name.size();
    if (shm_name_len != 0)
    {
        out_image_data.shm_name = static_cast<char*>(cucim_malloc(shm_name_len + 1));
        memcpy(out_image_data.shm_name, shm_name.c_str(), shm_name_len + 1);
    }
    else
    {
        out_image_data.shm_name = nullptr;
    }

    return true;
}

/////////////////////////////
// Iterator implementation //
/////////////////////////////

CuImage::iterator CuImage::begin()
{
    return iterator(shared_from_this());
}
CuImage::iterator CuImage::end()
{
    return iterator(shared_from_this(), true);
}

CuImage::const_iterator CuImage::begin() const
{
    return const_iterator(shared_from_this());
}

CuImage::const_iterator CuImage::end() const
{
    return const_iterator(shared_from_this(), true);
}

template <typename DataType>
CuImageIterator<DataType>::CuImageIterator(std::shared_ptr<DataType> cuimg, bool ending)
    : cuimg_(cuimg), loader_(nullptr), batch_index_(0), total_batch_count_(0)
{
    if (!cuimg_)
    {
        throw std::runtime_error("CuImageIterator: cuimg is nullptr!");
    }

    auto& image_data = cuimg_->image_data_;
    cucim::loader::ThreadBatchDataLoader* loader = nullptr;
    if (image_data)
    {
        loader = reinterpret_cast<cucim::loader::ThreadBatchDataLoader*>(image_data->loader);
        loader_ = loader;
    }

    if (ending) // point to the end
    {
        if (image_data)
        {
            if (loader)
            {
                total_batch_count_ = loader->total_batch_count();
                batch_index_ = total_batch_count_;
            }
            else
            {
                total_batch_count_ = 1;
                batch_index_ = 1;
            }
        }
        else
        {
            batch_index_ = 0;
        }
    }
    else
    {
        if (image_data)
        {
            if (loader)
            {
                total_batch_count_ = loader->total_batch_count();
                if (loader->size() > 1)
                {
                    batch_index_ = loader->processed_batch_count();
                }
                else
                {
                    batch_index_ = 0;
                }
            }
            else
            {
                total_batch_count_ = 1;
                batch_index_ = 0;
            }
        }
        else
        {
            throw std::out_of_range("Batch index out of range! ('image_data_' is null)");
        }
    }
}

template <typename DataType>
typename CuImageIterator<DataType>::reference CuImageIterator<DataType>::operator*() const
{
    return cuimg_;
}

template <typename DataType>
typename CuImageIterator<DataType>::pointer CuImageIterator<DataType>::operator->()
{
    return cuimg_.get();
}

template <typename DataType>
CuImageIterator<DataType>& CuImageIterator<DataType>::operator++()
{
    // Prefix increment
    increase_index_();
    return *this;
}

template <typename DataType>
CuImageIterator<DataType> CuImageIterator<DataType>::operator++(int)
{
    // Postfix increment
    auto temp(*this);
    increase_index_();
    return temp;
}

template <typename DataType>
bool CuImageIterator<DataType>::operator==(const CuImageIterator<DataType>& other)
{
    return cuimg_.get() == other.cuimg_.get() && batch_index_ == other.batch_index_;
};

template <typename DataType>
bool CuImageIterator<DataType>::operator!=(const CuImageIterator<DataType>& other)
{
    return cuimg_.get() != other.cuimg_.get() || batch_index_ != other.batch_index_;
};

template <typename DataType>
int64_t CuImageIterator<DataType>::index()
{
    auto loader = reinterpret_cast<cucim::loader::ThreadBatchDataLoader*>(loader_);
    if (loader && (loader->size() > 1))
    {
        batch_index_ = loader->processed_batch_count();
    }
    return batch_index_;
}

template <typename DataType>
uint64_t CuImageIterator<DataType>::size() const
{
    return total_batch_count_;
}

template <typename DataType>
void CuImageIterator<DataType>::increase_index_()
{
    auto loader = reinterpret_cast<cucim::loader::ThreadBatchDataLoader*>(loader_);
    if (loader)
    {
        auto next_data = loader->next_data();
        if (next_data)
        {
            auto image_data = reinterpret_cast<uint8_t**>(&(cuimg_->image_data_->container.data));
            if (*image_data)
            {
                delete[] * image_data;
            }
            *image_data = next_data;

            if (loader->batch_size() > 1)
            {
                // Set value for dimension 'N'
                cuimg_->image_data_->container.shape[0] = loader->data_batch_size();
                cuimg_->image_metadata_->shape[0] = loader->data_batch_size();
            }
        }
        if (loader->size() > 1)
        {
            batch_index_ = loader->processed_batch_count();
        }
        else
        {
            if (batch_index_ < static_cast<int64_t>(total_batch_count_))
            {
                ++batch_index_;
            }
        }
    }
    else
    {
        if (batch_index_ < static_cast<int64_t>(total_batch_count_))
        {
            ++batch_index_;
        }
    }
}

} // namespace cucim