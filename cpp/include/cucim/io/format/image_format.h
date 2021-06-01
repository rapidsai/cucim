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

#ifndef CUCIM_IMAGE_FORMAT_H
#define CUCIM_IMAGE_FORMAT_H

#include "cucim/core/interface.h"
#include "cucim/filesystem/file_handle.h"
#include "dlpack/dlpack.h"

#include <memory_resource>
#include <string>


namespace cucim::io::format
{

struct DimIndicesDesc
{
    int64_t indices[26]; /// Indices for each alphabet ('A'= 0, 'Z'= 25)
};

struct ResolutionInfoDesc
{
    uint16_t level_count;
    uint16_t level_ndim;
    int64_t* level_dimensions;
    float* level_downsamples;
    uint32_t* level_tile_sizes;
};

struct AssociatedImageInfoDesc
{
    uint16_t image_count;
    char** image_names;
};

struct ImageMetadataDesc
{
    void* handle; /// Handle for ImageMetadata object
    uint16_t ndim; /// Number of dimensions
    const char* dims; /// Dimension characters (E.g., "STCZYX")
    int64_t* shape; /// Size of each dimension
    DLDataType dtype; /// Data type of the array
    char** channel_names; /// Channel name list   TODO: 'S', 'T', and other dimension can have names so need to be
                                /// generalized.
    float* spacing; /// Physical size
    char** spacing_units; /// Units for each spacing element (size is same with `ndim`)
    float* origin; /// Physical location of (0, 0, 0) (size is always 3)
    float* direction; /// Direction cosines (size is always 3x3)
    const char* coord_sys; /// The coordinate frame in which the direction cosines are measured (either
                           /// 'LPS'(ITK/DICOM) or 'RAS'(NIfTI/3D Slicer))
    ResolutionInfoDesc resolution_info; /// Resolution information
    AssociatedImageInfoDesc associated_image_info; /// Associated image information
    const char* raw_data; /// Metadata in text format from the original image
    char* json_data; /// cucim & vendor's metadata in JSON format. Will be merged with above standard metadata. Memory
                     /// for this needs to be released manually.
};

// Without raw_data and json_data, metadata size is approximately 1104 bytes.
// It might be good to allocate 4k for that.
constexpr size_t IMAGE_METADATA_BUFFER_SIZE = 4096;
class EXPORT_VISIBLE ImageMetadata
{
public:
    ImageMetadata();
    ~ImageMetadata();
    void* allocate(size_t size);
    std::pmr::monotonic_buffer_resource& get_resource();
    constexpr uint8_t* get_buffer()
    {
        return buffer_.data();
    }

    ImageMetadataDesc& desc();

    ImageMetadata& ndim(uint16_t ndim);
    ImageMetadata& dims(const std::string_view& dims);
    ImageMetadata& shape(const std::pmr::vector<int64_t>& shape);
    ImageMetadata& dtype(const DLDataType& dtype);
    ImageMetadata& channel_names(const std::pmr::vector<std::string_view>& channel_names);

    ImageMetadata& spacing(const std::pmr::vector<float>& spacing);
    ImageMetadata& spacing_units(const std::pmr::vector<std::string_view>& spacing_units);

    ImageMetadata& origin(const std::pmr::vector<float>& origin);
    ImageMetadata& direction(const std::pmr::vector<float>& direction);
    ImageMetadata& coord_sys(const std::string_view& coord_sys);

    // ResolutionInfoDesc
    ImageMetadata& level_count(uint16_t level_count);
    ImageMetadata& level_ndim(uint16_t level_ndim);
    ImageMetadata& level_dimensions(const std::pmr::vector<int64_t>& level_dimensions);
    ImageMetadata& level_downsamples(const std::pmr::vector<float>& level_downsamples);
    ImageMetadata& level_tile_sizes(const std::pmr::vector<uint32_t>& level_tile_sizes);

    // AssociatedImageInfoDesc
    ImageMetadata& image_count(uint16_t image_count);
    ImageMetadata& image_names(const std::pmr::vector<std::string_view>& image_names);

    ImageMetadata& raw_data(const std::string_view& raw_data);
    ImageMetadata& json_data(const std::string_view& json_data);

private:
    ImageMetadataDesc desc_{};
    std::array<uint8_t, IMAGE_METADATA_BUFFER_SIZE> buffer_{};
    std::pmr::monotonic_buffer_resource res_{ buffer_.data(), sizeof(buffer_) };

// manylinux2014 requires gcc4-compatible libstdcxx-abi(gcc is configured with
// '--with-default-libstdcxx-abi=gcc4-compatible', https://gcc.gnu.org/onlinedocs/libstdc++/manual/configure.html) which
// forces to set _GLIBCXX_USE_CXX11_ABI=0 so std::pmr::string wouldn't be available on CentOS 7.
#if _GLIBCXX_USE_CXX11_ABI
    std::pmr::string dims_{ &res_ };
    std::pmr::vector<int64_t> shape_{ &res_ };
    std::pmr::vector<std::pmr::string> channel_names_{ &res_ };
    std::pmr::vector<float> spacing_{ &res_ };
    std::pmr::vector<std::pmr::string> spacing_units_{ &res_ };
    std::pmr::vector<float> origin_{ &res_ };
    std::pmr::vector<float> direction_{ &res_ };
    std::pmr::string coord_sys_{ &res_ };

    std::pmr::vector<int64_t> level_dimensions_{ &res_ };
    std::pmr::vector<float> level_downsamples_{ &res_ };
    std::pmr::vector<uint32_t> level_tile_sizes_{ &res_ };

    std::pmr::vector<std::pmr::string> image_names_{ &res_ };
#else
    std::string dims_;
    std::pmr::vector<int64_t> shape_{ &res_ };
    std::pmr::vector<std::string> channel_names_{ &res_ };
    std::pmr::vector<float> spacing_{ &res_ };
    std::pmr::vector<std::string> spacing_units_{ &res_ };
    std::pmr::vector<float> origin_{ &res_ };
    std::pmr::vector<float> direction_{ &res_ };
    std::string coord_sys_;

    std::pmr::vector<int64_t> level_dimensions_{ &res_ };
    std::pmr::vector<float> level_downsamples_{ &res_ };
    std::pmr::vector<uint32_t> level_tile_sizes_{ &res_ };

    std::pmr::vector<std::string> image_names_{ &res_ };
#endif
    // Memory for raw_data and json_data needs to be created with cucim_malloc();
};

struct ImageDataDesc
{
    DLTensor container;
    char* shm_name;
};

struct ImageCheckerDesc
{
    size_t header_start_offset; /// Start offset to look at the image header
    size_t header_read_size; /// Number of bytes from the start offset, needed to check image format
    /**
     * Returns true if the given file is valid for the format
     * @param file_name
     * @param buf
     * @return
     */
    bool(CUCIM_ABI* is_valid)(const char* file_name, const char* buf);
};

struct ImageParserDesc
{
    /**
     *
     * @param file_path
     * @return
     */
    CuCIMFileHandle(CUCIM_ABI* open)(const char* file_path);

    /**
     *
     * @param handle
     * @param out_metadata
     * @return
     */
    bool(CUCIM_ABI* parse)(CuCIMFileHandle* handle, ImageMetadataDesc* out_metadata);

    /**
     *
     * @param handle
     * @return
     */
    bool(CUCIM_ABI* close)(CuCIMFileHandle* handle);
};

struct ImageReaderRegionRequestDesc
{
    int64_t* location;
    int64_t* size;
    uint16_t level;
    DimIndicesDesc region_dim_indices;
    char* associated_image_name;
    char* device;
    DLTensor* buf;
    char* shm_name;
};

struct ImageReaderDesc
{
    /**
     *
     * @param handle
     * @param metadata
     * @param out_image_data
     * @param out_image_metadata needed for associated_image
     * @return
     */
    bool(CUCIM_ABI* read)(const CuCIMFileHandle* handle,
                          const ImageMetadataDesc* metadata,
                          const ImageReaderRegionRequestDesc* request,
                          ImageDataDesc* out_image_data,
                          ImageMetadataDesc* out_metadata);
};

struct ImageWriterDesc
{
    /**
     *
     * @param handle
     * @param metadata
     * @param image_data
     * @return
     */
    bool(CUCIM_ABI* write)(const CuCIMFileHandle* handle,
                           const ImageMetadataDesc* metadata,
                           const ImageDataDesc* image_data);
};

struct ImageFormatDesc
{
    void(CUCIM_ABI* set_enabled)(bool val); /// Sets if this format will be used in cucim (default: true).
    bool(CUCIM_ABI* is_enabled)(); /// true if this format is used when checking image compatibility.
    const char*(CUCIM_ABI* get_format_name)(); /// Returns the name of this format.
    ImageCheckerDesc image_checker;
    ImageParserDesc image_parser;
    ImageReaderDesc image_reader;
    ImageWriterDesc image_writer;
};

struct IImageFormat
{
    CUCIM_PLUGIN_INTERFACE("cucim::io::IImageFormat", 0, 1)
    ImageFormatDesc* formats;
    size_t format_count;
};

} // namespace cucim::io::format

#endif // CUCIM_IMAGE_FORMAT_H
