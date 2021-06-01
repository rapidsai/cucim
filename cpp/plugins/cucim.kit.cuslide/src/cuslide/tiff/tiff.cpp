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

#include "tiff.h"

#include <fcntl.h>

#include <algorithm>
#include <string_view>

#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <pugixml.hpp>
#include <tiffiop.h>

#include <cucim/codec/base64.h>
#include <cucim/logger/timer.h>
#include <cucim/memory/memory_manager.h>

#include "cuslide/jpeg/libjpeg_turbo.h"
#include "ifd.h"

static constexpr int DEFAULT_IFD_SIZE = 32;

using json = nlohmann::json;

namespace cuslide::tiff
{

// djb2 algorithm from http://www.cse.yorku.ca/~oz/hash.html
constexpr uint32_t hash_str(const char* str)
{
    uint32_t hash = 5381;
    uint32_t c = 0;
    while ((c = *str++))
        hash = ((hash << 5) + hash) + c; // hash * 33 + c
    return hash;
}

enum class PhilipsMetadataStage : uint8_t
{
    ROOT = 0,
    SCANNED_IMAGE,
    PIXEL_DATA_PRESENTATION,
    ELEMENT,
    ARRAY_ELEMENT
};
enum class PhilipsMetadataType : uint8_t
{
    IString = 0,
    IDouble,
    IUInt16,
    IUInt32,
    IUInt64
};
static void parse_string_array(const char* values, json& arr, PhilipsMetadataType type)
{
    std::string_view text(values);
    std::string_view::size_type pos = 0;
    while ((pos = text.find('"', pos)) != std::string_view::npos)
    {
        auto next_pos = text.find('"', pos + 1);
        if (next_pos != std::string_view::npos)
        {
            if (text[next_pos - 1] != '\\')
            {
                switch (type)
                {
                case PhilipsMetadataType::IString:
                    arr.emplace_back(std::string(text.substr(pos + 1, next_pos - pos - 1)));
                    break;
                case PhilipsMetadataType::IDouble:
                    arr.emplace_back(std::stod(std::string(text.substr(pos + 1, next_pos - pos - 1))));
                    break;
                case PhilipsMetadataType::IUInt16:
                case PhilipsMetadataType::IUInt32:
                case PhilipsMetadataType::IUInt64:
                    arr.emplace_back(std::stoul(std::string(text.substr(pos + 1, next_pos - pos - 1))));
                    break;
                }
                pos = next_pos + 1;
            }
        }
    }
}
static void parse_philips_tiff_metadata(const pugi::xml_node& node,
                                        json& metadata,
                                        const char* name,
                                        PhilipsMetadataStage stage)
{
    switch (stage)
    {
    case PhilipsMetadataStage::ROOT:
    case PhilipsMetadataStage::SCANNED_IMAGE:
    case PhilipsMetadataStage::PIXEL_DATA_PRESENTATION:
        for (pugi::xml_node attr = node.child("Attribute"); attr; attr = attr.next_sibling("Attribute"))
        {
            const pugi::xml_attribute& attr_attribute = attr.attribute("Name");
            if (attr_attribute)
            {
                parse_philips_tiff_metadata(attr, metadata, attr_attribute.value(), PhilipsMetadataStage::ELEMENT);
            }
        }
        break;
    case PhilipsMetadataStage::ARRAY_ELEMENT:
        break;
    case PhilipsMetadataStage::ELEMENT:
        const pugi::xml_attribute& attr_attribute = node.attribute("PMSVR");
        auto p_attr_name = attr_attribute.as_string();
        if (p_attr_name != nullptr && *p_attr_name != '\0')
        {
            if (name)
            {
                switch (hash_str(p_attr_name))
                {
                case hash_str("IString"):
                    metadata.emplace(name, node.text().as_string());
                    break;
                case hash_str("IDouble"):
                    metadata.emplace(name, node.text().as_double());
                    break;
                case hash_str("IUInt16"):
                    metadata.emplace(name, node.text().as_uint());
                    break;
                case hash_str("IUInt32"):
                    metadata.emplace(name, node.text().as_uint());
                    break;
                case hash_str("IUint64"):
                    metadata.emplace(name, node.text().as_ullong());
                    break;
                case hash_str("IStringArray"): { // Process text such as `"a" "b" "c"`
                    auto item_iter = metadata.emplace(name, json::array());
                    parse_string_array(node.child_value(), *(item_iter.first), PhilipsMetadataType::IString);
                    break;
                }
                case hash_str("IDoubleArray"): { // Process text such as `"0.0" "0.1" "0.2"`
                    auto item_iter = metadata.emplace(name, json::array());
                    parse_string_array(node.child_value(), *(item_iter.first), PhilipsMetadataType::IDouble);
                    break;
                }
                case hash_str("IUInt16Array"): { // Process text such as `"1" "2" "3"`
                    auto item_iter = metadata.emplace(name, json::array());
                    parse_string_array(node.child_value(), *(item_iter.first), PhilipsMetadataType::IUInt16);
                    break;
                }
                case hash_str("IUInt32Array"): { // Process text such as `"1" "2" "3"`
                    auto item_iter = metadata.emplace(name, json::array());
                    parse_string_array(node.child_value(), *(item_iter.first), PhilipsMetadataType::IUInt32);
                    break;
                }
                case hash_str("IUInt64Array"): { // Process text such as `"1" "2" "3"`
                    auto item_iter = metadata.emplace(name, json::array());
                    parse_string_array(node.child_value(), *(item_iter.first), PhilipsMetadataType::IUInt64);
                    break;
                }
                case hash_str("IDataObjectArray"):
                    if (strcmp(name, "PIIM_PIXEL_DATA_REPRESENTATION_SEQUENCE") == 0)
                    {
                        const auto& item_array_iter =
                            metadata.emplace(std::string("PIIM_PIXEL_DATA_REPRESENTATION_SEQUENCE"), json::array());
                        for (pugi::xml_node data_node = node.child("Array").child("DataObject"); data_node;
                             data_node = data_node.next_sibling("DataObject"))
                        {
                            auto& item_iter = item_array_iter.first->emplace_back(json{});
                            parse_philips_tiff_metadata(
                                data_node, item_iter, nullptr, PhilipsMetadataStage::PIXEL_DATA_PRESENTATION);
                        }
                    }
                    break;
                }
            }
        }
        break;
    }
}

TIFF::~TIFF()
{
    close();
}

TIFF::TIFF(const cucim::filesystem::Path& file_path, int mode) : file_path_(file_path)
{
    // Copy file path (Allocated memory would be freed at close() method.)
    char* file_path_cstr = static_cast<char*>(cucim_malloc(file_path.size() + 1));
    memcpy(file_path_cstr, file_path.c_str(), file_path.size());
    file_path_cstr[file_path.size()] = '\0';

    int fd = ::open(file_path_cstr, mode, 0666);
    if (fd == -1)
    {
        cucim_free(file_path_cstr);
        throw std::invalid_argument(fmt::format("Cannot open {}!", file_path));
    }
    tiff_client_ = ::TIFFFdOpen(fd, file_path_cstr, "rm"); // Add 'm' to disable memory-mapped file
    // TODO: make file_handle_ object to pointer
    file_handle_ = CuCIMFileHandle{ fd, nullptr, FileHandleType::kPosix, file_path_cstr, this };

    // TODO: warning if the file is big endian
    is_big_endian_ = ::TIFFIsBigEndian(tiff_client_);

    metadata_ = new json{};
}
TIFF::TIFF(const cucim::filesystem::Path& file_path, int mode, uint64_t read_config) : TIFF(file_path, mode)
{
    read_config_ = read_config;
}

std::shared_ptr<TIFF> TIFF::open(const cucim::filesystem::Path& file_path, int mode)
{
    auto tif = std::make_shared<TIFF>(file_path, mode);
    tif->construct_ifds();

    return tif;
}

std::shared_ptr<TIFF> TIFF::open(const cucim::filesystem::Path& file_path, int mode, uint64_t config)
{
    auto tif = std::make_shared<TIFF>(file_path, mode, config);
    tif->construct_ifds();

    return tif;
}

void TIFF::close()
{
    if (tiff_client_)
    {
        TIFFClose(tiff_client_);
        tiff_client_ = nullptr;
    }
    if (file_handle_.path)
    {
        cucim_free(file_handle_.path);
        file_handle_.path = nullptr;
    }
    if (file_handle_.client_data)
    {
        // Deleting file_handle_.client_data is parser_close()'s responsibility
        // Do not execute this: `delete static_cast<cuslide::tiff::TIFF*>(file_handle_.client_data);`
        file_handle_.client_data = nullptr;
    }
    if (metadata_)
    {
        delete reinterpret_cast<json*>(metadata_);
        metadata_ = nullptr;
    }
}

void TIFF::construct_ifds()
{
    ifd_offsets_.clear();
    ifd_offsets_.reserve(DEFAULT_IFD_SIZE);
    ifds_.clear();
    ifds_.reserve(DEFAULT_IFD_SIZE);

    uint16_t ifd_index = 0;
    do
    {
        uint64_t offset = TIFFCurrentDirOffset(tiff_client_);
        ifd_offsets_.push_back(offset);

        auto ifd = std::make_shared<cuslide::tiff::IFD>(this, ifd_index, offset);
        ifds_.emplace_back(std::move(ifd));
        ++ifd_index;
    } while (TIFFReadDirectory(tiff_client_));

    // Set index for each level
    level_to_ifd_idx_.reserve(ifd_index);
    for (size_t index = 0; index < ifd_index; ++index)
    {
        level_to_ifd_idx_.emplace_back(index);
    }

    // Resolve format and fix `level_to_ifds_idx_`
    resolve_vendor_format();

    // Sort index by resolution (the largest resolution is index 0)
    std::sort(level_to_ifd_idx_.begin(), level_to_ifd_idx_.end(), [this](const size_t& a, const size_t& b) {
        uint32_t width_a = this->ifds_[a]->width();
        uint32_t width_b = this->ifds_[b]->width();
        if (width_a > width_b)
        {
            return true;
        }
        else if (width_a < width_b)
        {
            return false;
        }
        else
        {
            uint32_t height_a = this->ifds_[a]->height();
            uint32_t height_b = this->ifds_[b]->height();
            return height_a > height_b;
        }
    });
}
void TIFF::resolve_vendor_format()
{
    uint16_t ifd_count = ifds_.size();
    if (ifd_count == 0)
    {
        return;
    }
    json* json_metadata = reinterpret_cast<json*>(metadata_);

    // Detect Philips TIFF
    auto& first_ifd = ifds_[0];
    std::string& software = first_ifd->software();
    std::string_view prefix("Philips");
    std::string_view macro_prefix("Macro");
    std::string_view label_prefix("Label");

    auto res = std::mismatch(prefix.begin(), prefix.end(), software.begin());
    if (res.first == prefix.end())
    {
        pugi::xml_document doc;
        const char* image_desc_cstr = first_ifd->image_description().c_str();
        pugi::xml_parse_result result = doc.load_string(image_desc_cstr);
        if (result)
        {
            const auto& data_object = doc.child("DataObject");
            if (std::string_view(data_object.attribute("ObjectType").as_string("")) != "DPUfsImport")
            {
                fmt::print(
                    stderr,
                    "[Warning] Failed to read as Philips TIFF. It looks like Philips TIFF but the image description of the first IFD doesn't have '<DataObject ObjectType=\"DPUfsImport\">' node!\n");
                return;
            }

            pugi::xpath_query PIM_DP_IMAGE_TYPE(
                "Attribute[@Name='PIM_DP_SCANNED_IMAGES']/Array/DataObject[Attribute/@Name='PIM_DP_IMAGE_TYPE' and Attribute/text()='WSI']");
            pugi::xpath_node_set wsi_nodes = PIM_DP_IMAGE_TYPE.evaluate_node_set(data_object);
            if (wsi_nodes.size() != 1)
            {
                fmt::print(
                    stderr,
                    "[Warning] Failed to read as Philips TIFF. Expected only one 'DPScannedImage' node with PIM_DP_IMAGE_TYPE='WSI'.\n");
                return;
            }

            pugi::xpath_query DICOM_PIXEL_SPACING(
                "Attribute[@Name='PIIM_PIXEL_DATA_REPRESENTATION_SEQUENCE']/Array/DataObject/Attribute[@Name='DICOM_PIXEL_SPACING']");
            pugi::xpath_node_set pixel_spacing_nodes = DICOM_PIXEL_SPACING.evaluate_node_set(wsi_nodes[0]);

            std::vector<std::pair<double, double>> pixel_spacings;
            pixel_spacings.reserve(pixel_spacings.size());

            for (const pugi::xpath_node& pixel_spacing : pixel_spacing_nodes)
            {
                std::string values = pixel_spacing.node().text().as_string();

                // Assume that 'values' has a '"<height spacing in mm>" "<width spacing in mm>"' form.
                double spacing_x = 0.0;
                double spacing_y = 0.0;

                std::string::size_type offset = values.find("\"");
                if (offset != std::string::npos)
                {
                    spacing_y = std::atof(&values.c_str()[offset + 1]);
                    offset = values.find(" \"", offset);
                    if (offset != std::string::npos)
                    {
                        spacing_x = std::atof(&values.c_str()[offset + 2]);
                    }
                }
                if (spacing_x == 0.0 || spacing_y == 0.0)
                {
                    fmt::print(stderr, "[Warning] Failed to read DICOM_PIXEL_SPACING: {}\n", values);
                    return;
                }
                pixel_spacings.emplace_back(std::pair{ spacing_x, spacing_y });
            }

            double spacing_x_l0 = pixel_spacings[0].first;
            double spacing_y_l0 = pixel_spacings[0].second;

            uint32_t width_l0 = first_ifd->width();
            uint32_t height_l0 = first_ifd->height();

            uint16_t spacing_index = 1;
            for (int index = 1, level_index = 1; index < ifd_count; ++index, ++level_index)
            {
                auto& ifd = ifds_[index];
                if (ifd->tile_width() == 0)
                {
                    // TODO: check macro and label
                    AssociatedImageBufferDesc buf_desc{};
                    buf_desc.type = AssociatedImageBufferType::IFD;
                    buf_desc.compression = static_cast<cucim::codec::CompressionMethod>(ifd->compression());
                    buf_desc.ifd_index = index;

                    auto& image_desc = ifd->image_description();
                    if (std::mismatch(macro_prefix.begin(), macro_prefix.end(), image_desc.begin()).first ==
                        macro_prefix.end())
                    {
                        associated_images_.emplace("macro", buf_desc);
                    }
                    else if (std::mismatch(label_prefix.begin(), label_prefix.end(), image_desc.begin()).first ==
                             label_prefix.end())
                    {
                        associated_images_.emplace("label", buf_desc);
                    }

                    // Remove item at index `ifd_index` from `level_to_ifd_idx_`
                    level_to_ifd_idx_.erase(level_to_ifd_idx_.begin() + level_index);
                    --level_index;
                    continue;
                }
                double downsample = std::round((pixel_spacings[spacing_index].first / spacing_x_l0 +
                                                pixel_spacings[spacing_index].second / spacing_y_l0) /
                                               2);
                // Fix width and height of IFD
                ifd->width_ = width_l0 / downsample;
                ifd->height_ = height_l0 / downsample;
                ++spacing_index;
            }

            constexpr int associated_image_type_count = 2;
            pugi::xpath_query ASSOCIATED_IMAGES[associated_image_type_count] = {
                pugi::xpath_query(
                    "Attribute[@Name='PIM_DP_SCANNED_IMAGES']/Array/DataObject[Attribute/@Name='PIM_DP_IMAGE_TYPE' and Attribute/text()='MACROIMAGE'][1]/Attribute[@Name='PIM_DP_IMAGE_DATA']"),
                pugi::xpath_query(
                    "Attribute[@Name='PIM_DP_SCANNED_IMAGES']/Array/DataObject[Attribute/@Name='PIM_DP_IMAGE_TYPE' and Attribute/text()='LABELIMAGE'][1]/Attribute[@Name='PIM_DP_IMAGE_DATA']")
            };
            constexpr const char* associated_image_names[associated_image_type_count] = { "macro", "label" };

            // Add associated image from XML if available (macro and label images)
            // : Refer to PIM_DP_IMAGE_TYPE in
            // https://www.openpathology.philips.com/wp-content/uploads/isyntax/4522%20207%2043941_2020_04_24%20Pathology%20iSyntax%20image%20format.pdf

            for (int associated_image_type_idx = 0; associated_image_type_idx < associated_image_type_count;
                 ++associated_image_type_idx)
            {
                pugi::xpath_node associated_node =
                    ASSOCIATED_IMAGES[associated_image_type_idx].evaluate_node(data_object);
                const char* associated_image_name = associated_image_names[associated_image_type_idx];

                // If the associated image doesn't exist
                if (associated_images_.find(associated_image_name) == associated_images_.end())
                {
                    if (associated_node)
                    {
                        auto node_offset = associated_node.node().offset_debug();

                        if (node_offset >= 0)
                        {
                            // `image_desc_cstr[node_offset]` would point to the following text:
                            //   Attribute Element="0x1004" Group="0x301D" Name="PIM_DP_IMAGE_DATA" PMSVR="IString">
                            //     (base64-encoded JPEG image)
                            //   </Attribute>
                            //

                            // 34 is from `Attribute Name="PIM_DP_IMAGE_DATA"`
                            char* data_ptr = const_cast<char*>(image_desc_cstr) + node_offset + 34;
                            uint32_t data_len = 0;
                            while (*data_ptr != '>' && *data_ptr != '\0')
                            {
                                ++data_ptr;
                            }
                            if (*data_ptr != '\0')
                            {
                                ++data_ptr; // start of base64-encoded data
                                char* data_end_ptr = data_ptr;
                                // Seek until it finds '<' for '</Attribute>'
                                while (*data_end_ptr != '<' && *data_end_ptr != '\0')
                                {
                                    ++data_end_ptr;
                                }
                                data_len = data_end_ptr - data_ptr;
                            }

                            if (data_len > 0)
                            {
                                AssociatedImageBufferDesc buf_desc{};
                                buf_desc.type = AssociatedImageBufferType::IFD_IMAGE_DESC;
                                buf_desc.compression = cucim::codec::CompressionMethod::JPEG;
                                buf_desc.desc_ifd_index = 0;
                                buf_desc.desc_offset = data_ptr - image_desc_cstr;
                                buf_desc.desc_size = data_len;

                                associated_images_.emplace(associated_image_name, buf_desc);
                            }
                        }
                    }
                }
            }

            // Set TIFF type
            tiff_type_ = TiffType::Philips;

            // Set background color
            background_value_ = 0xFF;

            // Get metadata
            if (json_metadata)
            {
                json philips_metadata;
                parse_philips_tiff_metadata(data_object, philips_metadata, nullptr, PhilipsMetadataStage::ROOT);
                parse_philips_tiff_metadata(
                    wsi_nodes[0].node(), philips_metadata, nullptr, PhilipsMetadataStage::SCANNED_IMAGE);
                (*json_metadata).emplace("philips", std::move(philips_metadata));
            }
        }
    }


    // Append TIFF metadata

    if (json_metadata)
    {
        json tiff_metadata;

        tiff_metadata.emplace("model", first_ifd->model());
        tiff_metadata.emplace("software", first_ifd->software());
        tiff_metadata.emplace("model", first_ifd->model());

        (*json_metadata).emplace("tiff", std::move(tiff_metadata));
    }
}

bool TIFF::read(const cucim::io::format::ImageMetadataDesc* metadata,
                const cucim::io::format::ImageReaderRegionRequestDesc* request,
                cucim::io::format::ImageDataDesc* out_image_data,
                cucim::io::format::ImageMetadataDesc* out_metadata)
{
    if (request->associated_image_name)
    {
        // 'out_metadata' is only needed for reading associated image
        return read_associated_image(metadata, request, out_image_data, out_metadata);
    }

    // TODO: assume length of location/size to 2.
    constexpr int32_t ndims = 2;

    if (request->level >= level_to_ifd_idx_.size())
    {
        throw std::invalid_argument(fmt::format(
            "Invalid level ({}) in the request! (Should be < {})", request->level, level_to_ifd_idx_.size()));
    }
    auto main_ifd = ifds_[level_to_ifd_idx_[0]];
    auto ifd = ifds_[level_to_ifd_idx_[request->level]];
    auto original_img_width = main_ifd->width();
    auto original_img_height = main_ifd->height();

    for (int32_t i = 0; i < ndims; ++i)
    {
        if (request->size[i] <= 0)
        {
            throw std::invalid_argument(
                fmt::format("Invalid size ({}) in the request! (Should be > 0)", request->size[i]));
        }
    }
    if (request->size[0] > original_img_width)
    {
        throw std::invalid_argument(
            fmt::format("Invalid size (it exceeds the original image width {})", original_img_width));
    }
    if (request->size[1] > original_img_height)
    {
        throw std::invalid_argument(
            fmt::format("Invalid size (it exceeds the original image height {})", original_img_height));
    }

    float downsample_factor = metadata->resolution_info.level_downsamples[request->level];

    // Change request based on downsample factor. (normalized value at level-0 -> real location at the requested level)
    for (int32_t i = 0; i < ndims; ++i)
    {
        request->location[i] /= downsample_factor;
    }
    return ifd->read(this, metadata, request, out_image_data);
}

bool TIFF::read_associated_image(const cucim::io::format::ImageMetadataDesc* metadata,
                                 const cucim::io::format::ImageReaderRegionRequestDesc* request,
                                 cucim::io::format::ImageDataDesc* out_image_data,
                                 cucim::io::format::ImageMetadataDesc* out_metadata_desc)
{
    // TODO: implement
    (void)metadata;

    std::string device_name(request->device);
    if (request->shm_name)
    {
        device_name = device_name + fmt::format("[{}]", request->shm_name); // TODO: check performance
    }
    cucim::io::Device out_device(device_name);

    uint8_t* raster = nullptr;
    size_t raster_size = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t samples_per_pixel = 0;

    // Raw metadata for the associated image
    const char* raw_data_ptr = nullptr;
    size_t raw_data_len = 0;
    // Json metadata for the associated image
    char* json_data_ptr = nullptr;

    auto associated_image = associated_images_.find(request->associated_image_name);
    if (associated_image != associated_images_.end())
    {
        auto& buf_desc = associated_image->second;

        switch (buf_desc.type)
        {
        case AssociatedImageBufferType::IFD: {
            const auto& image_ifd = ifd(buf_desc.ifd_index);

            auto& image_description = image_ifd->image_description();
            auto image_description_size = image_description.size();

            // Assign image description into raw_data_ptr
            raw_data_ptr = image_description.c_str();
            raw_data_len = image_description_size;

            width = image_ifd->width();
            height = image_ifd->height();
            samples_per_pixel = image_ifd->samples_per_pixel();
            raster_size = width * height * samples_per_pixel;

            raster = static_cast<uint8_t*>(cucim_malloc(raster_size)); // RGB image

            // TODO: here we assume that the image has a single strip.
            uint64_t offset = image_ifd->image_piece_offsets_[0];
            uint64_t size = image_ifd->image_piece_bytecounts_[0];
            const void* jpegtable_data = image_ifd->jpegtable_.data();
            uint32_t jpegtable_count = image_ifd->jpegtable_.size();

            if (!cuslide::jpeg::decode_libjpeg(file_handle_.fd, nullptr /*jpeg_buf*/, offset, size, jpegtable_data,
                                               jpegtable_count, &raster, out_device))
            {
                cucim_free(raster);
                fmt::print(stderr, "[Error] Failed to read region with libjpeg!\n");
                return false;
            }
            break;
        }
        case AssociatedImageBufferType::IFD_IMAGE_DESC: {
            const auto& image_ifd = ifd(buf_desc.desc_ifd_index);
            const char* image_desc_buf = image_ifd->image_description().data();
            char* decoded_buf = nullptr;
            int decoded_size = 0;

            if (!cucim::codec::base64::decode(
                    image_desc_buf, image_ifd->image_description().size(), &decoded_buf, &decoded_size))
            {
                fmt::print(stderr, "[Error] Failed to decode base64-encoded string from the metadata!\n");
                return false;
            }

            int image_width = 0;
            int image_height = 0;

            if (!cuslide::jpeg::get_dimension(decoded_buf, 0, decoded_size, &image_width, &image_height))
            {
                fmt::print(stderr, "[Error] Failed to read jpeg header for image dimension!\n");
                return false;
            }

            width = image_width;
            height = image_height;
            samples_per_pixel = 3; // NOTE: assumes RGB image
            raster_size = image_width * image_height * samples_per_pixel;

            raster = static_cast<uint8_t*>(cucim_malloc(raster_size)); // RGB image

            if (!cuslide::jpeg::decode_libjpeg(-1, reinterpret_cast<unsigned char*>(decoded_buf), 0 /*offset*/,
                                               decoded_size, nullptr /*jpegtable_data*/, 0 /*jpegtable_count*/, &raster,
                                               out_device))
            {
                cucim_free(raster);
                fmt::print(stderr, "[Error] Failed to read image from metadata with libjpeg!\n");
                return false;
            }
            break;
        }
        case AssociatedImageBufferType::FILE_OFFSET:
            // TODO: implement
            break;
        case AssociatedImageBufferType::BUFFER_POINTER:
            // TODO: implement
            break;
        case AssociatedImageBufferType::OWNED_BUFFER_POINTER:
            // TODO: implement
            break;
        }
    }

    // Populate image data
    const uint16_t ndim = 3;

    int64_t* container_shape = static_cast<int64_t*>(cucim_malloc(sizeof(int64_t) * ndim));
    container_shape[0] = height;
    container_shape[1] = width;
    container_shape[2] = 3; // TODO: hard-coded for 'C'

    // Copy the raster memory and free it if needed.
    cucim::memory::move_raster_from_host((void**)&raster, raster_size, out_device);

    auto& out_image_container = out_image_data->container;
    out_image_container.data = raster;
    out_image_container.ctx = DLContext{ static_cast<DLDeviceType>(out_device.type()), out_device.index() };
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
    }

    return true;
}

cucim::filesystem::Path TIFF::file_path() const
{
    return file_path_;
}

CuCIMFileHandle TIFF::file_handle() const
{
    return file_handle_;
}
::TIFF* TIFF::client() const
{
    return tiff_client_;
}
const std::vector<ifd_offset_t>& TIFF::ifd_offsets() const
{
    return ifd_offsets_;
}
std::shared_ptr<IFD> TIFF::ifd(size_t index) const
{
    return ifds_.at(index);
}
std::shared_ptr<IFD> TIFF::level_ifd(size_t level_index) const
{
    return ifds_.at(level_to_ifd_idx_.at(level_index));
}
size_t TIFF::ifd_count() const
{
    return ifd_offsets_.size();
}
size_t TIFF::level_count() const
{
    return level_to_ifd_idx_.size();
}
const std::map<std::string, AssociatedImageBufferDesc>& TIFF::associated_images() const
{
    return associated_images_;
}
size_t TIFF::associated_image_count() const
{
    return associated_images_.size();
}
bool TIFF::is_big_endian() const
{
    return is_big_endian_;
}

uint64_t TIFF::read_config() const
{
    return read_config_;
}

bool TIFF::is_in_read_config(uint64_t configs) const
{
    return (read_config_ & configs) == configs;
}

void TIFF::add_read_config(uint64_t configs)
{
    read_config_ |= configs;
}

TiffType TIFF::tiff_type()
{
    return tiff_type_;
}

std::string TIFF::metadata()
{
    json* metadata = reinterpret_cast<json*>(metadata_);

    if (metadata)
    {
        return metadata->dump();
    }
    else
    {
        return std::string{};
    }
}

void* TIFF::operator new(std::size_t sz)
{
    return cucim_malloc(sz);
}

void TIFF::operator delete(void* ptr)
{
    cucim_free(ptr);
}
} // namespace cuslide::tiff
