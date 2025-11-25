/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tiff.h"

#include <fcntl.h>

#include <algorithm>
#include <string>
#include <string_view>

#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <pugixml.hpp>
#include "tiff_constants.h"

#include <cucim/codec/base64.h>
#include <cucim/logger/timer.h>
#include <cucim/memory/memory_manager.h>
#include <cucim/profiler/nvtx3.h>

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

static std::vector<std::string> split_string(std::string_view s, std::string_view delim, size_t capacity = 0)
{
    size_t pos_start = 0;
    size_t pos_end = -1;
    size_t delim_len = delim.length();

    std::vector<std::string> result;
    std::string_view item;

    if (capacity != 0)
    {
        result.reserve(capacity);
    }

    while ((pos_end = s.find(delim, pos_start)) != std::string_view::npos)
    {
        item = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        result.emplace_back(item);
    }

    result.emplace_back(s.substr(pos_start));
    return result;
}

static std::string strip_string(const std::string& str)
{
    static const char* white_spaces = " \r\n\t";
    std::string::size_type start_pos = str.find_first_not_of(white_spaces);
    std::string::size_type end_pos = str.find_last_not_of(white_spaces);

    if (start_pos != std::string::npos)
    {
        return str.substr(start_pos, end_pos - start_pos + 1);
    }
    else
    {
        return std::string();
    }
}

static void parse_aperio_svs_metadata(std::shared_ptr<IFD>& first_ifd, json& metadata)
{
    (void)metadata;
    std::string& desc = first_ifd->image_description();

    // Assumes that metadata's image description starts with 'Aperio '.
    // It is handled by 'resolve_vendor_format()'
    std::vector<std::string> items = split_string(desc, "|");
    if (items.size() < 1)
    {
        return;
    }
    // Store the first item of the image description as 'Header'
    metadata.emplace("Header", items[0]);
    for (size_t i = 1; i < items.size(); ++i)
    {
        std::vector<std::string> key_value = split_string(items[i], " = ");
        if (key_value.size() == 2)
        {
            metadata.emplace(std::move(strip_string(key_value[0])), std::move(strip_string(key_value[1])));
        }
    }
}

TIFF::~TIFF()
{
    PROF_SCOPED_RANGE(PROF_EVENT(tiff__tiff));
    close();
}

// NEW CONSTRUCTOR: nvImageCodec-only (no libtiff)
TIFF::TIFF(const cucim::filesystem::Path& file_path) : file_path_(file_path)
{
    PROF_SCOPED_RANGE(PROF_EVENT_P(tiff_tiff, 1));
    
    #ifdef DEBUG
    fmt::print("📂 Opening TIFF file with nvImageCodec: {}\n", file_path);
    #endif // DEBUG
    
    // Step 1: Open file descriptor (needed for CuCIMFileHandle)
    // Copy file path (will be freed by CuCIMFileHandle destructor)
    char* file_path_cstr = static_cast<char*>(cucim_malloc(file_path.size() + 1));
    memcpy(file_path_cstr, file_path.c_str(), file_path.size());
    file_path_cstr[file_path.size()] = '\0';

    int fd = ::open(file_path_cstr, O_RDONLY, 0666);
    if (fd == -1)
    {
        cucim_free(file_path_cstr);
        throw std::invalid_argument(fmt::format("Cannot open {}!", file_path));
    }
    
    // Step 2: Create CuCIMFileHandle with 'this' as client_data
    // CRITICAL: The 5th parameter (client_data) must be 'this' so parser_parse() 
    // can retrieve the TIFF object later via handle->client_data
    file_handle_shared_ = std::make_shared<CuCIMFileHandle>(
        fd, nullptr, FileHandleType::kPosix, file_path_cstr, this);
    
    // Step 3: Set up deleter to clean up TIFF object when handle is destroyed
    // This is CRITICAL to prevent memory leaks and double-frees
    file_handle_shared_->set_deleter([](CuCIMFileHandle_ptr handle_ptr) -> bool {
        auto* handle = reinterpret_cast<CuCIMFileHandle*>(handle_ptr);
        if (handle && handle->client_data)
        {
            auto* tiff_obj = static_cast<cuslide::tiff::TIFF*>(handle->client_data);
            delete tiff_obj;
            handle->client_data = nullptr;
        }
        return true;
    });
    
    // Step 4: Initialize nvImageCodec TiffFileParser (MANDATORY)
    try {
        nvimgcodec_parser_ = std::make_unique<cuslide2::nvimgcodec::TiffFileParser>(
            file_path.c_str());
        
        if (!nvimgcodec_parser_->is_valid()) {
            throw std::runtime_error("TiffFileParser initialization failed");
        }
        
        #ifdef DEBUG
        fmt::print("✅ nvImageCodec TiffFileParser initialized successfully\n");
        #endif // DEBUG
        
        // Extract basic file properties from TiffFileParser
        std::string detected_format = nvimgcodec_parser_->get_detected_format();
        #ifdef DEBUG
        fmt::print("   Detected format: {}\n", detected_format);
        #endif // DEBUG
        
        #ifdef DEBUG
        uint32_t ifd_count = nvimgcodec_parser_->get_ifd_count();
        fmt::print("   IFD count: {}\n", ifd_count);
        #endif // DEBUG
        
        // Set default values (nvImageCodec handles endianness internally)
        is_big_endian_ = false;
        
    } catch (const std::exception& e) {
        #ifdef DEBUG
        fmt::print("❌ FATAL: Failed to initialize nvImageCodec TiffFileParser: {}\n", e.what());
        #endif // DEBUG
        #ifdef DEBUG
        fmt::print("   This library requires nvImageCodec for TIFF support.\n");
        #endif // DEBUG
        #ifdef DEBUG
        fmt::print("   Please ensure nvImageCodec is properly installed.\n");
        #endif // DEBUG
        // Cleanup file handle before re-throwing
        file_handle_shared_.reset();
        throw std::runtime_error(fmt::format(
            "Cannot open TIFF file '{}' without nvImageCodec: {}", 
            file_path, e.what()));
    }
    
    // Initialize metadata container
    metadata_ = new json{};
}

TIFF::TIFF(const cucim::filesystem::Path& file_path, uint64_t read_config) : TIFF(file_path)
{
    PROF_SCOPED_RANGE(PROF_EVENT_P(tiff_tiff, 2));
    read_config_ = read_config;
}

// Legacy libtiff-style constructors (for backward compatibility with tests)
TIFF::TIFF(const cucim::filesystem::Path& file_path, int mode) : TIFF(file_path)
{
    // mode parameter is ignored in pure nvImageCodec build
    (void)mode;
}

TIFF::TIFF(const cucim::filesystem::Path& file_path, int mode, uint64_t config) : TIFF(file_path, config)
{
    // mode parameter is ignored in pure nvImageCodec build
    (void)mode;
}

std::shared_ptr<TIFF> TIFF::open(const cucim::filesystem::Path& file_path)
{
    auto tif = std::make_shared<TIFF>(file_path);
    tif->construct_ifds();
    return tif;
}

std::shared_ptr<TIFF> TIFF::open(const cucim::filesystem::Path& file_path, uint64_t config)
{
    auto tif = std::make_shared<TIFF>(file_path, config);
    tif->construct_ifds();
    return tif;
}

// Legacy libtiff-style open methods (for backward compatibility with tests)
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
    // REMOVED: libtiff cleanup - no longer using tiff_client_
    
    // Clean up metadata
    if (metadata_)
    {
        delete reinterpret_cast<json*>(metadata_);
        metadata_ = nullptr;
    }
    
    // nvimgcodec_parser_ is automatically cleaned up by unique_ptr destructor
}

void TIFF::construct_ifds()
{
    PROF_SCOPED_RANGE(PROF_EVENT(tiff_construct_ifds));
    
    if (!nvimgcodec_parser_ || !nvimgcodec_parser_->is_valid()) {
        throw std::runtime_error("Cannot construct IFDs: nvImageCodec parser not available");
    }
    
    ifd_offsets_.clear();
    ifds_.clear();
    
    uint32_t ifd_count = nvimgcodec_parser_->get_ifd_count();
    #ifdef DEBUG
    fmt::print("📋 Constructing {} IFDs from nvImageCodec metadata\n", ifd_count);
    #endif // DEBUG
    
    ifd_offsets_.reserve(ifd_count);
    ifds_.reserve(ifd_count);
    
    for (uint32_t ifd_index = 0; ifd_index < ifd_count; ++ifd_index) {
        try {
#ifdef CUCIM_HAS_NVIMGCODEC
            // Get IFD metadata from TiffFileParser
            const auto& ifd_info = nvimgcodec_parser_->get_ifd(ifd_index);

            // Use IFD index as pseudo-offset (actual file offset not needed)
            ifd_offsets_.push_back(ifd_index);

            // Create IFD from nvImageCodec metadata using NEW constructor
            auto ifd = std::make_shared<cuslide::tiff::IFD>(this, ifd_index, ifd_info);
            ifds_.emplace_back(std::move(ifd));

            #ifdef DEBUG
            fmt::print("  ✅ IFD[{}]: {}x{}, {} channels, codec: {}\n",
                      ifd_index, ifd_info.width, ifd_info.height,
                      ifd_info.num_channels, ifd_info.codec);
            #endif // DEBUG
#else
            (void)ifd_index;
            throw std::runtime_error("cuslide2 requires nvImageCodec for IFD parsing");
#endif // CUCIM_HAS_NVIMGCODEC

        } catch (const std::exception& e) {
            #ifdef DEBUG
            fmt::print("  ⚠️  Failed to create IFD[{}]: {}\n", ifd_index, e.what());
            #endif // DEBUG
            // Continue with other IFDs - some may be corrupted
        }
    }
    
    if (ifds_.empty()) {
        throw std::runtime_error("No valid IFDs found in TIFF file");
    }
    
    #ifdef DEBUG
    fmt::print("✅ Successfully created {} out of {} IFDs\n", ifds_.size(), ifd_count);
    #endif // DEBUG
    
    // Initialize level-to-IFD mapping (will be updated by resolve_vendor_format)
    level_to_ifd_idx_.clear();
    level_to_ifd_idx_.reserve(ifds_.size());
    for (size_t index = 0; index < ifds_.size(); ++index) {
        level_to_ifd_idx_.emplace_back(index);
    }
    
    // Detect vendor format and classify IFDs
    resolve_vendor_format();
    
    // Sort resolution levels by size (largest first)
    std::sort(level_to_ifd_idx_.begin(), level_to_ifd_idx_.end(), 
             [this](const size_t& a, const size_t& b) {
        uint32_t width_a = this->ifds_[a]->width();
        uint32_t width_b = this->ifds_[b]->width();
        if (width_a != width_b) {
            return width_a > width_b;
        }
        return this->ifds_[a]->height() > this->ifds_[b]->height();
    });
    
    #ifdef DEBUG
    fmt::print("✅ TIFF initialization complete: {} levels, {} associated images\n",
              level_to_ifd_idx_.size(), associated_images_.size());
    #endif // DEBUG
}
void TIFF::resolve_vendor_format()
{
    PROF_SCOPED_RANGE(PROF_EVENT(tiff_resolve_vendor_format));
    uint16_t ifd_count = ifds_.size();
    if (ifd_count == 0)
    {
        return;
    }
    json* json_metadata = reinterpret_cast<json*>(metadata_);

    auto& first_ifd = ifds_[0];
    std::string& model = first_ifd->model();
    std::string& software = first_ifd->software();
    const uint16_t resolution_unit = first_ifd->resolution_unit();
    const float x_resolution = first_ifd->x_resolution();
    const float y_resolution = first_ifd->y_resolution();

    // Detect Aperio SVS format
    {
        bool is_aperio = false;
        
        // Method 1: Check ImageDescription starts with "Aperio "
        auto& image_desc = first_ifd->image_description();
        std::string_view prefix("Aperio ");
        auto res = std::mismatch(prefix.begin(), prefix.end(), image_desc.begin());
        if (res.first == prefix.end())
        {
            is_aperio = true;
        }
        
        // Method 2: Check metadata_blobs for Aperio (kind=1)
        // This includes the workaround for nvImageCodec 0.6.0 misclassifying Aperio as Leica
        if (!is_aperio && nvimgcodec_parser_)
        {
            const auto& metadata_blobs = nvimgcodec_parser_->get_metadata_blobs(0);
            if (metadata_blobs.find(1) != metadata_blobs.end())  // MED_APERIO = 1
            {
                is_aperio = true;
                #ifdef DEBUG
                fmt::print("✅ Aperio detected via metadata_blobs workaround\n");
                #endif
            }
        }
        
        if (is_aperio)
        {
            _populate_aperio_svs_metadata(ifd_count, json_metadata, first_ifd);
        }
    }

    // Detect Philips TIFF
    // NOTE: nvImageCodec 0.6.0 doesn't expose individual TIFF tags (like SOFTWARE)
    // Workaround: Check for Philips XML in ImageDescription or use nvImageCodec metadata kind
    {
        bool is_philips = false;
        
        // Method 1: Check SOFTWARE tag (available in nvImageCodec 0.7.0+)
        std::string_view prefix("Philips");
        auto res = std::mismatch(prefix.begin(), prefix.end(), software.begin());
        if (res.first == prefix.end())
        {
            is_philips = true;
        }
        
        // Method 2: Check for Philips XML structure in ImageDescription
        // (Workaround for nvImageCodec 0.6.0 where SOFTWARE tag is not available)
        if (!is_philips)
        {
            auto& image_desc = first_ifd->image_description();
            if (image_desc.find("<DataObject") != std::string::npos &&
                image_desc.find("ObjectType=\"DPUfsImport\"") != std::string::npos)
            {
                is_philips = true;
            }
        }
        
        // Method 3: Check metadata_blobs for Philips (kind=2)
        // This includes the workaround for nvImageCodec 0.6.0 misclassifying Philips as Ventana
        if (!is_philips && nvimgcodec_parser_)
        {
            const auto& metadata_blobs = nvimgcodec_parser_->get_metadata_blobs(0);
            if (metadata_blobs.find(2) != metadata_blobs.end())  // MED_PHILIPS = 2
            {
                is_philips = true;
                #ifdef DEBUG
                fmt::print("✅ Philips detected via metadata_blobs workaround\n");
                #endif
            }
        }
        
        // Method 4: Check if nvImageCodec detected it as Philips (format string)
        if (!is_philips && nvimgcodec_parser_)
        {
            std::string detected_format = nvimgcodec_parser_->get_detected_format();
            if (detected_format.find("Philips") != std::string::npos)
            {
                is_philips = true;
            }
        }
        
        if (is_philips)
        {
            _populate_philips_tiff_metadata(ifd_count, json_metadata, first_ifd);
        }
    }

    // Append TIFF metadata
    if (json_metadata)
    {
        json tiff_metadata;

        tiff_metadata.emplace("model", model);
        tiff_metadata.emplace("software", software);
        switch (resolution_unit)
        {
        case 2:
            tiff_metadata.emplace("resolution_unit", "inch");
            break;
        case 3:
            tiff_metadata.emplace("resolution_unit", "centimeter");
            break;
        default:
            tiff_metadata.emplace("resolution_unit", "");
            break;
        }
        tiff_metadata.emplace("x_resolution", x_resolution);
        tiff_metadata.emplace("y_resolution", y_resolution);

        (*json_metadata).emplace("tiff", std::move(tiff_metadata));
    }
}

void TIFF::_populate_philips_tiff_metadata(uint16_t ifd_count, void* metadata, std::shared_ptr<IFD>& first_ifd)
{
    json* json_metadata = reinterpret_cast<json*>(metadata);
    std::string_view macro_prefix("Macro");
    std::string_view label_prefix("Label");

    pugi::xml_document doc;
    const char* image_desc_cstr = first_ifd->image_description().c_str();
    pugi::xml_parse_result result = doc.load_string(image_desc_cstr);
    if (result)
    {
        const auto& data_object = doc.child("DataObject");
        if (std::string_view(data_object.attribute("ObjectType").as_string("")) != "DPUfsImport")
        {
            #ifdef DEBUG
            fmt::print(
                stderr,
                "[Warning] Failed to read as Philips TIFF. It looks like Philips TIFF but the image description of the first IFD doesn't have '<DataObject ObjectType=\"DPUfsImport\">' node!\n");
            #endif // DEBUG
            return;
        }

        pugi::xpath_query PIM_DP_IMAGE_TYPE(
            "Attribute[@Name='PIM_DP_SCANNED_IMAGES']/Array/DataObject[Attribute/@Name='PIM_DP_IMAGE_TYPE' and Attribute/text()='WSI']");
        pugi::xpath_node_set wsi_nodes = PIM_DP_IMAGE_TYPE.evaluate_node_set(data_object);
        if (wsi_nodes.size() != 1)
        {
            #ifdef DEBUG
            fmt::print(
                stderr,
                "[Warning] Failed to read as Philips TIFF. Expected only one 'DPScannedImage' node with PIM_DP_IMAGE_TYPE='WSI'.\n");
            #endif // DEBUG
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
                #ifdef DEBUG
                fmt::print(stderr, "[Warning] Failed to read DICOM_PIXEL_SPACING: {}\n", values);
                #endif // DEBUG
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
            
            // Check if this IFD is an associated image (macro/label) based on ImageDescription
            // NOTE: In Philips TIFF, pyramid levels can be strip-based (tile_width==0)
            // So we can't use tile_width to identify associated images
            auto& image_desc = ifd->image_description();
            bool is_macro = (std::mismatch(macro_prefix.begin(), macro_prefix.end(), image_desc.begin()).first == macro_prefix.end());
            bool is_label = (std::mismatch(label_prefix.begin(), label_prefix.end(), image_desc.begin()).first == label_prefix.end());
            
            if (is_macro || is_label)
            {
                // This is an associated image - add to associated_images_ map
                AssociatedImageBufferDesc buf_desc{};
                buf_desc.type = AssociatedImageBufferType::IFD;
                buf_desc.compression = static_cast<cucim::codec::CompressionMethod>(ifd->compression());
                buf_desc.ifd_index = index;

                if (is_macro)
                {
                    associated_images_.emplace("macro", buf_desc);
                }
                else if (is_label)
                {
                    associated_images_.emplace("label", buf_desc);
                }

                // Remove item at index `ifd_index` from `level_to_ifd_idx_`
                level_to_ifd_idx_.erase(level_to_ifd_idx_.begin() + level_index);
                --level_index;
                continue;
            }
            
            // This is a pyramid level - calculate downsample and fix dimensions
            if (spacing_index < pixel_spacings.size())
            {
                double downsample = std::round((pixel_spacings[spacing_index].first / spacing_x_l0 +
                                                pixel_spacings[spacing_index].second / spacing_y_l0) /
                                               2);
                // Fix width and height of IFD
                ifd->width_ = width_l0 / downsample;
                ifd->height_ = height_l0 / downsample;
                ++spacing_index;
            }
            else
            {
                // No pixel spacing metadata for this level - calculate from actual dimensions
                #ifdef DEBUG
                fmt::print("  ℹ️  No DICOM_PIXEL_SPACING for IFD[{}], using actual dimensions\n", index);
                #endif
                // Keep the actual dimensions from nvImageCodec
            }
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
            pugi::xpath_node associated_node = ASSOCIATED_IMAGES[associated_image_type_idx].evaluate_node(data_object);
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

void TIFF::_populate_aperio_svs_metadata(uint16_t ifd_count, void* metadata, std::shared_ptr<IFD>& first_ifd)
{
    (void)ifd_count;
    (void)metadata;
    (void)first_ifd;
    json* json_metadata = reinterpret_cast<json*>(metadata);
    (void)json_metadata;

    int32_t non_tile_image_count = 0;

    // Append associated images
    // NOTE: For Aperio SVS, associated images are identified by SubfileType:
    //   - SubfileType=0 at index 1: thumbnail (reduced resolution copy)
    //   - SubfileType=1: label image
    //   - SubfileType=9: macro image
    // Pyramid levels typically have SubfileType=0 and are tiled, but may be strip-based
    for (int index = 1, level_index = 1; index < ifd_count; ++index, ++level_index)
    {
        auto& ifd = ifds_[index];
        uint64_t subfile_type = ifd->subfile_type();
        
        // Check if this is an associated image based on SubfileType
        bool is_associated = false;
        std::string associated_name;
        
        if (index == 1 && subfile_type == 0 && ifd->tile_width() == 0)
        {
            // First non-main IFD with SubfileType=0 and strip-based: likely thumbnail
            is_associated = true;
            associated_name = "thumbnail";
        }
        else if (subfile_type == 1)
        {
            // SubfileType=1: label image
            is_associated = true;
            associated_name = "label";
        }
        else if (subfile_type == 9)
        {
            // SubfileType=9: macro image
            is_associated = true;
            associated_name = "macro";
        }
        
        if (is_associated)
        {
            ++non_tile_image_count;
            AssociatedImageBufferDesc buf_desc{};
            buf_desc.type = AssociatedImageBufferType::IFD;
            buf_desc.compression = static_cast<cucim::codec::CompressionMethod>(ifd->compression());
            buf_desc.ifd_index = index;
            associated_images_.emplace(associated_name, buf_desc);
            
            // Remove from pyramid levels
            level_to_ifd_idx_.erase(level_to_ifd_idx_.begin() + level_index);
            --level_index;
            continue;
        }
        // If not associated, keep as pyramid level (even if strip-based)
    }

    // Set TIFF type
    tiff_type_ = TiffType::Aperio;

    // Set background color
    background_value_ = 0xFF;

    // Get metadata
    if (json_metadata)
    {
        json aperio_metadata;
        parse_aperio_svs_metadata(first_ifd, aperio_metadata);
        (*json_metadata).emplace("aperio", std::move(aperio_metadata));
    }
}

bool TIFF::read(const cucim::io::format::ImageMetadataDesc* metadata,
                const cucim::io::format::ImageReaderRegionRequestDesc* request,
                cucim::io::format::ImageDataDesc* out_image_data,
                cucim::io::format::ImageMetadataDesc* out_metadata)
{
    PROF_SCOPED_RANGE(PROF_EVENT(tiff_read));
    if (request->associated_image_name)
    {
        // 'out_metadata' is only needed for reading associated image
        return read_associated_image(metadata, request, out_image_data, out_metadata);
    }

    const int32_t ndim = request->size_ndim;
    const uint64_t location_len = request->location_len;

    if (request->level >= level_to_ifd_idx_.size())
    {
        throw std::invalid_argument(fmt::format(
            "Invalid level ({}) in the request! (Should be < {})", request->level, level_to_ifd_idx_.size()));
    }
    auto main_ifd = ifds_[level_to_ifd_idx_[0]];
    auto ifd = ifds_[level_to_ifd_idx_[request->level]];
    auto original_img_width = main_ifd->width();
    auto original_img_height = main_ifd->height();

    for (int32_t i = 0; i < ndim; ++i)
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
    for (int64_t i = ndim * location_len - 1; i >= 0; --i)
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
    PROF_SCOPED_RANGE(PROF_EVENT(tiff_read_associated_image));
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

            width = image_ifd->width_;
            height = image_ifd->height_;
            samples_per_pixel = image_ifd->samples_per_pixel_;
            raster_size = width * height * samples_per_pixel;

            uint16_t compression_method = image_ifd->compression_;

            if (compression_method != COMPRESSION_JPEG && compression_method != COMPRESSION_LZW)
            {
                #ifdef DEBUG
                fmt::print(stderr,
                           "[Error] Unsupported compression method in read_associated_image()! (compression: {})\n",
                           compression_method);
                #endif // DEBUG
                return false;
            }

            // REMOVED: Legacy CPU decoder code for strips
            // In a pure nvImageCodec build, associated images should use nvImageCodec decoding
            // This legacy strip-based decoding path should not be used
            throw std::runtime_error(fmt::format(
                "INTERNAL ERROR: Legacy strip-based CPU decoder path reached. "
                "This should not happen in nvImageCodec build. "
                "Compression method: {}, IFD index: {}. "
                "Associated images should be decoded via nvImageCodec.",
                compression_method, buf_desc.desc_ifd_index));
            break;
        }
        case AssociatedImageBufferType::IFD_IMAGE_DESC: {
            // REMOVED: Legacy CPU decoder code for base64-encoded JPEG in ImageDescription
            // In a pure nvImageCodec build, this path should not be used
            // Base64-encoded images in metadata should be decoded via nvImageCodec
            throw std::runtime_error(
                "INTERNAL ERROR: Legacy IFD_IMAGE_DESC CPU decoder path reached. "
                "This should not happen in nvImageCodec build. "
                "Base64-encoded associated images should be decoded via nvImageCodec.");
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

cucim::filesystem::Path TIFF::file_path() const
{
    return file_path_;
}

std::shared_ptr<CuCIMFileHandle>& TIFF::file_handle()
{
    return file_handle_shared_;
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
