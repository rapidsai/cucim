/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "nvimgcodec_tiff_parser.h"

#ifdef CUCIM_HAS_NVIMGCODEC
#include <nvimgcodec.h>
#include <cuda_runtime.h>
#endif

#include <fmt/format.h>
#include <stdexcept>
#include <cstring>
#include <algorithm>

namespace cuslide2::nvimgcodec
{

#ifdef CUCIM_HAS_NVIMGCODEC

// ============================================================================
// IfdInfo Implementation
// ============================================================================

void IfdInfo::print() const
{
    fmt::print("  IFD[{}]: {}x{}, {} channels, {} bits/sample, codec: {}\n",
               index, width, height, num_channels, bits_per_sample, codec);
}

// ============================================================================
// NvImageCodecTiffParserManager Implementation
// ============================================================================

NvImageCodecTiffParserManager::NvImageCodecTiffParserManager() 
    : instance_(nullptr), decoder_(nullptr), initialized_(false)
{
    try
    {
        // Create nvImageCodec instance for TIFF parsing (separate from decoder instance)
        nvimgcodecInstanceCreateInfo_t create_info{};
        create_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.struct_size = sizeof(nvimgcodecInstanceCreateInfo_t);
        create_info.struct_next = nullptr;
        create_info.load_builtin_modules = 1;       // Load JPEG, PNG, etc.
        create_info.load_extension_modules = 1;     // Load JPEG2K, TIFF, etc.
        create_info.extension_modules_path = nullptr;
        create_info.create_debug_messenger = 0;     // Disable debug for TIFF parser
        create_info.debug_messenger_desc = nullptr;
        create_info.message_severity = 0;
        create_info.message_category = 0;
        
        nvimgcodecStatus_t status = nvimgcodecInstanceCreate(&instance_, &create_info);
        
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            status_message_ = fmt::format("Failed to create nvImageCodec instance for TIFF parsing (status: {})", 
                                         static_cast<int>(status));
            fmt::print("⚠️  {}\n", status_message_);
            return;
        }
        
        // Create decoder for metadata extraction (not for image decoding)
        // This decoder is used exclusively for nvimgcodecDecoderGetMetadata() calls
        nvimgcodecExecutionParams_t exec_params{};
        exec_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS;
        exec_params.struct_size = sizeof(nvimgcodecExecutionParams_t);
        exec_params.struct_next = nullptr;
        exec_params.device_allocator = nullptr;
        exec_params.pinned_allocator = nullptr;
        exec_params.max_num_cpu_threads = 0;
        exec_params.executor = nullptr;
        exec_params.device_id = NVIMGCODEC_DEVICE_CPU_ONLY;  // CPU-only for metadata extraction
        exec_params.pre_init = 0;
        exec_params.skip_pre_sync = 0;
        exec_params.num_backends = 0;
        exec_params.backends = nullptr;
        
        status = nvimgcodecDecoderCreate(instance_, &decoder_, &exec_params, nullptr);
        
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            nvimgcodecInstanceDestroy(instance_);
            instance_ = nullptr;
            status_message_ = fmt::format("Failed to create decoder for metadata extraction (status: {})", 
                                         static_cast<int>(status));
            fmt::print("⚠️  {}\n", status_message_);
            return;
        }
        
        initialized_ = true;
        status_message_ = "nvImageCodec TIFF parser initialized successfully (with metadata extraction support)";
        fmt::print("✅ {}\n", status_message_);
    }
    catch (const std::exception& e)
    {
        status_message_ = fmt::format("nvImageCodec TIFF parser initialization exception: {}", e.what());
        fmt::print("❌ {}\n", status_message_);
        initialized_ = false;
    }
}

NvImageCodecTiffParserManager::~NvImageCodecTiffParserManager()
{
    if (decoder_)
    {
        nvimgcodecDecoderDestroy(decoder_);
        decoder_ = nullptr;
    }
    
    if (instance_)
    {
        nvimgcodecInstanceDestroy(instance_);
        instance_ = nullptr;
    }
}

// ============================================================================
// TiffFileParser Implementation
// ============================================================================

TiffFileParser::TiffFileParser(const std::string& file_path)
    : file_path_(file_path), initialized_(false), 
      main_code_stream_(nullptr)
{
    auto& manager = NvImageCodecTiffParserManager::instance();
    
    if (!manager.is_available())
    {
        throw std::runtime_error(fmt::format("nvImageCodec not available: {}", 
                                            manager.get_status()));
    }
    
    try
    {
        // Step 1: Create code stream from TIFF file
        nvimgcodecStatus_t status = nvimgcodecCodeStreamCreateFromFile(
            manager.get_instance(),
            &main_code_stream_,
            file_path.c_str()
        );
        
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            throw std::runtime_error(fmt::format("Failed to create code stream from file: {} (status: {})",
                                                file_path, static_cast<int>(status)));
        }
        
        fmt::print("✅ Opened TIFF file: {}\n", file_path);
        
        // Step 2: Parse TIFF structure (metadata only)
        parse_tiff_structure();
        
        initialized_ = true;
        fmt::print("✅ TIFF parser initialized with {} IFDs\n", ifd_infos_.size());
    }
    catch (const std::exception& e)
    {
        // Cleanup on error
        if (main_code_stream_)
        {
            nvimgcodecCodeStreamDestroy(main_code_stream_);
            main_code_stream_ = nullptr;
        }
        
        throw;  // Re-throw
    }
}

TiffFileParser::~TiffFileParser()
{
    // IfdInfo destructors will destroy sub-code streams
    ifd_infos_.clear();
    
    // Destroy main code stream
    if (main_code_stream_)
    {
        nvimgcodecCodeStreamDestroy(main_code_stream_);
        main_code_stream_ = nullptr;
    }
}

void TiffFileParser::parse_tiff_structure()
{
    // Get TIFF structure information
    nvimgcodecCodeStreamInfo_t stream_info{};
    stream_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO;
    stream_info.struct_size = sizeof(nvimgcodecCodeStreamInfo_t);
    stream_info.struct_next = nullptr;
    
    nvimgcodecStatus_t status = nvimgcodecCodeStreamGetCodeStreamInfo(
        main_code_stream_, &stream_info);
    
    if (status != NVIMGCODEC_STATUS_SUCCESS)
    {
        throw std::runtime_error(fmt::format("Failed to get code stream info (status: {})",
                                            static_cast<int>(status)));
    }
    
    uint32_t num_ifds = stream_info.num_images;
    fmt::print("  TIFF has {} IFDs (resolution levels)\n", num_ifds);
    
    if (stream_info.codec_name)
    {
        fmt::print("  Codec: {}\n", stream_info.codec_name);
    }
    
    // Get information for each IFD
    for (uint32_t i = 0; i < num_ifds; ++i)
    {
        IfdInfo ifd_info;
        ifd_info.index = i;
        
        // Create view for this IFD
        nvimgcodecCodeStreamView_t view{};
        view.struct_type = NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_VIEW;
        view.struct_size = sizeof(nvimgcodecCodeStreamView_t);
        view.struct_next = nullptr;
        view.image_idx = i;  // Note: nvImageCodec uses 'image_idx' not 'image_index'
        
        // Get sub-code stream for this IFD
        status = nvimgcodecCodeStreamGetSubCodeStream(main_code_stream_,
                                                      &ifd_info.sub_code_stream,
                                                      &view);
        
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            fmt::print("⚠️  Failed to get sub-code stream for IFD {} (status: {})\n", 
                      i, static_cast<int>(status));
            continue;
        }
        
        // Get image information for this IFD
        nvimgcodecImageInfo_t image_info{};
        image_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
        image_info.struct_size = sizeof(nvimgcodecImageInfo_t);
        image_info.struct_next = nullptr;
        
        status = nvimgcodecCodeStreamGetImageInfo(ifd_info.sub_code_stream, &image_info);
        
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            fmt::print("⚠️  Failed to get image info for IFD {} (status: {})\n",
                      i, static_cast<int>(status));
            continue;
        }
        
        // Extract IFD metadata
        ifd_info.width = image_info.plane_info[0].width;
        ifd_info.height = image_info.plane_info[0].height;
        ifd_info.num_channels = image_info.num_planes;
        
        // Extract bits per sample from sample type
        auto sample_type = image_info.plane_info[0].sample_type;
        // sample_type encoding: bits = (type >> 11) & 0xFF
        ifd_info.bits_per_sample = static_cast<unsigned int>(sample_type) >> (8+3);
        
        if (image_info.codec_name)
        {
            ifd_info.codec = image_info.codec_name;
        }
        
        // Extract metadata for this IFD using nvimgcodecDecoderGetMetadata
        extract_ifd_metadata(ifd_info);
        
        ifd_info.print();
        
        ifd_infos_.push_back(std::move(ifd_info));
    }
}

void TiffFileParser::extract_ifd_metadata(IfdInfo& ifd_info)
{
    auto& manager = NvImageCodecTiffParserManager::instance();
    
    if (!manager.get_decoder() || !ifd_info.sub_code_stream)
    {
        return;  // No decoder or stream available
    }
    
    // Step 1: Get metadata count (first call with nullptr)
    int metadata_count = 0;
    nvimgcodecStatus_t status = nvimgcodecDecoderGetMetadata(
        manager.get_decoder(),
        ifd_info.sub_code_stream,
        nullptr,  // First call: get count only
        &metadata_count
    );
    
    if (status != NVIMGCODEC_STATUS_SUCCESS || metadata_count == 0)
    {
        return;  // No metadata or error
    }
    
    fmt::print("  Found {} metadata entries for IFD[{}]\n", metadata_count, ifd_info.index);
    
    // Step 2: Allocate array for metadata pointers
    std::vector<nvimgcodecMetadata_t*> metadata_ptrs(metadata_count, nullptr);
    
    // Step 3: Get actual metadata
    status = nvimgcodecDecoderGetMetadata(
        manager.get_decoder(),
        ifd_info.sub_code_stream,
        metadata_ptrs.data(),
        &metadata_count
    );
    
    if (status != NVIMGCODEC_STATUS_SUCCESS)
    {
        fmt::print("⚠️  Failed to retrieve metadata for IFD[{}] (status: {})\n",
                  ifd_info.index, static_cast<int>(status));
        return;
    }
    
    // Step 4: Process each metadata entry
    for (int j = 0; j < metadata_count; ++j)
    {
        if (!metadata_ptrs[j])
            continue;
        
        nvimgcodecMetadata_t* metadata = metadata_ptrs[j];
        
        // Extract metadata fields
        int kind = metadata->kind;
        int format = metadata->format;
        size_t buffer_size = metadata->buffer_size;
        const uint8_t* buffer = static_cast<const uint8_t*>(metadata->buffer);
        
        fmt::print("    Metadata[{}]: kind={}, format={}, size={}\n",
                  j, kind, format, buffer_size);
        
        // Store in metadata_blobs map
        if (buffer && buffer_size > 0)
        {
            IfdInfo::MetadataBlob blob;
            blob.format = format;
            blob.data.assign(buffer, buffer + buffer_size);
            ifd_info.metadata_blobs[kind] = std::move(blob);
            
            // Special handling: extract ImageDescription if it's a text format
            // nvimgcodecMetadataFormat_t: RAW=0, XML=1, JSON=2, etc.
            // For RAW format, treat as text if it looks like ASCII
            if (kind == 1 && ifd_info.image_description.empty())  // MED_APERIO = 1
            {
                // Aperio metadata is typically in RAW format as text
                ifd_info.image_description.assign(buffer, buffer + buffer_size);
            }
            else if (kind == 2)  // MED_PHILIPS = 2
            {
                // Philips metadata is typically XML
                ifd_info.image_description.assign(buffer, buffer + buffer_size);
            }
        }
    }
}

const IfdInfo& TiffFileParser::get_ifd(uint32_t index) const
{
    if (index >= ifd_infos_.size())
    {
        throw std::out_of_range(fmt::format("IFD index {} out of range (have {} IFDs)",
                                           index, ifd_infos_.size()));
    }
    return ifd_infos_[index];
}

ImageType TiffFileParser::classify_ifd(uint32_t ifd_index) const
{
    if (ifd_index >= ifd_infos_.size())
    {
        return ImageType::UNKNOWN;
    }
    
    const auto& ifd = ifd_infos_[ifd_index];
    const std::string& desc = ifd.image_description;
    
    // Aperio SVS classification based on ImageDescription keywords
    // Reference: https://docs.nvidia.com/cuda/nvimagecodec/samples/metadata.html
    // 
    // Examples from official nvImageCodec metadata sample:
    //   Label:     "Aperio Image Library v10.0.50\nlabel 415x422"
    //   Macro:     "Aperio Image Library v10.0.50\nmacro 1280x421"
    //   Thumbnail: "Aperio Image Library v10.0.50\n15374x17497 -> 674x768 - |..."
    //   Level:     "Aperio Image Library v10.0.50\n16000x17597 [0,100 15374x17497] (256x256) J2K/YUV16..."
    
    if (!desc.empty())
    {
        // Convert to lowercase for case-insensitive matching
        std::string desc_lower = desc;
        std::transform(desc_lower.begin(), desc_lower.end(), desc_lower.begin(),
                      [](unsigned char c){ return std::tolower(c); });
        
        // Check for explicit keywords
        if (desc_lower.find("label ") != std::string::npos || 
            desc_lower.find("\nlabel ") != std::string::npos)
        {
            return ImageType::LABEL;
        }
        
        if (desc_lower.find("macro ") != std::string::npos || 
            desc_lower.find("\nmacro ") != std::string::npos)
        {
            return ImageType::MACRO;
        }
        
        // Aperio thumbnail has dimension transformation: "WxH -> WxH"
        if (desc.find(" -> ") != std::string::npos && desc.find(" - ") != std::string::npos)
        {
            return ImageType::THUMBNAIL;
        }
    }
    
    // Fallback heuristics for formats without clear keywords
    // Small images are likely associated images
    if (ifd.width < 2000 && ifd.height < 2000)
    {
        // Convention: Second IFD (index 1) is often thumbnail
        if (ifd_index == 1)
        {
            return ImageType::THUMBNAIL;
        }
        
        // If description exists but no keywords matched, it's still likely associated
        if (!desc.empty())
        {
            return ImageType::UNKNOWN;  // Has description but can't classify
        }
    }
    
    // IFD 0 is always main resolution level
    if (ifd_index == 0)
    {
        return ImageType::RESOLUTION_LEVEL;
    }
    
    // Large images are resolution levels
    if (ifd.width >= 2000 || ifd.height >= 2000)
    {
        return ImageType::RESOLUTION_LEVEL;
    }
    
    return ImageType::UNKNOWN;
}

std::vector<uint32_t> TiffFileParser::get_resolution_levels() const
{
    std::vector<uint32_t> levels;
    
    for (const auto& ifd : ifd_infos_)
    {
        if (classify_ifd(ifd.index) == ImageType::RESOLUTION_LEVEL)
        {
            levels.push_back(ifd.index);
        }
    }
    
    return levels;
}

std::map<std::string, uint32_t> TiffFileParser::get_associated_images() const
{
    std::map<std::string, uint32_t> associated;
    
    for (const auto& ifd : ifd_infos_)
    {
        auto type = classify_ifd(ifd.index);
        switch (type)
        {
            case ImageType::THUMBNAIL:
                associated["thumbnail"] = ifd.index;
                break;
            case ImageType::LABEL:
                associated["label"] = ifd.index;
                break;
            case ImageType::MACRO:
                associated["macro"] = ifd.index;
                break;
            default:
                break;
        }
    }
    
    return associated;
}

void TiffFileParser::override_ifd_dimensions(uint32_t ifd_index, 
                                             uint32_t width, 
                                             uint32_t height)
{
    if (ifd_index >= ifd_infos_.size())
    {
        throw std::out_of_range(fmt::format("IFD index {} out of range (have {} IFDs)",
                                           ifd_index, ifd_infos_.size()));
    }
    
    auto& ifd = ifd_infos_[ifd_index];
    fmt::print("⚙️  Overriding IFD[{}] dimensions: {}x{} -> {}x{}\n",
              ifd_index, ifd.width, ifd.height, width, height);
    
    ifd.width = width;
    ifd.height = height;
}

std::string TiffFileParser::get_image_description(uint32_t ifd_index) const
{
    if (ifd_index >= ifd_infos_.size())
    {
        return "";
    }
    
    const auto& ifd = ifd_infos_[ifd_index];
    return ifd.image_description;
}

void TiffFileParser::print_info() const
{
    fmt::print("\nTIFF File Information:\n");
    fmt::print("  File: {}\n", file_path_);
    fmt::print("  Number of IFDs: {}\n", ifd_infos_.size());
    fmt::print("\nIFD Details:\n");
    
    for (const auto& ifd : ifd_infos_)
    {
        ifd.print();
    }
}

#endif // CUCIM_HAS_NVIMGCODEC

} // namespace cuslide2::nvimgcodec

