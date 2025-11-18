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
#include "nvimgcodec_manager.h"

#include <tiffio.h>
#include <cstring>  // for strlen

#ifdef CUCIM_HAS_NVIMGCODEC
#include <nvimgcodec.h>
#include <cuda_runtime.h>
#endif

#include <fmt/format.h>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <mutex>

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
    // Destroy sub-code streams first
    for (auto& ifd_info : ifd_infos_)
    {
        if (ifd_info.sub_code_stream)
        {
            nvimgcodecCodeStreamDestroy(ifd_info.sub_code_stream);
            ifd_info.sub_code_stream = nullptr;
        }
    }
    
    // Then destroy main code stream
    if (main_code_stream_)
    {
        nvimgcodecCodeStreamDestroy(main_code_stream_);
        main_code_stream_ = nullptr;
    }
    
    ifd_infos_.clear();
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
            fmt::print("❌ Failed to get sub-code stream for IFD {} (status: {})\n", 
                      i, static_cast<int>(status));
            fmt::print("   This IFD will be SKIPPED and cannot be decoded.\n");
            // Set sub_code_stream to nullptr explicitly to mark as invalid
            ifd_info.sub_code_stream = nullptr;
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
            fmt::print("❌ Failed to get image info for IFD {} (status: {})\n",
                      i, static_cast<int>(status));
            fmt::print("   This IFD will be SKIPPED and cannot be decoded.\n");
            // Clean up the sub_code_stream before continuing
            if (ifd_info.sub_code_stream)
            {
                nvimgcodecCodeStreamDestroy(ifd_info.sub_code_stream);
                ifd_info.sub_code_stream = nullptr;
            }
            continue;
        }
        
        // Extract IFD metadata
        ifd_info.width = image_info.plane_info[0].width;
        ifd_info.height = image_info.plane_info[0].height;
        ifd_info.num_channels = image_info.num_planes;
        
        // Extract bits per sample from sample type
        // sample_type encoding: bytes_per_element = (type >> 11) & 0xFF
        // Convert bytes to bits
        auto sample_type = image_info.plane_info[0].sample_type;
        int bytes_per_element = (static_cast<unsigned int>(sample_type) >> 11) & 0xFF;
        ifd_info.bits_per_sample = bytes_per_element * 8;  // Convert bytes to bits
        
        if (image_info.codec_name)
        {
            ifd_info.codec = image_info.codec_name;
        }
        
        // Extract metadata for this IFD using two complementary approaches:
        // 
        // 1. extract_ifd_metadata() - Uses nvImageCodec for vendor-specific metadata
        //    - Extracts Aperio, Philips, Leica metadata that libtiff doesn't support
        //    - Uses nvimgcodecDecoderGetMetadata() API
        // 
        // 2. extract_tiff_tags() - Uses libtiff for standard TIFF tags
        //    - Extracts standard TIFF tags (JPEGTables, Compression, ImageDescription)
        //    - Uses libtiff directly due to nvTIFF 0.6.0.77 metadata API limitations
        //    - Once nvTIFF metadata API is fixed, we can consolidate to nvImageCodec only
        // 
        // Both are needed because neither library alone provides complete metadata coverage.
        extract_ifd_metadata(ifd_info);  // Vendor metadata via nvImageCodec
        extract_tiff_tags(ifd_info);      // TIFF tags via libtiff
        
        ifd_info.print();
        
        ifd_infos_.push_back(std::move(ifd_info));
    }
    
    // Report parsing results
    if (ifd_infos_.size() == num_ifds)
    {
        fmt::print("✅ TIFF parser initialized with {} IFDs (all successful)\n", ifd_infos_.size());
    }
    else
    {
        fmt::print("⚠️  TIFF parser initialized with {} IFDs ({} out of {} total)\n", 
                  ifd_infos_.size(), ifd_infos_.size(), num_ifds);
        fmt::print("   {} IFDs were skipped due to parsing errors\n", num_ifds - ifd_infos_.size());
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

// ============================================================================
// nvImageCodec 0.7.0 Features Implementation
// ============================================================================

void TiffFileParser::extract_tiff_tags(IfdInfo& ifd_info)
{
    auto& manager = NvImageCodecTiffParserManager::instance();
    
    if (!manager.get_decoder())
    {
        fmt::print("  ⚠️  Cannot extract TIFF tags: decoder not available\n");
        return;
    }
    
    if (!ifd_info.sub_code_stream)
    {
        fmt::print("  ⚠️  Cannot extract TIFF tags: sub_code_stream is null\n");
        return;
    }
    
    // Map of TIFF tag IDs to names for common tags
    std::map<uint32_t, std::string> tiff_tag_names = {
        {254, "SUBFILETYPE"},      // Image type classification
        {256, "ImageWidth"},
        {257, "ImageLength"},
        {258, "BitsPerSample"},
        {259, "Compression"},
        {262, "PhotometricInterpretation"},
        {270, "ImageDescription"}, // Vendor metadata
        {271, "Make"},             // Scanner manufacturer
        {272, "Model"},            // Scanner model
        {305, "Software"},
        {306, "DateTime"},
        {322, "TileWidth"},
        {323, "TileLength"},
        {339, "SampleFormat"},
        {347, "JPEGTables"}        // Shared JPEG tables
    };
    
    fmt::print("  Extracting TIFF tags for IFD[{}]...\n", ifd_info.index);
    
    // NOTE: nvTIFF 0.6.0.77 metadata API is incompatible with our code
    // Skip nvImageCodec metadata extraction and use libtiff directly
    fmt::print("  ℹ️  Using libtiff for TIFF tag extraction (nvTIFF 0.6.0.77 compatibility)\n");
    
    // Use libtiff to extract TIFF tags directly
    int tiff_tag_count = 0;
    bool has_jpeg_tables = false;
    
    // Open TIFF file with libtiff to check for JPEGTables
    TIFF* tif = TIFFOpen(file_path_.c_str(), "r");
    if (tif)
    {
        // Set the directory to the IFD we're interested in
        if (TIFFSetDirectory(tif, ifd_info.index))
        {
            // Check for TIFFTAG_JPEGTABLES (tag 347)
            uint32_t jpegtables_count = 0;
            const void* jpegtables_data = nullptr;
            
            if (TIFFGetField(tif, TIFFTAG_JPEGTABLES, &jpegtables_count, &jpegtables_data))
            {
                has_jpeg_tables = true;
                ifd_info.tiff_tags["JPEGTables"] = "<detected by libtiff>";
                tiff_tag_count++;
                fmt::print("    🔍 Tag 347 (JPEGTables): [binary data, {} bytes] - ABBREVIATED JPEG DETECTED!\n", 
                          jpegtables_count);
            }
            
            // While we're here, extract other useful tags
            char* image_desc = nullptr;
            if (TIFFGetField(tif, TIFFTAG_IMAGEDESCRIPTION, &image_desc))
            {
                if (image_desc && strlen(image_desc) > 0)
                {
                    ifd_info.tiff_tags["ImageDescription"] = std::string(image_desc);
                    tiff_tag_count++;
                }
            }
            
            char* software = nullptr;
            if (TIFFGetField(tif, TIFFTAG_SOFTWARE, &software))
            {
                if (software && strlen(software) > 0)
                {
                    ifd_info.tiff_tags["Software"] = std::string(software);
                    tiff_tag_count++;
                }
            }
            
            uint16_t compression = 0;
            if (TIFFGetField(tif, TIFFTAG_COMPRESSION, &compression))
            {
                ifd_info.tiff_tags["Compression"] = std::to_string(compression);
                tiff_tag_count++;
            }
        }
        
        TIFFClose(tif);
    }
    else
    {
        fmt::print("  ⚠️  Failed to open TIFF file with libtiff: {}\n", file_path_);
    }
    
    if (tiff_tag_count > 0)
    {
        fmt::print("  ✅ Extracted {} TIFF tags for IFD[{}]\n", tiff_tag_count, ifd_info.index);
        if (has_jpeg_tables)
        {
            fmt::print("  ℹ️  IFD[{}] uses abbreviated JPEG (JPEGTables present)\n", ifd_info.index);
            fmt::print("  ✅ nvTIFF 0.6.0.77 will handle JPEGTables automatically with GPU acceleration\n");
        }
    }
    else
    {
        fmt::print("  ℹ️  No recognized TIFF tags found for IFD[{}]\n", ifd_info.index);
    }
}

std::string TiffFileParser::get_tiff_tag(uint32_t ifd_index, const std::string& tag_name) const
{
    if (ifd_index >= ifd_infos_.size())
        return "";
    
    auto it = ifd_infos_[ifd_index].tiff_tags.find(tag_name);
    if (it != ifd_infos_[ifd_index].tiff_tags.end())
        return it->second;
    
    return "";
}

int TiffFileParser::get_subfile_type(uint32_t ifd_index) const
{
    std::string subfile_str = get_tiff_tag(ifd_index, "SUBFILETYPE");
    if (subfile_str.empty())
        return -1;
    
    try {
        return std::stoi(subfile_str);
    } catch (...) {
        return -1;
    }
}

std::vector<int> TiffFileParser::query_metadata_kinds(uint32_t ifd_index) const
{
    std::vector<int> kinds;
    
    if (ifd_index >= ifd_infos_.size())
        return kinds;
    
    // Return all metadata kinds found in this IFD
    for (const auto& [kind, blob] : ifd_infos_[ifd_index].metadata_blobs)
    {
        kinds.push_back(kind);
    }
    
    // Also add TIFF_TAG kind (0) if any tags were extracted
    if (!ifd_infos_[ifd_index].tiff_tags.empty())
    {
        kinds.insert(kinds.begin(), 0);  // NVIMGCODEC_METADATA_KIND_TIFF_TAG = 0
    }
    
    return kinds;
}

std::string TiffFileParser::get_detected_format() const
{
    if (ifd_infos_.empty())
        return "Unknown";
    
    // Check first IFD for vendor-specific metadata
    const auto& kinds = query_metadata_kinds(0);
    
    for (int kind : kinds)
    {
        switch (kind)
        {
            case 1:  // NVIMGCODEC_METADATA_KIND_MED_APERIO
                return "Aperio SVS";
            case 2:  // NVIMGCODEC_METADATA_KIND_MED_PHILIPS
                return "Philips TIFF";
            case 3:  // NVIMGCODEC_METADATA_KIND_MED_LEICA (if available)
                return "Leica SCN";
            case 4:  // NVIMGCODEC_METADATA_KIND_MED_VENTANA
                return "Ventana";
            case 5:  // NVIMGCODEC_METADATA_KIND_MED_TRESTLE
                return "Trestle";
            default:
                break;
        }
    }
    
    // Fallback: Generic TIFF with detected codec
    if (!ifd_infos_.empty() && !ifd_infos_[0].codec.empty())
    {
        return fmt::format("Generic TIFF ({})", ifd_infos_[0].codec);
    }
    
    return "Generic TIFF";
}

// ============================================================================
// ROI-Based Decoding Implementation (nvTiff File-Level API)
// ============================================================================

uint8_t* TiffFileParser::decode_region(
    uint32_t ifd_index,
    uint32_t x, uint32_t y,
    uint32_t width, uint32_t height,
    uint8_t* output_buffer,
    const cucim::io::Device& device)
{
    if (!initialized_)
    {
        throw std::runtime_error("TIFF parser not initialized");
    }
    
    if (ifd_index >= ifd_infos_.size())
    {
        throw std::out_of_range(fmt::format("IFD index {} out of range (max: {})",
                                            ifd_index, ifd_infos_.size() - 1));
    }
    
    const auto& ifd = ifd_infos_[ifd_index];
    
    // Validate that sub_code_stream is valid (parsing must have succeeded)
    if (!ifd.sub_code_stream)
    {
        throw std::runtime_error(fmt::format(
            "IFD[{}] has invalid sub_code_stream - TIFF parsing may have failed during initialization. "
            "This IFD cannot be decoded.", ifd_index));
    }
    
    // nvImageCodec supports out-of-bounds ROI decoding (will pad with boundary pixels)
    // Just warn if ROI extends beyond IFD bounds
    if (x + width > ifd.width || y + height > ifd.height)
    {
        fmt::print("⚠️  ROI ({},{} {}x{}) extends beyond IFD dimensions ({}x{}) - "
                   "nvImageCodec will pad with boundary pixels\n",
                   x, y, width, height, ifd.width, ifd.height);
    }
    
    // NOTE: nvTIFF 0.6.0.77 CAN handle JPEGTables (TIFFTAG_JPEGTABLES)!
    // Previous documentation suggested nvImageCodec couldn't handle abbreviated JPEG,
    // but testing confirms nvTIFF 0.6.0.77 successfully decodes with automatic JPEG table handling.
    // The "📋 nvTiff: Decoding with automatic JPEG table handling..." message confirms this.
    // 
    // Benefit: GPU-accelerated decoding for Aperio SVS files instead of CPU libjpeg-turbo fallback!
    
    if (ifd.tiff_tags.find("JPEGTables") != ifd.tiff_tags.end())
    {
        fmt::print("ℹ️  JPEG with JPEGTables detected - nvTIFF 0.6.0.77 will handle automatically\n");
    }
    
    fmt::print("✓ Proceeding with nvTIFF/nvImageCodec decode (codec='{}')\n", ifd.codec);
    
    fmt::print("🎯 nvTiff ROI Decode: IFD[{}] region ({},{}) {}x{}, device={}\n",
              ifd_index, x, y, width, height, std::string(device));
    
    // CRITICAL: Must use the same manager that created main_code_stream_!
    // Using a decoder from a different nvImageCodec instance causes segfaults.
    auto& manager = NvImageCodecTiffParserManager::instance();
    if (!manager.is_available())
    {
        throw std::runtime_error("nvImageCodec not available for ROI decoding");
    }
    
    try
    {
        // Use decoder from the same manager instance that created main_code_stream_
        nvimgcodecDecoder_t decoder = manager.get_decoder();
        
        // Prepare decode parameters
        nvimgcodecDecodeParams_t decode_params{};
        decode_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS;
        decode_params.struct_size = sizeof(nvimgcodecDecodeParams_t);
        decode_params.struct_next = nullptr;
        decode_params.apply_exif_orientation = 0;
        
        // Create a code stream view with ROI region
        nvimgcodecRegion_t region{};
        region.struct_type = NVIMGCODEC_STRUCTURE_TYPE_REGION;
        region.struct_size = sizeof(nvimgcodecRegion_t);
        region.struct_next = nullptr;
        region.ndim = 2;
        region.start[0] = y;  // Height dimension
        region.start[1] = x;  // Width dimension
        region.end[0] = y + height;
        region.end[1] = x + width;
        // out_of_bounds_policy and out_of_bounds_samples are zero-initialized by {} above
        
        // Create code stream view for ROI
        // CRITICAL: Must create ROI stream from main_code_stream, not from ifd.sub_code_stream!
        // Nested sub-streams don't properly handle JPEG tables in TIFF files.
        nvimgcodecCodeStreamView_t view{};
        view.struct_type = NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_VIEW;
        view.struct_size = sizeof(nvimgcodecCodeStreamView_t);
        view.struct_next = nullptr;
        view.image_idx = ifd_index;  // Specify which IFD in the main stream
        view.region = region;         // AND the ROI region within that IFD
        
        // Get ROI-specific code stream directly from main stream (not from IFD sub-stream!)
        nvimgcodecCodeStream_t roi_stream = nullptr;
        fmt::print("📍 Creating ROI sub-stream: IFD[{}] ROI=[{},{}:{}x{}] from main stream\n",
                  ifd_index, x, y, width, height);
        
        nvimgcodecStatus_t status = nvimgcodecCodeStreamGetSubCodeStream(
            main_code_stream_, &roi_stream, &view);
        
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            throw std::runtime_error(fmt::format(
                "Failed to create ROI code stream for IFD[{}] ROI=[{},{}:{}x{}]: status={}\n"
                "  IFD dimensions: {}x{}, codec: {}\n"
                "  This may indicate an issue with nvImageCodec ROI support for this codec.",
                ifd_index, x, y, width, height, static_cast<int>(status),
                ifd.width, ifd.height, ifd.codec));
        }
        
        fmt::print("✅ ROI sub-stream created successfully\n");
        
        // Get input image info from ROI code stream
        fmt::print("🔍 Getting image info from ROI stream...\n");
        nvimgcodecImageInfo_t input_image_info{};
        input_image_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
        input_image_info.struct_size = sizeof(nvimgcodecImageInfo_t);
        input_image_info.struct_next = nullptr;
        
        status = nvimgcodecCodeStreamGetImageInfo(roi_stream, &input_image_info);
        
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            nvimgcodecCodeStreamDestroy(roi_stream);
            throw std::runtime_error(fmt::format(
                "Failed to get image info for IFD[{}]: status={}", ifd_index, static_cast<int>(status)));
        }
        
        // Validate image info
        if (input_image_info.num_planes == 0)
        {
            nvimgcodecCodeStreamDestroy(roi_stream);
            throw std::runtime_error(fmt::format(
                "IFD[{}] ROI image info has 0 planes", ifd_index));
        }
        
        fmt::print("✅ Got image info: {}x{}, {} channels, sample_format={}, color_spec={}\n", 
                  input_image_info.plane_info[0].width,
                  input_image_info.plane_info[0].height,
                  input_image_info.num_planes,
                  static_cast<int>(input_image_info.sample_format),
                  static_cast<int>(input_image_info.color_spec));
        
        fmt::print("⚠️  Note: ROI stream returns full image dimensions, will use requested ROI: {}x{}\n",
                  width, height);
        
        // Prepare output image info (use requested ROI dimensions, not input_image_info)
        fmt::print("📝 Preparing output image info...\n");
        
        // CRITICAL: Use zero-initialization to avoid copying codec-specific internal fields
        // Copying from input_image_info can cause segfault because it includes fields
        // (like codec_name, internal pointers) that are only valid for the input stream
        nvimgcodecImageInfo_t output_image_info{};
        
        // Set struct metadata
        output_image_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
        output_image_info.struct_size = sizeof(nvimgcodecImageInfo_t);
        output_image_info.struct_next = nullptr;
        
        // Set output format - IMPORTANT: For interleaved RGB, num_planes = 1
        output_image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
        output_image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        output_image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
        output_image_info.num_planes = 1;  // Interleaved RGB is a single plane with multiple channels
        
        // Set plane info (dimensions and channels)
        output_image_info.plane_info[0].width = width;
        output_image_info.plane_info[0].height = height;
        output_image_info.plane_info[0].num_channels = ifd.num_channels;
        output_image_info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        output_image_info.plane_info[0].precision = 0;  // Use default precision
        
        // IMPORTANT: Do NOT explicitly initialize orientation struct
        // The struct is already zero-initialized, and explicit initialization can cause
        // nvImageCodec to misinterpret the struct or access invalid memory.
        // Orientation handling is done via decode_params.apply_exif_orientation instead.
        
        // Set buffer kind based on device
        bool use_gpu = (device.type() == cucim::io::DeviceType::kCUDA);
        output_image_info.buffer_kind = use_gpu ? 
            NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE :
            NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
        
        // Calculate buffer requirements for interleaved RGB
        // We're using UINT8 format (1 byte per element)
        int bytes_per_element = 1;  // NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8
        
        // For interleaved RGB: row_stride = width * channels * bytes_per_element
        size_t row_stride = width * ifd.num_channels * bytes_per_element;
        size_t output_size = row_stride * height;
        
        fmt::print("💾 Allocating output buffer: {} bytes on {} ({}x{}x{}x{} bytes/element)\n", 
                  output_size, use_gpu ? "GPU" : "CPU",
                  width, height, ifd.num_channels, bytes_per_element);
        
        // Allocate output buffer if not provided
        bool buffer_was_preallocated = (output_buffer != nullptr);
        
        if (!buffer_was_preallocated)
        {
            if (use_gpu)
            {
                cudaError_t cuda_err = cudaMalloc(&output_buffer, output_size);
                if (cuda_err != cudaSuccess)
                {
                    throw std::runtime_error(fmt::format(
                        "Failed to allocate {} bytes on GPU: {}",
                        output_size, cudaGetErrorString(cuda_err)));
                }
            }
            else
            {
                output_buffer = static_cast<uint8_t*>(malloc(output_size));
                if (!output_buffer)
                {
                    throw std::runtime_error(fmt::format(
                        "Failed to allocate {} bytes on host", output_size));
                }
            }
            fmt::print("✅ Buffer allocated successfully\n");
        }
        else
        {
            fmt::print("ℹ️  Using pre-allocated buffer\n");
        }
        
        // Set buffer info with correct row stride
        output_image_info.buffer = output_buffer;
        output_image_info.buffer_size = output_size;
        output_image_info.plane_info[0].row_stride = row_stride;
        output_image_info.cuda_stream = 0;  // CRITICAL: Default CUDA stream (must be set!)
        
        // Create nvImageCodec image object
        fmt::print("🖼️  Creating nvImageCodec image object...\n");
        fmt::print("   Image config: {}x{}, {} planes, {} channels/plane, buffer_size={}, row_stride={}\n",
                  output_image_info.plane_info[0].width,
                  output_image_info.plane_info[0].height,
                  output_image_info.num_planes,
                  output_image_info.plane_info[0].num_channels,
                  output_image_info.buffer_size,
                  output_image_info.plane_info[0].row_stride);
        fmt::print("   Buffer kind: {}, sample_format: {}, color_spec: {}\n",
                  static_cast<int>(output_image_info.buffer_kind),
                  static_cast<int>(output_image_info.sample_format),
                  static_cast<int>(output_image_info.color_spec));
        
        nvimgcodecImage_t image;
        status = nvimgcodecImageCreate(manager.get_instance(), &image, &output_image_info);
        
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            nvimgcodecCodeStreamDestroy(roi_stream);
            if (!buffer_was_preallocated)
            {
                if (use_gpu)
                    cudaFree(output_buffer);
                else
                    free(output_buffer);
            }
            throw std::runtime_error(fmt::format(
                "Failed to create nvImageCodec image: status={}", static_cast<int>(status)));
        }
        
        fmt::print("✅ Image object created successfully\n");
        
        // Perform decode - nvTiff handles JPEG tables automatically!
        fmt::print("📋 nvTiff: Decoding with automatic JPEG table handling...\n");
        fmt::print("   Decoder: {}, ROI stream: {}, Image: {}\n",
                  static_cast<void*>(decoder),
                  static_cast<void*>(roi_stream),
                  static_cast<void*>(image));
        
        nvimgcodecFuture_t decode_future;
        {
            std::lock_guard<std::mutex> lock(manager.get_mutex());
            fmt::print("   Calling nvimgcodecDecoderDecode()...\n");
            status = nvimgcodecDecoderDecode(
                decoder,
                &roi_stream,  // Use ROI stream instead of full IFD stream
                &image,
                1,
                &decode_params,
                &decode_future);
            fmt::print("   Decode scheduled, status={}\n", static_cast<int>(status));
        }
        
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            nvimgcodecImageDestroy(image);
            nvimgcodecCodeStreamDestroy(roi_stream);
            if (!buffer_was_preallocated)
            {
                if (use_gpu)
                    cudaFree(output_buffer);
                else
                    free(output_buffer);
            }
            throw std::runtime_error(fmt::format(
                "Failed to schedule decode: status={}", static_cast<int>(status)));
        }
        
        // Wait for decode completion
        fmt::print("⏳ Waiting for decode to complete...\n");
        size_t status_size = 1;
        nvimgcodecProcessingStatus_t decode_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
        status = nvimgcodecFutureGetProcessingStatus(decode_future, &decode_status, &status_size);
        fmt::print("   Future status: {}, Processing status: {}\n",
                  static_cast<int>(status), static_cast<int>(decode_status));
        
        if (use_gpu)
        {
            cudaDeviceSynchronize();
            fmt::print("   GPU synchronized\n");
        }
        
        // Check for decode failure BEFORE cleanup
        bool decode_failed = (status != NVIMGCODEC_STATUS_SUCCESS || 
                             decode_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
        
        if (decode_failed)
        {
            fmt::print("⚠️  nvImageCodec decode failed (status={}, decode_status={})\n",
                      static_cast<int>(status), static_cast<int>(decode_status));
            
            // CRITICAL: Detach buffer ownership before destroying image object
            // This prevents nvImageCodec from trying to access/free the buffer
            output_image_info.buffer = nullptr;
            
            fmt::print("🧹 Cleaning up after failed decode...\n");
            nvimgcodecFutureDestroy(decode_future);
            nvimgcodecImageDestroy(image);
            nvimgcodecCodeStreamDestroy(roi_stream);
            
            // Safely free buffer if we allocated it
            if (!buffer_was_preallocated && output_buffer != nullptr)
            {
                fmt::print("   Freeing allocated buffer...\n");
                if (use_gpu)
                    cudaFree(output_buffer);
                else
                    free(output_buffer);
                output_buffer = nullptr;  // Prevent double-free
            }
            
            // Decode failure likely means abbreviated JPEG not supported by nvImageCodec
            // Return nullptr to trigger fallback to libjpeg-turbo
            fmt::print("💡 Returning nullptr to trigger libjpeg-turbo fallback\n");
            return nullptr;
        }
        
        // Success path: Normal cleanup
        fmt::print("🧹 Cleaning up nvImageCodec objects...\n");
        fmt::print("   Destroying future...\n");
        nvimgcodecFutureDestroy(decode_future);
        fmt::print("   Destroying image...\n");
        nvimgcodecImageDestroy(image);
        fmt::print("   Destroying ROI stream...\n");
        nvimgcodecCodeStreamDestroy(roi_stream);
        fmt::print("✅ Cleanup complete\n");
        
        fmt::print("✅ nvTiff ROI Decode: Success! {}x{} decoded\n", width, height);
        return output_buffer;
    }
    catch (const std::exception& e)
    {
        fmt::print("❌ nvTiff ROI Decode failed: {}\n", e.what());
        throw;
    }
}

uint8_t* TiffFileParser::decode_ifd(
    uint32_t ifd_index,
    uint8_t* output_buffer,
    const cucim::io::Device& device)
{
    if (ifd_index >= ifd_infos_.size())
    {
        throw std::out_of_range(fmt::format("IFD index {} out of range", ifd_index));
    }
    
    const auto& ifd = ifd_infos_[ifd_index];
    return decode_region(ifd_index, 0, 0, ifd.width, ifd.height, output_buffer, device);
}

bool TiffFileParser::has_roi_decode_support() const
{
    auto& manager = NvImageCodecManager::instance();
    return manager.is_initialized();
}

#endif // CUCIM_HAS_NVIMGCODEC

} // namespace cuslide2::nvimgcodec

