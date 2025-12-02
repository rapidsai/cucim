/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nvimgcodec_tiff_parser.h"

#include <algorithm>  // for std::transform
#include <cstring>    // for strlen

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

// Helper function to convert TiffTagValue variant to string representation
static std::string tiff_tag_value_to_string(const TiffTagValue& value)
{
    return std::visit([](const auto& v) -> std::string {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::monostate>)
        {
            return "";  // Empty/unset
        }
        else if constexpr (std::is_same_v<T, std::string>)
        {
            return v;
        }
        else if constexpr (std::is_same_v<T, std::vector<uint8_t>>)
        {
            return fmt::format("[{} bytes]", v.size());
        }
        else if constexpr (std::is_same_v<T, std::vector<uint16_t>>)
        {
            std::string result;
            for (size_t i = 0; i < v.size() && i < 10; ++i)
            {
                if (i > 0) result += ",";
                result += std::to_string(v[i]);
            }
            if (v.size() > 10) result += ",...";
            return result;
        }
        else if constexpr (std::is_same_v<T, std::vector<uint32_t>>)
        {
            std::string result;
            for (size_t i = 0; i < v.size() && i < 10; ++i)
            {
                if (i > 0) result += ",";
                result += std::to_string(v[i]);
            }
            if (v.size() > 10) result += ",...";
            return result;
        }
        else if constexpr (std::is_same_v<T, std::vector<uint64_t>>)
        {
            std::string result;
            for (size_t i = 0; i < v.size() && i < 10; ++i)
            {
                if (i > 0) result += ",";
                result += std::to_string(v[i]);
            }
            if (v.size() > 10) result += ",...";
            return result;
        }
        else if constexpr (std::is_same_v<T, std::vector<float>>)
        {
            std::string result;
            for (size_t i = 0; i < v.size() && i < 10; ++i)
            {
                if (i > 0) result += ",";
                result += std::to_string(v[i]);
            }
            if (v.size() > 10) result += ",...";
            return result;
        }
        else if constexpr (std::is_same_v<T, std::vector<double>>)
        {
            std::string result;
            for (size_t i = 0; i < v.size() && i < 10; ++i)
            {
                if (i > 0) result += ",";
                result += std::to_string(v[i]);
            }
            if (v.size() > 10) result += ",...";
            return result;
        }
        else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
        {
            return fmt::format("{}", v);
        }
        else
        {
            return std::to_string(v);
        }
    }, value);
}

// Template helper to extract single scalar value from TIFF tag metadata
// Per nvImageCodec team: value_count check is sufficient, buffer_size check is redundant
template<typename T>
static bool extract_single_value(const std::vector<uint8_t>& buffer, 
                                 int value_count,
                                 TiffTagValue& out_value)
{
    if (value_count == 1)
    {
        T val = *reinterpret_cast<const T*>(buffer.data());
        out_value = val;
        return true;
    }
    return false;
}

// Template helper to extract array of values as vector
template<typename T>
static bool extract_value_array(const std::vector<uint8_t>& buffer,
                                int value_count,
                                TiffTagValue& out_value)
{
    if (value_count > 1)
    {
        const T* vals = reinterpret_cast<const T*>(buffer.data());
        out_value = std::vector<T>(vals, vals + value_count);
        return true;
    }
    return false;
}

// ============================================================================
// NvImageCodecTiffParserManager Implementation
// ============================================================================

NvImageCodecTiffParserManager::NvImageCodecTiffParserManager() 
    : instance_(nullptr), decoder_(nullptr), cpu_decoder_(nullptr), initialized_(false)
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
            #ifdef DEBUG
            fmt::print("⚠️  {}\n", status_message_);
            #endif // DEBUG
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
            #ifdef DEBUG
            fmt::print("⚠️  {}\n", status_message_);
            #endif // DEBUG
            return;
        }
        
        // Create CPU-only decoder for native CPU decoding
        nvimgcodecBackendKind_t cpu_backend_kind = NVIMGCODEC_BACKEND_KIND_CPU_ONLY;
        nvimgcodecBackendParams_t cpu_backend_params{};
        cpu_backend_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS;
        cpu_backend_params.struct_size = sizeof(nvimgcodecBackendParams_t);
        cpu_backend_params.struct_next = nullptr;
        
        nvimgcodecBackend_t cpu_backend{};
        cpu_backend.struct_type = NVIMGCODEC_STRUCTURE_TYPE_BACKEND;
        cpu_backend.struct_size = sizeof(nvimgcodecBackend_t);
        cpu_backend.struct_next = nullptr;
        cpu_backend.kind = cpu_backend_kind;
        cpu_backend.params = cpu_backend_params;
        
        nvimgcodecExecutionParams_t cpu_exec_params = exec_params;
        cpu_exec_params.num_backends = 1;
        cpu_exec_params.backends = &cpu_backend;
        
        if (nvimgcodecDecoderCreate(instance_, &cpu_decoder_, &cpu_exec_params, nullptr) == NVIMGCODEC_STATUS_SUCCESS)
        {
            #ifdef DEBUG
            fmt::print("✅ CPU-only decoder created successfully (TIFF parser)\n");
            #endif // DEBUG
        }
        else
        {
            #ifdef DEBUG
            fmt::print("⚠️  Failed to create CPU-only decoder (CPU decoding will use fallback)\n");
            #endif // DEBUG
            cpu_decoder_ = nullptr;
        }
        
        initialized_ = true;
        status_message_ = "nvImageCodec TIFF parser initialized successfully (with metadata extraction support)";
        #ifdef DEBUG
        fmt::print("✅ {}\n", status_message_);
        #endif // DEBUG
    }
    catch (const std::exception& e)
    {
        status_message_ = fmt::format("nvImageCodec TIFF parser initialization exception: {}", e.what());
        #ifdef DEBUG
        fmt::print("❌ {}\n", status_message_);
        #endif // DEBUG
        initialized_ = false;
    }
}

NvImageCodecTiffParserManager::~NvImageCodecTiffParserManager()
{
    // Proper cleanup: destroy decoders first, then instance
    // Per nvImageCodec team: all code streams should be destroyed before this point
    // (handled by TiffFileParser destructors which are called before singleton destruction)
    
    if (cpu_decoder_)
    {
        nvimgcodecDecoderDestroy(cpu_decoder_);
        cpu_decoder_ = nullptr;
    }
    
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
        
        #ifdef DEBUG
        fmt::print("✅ Opened TIFF file: {}\n", file_path);
        #endif // DEBUG
        
        // Step 2: Parse TIFF structure (metadata only)
        parse_tiff_structure();
        
        initialized_ = true;
        #ifdef DEBUG
        fmt::print("✅ TIFF parser initialized with {} IFDs\n", ifd_infos_.size());
        #endif // DEBUG
    }
    catch (const std::exception& e)
    {
        // Don't explicitly destroy main_code_stream_ here - let instance cleanup handle it
        // (See destructor comment for explanation of static destruction order issues)
        main_code_stream_ = nullptr;
        
        throw;  // Re-throw
    }
}

TiffFileParser::~TiffFileParser()
{
    // Per nvImageCodec team: each code stream (parent or sub) has its own state
    // and MUST be explicitly destroyed. Sub-streams are NOT automatically cleaned
    // up when the parent is destroyed.
    
    // Destroy sub-code streams first (IFD streams)
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
    #ifdef DEBUG
    fmt::print("  TIFF has {} IFDs (resolution levels)\n", num_ifds);
    #endif // DEBUG
    
    if (stream_info.codec_name[0] != '\0')
    {
        #ifdef DEBUG
        fmt::print("  Codec: {}\n", stream_info.codec_name);
        #endif // DEBUG
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
            #ifdef DEBUG
            fmt::print("❌ Failed to get sub-code stream for IFD {} (status: {})\n", 
                      i, static_cast<int>(status));
            #endif // DEBUG
            #ifdef DEBUG
            fmt::print("   This IFD will be SKIPPED and cannot be decoded.\n");
            #endif // DEBUG
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
            #ifdef DEBUG
            fmt::print("❌ Failed to get image info for IFD {} (status: {})\n",
                      i, static_cast<int>(status));
            #endif // DEBUG
            #ifdef DEBUG
            fmt::print("   This IFD will be SKIPPED and cannot be decoded.\n");
            #endif // DEBUG
            // NOTE: Do NOT destroy sub_code_stream here - it's a view into main_code_stream
            // Main stream destruction will handle cleanup. Just mark as invalid.
            ifd_info.sub_code_stream = nullptr;
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
        
        // NOTE: image_info.codec_name typically contains "tiff" (the container format)
        // We need to determine the actual compression codec (jpeg2000, jpeg, etc.)
        if (image_info.codec_name[0] != '\0')
        {
            ifd_info.codec = image_info.codec_name;
        }
        
        // Extract metadata for this IFD using nvimgcodecDecoderGetMetadata
        // Extract vendor-specific metadata (Aperio, Philips, etc.)
        extract_ifd_metadata(ifd_info);
        
        // Extract TIFF metadata using available methods
        extract_tiff_tags(ifd_info);
        
        // TODO(nvImageCodec 0.7.0): Use direct TIFF tag queries when 0.7.0 is released
        // Individual TIFF tag access (e.g., COMPRESSION tag 259) will be available in 0.7.0
        // Example: metadata = decoder.get_metadata(scs, name="Compression")
        //
        // Current limitation (0.6.0):
        // - codec_name returns "tiff" (container format) not compression type
        // - Individual TIFF tags not exposed through metadata API
        // - Only vendor-specific metadata blobs available (MED_APERIO, MED_PHILIPS, etc.)
        //
        // Workaround: Infer compression from chroma_subsampling and file extension
        // Reference: https://nvidia.slack.com/archives/C092X06LK9U (Oct 27, 2024)
        if (ifd_info.codec == "tiff")
        {
            // Try to infer compression from TIFF metadata first
            bool compression_inferred = false;
            
            // Check if we have TIFF Compression tag (stored as typed value)
            auto compression_it = ifd_info.tiff_tags.find("COMPRESSION");
            if (compression_it != ifd_info.tiff_tags.end())
            {
                // COMPRESSION tag is always SHORT (uint16_t) per TIFF spec
                // Check type before extracting to avoid exceptions
                if (std::holds_alternative<uint16_t>(compression_it->second))
                {
                    uint16_t compression_value = std::get<uint16_t>(compression_it->second);
                    
                    switch (compression_value)
                    {
                        case 1:    // COMPRESSION_NONE
                            // Keep as "tiff" for uncompressed
                            #ifdef DEBUG
                            fmt::print("  ℹ️  Detected uncompressed TIFF\n");
                            #endif // DEBUG
                            compression_inferred = true;
                            break;
                        case 5:    // COMPRESSION_LZW
                            ifd_info.codec = "tiff";  // nvImageCodec handles as tiff
                            compression_inferred = true;
                            #ifdef DEBUG
                            fmt::print("  ℹ️  Detected LZW compression (TIFF codec)\n");
                            #endif // DEBUG
                            break;
                        case 7:    // COMPRESSION_JPEG
                            ifd_info.codec = "jpeg";  // Use JPEG decoder!
                            compression_inferred = true;
                            #ifdef DEBUG
                            fmt::print("  ℹ️  Detected JPEG compression → using JPEG codec\n");
                            #endif // DEBUG
                            break;
                        case 8:    // COMPRESSION_DEFLATE (Adobe-style)
                        case 32946: // COMPRESSION_DEFLATE (old-style)
                            ifd_info.codec = "tiff";
                            compression_inferred = true;
                            #ifdef DEBUG
                            fmt::print("  ℹ️  Detected DEFLATE compression (TIFF codec)\n");
                            #endif // DEBUG
                            break;
                        case 33003: // Aperio JPEG2000 YCbCr
                        case 33005: // Aperio JPEG2000 RGB
                        case 34712: // JPEG2000
                            ifd_info.codec = "jpeg2000";
                            compression_inferred = true;
                            #ifdef DEBUG
                            fmt::print("  ℹ️  Detected JPEG2000 compression\n");
                            #endif // DEBUG
                            break;
                        default:
                            #ifdef DEBUG
                            fmt::print("  ⚠️  Unknown TIFF compression value: {}\n", compression_value);
                            #endif // DEBUG
                            break;
                    }
                }
                else
                {
                    #ifdef DEBUG
                    fmt::print("  ⚠️  COMPRESSION tag is not uint16_t (unexpected type)\n");
                    #endif // DEBUG
                }
            }
            
            // Fallback to filename-based heuristics if metadata didn't help
            if (!compression_inferred)
            {
                // Aperio JPEG2000 files typically have "JP2K" in filename
                if (file_path_.find("JP2K") != std::string::npos || 
                    file_path_.find("jp2k") != std::string::npos)
                {
                    ifd_info.codec = "jpeg2000";
                    #ifdef DEBUG
                    fmt::print("  ℹ️  Inferred codec 'jpeg2000' from filename (JP2K pattern)\n");
                    #endif // DEBUG
                    compression_inferred = true;
                }
            }
            
            // Warning if we still couldn't infer compression
            if (!compression_inferred && ifd_info.tiff_tags.empty())
            {
                #ifdef DEBUG
                fmt::print("  ⚠️  Warning: codec is 'tiff' but could not infer compression.\n");
                fmt::print("     File: {}\n", file_path_);
                fmt::print("     This may limit CPU decoder availability.\n");
                #endif
            }
        }
        
        ifd_infos_.push_back(std::move(ifd_info));
    }
    
    // Report parsing results
    if (ifd_infos_.size() == num_ifds)
    {
        #ifdef DEBUG
        fmt::print("✅ TIFF parser initialized with {} IFDs (all successful)\n", ifd_infos_.size());
        #endif // DEBUG
    }
    else
    {
        #ifdef DEBUG
        fmt::print("⚠️  TIFF parser initialized with {} IFDs ({} out of {} total)\n", 
                  ifd_infos_.size(), ifd_infos_.size(), num_ifds);
        #endif // DEBUG
        #ifdef DEBUG
        fmt::print("   {} IFDs were skipped due to parsing errors\n", num_ifds - ifd_infos_.size());
        #endif // DEBUG
    }
}

void TiffFileParser::extract_ifd_metadata(IfdInfo& ifd_info)
{
    auto& manager = NvImageCodecTiffParserManager::instance();
    
    #ifdef DEBUG
    fmt::print("🔍 Extracting metadata for IFD[{}]...\n", ifd_info.index);
    #endif
    
    if (!manager.get_decoder() || !ifd_info.sub_code_stream)
    {
        if (!manager.get_decoder())
            fmt::print("  ⚠️  Decoder not available\n");
        if (!ifd_info.sub_code_stream)
            fmt::print("  ⚠️  No sub-code stream for this IFD\n");
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
    
    if (status != NVIMGCODEC_STATUS_SUCCESS)
    {
        #ifdef DEBUG
        fmt::print("  ⚠️  Metadata query failed with status: {}\n", static_cast<int>(status));
        #endif
        return;
    }
    
    if (metadata_count == 0)
    {
        #ifdef DEBUG
        fmt::print("  ℹ️  No metadata entries found for this IFD\n");
        #endif
        return;  // No metadata
    }
    
    #ifdef DEBUG
    fmt::print("  ✅ Found {} metadata entries for IFD[{}]\n", metadata_count, ifd_info.index);
    #endif
    
    // Step 2: Allocate metadata structures AND buffers
    // nvImageCodec requires us to allocate buffers based on buffer_size from first call
    std::vector<nvimgcodecMetadata_t> metadata_structs(metadata_count);
    std::vector<nvimgcodecMetadata_t*> metadata_ptrs(metadata_count);
    std::vector<std::vector<uint8_t>> metadata_buffers(metadata_count);  // Storage for actual data
    
    // First, query to get buffer sizes (metadata structs must be initialized)
    for (int i = 0; i < metadata_count; i++)
    {
        metadata_structs[i].struct_type = NVIMGCODEC_STRUCTURE_TYPE_METADATA;
        metadata_structs[i].struct_size = sizeof(nvimgcodecMetadata_t);
        metadata_structs[i].struct_next = nullptr;
        metadata_structs[i].buffer = nullptr;  // Query mode: get sizes
        metadata_structs[i].buffer_size = 0;
        metadata_ptrs[i] = &metadata_structs[i];
    }
    
    // Query call to get buffer sizes
    status = nvimgcodecDecoderGetMetadata(
        manager.get_decoder(),
        ifd_info.sub_code_stream,
        metadata_ptrs.data(),
        &metadata_count
    );
    
    if (status != NVIMGCODEC_STATUS_SUCCESS)
    {
        #ifdef DEBUG
        fmt::print("  ⚠️  Failed to query metadata sizes (status: {})\n", static_cast<int>(status));
        #endif
        return;
    }
    
    // Now allocate buffers based on reported sizes
    for (int i = 0; i < metadata_count; i++)
    {
        size_t required_size = metadata_structs[i].buffer_size;
        if (required_size > 0)
        {
            metadata_buffers[i].resize(required_size);
            metadata_structs[i].buffer = metadata_buffers[i].data();
            #ifdef DEBUG
            fmt::print("    📦 Allocated {} bytes for metadata[{}]\n", required_size, i);
            #endif
        }
    }
    
    // Step 3: Get actual metadata content (buffers now allocated)
    status = nvimgcodecDecoderGetMetadata(
        manager.get_decoder(),
        ifd_info.sub_code_stream,
        metadata_ptrs.data(),
        &metadata_count
    );
    
    if (status != NVIMGCODEC_STATUS_SUCCESS)
    {
        #ifdef DEBUG
        fmt::print("  ⚠️  Failed to retrieve metadata content (status: {})\n", static_cast<int>(status));
        #endif
        return;
    }
    
    #ifdef DEBUG
    fmt::print("  ✅ Successfully retrieved {} metadata entries with content\n", metadata_count);
    #endif
    
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
        
        #ifdef DEBUG
        // Map kind to human-readable name for debugging
        const char* kind_name = "UNKNOWN";
        switch (kind) {
            case NVIMGCODEC_METADATA_KIND_UNKNOWN: kind_name = "UNKNOWN"; break;
            case NVIMGCODEC_METADATA_KIND_TIFF_TAG: kind_name = "TIFF_TAG"; break;
            case NVIMGCODEC_METADATA_KIND_ICC_PROFILE: kind_name = "ICC_PROFILE"; break;
            case NVIMGCODEC_METADATA_KIND_EXIF: kind_name = "EXIF"; break;
            case NVIMGCODEC_METADATA_KIND_GEO: kind_name = "GEO"; break;
            case NVIMGCODEC_METADATA_KIND_MED_APERIO: kind_name = "MED_APERIO"; break;
            case NVIMGCODEC_METADATA_KIND_MED_PHILIPS: kind_name = "MED_PHILIPS"; break;
            case NVIMGCODEC_METADATA_KIND_MED_VENTANA: kind_name = "MED_VENTANA"; break;
            case NVIMGCODEC_METADATA_KIND_MED_LEICA: kind_name = "MED_LEICA"; break;
            case NVIMGCODEC_METADATA_KIND_MED_TRESTLE: kind_name = "MED_TRESTLE"; break;
        }
        fmt::print("    Metadata[{}]: kind={} ({}), format={}, size={}\n",
                  j, kind, kind_name, format, buffer_size);
        #endif
        
        // Store in metadata_blobs map
        if (buffer && buffer_size > 0)
        {
            IfdInfo::MetadataBlob blob;
            blob.format = format;
            blob.data.assign(buffer, buffer + buffer_size);
            ifd_info.metadata_blobs[kind] = std::move(blob);
            
            // Note: ImageDescription is now extracted directly via TIFF tag 270
            // in extract_tiff_tags() using nvImageCodec 0.7.0's direct tag query API.
            // The vendor metadata blobs (MED_APERIO, MED_PHILIPS, etc.) are stored
            // above for format detection and vendor-specific parsing.
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

std::string TiffFileParser::get_tiff_tag(uint32_t ifd_index, const std::string& tag_name) const
{
    if (ifd_index >= ifd_infos_.size())
        return "";
    
    auto it = ifd_infos_[ifd_index].tiff_tags.find(tag_name);
    if (it != ifd_infos_[ifd_index].tiff_tags.end())
        return tiff_tag_value_to_string(it->second);
    
    return "";
}

void TiffFileParser::extract_tiff_tags(IfdInfo& ifd_info)
{
    auto& manager = NvImageCodecTiffParserManager::instance();
    
    if (!manager.get_decoder())
    {
        #ifdef DEBUG
        fmt::print("  ⚠️  Cannot extract TIFF tags: decoder not available\n");
        #endif // DEBUG
        return;
    }
    
    if (!ifd_info.sub_code_stream)
    {
        #ifdef DEBUG
        fmt::print("  ⚠️  Cannot extract TIFF tags: sub_code_stream is null\n");
        #endif // DEBUG
        return;
    }
    
    // ========================================================================
    // nvImageCodec 0.7.0: Direct TIFF Tag Retrieval by ID
    // ========================================================================
    // Python API example:
    //   tag_value = decoder.get_metadata(scs, id=tag_id).value
    //
    // C API equivalent:
    //   1. Set metadata.kind = NVIMGCODEC_METADATA_KIND_TIFF_TAG
    //   2. Set metadata.id = <tag_id> (e.g., 270 for ImageDescription)
    //   3. Call nvimgcodecDecoderGetMetadata() to retrieve the specific tag
    
    // Map of TIFF tag IDs to names for tags we want to extract
    std::vector<std::pair<uint16_t, std::string>> tiff_tags_to_query = {
        {254, "SUBFILETYPE"},      // Image type classification (0=full, 1=reduced, etc.)
        {256, "IMAGEWIDTH"},
        {257, "IMAGELENGTH"},
        {258, "BITSPERSAMPLE"},
        {259, "COMPRESSION"},      // Critical for codec detection!
        {262, "PHOTOMETRIC"},
        {270, "IMAGEDESCRIPTION"}, // Vendor metadata
        {271, "MAKE"},             // Scanner manufacturer
        {272, "MODEL"},            // Scanner model
        {277, "SAMPLESPERPIXEL"},
        {305, "SOFTWARE"},
        {306, "DATETIME"},
        {322, "TILEWIDTH"},
        {323, "TILELENGTH"},
        {330, "SUBIFD"},           // SubIFD offsets (for OME-TIFF, etc.)
        {339, "SAMPLEFORMAT"},
        {347, "JPEGTABLES"}        // Shared JPEG tables
    };
    
    #ifdef DEBUG
    fmt::print("  📋 Extracting TIFF tags (nvImageCodec 0.7.0 - query by ID)...\n");
    #endif // DEBUG
    
    int extracted_count = 0;
    
    // Query each tag individually by ID (following Python API pattern)
    for (const auto& [tag_id, tag_name] : tiff_tags_to_query)
    {
        // Set up metadata request for specific tag
        nvimgcodecMetadata_t metadata{};
        metadata.struct_type = NVIMGCODEC_STRUCTURE_TYPE_METADATA;
        metadata.struct_size = sizeof(nvimgcodecMetadata_t);
        metadata.struct_next = nullptr;
        metadata.kind = NVIMGCODEC_METADATA_KIND_TIFF_TAG;
        metadata.id = tag_id;
        metadata.buffer = nullptr;
        metadata.buffer_size = 0;
        
        nvimgcodecMetadata_t* metadata_ptr = &metadata;
        int metadata_count = 1;
        
        // First call: query buffer size
        nvimgcodecStatus_t status = nvimgcodecDecoderGetMetadata(
            manager.get_decoder(),
            ifd_info.sub_code_stream,
            &metadata_ptr,
            &metadata_count
        );
        
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            // API error - log warning for unexpected failures
            // Note: Some status codes may indicate "tag not found" which is normal
            #ifdef DEBUG
            fmt::print("  ⚠️  TIFF tag {} query failed (status: {})\n", tag_id, static_cast<int>(status));
            #endif
            continue;
        }
        
        if (metadata.buffer_size == 0)
        {
            // Tag not present in this IFD - this is normal, not all tags exist
            continue;
        }
        
        // Allocate buffer for tag value
        std::vector<uint8_t> buffer(metadata.buffer_size);
        metadata.buffer = buffer.data();
        
        // Second call: retrieve actual value
        status = nvimgcodecDecoderGetMetadata(
            manager.get_decoder(),
            ifd_info.sub_code_stream,
            &metadata_ptr,
            &metadata_count
        );
        
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            // Unexpected: first call succeeded but second failed
            #ifdef DEBUG
            fmt::print("  ⚠️  TIFF tag {} retrieval failed (status: {})\n", tag_id, static_cast<int>(status));
            #endif
            continue;
        }
        
        if (metadata.buffer_size == 0)
        {
            // Unexpected: buffer was allocated but size is now 0
            continue;
        }
        
        // Convert value based on type and store as typed variant
        // The variant is initialized to std::monostate by default
        TiffTagValue tag_value;
        
        switch (metadata.value_type)
        {
            case NVIMGCODEC_METADATA_VALUE_TYPE_ASCII:
            {
                // ASCII string
                std::string str_val;
                str_val.assign(reinterpret_cast<const char*>(buffer.data()), metadata.buffer_size);
                // Remove trailing null(s) if present
                while (!str_val.empty() && str_val.back() == '\0')
                    str_val.pop_back();
                if (!str_val.empty())
                {
                    tag_value = std::move(str_val);
                }
                break;
            }
                
            case NVIMGCODEC_METADATA_VALUE_TYPE_SHORT:
                extract_single_value<uint16_t>(buffer, metadata.value_count, tag_value) ||
                extract_value_array<uint16_t>(buffer, metadata.value_count, tag_value);
                break;
                
            case NVIMGCODEC_METADATA_VALUE_TYPE_LONG:
                extract_single_value<uint32_t>(buffer, metadata.value_count, tag_value) ||
                extract_value_array<uint32_t>(buffer, metadata.value_count, tag_value);
                break;
                
            case NVIMGCODEC_METADATA_VALUE_TYPE_BYTE:
                if (metadata.value_count == 1)
                {
                    tag_value = buffer[0];
                }
                else
                {
                    // Binary data - store as vector<uint8_t>
                    std::vector<uint8_t> vec(buffer.begin(), buffer.begin() + metadata.buffer_size);
                    tag_value = std::move(vec);
                }
                break;
                
            case NVIMGCODEC_METADATA_VALUE_TYPE_SBYTE:
                if (metadata.value_count == 1)
                {
                    tag_value = static_cast<int8_t>(buffer[0]);
                }
                else
                {
                    // Signed byte array - store as vector<uint8_t> (reinterpret as needed)
                    std::vector<uint8_t> vec(buffer.begin(), buffer.begin() + metadata.buffer_size);
                    tag_value = std::move(vec);
                }
                break;
                
            case NVIMGCODEC_METADATA_VALUE_TYPE_UNDEFINED:
                // UNDEFINED type - binary data, store as vector<uint8_t>
                {
                    std::vector<uint8_t> vec(buffer.begin(), buffer.begin() + metadata.buffer_size);
                    tag_value = std::move(vec);
                }
                break;
                
            case NVIMGCODEC_METADATA_VALUE_TYPE_SSHORT:
                extract_single_value<int16_t>(buffer, metadata.value_count, tag_value) ||
                extract_value_array<uint16_t>(buffer, metadata.value_count, tag_value);
                break;
                
            case NVIMGCODEC_METADATA_VALUE_TYPE_SLONG:
                extract_single_value<int32_t>(buffer, metadata.value_count, tag_value) ||
                extract_value_array<uint32_t>(buffer, metadata.value_count, tag_value);
                break;
                
            case NVIMGCODEC_METADATA_VALUE_TYPE_LONG8:
            case NVIMGCODEC_METADATA_VALUE_TYPE_IFD8:
                extract_single_value<uint64_t>(buffer, metadata.value_count, tag_value) ||
                extract_value_array<uint64_t>(buffer, metadata.value_count, tag_value);
                break;
                
            case NVIMGCODEC_METADATA_VALUE_TYPE_SLONG8:
                extract_single_value<int64_t>(buffer, metadata.value_count, tag_value) ||
                extract_value_array<uint64_t>(buffer, metadata.value_count, tag_value);
                break;
                
            case NVIMGCODEC_METADATA_VALUE_TYPE_FLOAT:
                extract_single_value<float>(buffer, metadata.value_count, tag_value) ||
                extract_value_array<float>(buffer, metadata.value_count, tag_value);
                break;
                
            case NVIMGCODEC_METADATA_VALUE_TYPE_DOUBLE:
                extract_single_value<double>(buffer, metadata.value_count, tag_value) ||
                extract_value_array<double>(buffer, metadata.value_count, tag_value);
                break;
                
            case NVIMGCODEC_METADATA_VALUE_TYPE_RATIONAL:
                if (metadata.value_count == 1 && metadata.buffer_size >= 8)
                {
                    // Single Rational = two LONGs (numerator, denominator) - store as string
                    uint32_t num = *reinterpret_cast<const uint32_t*>(buffer.data());
                    uint32_t den = *reinterpret_cast<const uint32_t*>(buffer.data() + 4);
                    if (den != 0)
                        tag_value = fmt::format("{}/{}", num, den);
                    else
                        tag_value = std::to_string(num);
                }
                else if (metadata.value_count > 1)
                {
                    // Array of Rationals - store as comma-separated string
                    size_t rational_size = 8;  // 2 × uint32_t
                    std::string result;
                    for (size_t i = 0; i < metadata.value_count; ++i)
                    {
                        const uint8_t* ptr = buffer.data() + i * rational_size;
                        uint32_t num = *reinterpret_cast<const uint32_t*>(ptr);
                        uint32_t den = *reinterpret_cast<const uint32_t*>(ptr + 4);
                        if (i > 0) result += ", ";
                        if (den != 0)
                            result += fmt::format("{}/{}", num, den);
                        else
                            result += std::to_string(num);
                    }
                    tag_value = std::move(result);
                }
                break;
                
            case NVIMGCODEC_METADATA_VALUE_TYPE_SRATIONAL:
                if (metadata.value_count == 1 && metadata.buffer_size >= 8)
                {
                    // Single Signed Rational = two SLONGs (numerator, denominator) - store as string
                    int32_t num = *reinterpret_cast<const int32_t*>(buffer.data());
                    int32_t den = *reinterpret_cast<const int32_t*>(buffer.data() + 4);
                    if (den != 0)
                        tag_value = fmt::format("{}/{}", num, den);
                    else
                        tag_value = std::to_string(num);
                }
                else if (metadata.value_count > 1)
                {
                    // Array of Signed Rationals - store as comma-separated string
                    size_t rational_size = 8;  // 2 × int32_t
                    std::string result;
                    for (size_t i = 0; i < metadata.value_count; ++i)
                    {
                        const uint8_t* ptr = buffer.data() + i * rational_size;
                        int32_t num = *reinterpret_cast<const int32_t*>(ptr);
                        int32_t den = *reinterpret_cast<const int32_t*>(ptr + 4);
                        if (i > 0) result += ", ";
                        if (den != 0)
                            result += fmt::format("{}/{}", num, den);
                        else
                            result += std::to_string(num);
                    }
                    tag_value = std::move(result);
                }
                break;
                
            default:
                // For unknown types, store as binary data or string
                if (metadata.buffer_size > 0)
                {
                    if (metadata.buffer_size <= 8 && metadata.value_count == 1)
                    {
                        // Small value - try to interpret as number and store as string
                        uint64_t val = 0;
                        std::memcpy(&val, buffer.data(), std::min(metadata.buffer_size, sizeof(val)));
                        tag_value = std::to_string(val);
                    }
                    else
                    {
                        // Store raw bytes - optionally limit size to prevent storing huge blobs
                        // Use configurable limit (0 = unlimited, default)
                        size_t store_size = metadata.buffer_size;
                        if (max_binary_tag_size_ > 0 && metadata.buffer_size > max_binary_tag_size_)
                        {
                            store_size = max_binary_tag_size_;
                            #ifdef DEBUG
                            fmt::print("  ℹ️  TIFF tag {} binary data truncated: {} -> {} bytes\n",
                                      tag_id, metadata.buffer_size, store_size);
                            #endif
                        }
                        std::vector<uint8_t> vec(buffer.begin(), buffer.begin() + store_size);
                        tag_value = std::move(vec);
                    }
                }
                break;
        }
        
        // Check if a value was successfully stored (not monostate)
        if (!std::holds_alternative<std::monostate>(tag_value))
        {
            ifd_info.tiff_tags[tag_name] = std::move(tag_value);
            extracted_count++;
            
            #ifdef DEBUG
            // Format value for debug output
            std::string debug_str = std::visit([](const auto& v) -> std::string {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, std::monostate>)
                    return "<unset>";
                else if constexpr (std::is_same_v<T, std::string>)
                    return v.length() > 60 ? v.substr(0, 60) + "..." : v;
                else if constexpr (std::is_same_v<T, std::vector<uint8_t>>)
                    return fmt::format("[{} bytes]", v.size());
                else if constexpr (std::is_same_v<T, std::vector<uint16_t>>)
                    return fmt::format("[{} uint16 values]", v.size());
                else if constexpr (std::is_same_v<T, std::vector<uint32_t>>)
                    return fmt::format("[{} uint32 values]", v.size());
                else if constexpr (std::is_same_v<T, std::vector<uint64_t>>)
                    return fmt::format("[{} uint64 values]", v.size());
                else if constexpr (std::is_same_v<T, std::vector<float>>)
                    return fmt::format("[{} float values]", v.size());
                else if constexpr (std::is_same_v<T, std::vector<double>>)
                    return fmt::format("[{} double values]", v.size());
                else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
                    return fmt::format("{}", v);
                else
                    return std::to_string(v);
            }, ifd_info.tiff_tags[tag_name]);
            fmt::print("    ✅ Tag {} ({}): {}\n", tag_id, tag_name, debug_str);
            #endif // DEBUG
        }
    }
    
    if (extracted_count > 0)
    {
        #ifdef DEBUG
        fmt::print("  ✅ Extracted {} TIFF tags using nvImageCodec 0.7.0 API\n", extracted_count);
        #endif // DEBUG
        
        // Store ImageDescription if available from tags
        auto desc_it = ifd_info.tiff_tags.find("IMAGEDESCRIPTION");
        if (desc_it != ifd_info.tiff_tags.end() && ifd_info.image_description.empty())
        {
            ifd_info.image_description = tiff_tag_value_to_string(desc_it->second);
        }
        
        return;  // Success
    }
    
    // Fallback: File extension heuristics for older nvImageCodec versions
    #ifdef DEBUG
    fmt::print("  ⚠️  Using file extension heuristics (no TIFF tags retrieved)\n");
    #endif // DEBUG
    
    std::string ext;
    size_t dot_pos = file_path_.rfind('.');
    if (dot_pos != std::string::npos)
    {
        ext = file_path_.substr(dot_pos);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    }
    
    // Aperio SVS, Hamamatsu NDPI, Hamamatsu VMS/VMU typically use JPEG compression
    if (ext == ".svs" || ext == ".ndpi" || ext == ".vms" || ext == ".vmu")
    {
        ifd_info.tiff_tags["COMPRESSION"] = static_cast<uint16_t>(7);  // TIFF_COMPRESSION_JPEG
        #ifdef DEBUG
        fmt::print("  ✅ Inferred JPEG compression (WSI format: {})\n", ext);
        #endif // DEBUG
    }
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
    
    // Also add TIFF_TAG kind if any tags were extracted
    // nvImageCodec 0.7.0: TIFF_TAG = 1 (not 0!)
    if (!ifd_infos_[ifd_index].tiff_tags.empty())
    {
        kinds.insert(kinds.begin(), NVIMGCODEC_METADATA_KIND_TIFF_TAG);
    }
    
    return kinds;
}

std::string TiffFileParser::get_detected_format() const
{
    if (ifd_infos_.empty())
        return "Unknown";
    
    // Check first IFD for vendor-specific metadata
    // nvImageCodec 0.7.0: Use proper enum values
    const auto& kinds = query_metadata_kinds(0);
    
    for (int kind : kinds)
    {
        switch (kind)
        {
            case NVIMGCODEC_METADATA_KIND_MED_APERIO:
                return "Aperio SVS";
            case NVIMGCODEC_METADATA_KIND_MED_PHILIPS:
                return "Philips TIFF";
            case NVIMGCODEC_METADATA_KIND_MED_LEICA:
                return "Leica SCN";
            case NVIMGCODEC_METADATA_KIND_MED_VENTANA:
                return "Ventana";
            case NVIMGCODEC_METADATA_KIND_MED_TRESTLE:
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

#endif // CUCIM_HAS_NVIMGCODEC

} // namespace cuslide2::nvimgcodec

