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

// ============================================================================
// IfdInfo Implementation
// ============================================================================

void IfdInfo::print() const
{
    #ifdef DEBUG
    fmt::print("  IFD[{}]: {}x{}, {} channels, {} bits/sample, codec: {}\n",
               index, width, height, num_channels, bits_per_sample, codec);
    #endif // DEBUG
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
            
            // Check if we have TIFF Compression tag (stored as string key "COMPRESSION")
            auto compression_it = ifd_info.tiff_tags.find("COMPRESSION");
            if (compression_it != ifd_info.tiff_tags.end())
            {
                try
                {
                    // Parse compression value from string
                    uint16_t compression_value = static_cast<uint16_t>(std::stoi(compression_it->second));
                    
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
                catch (const std::exception& e)
                {
                    #ifdef DEBUG
                    fmt::print("  ⚠️  Failed to parse COMPRESSION tag value: {}\n", e.what());
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
                #endif // DEBUG
                #ifdef DEBUG
                fmt::print("     File: {}\n", file_path_);
                #endif // DEBUG
                #ifdef DEBUG
                fmt::print("     This may limit CPU decoder availability.\n");
                #endif // DEBUG
            }
        }
        
        ifd_info.print();
        
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
            case 0: kind_name = "TIFF_TAG"; break;
            case 1: kind_name = "MED_APERIO"; break;
            case 2: kind_name = "MED_PHILIPS"; break;
            case 3: kind_name = "MED_LEICA"; break;
            case 4: kind_name = "MED_VENTANA"; break;
            case 5: kind_name = "MED_TRESTLE"; break;
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
            
            // Special handling: extract ImageDescription if it's a text format
            // nvimgcodecMetadataFormat_t: RAW=0, XML=1, JSON=2, etc.
            // For RAW format, treat as text if it looks like ASCII
            if (kind == 1 && ifd_info.image_description.empty())  // MED_APERIO = 1
            {
                // Aperio metadata is typically in RAW format as text
                ifd_info.image_description.assign(buffer, buffer + buffer_size);
                #ifdef DEBUG
                fmt::print("  ✅ Extracted Aperio ImageDescription ({} bytes)\n", buffer_size);
                #endif
            }
            else if (kind == 2)  // MED_PHILIPS = 2
            {
                // Philips metadata is typically XML
                ifd_info.image_description.assign(buffer, buffer + buffer_size);
                #ifdef DEBUG
                fmt::print("  ✅ Extracted Philips ImageDescription XML ({} bytes)\n", buffer_size);
                
                // Show preview of XML
                if (buffer_size > 0) {
                    std::string preview(buffer, buffer + std::min(buffer_size, size_t(100)));
                    fmt::print("     XML preview: {}...\n", preview);
                }
                #endif
            }
            else if (kind == 3)  // MED_LEICA = 3 (but might be misclassified Aperio!)
            {
                // WORKAROUND: nvImageCodec 0.6.0 sometimes misclassifies Aperio as Leica
                // Check if this is actually Aperio by looking for "Aperio Image Library" text
                if (buffer_size > 20)
                {
                    std::string content(buffer, buffer + std::min(buffer_size, size_t(200)));
                    
                    if (content.find("Aperio Image Library") != std::string::npos ||
                        content.find("Aperio") == 0)  // Starts with "Aperio"
                    {
                        // This is actually Aperio misclassified as Leica!
                        #ifdef DEBUG
                        fmt::print("  ⚠️  nvImageCodec 0.6.0: Aperio misclassified as Leica (corrected)\n");
                        #endif
                        ifd_info.image_description.assign(buffer, buffer + buffer_size);
                        
                        // Also store it as kind=1 (Aperio) for proper detection
                        IfdInfo::MetadataBlob aperio_blob;
                        aperio_blob.format = format;
                        aperio_blob.data.assign(buffer, buffer + buffer_size);
                        ifd_info.metadata_blobs[1] = std::move(aperio_blob);  // Store as MED_APERIO
                    }
                }
            }
            else if (kind == 4)  // MED_VENTANA = 4 (but might be misclassified Philips!)
            {
                // WORKAROUND: nvImageCodec 0.6.0 sometimes misclassifies Philips as Ventana
                // Check if this is actually Philips XML by looking for DataObject/DPUfsImport
                if (buffer_size > 100)
                {
                    std::string content(buffer, buffer + std::min(buffer_size, size_t(500)));
                    
                    if (content.find("<?xml") != std::string::npos && 
                        content.find("DataObject") != std::string::npos &&
                        content.find("DPUfsImport") != std::string::npos)
                    {
                        // This is actually Philips XML misclassified as Ventana!
                        #ifdef DEBUG
                        fmt::print("  ⚠️  nvImageCodec 0.6.0: Philips misclassified as Ventana (corrected)\n");
                        #endif
                        ifd_info.image_description.assign(buffer, buffer + buffer_size);
                        
                        // Also store it as kind=2 (Philips) for proper detection
                        IfdInfo::MetadataBlob philips_blob;
                        philips_blob.format = format;
                        philips_blob.data.assign(buffer, buffer + buffer_size);
                        ifd_info.metadata_blobs[2] = std::move(philips_blob);  // Store as MED_PHILIPS
                    }
                }
            }
        }
    }
    
    // WORKAROUND for nvImageCodec 0.6.0: Philips TIFF metadata limitation
    // ========================================================================
    // nvImageCodec 0.6.0 does NOT expose:
    // 1. Individual TIFF tags (SOFTWARE, ImageDescription, etc.)
    // 2. Philips format detection for some files
    //
    // This means:
    // - If nvImageCodec doesn't detect the file as Philips (kind=2), we can't get the XML
    // - The SOFTWARE tag is not available to trigger Philips detection
    // - ImageDescription tag (270) with Philips XML is not accessible
    //
    // SOLUTION: Upgrade to nvImageCodec 0.7.0+ which supports:
    // - Individual TIFF tag queries (e.g., decoder.get_metadata(scs, name="ImageDescription"))
    // - Better vendor format detection
    // - Full TIFF tag access via kind=0 (TIFF_TAG)
    //
    // Reference: Slack thread Oct 21-Nov 12, 2024 (swdl-image-codecs)
    // - Michal Kepa: "retrieving individual tiff tag is currently only on the internal branch"
    // - Sebastian Matysik: "this is added in 0.7.0 which was not released yet"
    //
    // Until 0.7.0 is available, Philips metadata parsing will only work for files that
    // nvImageCodec successfully detects as Philips format (exposes MED_PHILIPS metadata).
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

// Internal helper for get_detected_format()
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

std::string TiffFileParser::get_tiff_tag(uint32_t ifd_index, const std::string& tag_name) const
{
    if (ifd_index >= ifd_infos_.size())
        return "";
    
    auto it = ifd_infos_[ifd_index].tiff_tags.find(tag_name);
    if (it != ifd_infos_[ifd_index].tiff_tags.end())
        return it->second;
    
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
    
    // Map of TIFF tag IDs to names for common tags we want to extract
    std::map<uint32_t, std::string> tiff_tag_names = {
        {254, "SUBFILETYPE"},      // Image type classification
        {256, "IMAGEWIDTH"},
        {257, "IMAGELENGTH"},
        {258, "BITSPERSAMPLE"},
        {259, "COMPRESSION"},      // ← Critical for codec detection!
        {262, "PHOTOMETRIC"},
        {270, "IMAGEDESCRIPTION"}, // Vendor metadata
        {271, "MAKE"},             // Scanner manufacturer
        {272, "MODEL"},            // Scanner model
        {305, "SOFTWARE"},
        {306, "DATETIME"},
        {322, "TILEWIDTH"},
        {323, "TILELENGTH"},
        {339, "SAMPLEFORMAT"},
        {347, "JPEGTABLES"}        // Shared JPEG tables
    };
    
    // NOTE: nvImageCodec 0.6.0 Limitation (confirmed by NVIDIA team)
    // ================================================================
    // Individual TIFF tag access (kind=0, TIFF_TAG) is NOT available in 0.6.0
    // Only vendor-specific metadata blobs are exposed (MED_APERIO, MED_PHILIPS, etc.)
    // 
    // Individual TIFF tag queries will be available in nvImageCodec 0.7.0 (mid-November 2024):
    // Example: metadata = decoder.get_metadata(scs, name="Compression")
    // 
    // Reference: https://nvidia.slack.com/archives/C092X06LK9U (Oct 27, 2024)
    // - Michal Kepa: "retrieving individual tiff tag is currently only on the internal branch"
    // - Sebastian Matysik: "this is added in 0.7.0 which was not released yet"
    //
    // WORKAROUND for 0.6.0: Use file extension heuristics
    
    int extracted_count = 0;
    
    // File extension heuristics for known WSI (Whole Slide Imaging) formats
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
        ifd_info.tiff_tags["COMPRESSION"] = "7";  // TIFF_COMPRESSION_JPEG
        #ifdef DEBUG
        fmt::print("  ✅ Inferred JPEG compression (WSI format: {})\n", ext);
        #endif // DEBUG
        extracted_count++;
    }
    
    // Store ImageDescription if available
    if (!ifd_info.image_description.empty())
    {
        ifd_info.tiff_tags["IMAGEDESCRIPTION"] = ifd_info.image_description;
    }
    
    // Summary
    if (extracted_count > 0)
    {
        #ifdef DEBUG
        fmt::print("  ✅ Compression detection successful (nvImageCodec 0.6.0 heuristics)\n");
        #endif // DEBUG
    }
    else
    {
        #ifdef DEBUG
        fmt::print("  ⚠️  Unable to determine compression type from file extension\n");
        #endif // DEBUG
        #ifdef DEBUG
        fmt::print("      Upgrade to nvImageCodec 0.7.0 for direct TIFF tag access\n");
        #endif // DEBUG
    }
    
    (void)tiff_tag_names;  // Suppress unused variable warning
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

#endif // CUCIM_HAS_NVIMGCODEC

} // namespace cuslide2::nvimgcodec

