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
    : instance_(nullptr), initialized_(false)
{
    try
    {
        // Create nvImageCodec instance for TIFF parsing
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
        
        initialized_ = true;
        status_message_ = "nvImageCodec TIFF parser initialized successfully";
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
      main_code_stream_(nullptr), decoder_(nullptr)
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
        
        // Step 2: Parse TIFF structure
        parse_tiff_structure();
        
        // Step 3: Create decoder for decoding operations
        nvimgcodecExecutionParams_t exec_params{};
        exec_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS;
        exec_params.struct_size = sizeof(nvimgcodecExecutionParams_t);
        exec_params.struct_next = nullptr;
        exec_params.device_allocator = nullptr;
        exec_params.pinned_allocator = nullptr;
        exec_params.max_num_cpu_threads = 0;  // Use default
        exec_params.executor = nullptr;
        exec_params.device_id = NVIMGCODEC_DEVICE_CURRENT;
        exec_params.pre_init = 0;
        exec_params.skip_pre_sync = 0;
        exec_params.num_backends = 0;
        exec_params.backends = nullptr;
        
        status = nvimgcodecDecoderCreate(manager.get_instance(), &decoder_, 
                                         &exec_params, nullptr);
        
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            throw std::runtime_error(fmt::format("Failed to create decoder (status: {})",
                                                static_cast<int>(status)));
        }
        
        initialized_ = true;
        fmt::print("✅ TIFF parser initialized with {} IFDs\n", ifd_infos_.size());
    }
    catch (const std::exception& e)
    {
        // Cleanup on error
        if (decoder_)
        {
            nvimgcodecDecoderDestroy(decoder_);
            decoder_ = nullptr;
        }
        
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
    // Destroy decoder
    if (decoder_)
    {
        nvimgcodecDecoderDestroy(decoder_);
        decoder_ = nullptr;
    }
    
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
        
        ifd_info.print();
        
        ifd_infos_.push_back(std::move(ifd_info));
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

bool TiffFileParser::decode_ifd(uint32_t ifd_index,
                                uint8_t** output_buffer,
                                const cucim::io::Device& out_device)
{
    if (!initialized_)
    {
        fmt::print("❌ TIFF parser not initialized\n");
        return false;
    }
    
    if (ifd_index >= ifd_infos_.size())
    {
        fmt::print("❌ IFD index {} out of range (have {} IFDs)\n", 
                  ifd_index, ifd_infos_.size());
        return false;
    }
    
    const auto& ifd = ifd_infos_[ifd_index];
    
    fmt::print("🚀 Decoding IFD[{}]: {}x{}, codec: {}\n",
              ifd_index, ifd.width, ifd.height, ifd.codec);
    
    try
    {
        // Step 1: Prepare output image info
        nvimgcodecImageInfo_t output_image_info{};
        output_image_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
        output_image_info.struct_size = sizeof(nvimgcodecImageInfo_t);
        output_image_info.struct_next = nullptr;
        
        // Use interleaved RGB format (learned from bug fix!)
        output_image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
        output_image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        output_image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
        output_image_info.num_planes = 1;  // Interleaved RGB is a single plane
        
        // Set buffer kind based on output device
        std::string device_str = std::string(out_device);
        if (device_str.find("cuda") != std::string::npos)
        {
            output_image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
        }
        else
        {
            output_image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
        }
        
        // Calculate buffer requirements for interleaved RGB
        uint32_t num_channels = 3;  // RGB
        size_t row_stride = ifd.width * num_channels;  // Correct stride!
        size_t buffer_size = row_stride * ifd.height;
        
        output_image_info.plane_info[0].height = ifd.height;
        output_image_info.plane_info[0].width = ifd.width;
        output_image_info.plane_info[0].num_channels = num_channels;
        output_image_info.plane_info[0].row_stride = row_stride;
        output_image_info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        output_image_info.buffer_size = buffer_size;
        output_image_info.cuda_stream = 0;  // Default stream
        
        fmt::print("  Buffer: {}x{} RGB, stride={}, size={} bytes\n",
                  ifd.width, ifd.height, row_stride, buffer_size);
        
        // Step 2: Allocate output buffer
        void* buffer = nullptr;
        if (output_image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE)
        {
            cudaError_t cuda_status = cudaMalloc(&buffer, buffer_size);
            if (cuda_status != cudaSuccess)
            {
                fmt::print("❌ Failed to allocate GPU memory: {}\n", 
                          cudaGetErrorString(cuda_status));
                return false;
            }
            fmt::print("  Allocated GPU buffer\n");
        }
        else
        {
            buffer = malloc(buffer_size);
            if (!buffer)
            {
                fmt::print("❌ Failed to allocate host memory\n");
                return false;
            }
            fmt::print("  Allocated CPU buffer\n");
        }
        
        output_image_info.buffer = buffer;
        
        // Step 3: Create image object
        nvimgcodecImage_t image;
        nvimgcodecStatus_t status = nvimgcodecImageCreate(
            NvImageCodecTiffParserManager::instance().get_instance(),
            &image,
            &output_image_info
        );
        
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            fmt::print("❌ Failed to create image object (status: {})\n",
                      static_cast<int>(status));
            if (output_image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE)
            {
                cudaFree(buffer);
            }
            else
            {
                free(buffer);
            }
            return false;
        }
        
        // Step 4: Prepare decode parameters
        nvimgcodecDecodeParams_t decode_params{};
        decode_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS;
        decode_params.struct_size = sizeof(nvimgcodecDecodeParams_t);
        decode_params.struct_next = nullptr;
        decode_params.apply_exif_orientation = 1;
        
        // Step 5: Schedule decoding
        nvimgcodecFuture_t decode_future;
        status = nvimgcodecDecoderDecode(decoder_,
                                        &ifd.sub_code_stream,
                                        &image,
                                        1,
                                        &decode_params,
                                        &decode_future);
        
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            fmt::print("❌ Failed to schedule decoding (status: {})\n",
                      static_cast<int>(status));
            nvimgcodecImageDestroy(image);
            if (output_image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE)
            {
                cudaFree(buffer);
            }
            else
            {
                free(buffer);
            }
            return false;
        }
        
        // Step 6: Wait for completion
        nvimgcodecProcessingStatus_t decode_status;
        size_t status_size;
        nvimgcodecFutureGetProcessingStatus(decode_future, &decode_status, &status_size);
        
        if (output_image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE)
        {
            cudaDeviceSynchronize();  // Wait for GPU operations
        }
        
        // Cleanup
        nvimgcodecFutureDestroy(decode_future);
        nvimgcodecImageDestroy(image);
        
        if (decode_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
        {
            fmt::print("❌ Decoding failed (status: {})\n", static_cast<int>(decode_status));
            if (output_image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE)
            {
                cudaFree(buffer);
            }
            else
            {
                free(buffer);
            }
            return false;
        }
        
        // Success! Return buffer to caller
        *output_buffer = static_cast<uint8_t*>(buffer);
        
        fmt::print("✅ Successfully decoded IFD[{}]\n", ifd_index);
        return true;
    }
    catch (const std::exception& e)
    {
        fmt::print("❌ Exception during decode: {}\n", e.what());
        return false;
    }
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

