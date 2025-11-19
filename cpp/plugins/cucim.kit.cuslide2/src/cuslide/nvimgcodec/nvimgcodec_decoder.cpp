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

#include "nvimgcodec_decoder.h"
#include "nvimgcodec_tiff_parser.h"
#include "nvimgcodec_manager.h"

#ifdef CUCIM_HAS_NVIMGCODEC
#include <nvimgcodec.h>
#endif

#include <memory>
#include <vector>
#include <cstring>
#include <string>
#include <stdexcept>
#include <vector>
#include <unistd.h>
#include <mutex>
#include <fmt/format.h>

#ifdef CUCIM_HAS_NVIMGCODEC
#include <cuda_runtime.h>
#endif

namespace cuslide2::nvimgcodec
{

#ifdef CUCIM_HAS_NVIMGCODEC

// NvImageCodecManager is now defined in nvimgcodec_manager.h (shared across decoder and tiff_parser)

// Global TiffFileParser cache for nvTiff file-level API
// This avoids re-parsing the same TIFF file for every tile
static std::mutex parser_cache_mutex;
static std::map<std::string, std::shared_ptr<TiffFileParser>> parser_cache;

bool decode_tile_nvtiff_roi(const char* file_path,
                            uint32_t ifd_index,
                            uint32_t tile_x, uint32_t tile_y,
                            uint32_t tile_width, uint32_t tile_height,
                            uint8_t** dest,
                            const cucim::io::Device& out_device)
{
    if (!file_path || !dest)
    {
        return false;
    }
    
    try
    {
        // Get or create TiffFileParser for this file
        std::shared_ptr<TiffFileParser> parser;
        {
            std::lock_guard<std::mutex> lock(parser_cache_mutex);
            auto it = parser_cache.find(file_path);
            if (it != parser_cache.end())
            {
                parser = it->second;
            }
            else
            {
                parser = std::make_shared<TiffFileParser>(file_path);
                if (!parser->is_valid())
                {
                    #ifdef DEBUG
                    fmt::print("⚠️  nvTiff ROI: Failed to parse TIFF file: {}\n", file_path);
                    #endif // DEBUG
                    return false;
                }
                parser_cache[file_path] = parser;
                #ifdef DEBUG
                fmt::print("✅ nvTiff ROI: Cached TIFF parser for {}\n", file_path);
                #endif // DEBUG
            }
        }
        
        // Check if IFD index is valid
        if (ifd_index >= parser->get_ifd_count())
        {
            #ifdef DEBUG
            fmt::print("⚠️  nvTiff ROI: Invalid IFD index {} (max: {})\n", 
                      ifd_index, parser->get_ifd_count() - 1);
            #endif // DEBUG
            return false;
        }
        
        // Decode the tile region using nvTiff file-level API
        *dest = parser->decode_region(ifd_index, tile_x, tile_y, 
                                      tile_width, tile_height, 
                                      nullptr, out_device);
        
        return (*dest != nullptr);
    }
    catch (const std::exception& e)
    {
        #ifdef DEBUG
        fmt::print("❌ nvTiff ROI decode failed: {}\n", e.what());
        #endif // DEBUG
        return false;
    }
}

bool decode_jpeg_nvimgcodec(int fd,
                            unsigned char* jpeg_buf,
                            uint64_t offset,
                            uint64_t size,
                            const void* jpegtable_data,
                            uint32_t jpegtable_count,
                            uint8_t** dest,
                            const cucim::io::Device& out_device,
                            int jpeg_color_space)
{
    // Get nvImageCodec manager instance
    auto& manager = NvImageCodecManager::instance();
    
    if (!manager.is_initialized())
    {
        #ifdef DEBUG
        fmt::print("⚠️  nvImageCodec JPEG decode: API not available - {}\n", manager.get_status());
        #endif // DEBUG
        return false; // Fallback to original decoder
    }
    
    // NOTE: JPEG-compressed TIFF files with JPEGTables (abbreviated JPEG) are now handled
    // by the nvTiff decoder path (decode_ifd_region_nvimgcodec) which automatically manages
    // JPEG tables. This function is primarily for standalone JPEG files.
    // 
    // If called with JPEG tables present, fall back to libjpeg-turbo which handles them natively.
    if (jpegtable_data && jpegtable_count > 0) {
        #ifdef DEBUG
        fmt::print("ℹ️  JPEG with tables detected - falling back to libjpeg-turbo\n");
        #endif // DEBUG
        #ifdef DEBUG
        fmt::print("   (TIFF files should use nvTiff path for automatic table handling)\n");
        #endif // DEBUG
        return false; // Fallback to libjpeg-turbo
    }
    
    #ifdef DEBUG
    fmt::print("🚀 nvImageCodec JPEG decode: Starting, size={} bytes, device={}\n", 
              size, std::string(out_device));
    #endif // DEBUG
    
    try {
        // Step 1: Create code stream from memory buffer (following official API pattern)
        nvimgcodecCodeStream_t code_stream;
        
        // Read JPEG data into buffer if needed
        std::vector<uint8_t> jpeg_data;
        if (jpeg_buf) {
            jpeg_data.assign(jpeg_buf, jpeg_buf + size);
        } else {
            // Read from file descriptor at offset
            jpeg_data.resize(size);
            if (lseek(fd, offset, SEEK_SET) == -1) {
                #ifdef DEBUG
                fmt::print("❌ nvImageCodec JPEG decode: Failed to seek in file\n");
                #endif // DEBUG
                return false;
            }
            if (read(fd, jpeg_data.data(), size) != static_cast<ssize_t>(size)) {
                #ifdef DEBUG
                fmt::print("❌ nvImageCodec JPEG decode: Failed to read JPEG data\n");
                #endif // DEBUG
                return false;
            }
        }
        
        // Validate JPEG data before creating code stream
        if (jpeg_data.size() < 4 || jpeg_data.empty()) {
            #ifdef DEBUG
            fmt::print("❌ nvImageCodec JPEG decode: Invalid JPEG data size: {} bytes\n", jpeg_data.size());
            #endif // DEBUG
            return false;
        }
        
        // Create code stream from memory
        nvimgcodecStatus_t status = nvimgcodecCodeStreamCreateFromHostMem(
            manager.get_instance(), &code_stream, jpeg_data.data(), jpeg_data.size());
            
        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            #ifdef DEBUG
            fmt::print("❌ nvImageCodec JPEG decode: Failed to create code stream (status: {})\n", 
                      static_cast<int>(status));
            #endif // DEBUG
            #ifdef DEBUG
            fmt::print("💡 Falling back to libjpeg-turbo decoder\n");
            #endif // DEBUG
            return false; // Fallback to libjpeg-turbo
        }
        
        // Step 2: Get image information (following official API pattern)
        nvimgcodecImageInfo_t input_image_info{};
        input_image_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
        input_image_info.struct_size = sizeof(nvimgcodecImageInfo_t);
        input_image_info.struct_next = nullptr;
        if (nvimgcodecCodeStreamGetImageInfo(code_stream, &input_image_info) != NVIMGCODEC_STATUS_SUCCESS) {
            #ifdef DEBUG
            fmt::print("❌ nvImageCodec JPEG decode: Failed to get image info\n");
            #endif // DEBUG
            nvimgcodecCodeStreamDestroy(code_stream);
            return false;
        }
        
        #ifdef DEBUG
        fmt::print("✅ nvImageCodec JPEG decode: Image info - {}x{}, {} planes, codec: {}\n",
                  input_image_info.plane_info[0].width, input_image_info.plane_info[0].height,
                  input_image_info.num_planes, input_image_info.codec_name);
        #endif // DEBUG
        
        // Step 3: Prepare output image info (following official API pattern)
        nvimgcodecImageInfo_t output_image_info(input_image_info);
        // FIX: Use interleaved RGB format instead of planar
        output_image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
        
        // Map jpeg_color_space to nvImageCodec color spec
        // JPEG color spaces: JPEG_CS_UNKNOWN=0, JPEG_CS_GRAYSCALE=1, JPEG_CS_RGB=2, JPEG_CS_YCbCr=3, JPEG_CS_CMYK=4, JPEG_CS_YCCK=5
        switch (jpeg_color_space) {
            case 1: // Grayscale
                output_image_info.color_spec = NVIMGCODEC_COLORSPEC_GRAY;
                break;
            case 2: // RGB
                output_image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
                break;
            case 3: // YCbCr
                output_image_info.color_spec = NVIMGCODEC_COLORSPEC_SYCC;
                break;
            default: // Unknown or other
                output_image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
                #ifdef DEBUG
                fmt::print("⚠️  nvImageCodec JPEG decode: Unknown color space {}, defaulting to sRGB\n", jpeg_color_space);
                #endif // DEBUG
                break;
        }
        #ifdef DEBUG
        fmt::print("📋 nvImageCodec JPEG decode: Using color space {} (input JPEG color space: {})\n", 
                  static_cast<int>(output_image_info.color_spec), jpeg_color_space);
        #endif // DEBUG
        
        output_image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
        output_image_info.num_planes = 1;  // Interleaved RGB is a single plane
        
        // Set buffer kind based on output device
        std::string device_str = std::string(out_device);
        if (device_str.find("cuda") != std::string::npos) {
            output_image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
        } else {
            output_image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
        }
        
        // Calculate buffer requirements for interleaved RGB
        // Note: width/height already copied from input_image_info via copy constructor
        // Reference: Michal Kepa feedback - only update row_stride, buffer_size, num_channels, buffer
        auto sample_type = output_image_info.plane_info[0].sample_type;
        int bytes_per_element = static_cast<unsigned int>(sample_type) >> (8+3);
        uint32_t width = output_image_info.plane_info[0].width;   // Already set from input_image_info
        uint32_t height = output_image_info.plane_info[0].height; // Already set from input_image_info
        uint32_t num_channels = 3;  // RGB
        
        // For interleaved RGB: row_stride = width * channels * bytes_per_element
        size_t row_stride = width * num_channels * bytes_per_element;
        
        // Update only the fields that differ from input (width/height already correct)
        output_image_info.plane_info[0].num_channels = num_channels;
        output_image_info.plane_info[0].row_stride = row_stride;
        output_image_info.buffer_size = row_stride * height;
        output_image_info.cuda_stream = 0; // Default stream
        
        // Use pre-allocated buffer if provided, otherwise allocate new buffer
        void* output_buffer = *dest;  // Check if caller provided a pre-allocated buffer
        bool buffer_was_preallocated = (output_buffer != nullptr);
        
        if (!buffer_was_preallocated) {
            // Allocate output buffer only if not pre-allocated
            if (output_image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
                if (cudaMalloc(&output_buffer, output_image_info.buffer_size) != cudaSuccess) {
                    #ifdef DEBUG
                    fmt::print("❌ nvImageCodec JPEG decode: Failed to allocate GPU memory\n");
                    #endif // DEBUG
                    nvimgcodecCodeStreamDestroy(code_stream);
                    return false;
                }
            } else {
                output_buffer = malloc(output_image_info.buffer_size);
                if (!output_buffer) {
                    #ifdef DEBUG
                    fmt::print("❌ nvImageCodec JPEG decode: Failed to allocate host memory\n");
                    #endif // DEBUG
                    nvimgcodecCodeStreamDestroy(code_stream);
                    return false;
                }
            }
        }
        
        output_image_info.buffer = output_buffer;
        
        // Step 4: Create image object (following official API pattern)
        nvimgcodecImage_t image;
        if (nvimgcodecImageCreate(manager.get_instance(), &image, &output_image_info) != NVIMGCODEC_STATUS_SUCCESS) {
            #ifdef DEBUG
            fmt::print("❌ nvImageCodec JPEG decode: Failed to create image object\n");
            #endif // DEBUG
            if (!buffer_was_preallocated) {
                if (output_image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
                    cudaFree(output_buffer);
                } else {
                    free(output_buffer);
                }
            }
            nvimgcodecCodeStreamDestroy(code_stream);
            return false;
        }
        
        // Step 5: Prepare decode parameters (following official API pattern)
        nvimgcodecDecodeParams_t decode_params{};
        decode_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS;
        decode_params.struct_size = sizeof(nvimgcodecDecodeParams_t);
        decode_params.struct_next = nullptr;
        decode_params.apply_exif_orientation = 1;
        
        // Step 6: Schedule decoding (following official API pattern)
        // THREAD-SAFETY: Lock the decoder to prevent concurrent access from multiple threads
        nvimgcodecFuture_t decode_future;
        {
            std::lock_guard<std::mutex> lock(manager.get_mutex());
            if (nvimgcodecDecoderDecode(manager.get_decoder(), &code_stream, &image, 1, &decode_params, &decode_future) != NVIMGCODEC_STATUS_SUCCESS) {
                #ifdef DEBUG
                fmt::print("❌ nvImageCodec JPEG decode: Failed to schedule decoding\n");
                #endif // DEBUG
                nvimgcodecImageDestroy(image);
                if (!buffer_was_preallocated) {
                    if (output_image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
                        cudaFree(output_buffer);
                    } else {
                        free(output_buffer);
                    }
                }
                nvimgcodecCodeStreamDestroy(code_stream);
                return false;
            }
        }
        
        // Step 7: Wait for decoding to finish (following official API pattern)
        size_t status_size = 1;
        nvimgcodecProcessingStatus_t decode_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
        
        // Safely get processing status with validation
        nvimgcodecStatus_t future_status = nvimgcodecFutureGetProcessingStatus(
            decode_future, &decode_status, &status_size);
            
        if (future_status != NVIMGCODEC_STATUS_SUCCESS) {
            #ifdef DEBUG
            fmt::print("❌ nvImageCodec JPEG decode: Failed to get future status (code: {})\n", 
                      static_cast<int>(future_status));
            #endif // DEBUG
            nvimgcodecFutureDestroy(decode_future);
            nvimgcodecImageDestroy(image);
            if (!buffer_was_preallocated) {
                if (output_image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
                    cudaFree(output_buffer);
                } else {
                    free(output_buffer);
                }
            }
            nvimgcodecCodeStreamDestroy(code_stream);
            #ifdef DEBUG
            fmt::print("💡 Falling back to libjpeg-turbo decoder\n");
            #endif // DEBUG
            return false;
        }
        
        // Synchronize only if we're on GPU
        if (output_image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
            cudaError_t cuda_err = cudaDeviceSynchronize();
            if (cuda_err != cudaSuccess) {
                #ifdef DEBUG
                fmt::print("⚠️  CUDA synchronization warning: {}\n", cudaGetErrorString(cuda_err));
                #endif // DEBUG
            }
        }
        
        if (decode_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
            #ifdef DEBUG
            fmt::print("❌ nvImageCodec JPEG decode: Processing failed with status: {}\n", 
                      static_cast<int>(decode_status));
            #endif // DEBUG
            nvimgcodecFutureDestroy(decode_future);
            nvimgcodecImageDestroy(image);
            if (!buffer_was_preallocated) {
                if (output_image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
                    cudaFree(output_buffer);
                } else {
                    free(output_buffer);
                }
            }
            nvimgcodecCodeStreamDestroy(code_stream);
            #ifdef DEBUG
            fmt::print("💡 Falling back to libjpeg-turbo decoder\n");
            #endif // DEBUG
            return false;
        }
        
        // Success! Set output pointer
        *dest = static_cast<uint8_t*>(output_buffer);
        
        #ifdef DEBUG
        fmt::print("✅ nvImageCodec JPEG decode: Successfully decoded {}x{} image\n",
                  output_image_info.plane_info[0].width, output_image_info.plane_info[0].height);
        #endif // DEBUG
        
        // Cleanup (but keep the output buffer for caller)
        nvimgcodecFutureDestroy(decode_future);
        nvimgcodecImageDestroy(image);
        nvimgcodecCodeStreamDestroy(code_stream);
        
        return true; // Success!
        
    } catch (const std::exception& e) {
        #ifdef DEBUG
        fmt::print("❌ nvImageCodec JPEG decode: Exception - {}\n", e.what());
        #endif // DEBUG
        return false;
    }
}

bool decode_jpeg2k_nvimgcodec(int fd,
                              unsigned char* jpeg2k_buf,
                              uint64_t offset,
                              uint64_t size,
                              uint8_t** dest,
                              size_t dest_size,
                              const cucim::io::Device& out_device,
                              int color_space)
{
    #ifdef DEBUG
    fmt::print("🔍 decode_jpeg2k_nvimgcodec: ENTRY - fd={}, offset={}, size={}\n", fd, offset, size);
    #endif // DEBUG
    
    // Get nvImageCodec manager instance
    #ifdef DEBUG
    fmt::print("🔍 decode_jpeg2k_nvimgcodec: Getting manager instance...\n");
    #endif // DEBUG
    auto& manager = NvImageCodecManager::instance();
    #ifdef DEBUG
    fmt::print("🔍 decode_jpeg2k_nvimgcodec: Got manager instance\n");
    #endif // DEBUG
    
    if (!manager.is_initialized())
    {
        #ifdef DEBUG
        fmt::print("⚠️  nvImageCodec JPEG2000 decode: API not available - {}\n", manager.get_status());
        #endif // DEBUG
        return false; // Fallback to original decoder
    }
    
    #ifdef DEBUG
    fmt::print("🚀 nvImageCodec JPEG2000 decode: Starting, size={} bytes, device={}\n", 
              size, std::string(out_device));
    #endif // DEBUG
    
    try {
        // Step 1: Create code stream from memory buffer (following official API pattern)
        nvimgcodecCodeStream_t code_stream;
        
        // Read JPEG2000 data into buffer if needed
        std::vector<uint8_t> jpeg2k_data;
        if (jpeg2k_buf) {
            jpeg2k_data.assign(jpeg2k_buf, jpeg2k_buf + size);
        } else {
            // Read from file descriptor at offset
            jpeg2k_data.resize(size);
            if (lseek(fd, offset, SEEK_SET) == -1) {
                #ifdef DEBUG
                fmt::print("❌ nvImageCodec JPEG2000 decode: Failed to seek in file\n");
                #endif // DEBUG
                return false;
            }
            if (read(fd, jpeg2k_data.data(), size) != static_cast<ssize_t>(size)) {
                #ifdef DEBUG
                fmt::print("❌ nvImageCodec JPEG2000 decode: Failed to read JPEG2000 data\n");
                #endif // DEBUG
                return false;
            }
        }
        
        // Create code stream from memory
        if (nvimgcodecCodeStreamCreateFromHostMem(manager.get_instance(), &code_stream, 
                                                 jpeg2k_data.data(), jpeg2k_data.size()) != NVIMGCODEC_STATUS_SUCCESS) {
            #ifdef DEBUG
            fmt::print("❌ nvImageCodec JPEG2000 decode: Failed to create code stream\n");
            #endif // DEBUG
            return false;
        }
        
        // Step 2: Get image information (following official API pattern)
        nvimgcodecImageInfo_t input_image_info{};
        input_image_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
        input_image_info.struct_size = sizeof(nvimgcodecImageInfo_t);
        input_image_info.struct_next = nullptr;
        if (nvimgcodecCodeStreamGetImageInfo(code_stream, &input_image_info) != NVIMGCODEC_STATUS_SUCCESS) {
            #ifdef DEBUG
            fmt::print("❌ nvImageCodec JPEG2000 decode: Failed to get image info\n");
            #endif // DEBUG
            nvimgcodecCodeStreamDestroy(code_stream);
            return false;
        }
        
        #ifdef DEBUG
        fmt::print("✅ nvImageCodec JPEG2000 decode: Image info - {}x{}, {} planes, codec: {}\n",
                  input_image_info.plane_info[0].width, input_image_info.plane_info[0].height,
                  input_image_info.num_planes, input_image_info.codec_name);
        #endif // DEBUG
        
        // Step 3: Prepare output image info (following official API pattern)
        nvimgcodecImageInfo_t output_image_info(input_image_info);
        // FIX: Use interleaved RGB format instead of planar
        output_image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
        
        // Map color_space to nvImageCodec color spec
        // Caller convention (from ifd.cpp): 0=RGB, 1=YCbCr
        switch (color_space) {
            case 0: // RGB (Aperio JPEG2000 RGB format - 33005)
                output_image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
                #ifdef DEBUG
                fmt::print("📋 nvImageCodec JPEG2000 decode: Using sRGB color space\n");
                #endif // DEBUG
                break;
            case 1: // YCbCr (Aperio JPEG2000 YCbCr format - 33003)
                output_image_info.color_spec = NVIMGCODEC_COLORSPEC_SYCC;
                #ifdef DEBUG
                fmt::print("📋 nvImageCodec JPEG2000 decode: Using YCbCr color space\n");
                #endif // DEBUG
                break;
            default: // Unknown or other
                output_image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
                #ifdef DEBUG
                fmt::print("⚠️  nvImageCodec JPEG2000 decode: Unknown color space {}, defaulting to sRGB\n", color_space);
                #endif // DEBUG
                break;
        }
        
        output_image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
        output_image_info.num_planes = 1;  // Interleaved RGB is a single plane
        
        // Set buffer kind based on output device
        std::string device_str = std::string(out_device);
        if (device_str.find("cuda") != std::string::npos) {
            output_image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
        } else {
            output_image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
        }
        
        // Calculate buffer requirements for interleaved RGB
        // Note: width/height already copied from input_image_info via copy constructor
        // Reference: Michal Kepa feedback - only update row_stride, buffer_size, num_channels, buffer
        auto sample_type = output_image_info.plane_info[0].sample_type;
        int bytes_per_element = static_cast<unsigned int>(sample_type) >> (8+3);
        uint32_t width = output_image_info.plane_info[0].width;   // Already set from input_image_info
        uint32_t height = output_image_info.plane_info[0].height; // Already set from input_image_info
        uint32_t num_channels = 3;  // RGB
        
        // For interleaved RGB: row_stride = width * channels * bytes_per_element
        size_t row_stride = width * num_channels * bytes_per_element;
        
        // Update only the fields that differ from input (width/height already correct)
        output_image_info.plane_info[0].num_channels = num_channels;
        output_image_info.plane_info[0].row_stride = row_stride;
        output_image_info.buffer_size = row_stride * height;
        output_image_info.cuda_stream = 0; // Default stream
        
        // Use pre-allocated buffer if provided, otherwise allocate new buffer
        void* output_buffer = *dest;  // Check if caller provided a pre-allocated buffer
        bool buffer_was_preallocated = (output_buffer != nullptr);
        
        if (!buffer_was_preallocated) {
            // Allocate output buffer only if not pre-allocated
            if (output_image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
                if (cudaMalloc(&output_buffer, output_image_info.buffer_size) != cudaSuccess) {
                    #ifdef DEBUG
                    fmt::print("❌ nvImageCodec JPEG2000 decode: Failed to allocate GPU memory\n");
                    #endif // DEBUG
                    nvimgcodecCodeStreamDestroy(code_stream);
                    return false;
                }
            } else {
                output_buffer = malloc(output_image_info.buffer_size);
                if (!output_buffer) {
                    #ifdef DEBUG
                    fmt::print("❌ nvImageCodec JPEG2000 decode: Failed to allocate host memory\n");
                    #endif // DEBUG
                    nvimgcodecCodeStreamDestroy(code_stream);
                    return false;
                }
            }
        }
        
        output_image_info.buffer = output_buffer;
        
        // Step 4: Create image object (following official API pattern)
        nvimgcodecImage_t image;
        if (nvimgcodecImageCreate(manager.get_instance(), &image, &output_image_info) != NVIMGCODEC_STATUS_SUCCESS) {
            #ifdef DEBUG
            fmt::print("❌ nvImageCodec JPEG2000 decode: Failed to create image object\n");
            #endif // DEBUG
            if (!buffer_was_preallocated) {
                if (output_image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
                    cudaFree(output_buffer);
                } else {
                    free(output_buffer);
                }
            }
            nvimgcodecCodeStreamDestroy(code_stream);
            return false;
        }
        
        // Step 5: Prepare decode parameters (following official API pattern)
        nvimgcodecDecodeParams_t decode_params{};
        decode_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS;
        decode_params.struct_size = sizeof(nvimgcodecDecodeParams_t);
        decode_params.struct_next = nullptr;
        decode_params.apply_exif_orientation = 1;
        
        // Step 6: Schedule decoding (following official API pattern)
        // THREAD-SAFETY: Lock the decoder to prevent concurrent access from multiple threads
        nvimgcodecFuture_t decode_future;
        {
            std::lock_guard<std::mutex> lock(manager.get_mutex());
            #ifdef DEBUG
            fmt::print("📍 About to call nvimgcodecDecoderDecode...\n");
            #endif // DEBUG
            if (nvimgcodecDecoderDecode(manager.get_decoder(), &code_stream, &image, 1, &decode_params, &decode_future) != NVIMGCODEC_STATUS_SUCCESS) {
                #ifdef DEBUG
                fmt::print("❌ nvImageCodec JPEG2000 decode: Failed to schedule decoding\n");
                #endif // DEBUG
                nvimgcodecImageDestroy(image);
                if (!buffer_was_preallocated) {
                    if (output_image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
                        cudaFree(output_buffer);
                    } else {
                        free(output_buffer);
                    }
                }
                nvimgcodecCodeStreamDestroy(code_stream);
                return false;
            }
            #ifdef DEBUG
            fmt::print("📍 nvimgcodecDecoderDecode returned successfully\n");
            #endif // DEBUG
        }
        
        // Step 7: Wait for decoding to finish (following official API pattern)
        #ifdef DEBUG
        fmt::print("📍 Getting processing status...\n");
        #endif // DEBUG
        size_t status_size = 1;
        nvimgcodecProcessingStatus_t decode_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
        nvimgcodecFutureGetProcessingStatus(decode_future, &decode_status, &status_size);
        #ifdef DEBUG
        fmt::print("📍 Got processing status: {}\n", static_cast<int>(decode_status));
        #endif // DEBUG
        
        // Only synchronize if decoding to GPU memory
        if (output_image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
            #ifdef DEBUG
            fmt::print("📍 Calling cudaDeviceSynchronize for GPU buffer...\n");
            #endif // DEBUG
            cudaDeviceSynchronize(); // Wait for GPU operations to complete
            #ifdef DEBUG
            fmt::print("📍 cudaDeviceSynchronize completed\n");
            #endif // DEBUG
        } else {
            #ifdef DEBUG
            fmt::print("📍 Skipping cudaDeviceSynchronize for CPU buffer\n");
            #endif // DEBUG
        }
        
        if (decode_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
            #ifdef DEBUG
            fmt::print("❌ nvImageCodec JPEG2000 decode: Processing failed with status: {}\n", decode_status);
            #endif // DEBUG
            nvimgcodecFutureDestroy(decode_future);
            nvimgcodecImageDestroy(image);
            if (!buffer_was_preallocated) {
                if (output_image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
                    cudaFree(output_buffer);
                } else {
                    free(output_buffer);
                }
            }
            nvimgcodecCodeStreamDestroy(code_stream);
            return false;
        }
        
        // Success! Set output pointer
        *dest = static_cast<uint8_t*>(output_buffer);
        
        #ifdef DEBUG
        fmt::print("✅ nvImageCodec JPEG2000 decode: Successfully decoded {}x{} image\n",
                  output_image_info.plane_info[0].width, output_image_info.plane_info[0].height);
        #endif // DEBUG
        
        // Cleanup (but keep the output buffer for caller)
        nvimgcodecFutureDestroy(decode_future);
        nvimgcodecImageDestroy(image);
        nvimgcodecCodeStreamDestroy(code_stream);
        
        return true; // Success!
        
    } catch (const std::exception& e) {
        #ifdef DEBUG
        fmt::print("❌ nvImageCodec JPEG2000 decode: Exception - {}\n", e.what());
        #endif // DEBUG
        return false;
    }
    
    // Suppress unused parameter warning (dest_size not currently used)
    (void)dest_size;
}

// ============================================================================
// IFD-Level Decoding Functions (Parsing-Decoder Separation)
// ============================================================================

bool decode_ifd_nvimgcodec(const IfdInfo& ifd_info,
                           uint8_t** output_buffer,
                           const cucim::io::Device& out_device)
{
    if (!ifd_info.sub_code_stream)
    {
        #ifdef DEBUG
        fmt::print("❌ IFD info has no sub_code_stream\n");
        #endif // DEBUG
        return false;
    }
    
    #ifdef DEBUG
    fmt::print("🚀 Decoding IFD[{}]: {}x{}, codec: {}\n",
              ifd_info.index, ifd_info.width, ifd_info.height, ifd_info.codec);
    #endif // DEBUG
    
    try
    {
        // CRITICAL: Must use the same manager that created the sub_code_stream
        // Using a decoder from a different nvImageCodec instance causes segfaults.
        auto& manager = NvImageCodecTiffParserManager::instance();
        if (!manager.is_available())
        {
            #ifdef DEBUG
            fmt::print("❌ nvImageCodec TIFF parser manager not initialized\n");
            #endif // DEBUG
            return false;
        }
        
        // Intelligent decoder selection:
        // 1. Use CPU decoder if: (a) CPU output requested, OR (b) GPU not available
        // 2. Use hybrid decoder (GPU) only when GPU is available and needed
        std::string device_str = std::string(out_device);
        bool target_is_gpu = (device_str.find("cuda") != std::string::npos);
        bool needs_cpu_fallback = !target_is_gpu;
        
        // Check if GPU is available
        int device_count = 0;
        cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
        bool gpu_available = (cuda_err == cudaSuccess && device_count > 0);
        
        nvimgcodecDecoder_t decoder;
        
        if (!target_is_gpu || !gpu_available)
        {
            // Prefer CPU decoder when: requesting CPU output OR GPU unavailable
            if (manager.has_cpu_decoder())
            {
                decoder = manager.get_cpu_decoder();
                if (!gpu_available)
                {
                    #ifdef DEBUG
                    fmt::print("  💡 GPU unavailable, using CPU-only decoder (native libjpeg-turbo)\n");
                    #endif // DEBUG
                }
                else
                {
                    #ifdef DEBUG
                    fmt::print("  💡 CPU output requested, using CPU-only decoder (native libjpeg-turbo)\n");
                    #endif // DEBUG
                }
            }
            else
            {
                decoder = manager.get_decoder();
                #ifdef DEBUG
                fmt::print("  ⚠️  CPU decoder not available, will use hybrid decoder\n");
                #endif // DEBUG
            }
        }
        else
        {
            // GPU is available and GPU output requested
            decoder = manager.get_decoder();
            #ifdef DEBUG
            fmt::print("  💡 Using hybrid decoder (GPU-accelerated)\n");
            #endif // DEBUG
        }
        
        // Step 1: Determine buffer kind
        nvimgcodecImageBufferKind_t buffer_kind;
        if (target_is_gpu)
        {
            buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
        }
        else
        {
            // CPU output - will use CPU decoder if available
            buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
        }
        
        // Step 2: Prepare output image info
        nvimgcodecImageInfo_t output_image_info{};
        output_image_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
        output_image_info.struct_size = sizeof(nvimgcodecImageInfo_t);
        output_image_info.struct_next = nullptr;
        
        // Use interleaved RGB format
        output_image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
        output_image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        output_image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
        output_image_info.num_planes = 1;  // Interleaved RGB is a single plane
        output_image_info.buffer_kind = buffer_kind;
        
        // Calculate buffer requirements for interleaved RGB
        uint32_t num_channels = 3;  // RGB
        size_t row_stride = ifd_info.width * num_channels;
        size_t buffer_size = row_stride * ifd_info.height;
        
        output_image_info.plane_info[0].height = ifd_info.height;
        output_image_info.plane_info[0].width = ifd_info.width;
        output_image_info.plane_info[0].num_channels = num_channels;
        output_image_info.plane_info[0].row_stride = row_stride;
        output_image_info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        output_image_info.buffer_size = buffer_size;
        output_image_info.cuda_stream = 0;  // Default stream
        
        #ifdef DEBUG
        fmt::print("  Buffer: {}x{} RGB, stride={}, size={} bytes\n",
                  ifd_info.width, ifd_info.height, row_stride, buffer_size);
        #endif // DEBUG
        
        // Step 3: Allocate output buffer
        void* buffer = nullptr;
        if (buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE)
        {
            cudaError_t cuda_status = cudaMalloc(&buffer, buffer_size);
            if (cuda_status != cudaSuccess)
            {
                #ifdef DEBUG
                fmt::print("❌ Failed to allocate GPU memory: {}\n", 
                          cudaGetErrorString(cuda_status));
                #endif // DEBUG
                return false;
            }
            #ifdef DEBUG
            fmt::print("  Allocated GPU buffer\n");
            #endif // DEBUG
        }
        else
        {
            buffer = malloc(buffer_size);
            if (!buffer)
            {
                #ifdef DEBUG
                fmt::print("❌ Failed to allocate host memory\n");
                #endif // DEBUG
                return false;
            }
            #ifdef DEBUG
            fmt::print("  Allocated CPU buffer\n");
            #endif // DEBUG
        }
        
        output_image_info.buffer = buffer;
        
        // Step 4: Create image object
        nvimgcodecImage_t image;
        nvimgcodecStatus_t status = nvimgcodecImageCreate(
            manager.get_instance(),
            &image,
            &output_image_info
        );
        
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            #ifdef DEBUG
            fmt::print("❌ Failed to create image object (status: {})\n",
                      static_cast<int>(status));
            #endif // DEBUG
            if (buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE)
            {
                cudaFree(buffer);
            }
            else
            {
                free(buffer);
            }
            return false;
        }
        
        // Step 5: Prepare decode parameters
        nvimgcodecDecodeParams_t decode_params{};
        decode_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS;
        decode_params.struct_size = sizeof(nvimgcodecDecodeParams_t);
        decode_params.struct_next = nullptr;
        decode_params.apply_exif_orientation = 1;
        
        // Step 6: Schedule decoding
        nvimgcodecFuture_t decode_future;
        nvimgcodecCodeStream_t stream = ifd_info.sub_code_stream;
        status = nvimgcodecDecoderDecode(decoder,
                                        &stream,
                                        &image,
                                        1,
                                        &decode_params,
                                        &decode_future);
        
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            #ifdef DEBUG
            fmt::print("❌ Failed to schedule decoding (status: {})\n",
                      static_cast<int>(status));
            #endif // DEBUG
            nvimgcodecImageDestroy(image);
            if (buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE)
            {
                cudaFree(buffer);
            }
            else
            {
                free(buffer);
            }
            return false;
        }
        
        // Step 7: Wait for completion
        nvimgcodecProcessingStatus_t decode_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
        size_t status_size = 1;
        nvimgcodecFutureGetProcessingStatus(decode_future, &decode_status, &status_size);
        
        if (buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE)
        {
            cudaDeviceSynchronize();  // Wait for GPU operations
        }
        
        // Cleanup partial resources
        nvimgcodecFutureDestroy(decode_future);
        nvimgcodecImageDestroy(image);
        
        // Step 8: Check decode status and handle CPU fallback
        if (decode_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
        {
            #ifdef DEBUG
            fmt::print("❌ Decoding failed (status: {})\n", static_cast<int>(decode_status));
            #endif // DEBUG
            
            // If CPU decoding failed and we need CPU output, try GPU decode + copy
            if (needs_cpu_fallback && buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST)
            {
                #ifdef DEBUG
                fmt::print("  💡 CPU decoder not available, falling back to GPU decode + CPU copy...\n");
                #endif // DEBUG
                free(buffer);  // Free CPU buffer
                
                // Retry with GPU buffer
                void* gpu_buffer = nullptr;
                cudaError_t cuda_status = cudaMalloc(&gpu_buffer, buffer_size);
                if (cuda_status != cudaSuccess)
                {
                    #ifdef DEBUG
                    fmt::print("❌ Failed to allocate GPU memory for fallback: {}\n", 
                              cudaGetErrorString(cuda_status));
                    #endif // DEBUG
                    return false;
                }
                
                // Update image info for GPU
                output_image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
                output_image_info.buffer = gpu_buffer;
                
                // Create GPU image
                status = nvimgcodecImageCreate(manager.get_instance(), &image, &output_image_info);
                if (status != NVIMGCODEC_STATUS_SUCCESS)
                {
                    #ifdef DEBUG
                    fmt::print("❌ Failed to create GPU image for fallback (status: {})\n",
                              static_cast<int>(status));
                    #endif // DEBUG
                    cudaFree(gpu_buffer);
                    return false;
                }
                
                // Decode to GPU
                status = nvimgcodecDecoderDecode(decoder, &stream, &image, 1, 
                                                &decode_params, &decode_future);
                if (status != NVIMGCODEC_STATUS_SUCCESS)
                {
                    #ifdef DEBUG
                    fmt::print("❌ Failed to schedule GPU fallback decoding (status: {})\n",
                              static_cast<int>(status));
                    #endif // DEBUG
                    nvimgcodecImageDestroy(image);
                    cudaFree(gpu_buffer);
                    return false;
                }
                
                // Wait for GPU decode
                nvimgcodecFutureGetProcessingStatus(decode_future, &decode_status, &status_size);
                cudaDeviceSynchronize();
                
                nvimgcodecFutureDestroy(decode_future);
                nvimgcodecImageDestroy(image);
                
                if (decode_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
                {
                    #ifdef DEBUG
                    fmt::print("❌ GPU fallback decoding failed (status: {})\n", 
                              static_cast<int>(decode_status));
                    #endif // DEBUG
                    cudaFree(gpu_buffer);
                    return false;
                }
                
                // Copy from GPU to CPU
                #ifdef DEBUG
                fmt::print("  📥 Copying decoded data from GPU to CPU...\n");
                #endif // DEBUG
                buffer = malloc(buffer_size);
                if (!buffer)
                {
                    #ifdef DEBUG
                    fmt::print("❌ Failed to allocate CPU memory for copy\n");
                    #endif // DEBUG
                    cudaFree(gpu_buffer);
                    return false;
                }
                
                cuda_status = cudaMemcpy(buffer, gpu_buffer, buffer_size, cudaMemcpyDeviceToHost);
                cudaFree(gpu_buffer);  // Free GPU buffer after copy
                
                if (cuda_status != cudaSuccess)
                {
                    #ifdef DEBUG
                    fmt::print("❌ Failed to copy GPU data to CPU: {}\n", 
                              cudaGetErrorString(cuda_status));
                    #endif // DEBUG
                    free(buffer);
                    return false;
                }
                
                #ifdef DEBUG
                fmt::print("  ✅ CPU fallback successful (GPU decode + CPU copy)\n");
                #endif // DEBUG
            }
            else
            {
                // No fallback available
                if (buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE)
                {
                    cudaFree(buffer);
                }
                else
                {
                    free(buffer);
                }
                return false;
            }
        }
        
        // Success! Return buffer to caller
        *output_buffer = static_cast<uint8_t*>(buffer);
        
        #ifdef DEBUG
        fmt::print("✅ Successfully decoded IFD[{}]\n", ifd_info.index);
        #endif // DEBUG
        return true;
    }
    catch (const std::exception& e)
    {
        #ifdef DEBUG
        fmt::print("❌ Exception during decode: {}\n", e.what());
        #endif // DEBUG
        return false;
    }
}

bool decode_ifd_region_nvimgcodec(const IfdInfo& ifd_info,
                                  nvimgcodecCodeStream_t main_code_stream,
                                  uint32_t x, uint32_t y,
                                  uint32_t width, uint32_t height,
                                  uint8_t** output_buffer,
                                  const cucim::io::Device& out_device)
{
    if (!main_code_stream)
    {
        #ifdef DEBUG
        fmt::print("❌ Invalid main_code_stream\n");
        #endif // DEBUG
        return false;
    }
    
    #ifdef DEBUG
    fmt::print("🚀 Decoding IFD[{}] region: [{},{}] {}x{}, codec: {}\n",
              ifd_info.index, x, y, width, height, ifd_info.codec);
    #endif // DEBUG
    
    try
    {
        // CRITICAL: Must use the same manager that created main_code_stream!
        // Using a decoder from a different nvImageCodec instance causes segfaults.
        auto& manager = NvImageCodecTiffParserManager::instance();
        if (!manager.is_available())
        {
            #ifdef DEBUG
            fmt::print("❌ nvImageCodec TIFF parser manager not initialized\n");
            #endif // DEBUG
            return false;
        }
        
        // ROI decoding from TIFF requires nvTiff extension (not available in CPU-only backend)
        // Therefore, always use hybrid decoder for ROI operations
        // CPU-only decoder is only used for full IFD decoding (see decode_ifd_nvimgcodec)
        std::string device_str = std::string(out_device);
        bool target_is_cpu = (device_str.find("cpu") != std::string::npos);
        
        nvimgcodecDecoder_t decoder = manager.get_decoder();  // Always use hybrid for ROI
        #ifdef DEBUG
        fmt::print("  💡 Using hybrid decoder for ROI (nvTiff required for TIFF sub-regions)\n");
        #endif // DEBUG
        
        // Step 1: Create view with ROI for this IFD
        nvimgcodecRegion_t region{};
        region.struct_type = NVIMGCODEC_STRUCTURE_TYPE_REGION;
        region.struct_size = sizeof(nvimgcodecRegion_t);
        region.struct_next = nullptr;
        region.ndim = 2;
        region.start[0] = y;  // row
        region.start[1] = x;  // col
        region.end[0] = y + height;
        region.end[1] = x + width;
        
        nvimgcodecCodeStreamView_t view{};
        view.struct_type = NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_VIEW;
        view.struct_size = sizeof(nvimgcodecCodeStreamView_t);
        view.struct_next = nullptr;
        view.image_idx = ifd_info.index;
        view.region = region;
        
        // Get sub-code stream for this ROI
        nvimgcodecCodeStream_t roi_stream;
        nvimgcodecStatus_t status = nvimgcodecCodeStreamGetSubCodeStream(
            main_code_stream,
            &roi_stream,
            &view
        );
        
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            #ifdef DEBUG
            fmt::print("❌ Failed to create ROI sub-stream (status: {})\n",
                      static_cast<int>(status));
            #endif // DEBUG
            return false;
        }
        
        // Step 2: Determine buffer kind
        // For ROI decoding, hybrid decoder works best with GPU buffers (nvTiff uses GPU)
        // We'll decode to GPU and copy to CPU if needed
        int device_count = 0;
        cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
        bool gpu_available = (cuda_err == cudaSuccess && device_count > 0);
        
        nvimgcodecImageBufferKind_t buffer_kind;
        if (gpu_available)
        {
            // Use GPU buffer for decoding (best performance with hybrid decoder)
            buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
            if (target_is_cpu)
            {
                #ifdef DEBUG
                fmt::print("  ℹ️  Will decode to GPU then copy to CPU (ROI requires nvTiff)\n");
                #endif // DEBUG
            }
        }
        else
        {
            // No GPU available, try CPU buffer (may not work for TIFF ROI)
            buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
            #ifdef DEBUG
            fmt::print("  ⚠️  No GPU available, attempting CPU buffer (may fail for TIFF ROI)\n");
            #endif // DEBUG
        }
        
        // Step 3: Prepare output image info for the region
        nvimgcodecImageInfo_t output_image_info{};
        output_image_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
        output_image_info.struct_size = sizeof(nvimgcodecImageInfo_t);
        output_image_info.struct_next = nullptr;
        
        // Use interleaved RGB format
        output_image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
        output_image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        output_image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
        output_image_info.num_planes = 1;
        output_image_info.buffer_kind = buffer_kind;
        
        // Calculate buffer requirements for the region
        uint32_t num_channels = 3;  // RGB
        size_t row_stride = width * num_channels;
        size_t buffer_size = row_stride * height;
        
        output_image_info.plane_info[0].height = height;
        output_image_info.plane_info[0].width = width;
        output_image_info.plane_info[0].num_channels = num_channels;
        output_image_info.plane_info[0].row_stride = row_stride;
        output_image_info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        output_image_info.buffer_size = buffer_size;
        output_image_info.cuda_stream = 0;
        
        #ifdef DEBUG
        fmt::print("  Buffer: {}x{} RGB, stride={}, size={} bytes\n",
                  width, height, row_stride, buffer_size);
        #endif // DEBUG
        
        // Step 4: Allocate output buffer
        void* buffer = nullptr;
        if (buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE)
        {
            cudaError_t cuda_status = cudaMalloc(&buffer, buffer_size);
            if (cuda_status != cudaSuccess)
            {
                #ifdef DEBUG
                fmt::print("❌ Failed to allocate GPU memory: {}\n", 
                          cudaGetErrorString(cuda_status));
                #endif // DEBUG
                nvimgcodecCodeStreamDestroy(roi_stream);
                return false;
            }
            #ifdef DEBUG
            fmt::print("  Allocated GPU buffer\n");
            #endif // DEBUG
        }
        else
        {
            buffer = malloc(buffer_size);
            if (!buffer)
            {
                #ifdef DEBUG
                fmt::print("❌ Failed to allocate host memory\n");
                #endif // DEBUG
                nvimgcodecCodeStreamDestroy(roi_stream);
                return false;
            }
            #ifdef DEBUG
            fmt::print("  Allocated CPU buffer\n");
            #endif // DEBUG
        }
        
        output_image_info.buffer = buffer;
        
        // Step 5: Create image object
        nvimgcodecImage_t image;
        status = nvimgcodecImageCreate(
            manager.get_instance(),
            &image,
            &output_image_info
        );
        
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            #ifdef DEBUG
            fmt::print("❌ Failed to create image object (status: {})\n",
                      static_cast<int>(status));
            #endif // DEBUG
            if (buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE)
            {
                cudaFree(buffer);
            }
            else
            {
                free(buffer);
            }
            nvimgcodecCodeStreamDestroy(roi_stream);
            return false;
        }
        
        // Step 6: Prepare decode parameters
        nvimgcodecDecodeParams_t decode_params{};
        decode_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS;
        decode_params.struct_size = sizeof(nvimgcodecDecodeParams_t);
        decode_params.struct_next = nullptr;
        decode_params.apply_exif_orientation = 1;
        
        // Step 7: Schedule decoding
        nvimgcodecFuture_t decode_future;
        status = nvimgcodecDecoderDecode(decoder,
                                        &roi_stream,
                                        &image,
                                        1,
                                        &decode_params,
                                        &decode_future);
        
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            #ifdef DEBUG
            fmt::print("❌ Failed to schedule decoding (status: {})\n",
                      static_cast<int>(status));
            #endif // DEBUG
            nvimgcodecImageDestroy(image);
            if (buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE)
            {
                cudaFree(buffer);
            }
            else
            {
                free(buffer);
            }
            nvimgcodecCodeStreamDestroy(roi_stream);
            return false;
        }
        
        // Step 8: Wait for completion
        nvimgcodecProcessingStatus_t decode_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
        size_t status_size = 1;
        nvimgcodecFutureGetProcessingStatus(decode_future, &decode_status, &status_size);
        
        if (buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE)
        {
            cudaDeviceSynchronize();
        }
        
        // Cleanup partial resources
        nvimgcodecFutureDestroy(decode_future);
        nvimgcodecImageDestroy(image);
        
        // Step 9: Check decode status
        if (decode_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
        {
            #ifdef DEBUG
            fmt::print("❌ Decoding failed (status: {})\n", static_cast<int>(decode_status));
            #endif // DEBUG
            
            // Decoding failed - clean up and return error
            if (buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE)
            {
                cudaFree(buffer);
            }
            else
            {
                free(buffer);
            }
            nvimgcodecCodeStreamDestroy(roi_stream);
            return false;
        }
        
        // Step 10: Handle GPU-to-CPU copy if target is CPU but we decoded to GPU
        if (target_is_cpu && buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE)
        {
            // Successful GPU decode, now copy to CPU
            #ifdef DEBUG
            fmt::print("✅ Successfully decoded IFD[{}] region\n", ifd_info.index);
            #endif // DEBUG
            #ifdef DEBUG
            fmt::print("  📥 Copying decoded data from GPU to CPU...\n");
            #endif // DEBUG
            
            void* gpu_buffer = buffer;
            buffer = malloc(buffer_size);
            if (!buffer)
            {
                #ifdef DEBUG
                fmt::print("❌ Failed to allocate CPU memory for copy\n");
                #endif // DEBUG
                cudaFree(gpu_buffer);
                nvimgcodecCodeStreamDestroy(roi_stream);
                return false;
            }
            
            cudaError_t cuda_status = cudaMemcpy(buffer, gpu_buffer, buffer_size, cudaMemcpyDeviceToHost);
            cudaFree(gpu_buffer);  // Free GPU buffer after copy
            
            if (cuda_status != cudaSuccess)
            {
                #ifdef DEBUG
                fmt::print("❌ Failed to copy from GPU to CPU: {}\n", 
                          cudaGetErrorString(cuda_status));
                #endif // DEBUG
                free(buffer);
                nvimgcodecCodeStreamDestroy(roi_stream);
                return false;
            }
            #ifdef DEBUG
            fmt::print("  ✅ GPU-to-CPU copy completed\n");
            #endif // DEBUG
        }
        else
        {
            #ifdef DEBUG
            fmt::print("✅ Successfully decoded IFD[{}] region\n", ifd_info.index);
            #endif // DEBUG
        }
        
        // Clean up
        nvimgcodecCodeStreamDestroy(roi_stream);
        
        // Assign output buffer
        *output_buffer = reinterpret_cast<uint8_t*>(buffer);
        #ifdef DEBUG
        fmt::print("✅ nvImageCodec ROI decode successful: {}x{} at ({}, {})\n", 
                  width, height, x, y);
        #endif // DEBUG
        return true;
    }
    catch (const std::exception& e)
    {
        #ifdef DEBUG
        fmt::print("❌ Exception in ROI decoding: {}\n", e.what());
        #endif // DEBUG
        return false;
    }
}

#else // !CUCIM_HAS_NVIMGCODEC

// Fallback implementations when nvImageCodec is not available
bool decode_jpeg_nvimgcodec(int fd,
                            unsigned char* jpeg_buf,
                            uint64_t offset,
                            uint64_t size,
                            const void* jpegtable_data,
                            uint32_t jpegtable_count,
                            uint8_t** dest,
                            const cucim::io::Device& out_device,
                            int jpeg_color_space)
{
    (void)fd; (void)jpeg_buf; (void)offset; (void)size;
    (void)jpegtable_data; (void)jpegtable_count; (void)dest;
    (void)out_device; (void)jpeg_color_space;
    
    #ifdef DEBUG
    fmt::print(stderr, "nvImageCodec not available - falling back to original decoder\n");
    #endif // DEBUG
    return false;
}

bool decode_jpeg2k_nvimgcodec(int fd,
                              unsigned char* jpeg2k_buf,
                              uint64_t offset,
                              uint64_t size,
                              uint8_t** dest,
                              size_t dest_size,
                              const cucim::io::Device& out_device,
                              int color_space)
{
    (void)fd; (void)jpeg2k_buf; (void)offset; (void)size;
    (void)dest; (void)dest_size; (void)out_device; (void)color_space;
    
    #ifdef DEBUG
    fmt::print(stderr, "nvImageCodec not available - falling back to original decoder\n");
    #endif // DEBUG
    return false;
}

#endif // CUCIM_HAS_NVIMGCODEC

} // namespace cuslide2::nvimgcodec
