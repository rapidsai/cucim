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

// Global nvImageCodec instance (singleton pattern for efficiency)
class NvImageCodecManager
{
public:
    static NvImageCodecManager& instance()
    {
        static NvImageCodecManager instance;
        return instance;
    }

    nvimgcodecInstance_t get_instance() const { return instance_; }
    nvimgcodecDecoder_t get_decoder() const { return decoder_; }
    bool is_initialized() const { return initialized_; }
    const std::string& get_status() const { return status_message_; }
    std::mutex& get_mutex() { return decoder_mutex_; }

    // Quick API validation test
    bool test_nvimagecodec_api()
    {
        if (!initialized_) return false;
        
        try {
            // Test 1: Get nvImageCodec properties
            nvimgcodecProperties_t props{};
            props.struct_type = NVIMGCODEC_STRUCTURE_TYPE_PROPERTIES;
            props.struct_size = sizeof(nvimgcodecProperties_t);
            props.struct_next = nullptr;
            
            if (nvimgcodecGetProperties(&props) == NVIMGCODEC_STATUS_SUCCESS)
            {
                uint32_t version = props.version;
                uint32_t major = (version >> 16) & 0xFF;
                uint32_t minor = (version >> 8) & 0xFF;
                uint32_t patch = version & 0xFF;
                
                fmt::print("✅ nvImageCodec API Test: Version {}.{}.{}\n", major, minor, patch);
                
                // Test 2: Check decoder capabilities
                if (decoder_)
                {
                    fmt::print("✅ nvImageCodec Decoder: Ready\n");
                    return true;
                }
            }
        }
        catch (const std::exception& e)
        {
            fmt::print("⚠️  nvImageCodec API Test failed: {}\n", e.what());
        }
        
        return false;
    }

private:
    NvImageCodecManager() : initialized_(false)
    {
        try {
            // Create nvImageCodec instance following official API pattern
            nvimgcodecInstanceCreateInfo_t create_info{};
            create_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
            create_info.struct_size = sizeof(nvimgcodecInstanceCreateInfo_t);
            create_info.struct_next = nullptr;
            create_info.load_builtin_modules = 1;
            create_info.load_extension_modules = 1;
            create_info.extension_modules_path = nullptr;
            create_info.create_debug_messenger = 1;
            create_info.debug_messenger_desc = nullptr;
            create_info.message_severity = 0;
            create_info.message_category = 0;
        
        if (nvimgcodecInstanceCreate(&instance_, &create_info) != NVIMGCODEC_STATUS_SUCCESS)
        {
                status_message_ = "Failed to create nvImageCodec instance";
                fmt::print("❌ {}\n", status_message_);
                return;
            }

            // Create decoder with execution parameters following official API pattern
            nvimgcodecExecutionParams_t exec_params{};
            exec_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS;
            exec_params.struct_size = sizeof(nvimgcodecExecutionParams_t);
            exec_params.struct_next = nullptr;
            exec_params.device_allocator = nullptr;
            exec_params.pinned_allocator = nullptr;
            exec_params.max_num_cpu_threads = 0; // Use default
            exec_params.executor = nullptr;
            exec_params.device_id = NVIMGCODEC_DEVICE_CURRENT;
            exec_params.pre_init = 0;
            exec_params.skip_pre_sync = 0;
            exec_params.num_backends = 0;
            exec_params.backends = nullptr;
        
        if (nvimgcodecDecoderCreate(instance_, &decoder_, &exec_params, nullptr) != NVIMGCODEC_STATUS_SUCCESS)
        {
            nvimgcodecInstanceDestroy(instance_);
            instance_ = nullptr;
            status_message_ = "Failed to create nvImageCodec decoder";
            fmt::print("❌ {}\n", status_message_);
            return;
            }
            
        initialized_ = true;
        status_message_ = "nvImageCodec initialized successfully";
        fmt::print("✅ {}\n", status_message_);
            
        // Run quick API test
        test_nvimagecodec_api();
        }
        catch (const std::exception& e)
        {
            status_message_ = fmt::format("nvImageCodec initialization exception: {}", e.what());
            fmt::print("❌ {}\n", status_message_);
            initialized_ = false;
        }
    }

    ~NvImageCodecManager()
    {
        // Intentionally NOT destroying resources to avoid crashes during Python interpreter shutdown
        // The OS will reclaim these resources when the process exits.
        // This is a workaround for nvJPEG2000 cleanup issues during static destruction.
        // Resources are only held in a singleton that lives for the entire program lifetime anyway.
    }

    nvimgcodecInstance_t instance_{nullptr};
    nvimgcodecDecoder_t decoder_{nullptr};
    bool initialized_{false};
    std::string status_message_;
    std::mutex decoder_mutex_;  // Protect decoder operations from concurrent access
};

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
        fmt::print("⚠️  nvImageCodec JPEG decode: API not available - {}\n", manager.get_status());
        return false; // Fallback to original decoder
    }
    
    fmt::print("🚀 nvImageCodec JPEG decode: Starting, size={} bytes, device={}\n", 
              size, std::string(out_device));
    
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
                fmt::print("❌ nvImageCodec JPEG decode: Failed to seek in file\n");
                return false;
            }
            if (read(fd, jpeg_data.data(), size) != static_cast<ssize_t>(size)) {
                fmt::print("❌ nvImageCodec JPEG decode: Failed to read JPEG data\n");
                return false;
            }
        }
        
        // Handle JPEG tables (common in Aperio SVS files)
        // JPEG tables contain quantization and Huffman tables that need to be prepended to each tile
        if (jpegtable_data && jpegtable_count > 0) {
            fmt::print("📋 nvImageCodec JPEG decode: Merging JPEG tables ({} bytes) with tile data\n", jpegtable_count);
            
            // Create a new buffer with tables prepended
            std::vector<uint8_t> jpeg_with_tables;
            jpeg_with_tables.reserve(jpegtable_count + size);
            
            // Copy JPEG tables, but remove trailing EOI marker (0xFFD9) if present
            const uint8_t* table_ptr = static_cast<const uint8_t*>(jpegtable_data);
            size_t table_size = jpegtable_count;
            
            // Check if tables end with EOI marker and remove it
            if (table_size >= 2 && table_ptr[table_size - 2] == 0xFF && table_ptr[table_size - 1] == 0xD9) {
                table_size -= 2;  // Remove EOI marker
                fmt::print("📋 nvImageCodec JPEG decode: Removed EOI marker from tables\n");
            }
            
            jpeg_with_tables.insert(jpeg_with_tables.end(), table_ptr, table_ptr + table_size);
            
            // Append JPEG tile data (skip SOI marker 0xFFD8 if present)
            if (jpeg_data.size() >= 2 && jpeg_data[0] == 0xFF && jpeg_data[1] == 0xD8) {
                // Skip SOI marker in tile data since tables already have it
                jpeg_with_tables.insert(jpeg_with_tables.end(), jpeg_data.begin() + 2, jpeg_data.end());
                fmt::print("📋 nvImageCodec JPEG decode: Skipped SOI marker in tile data\n");
            } else {
                // No SOI in tile data, append as-is
                jpeg_with_tables.insert(jpeg_with_tables.end(), jpeg_data.begin(), jpeg_data.end());
            }
            
            jpeg_data = std::move(jpeg_with_tables);
            fmt::print("📋 nvImageCodec JPEG decode: Final JPEG stream size: {} bytes\n", jpeg_data.size());
        }
        
        // Create code stream from memory
        if (nvimgcodecCodeStreamCreateFromHostMem(manager.get_instance(), &code_stream, 
                                                 jpeg_data.data(), jpeg_data.size()) != NVIMGCODEC_STATUS_SUCCESS) {
            fmt::print("❌ nvImageCodec JPEG decode: Failed to create code stream\n");
            return false;
        }
        
        // Step 2: Get image information (following official API pattern)
        nvimgcodecImageInfo_t input_image_info{};
        input_image_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
        input_image_info.struct_size = sizeof(nvimgcodecImageInfo_t);
        input_image_info.struct_next = nullptr;
        if (nvimgcodecCodeStreamGetImageInfo(code_stream, &input_image_info) != NVIMGCODEC_STATUS_SUCCESS) {
            fmt::print("❌ nvImageCodec JPEG decode: Failed to get image info\n");
            nvimgcodecCodeStreamDestroy(code_stream);
            return false;
        }
        
        fmt::print("✅ nvImageCodec JPEG decode: Image info - {}x{}, {} planes, codec: {}\n",
                  input_image_info.plane_info[0].width, input_image_info.plane_info[0].height,
                  input_image_info.num_planes, input_image_info.codec_name);
        
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
                fmt::print("⚠️  nvImageCodec JPEG decode: Unknown color space {}, defaulting to sRGB\n", jpeg_color_space);
                break;
        }
        fmt::print("📋 nvImageCodec JPEG decode: Using color space {} (input JPEG color space: {})\n", 
                  static_cast<int>(output_image_info.color_spec), jpeg_color_space);
        
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
        auto sample_type = output_image_info.plane_info[0].sample_type;
        int bytes_per_element = static_cast<unsigned int>(sample_type) >> (8+3);
        uint32_t width = input_image_info.plane_info[0].width;
        uint32_t height = input_image_info.plane_info[0].height;
        uint32_t num_channels = 3;  // RGB
        
        // For interleaved RGB: row_stride = width * channels * bytes_per_element
        size_t row_stride = width * num_channels * bytes_per_element;
        
        // Set plane info for single interleaved plane
        output_image_info.plane_info[0].height = height;
        output_image_info.plane_info[0].width = width;
        output_image_info.plane_info[0].num_channels = num_channels;
        output_image_info.plane_info[0].row_stride = row_stride;
        
        // Total buffer size for interleaved RGB
        output_image_info.buffer_size = row_stride * height;
        output_image_info.cuda_stream = 0; // Default stream
        
        // Use pre-allocated buffer if provided, otherwise allocate new buffer
        void* output_buffer = *dest;  // Check if caller provided a pre-allocated buffer
        bool buffer_was_preallocated = (output_buffer != nullptr);
        
        if (!buffer_was_preallocated) {
            // Allocate output buffer only if not pre-allocated
            if (output_image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
                if (cudaMalloc(&output_buffer, output_image_info.buffer_size) != cudaSuccess) {
                    fmt::print("❌ nvImageCodec JPEG decode: Failed to allocate GPU memory\n");
                    nvimgcodecCodeStreamDestroy(code_stream);
                    return false;
                }
            } else {
                output_buffer = malloc(output_image_info.buffer_size);
                if (!output_buffer) {
                    fmt::print("❌ nvImageCodec JPEG decode: Failed to allocate host memory\n");
                    nvimgcodecCodeStreamDestroy(code_stream);
                    return false;
                }
            }
        }
        
        output_image_info.buffer = output_buffer;
        
        // Step 4: Create image object (following official API pattern)
        nvimgcodecImage_t image;
        if (nvimgcodecImageCreate(manager.get_instance(), &image, &output_image_info) != NVIMGCODEC_STATUS_SUCCESS) {
            fmt::print("❌ nvImageCodec JPEG decode: Failed to create image object\n");
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
                fmt::print("❌ nvImageCodec JPEG decode: Failed to schedule decoding\n");
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
        size_t status_size;
        nvimgcodecProcessingStatus_t decode_status;
        nvimgcodecFutureGetProcessingStatus(decode_future, &decode_status, &status_size);
        cudaDeviceSynchronize(); // Wait for GPU operations to complete
        
        if (decode_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
            fmt::print("❌ nvImageCodec JPEG decode: Processing failed with status: {}\n", decode_status);
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
        
        fmt::print("✅ nvImageCodec JPEG decode: Successfully decoded {}x{} image\n",
                  output_image_info.plane_info[0].width, output_image_info.plane_info[0].height);
        
        // Cleanup (but keep the output buffer for caller)
        nvimgcodecFutureDestroy(decode_future);
        nvimgcodecImageDestroy(image);
        nvimgcodecCodeStreamDestroy(code_stream);
        
        return true; // Success!
        
    } catch (const std::exception& e) {
        fmt::print("❌ nvImageCodec JPEG decode: Exception - {}\n", e.what());
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
    // Get nvImageCodec manager instance
    auto& manager = NvImageCodecManager::instance();
    
    if (!manager.is_initialized())
    {
        fmt::print("⚠️  nvImageCodec JPEG2000 decode: API not available - {}\n", manager.get_status());
        return false; // Fallback to original decoder
    }
    
    fmt::print("🚀 nvImageCodec JPEG2000 decode: Starting, size={} bytes, device={}\n", 
              size, std::string(out_device));
    
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
                fmt::print("❌ nvImageCodec JPEG2000 decode: Failed to seek in file\n");
                return false;
            }
            if (read(fd, jpeg2k_data.data(), size) != static_cast<ssize_t>(size)) {
                fmt::print("❌ nvImageCodec JPEG2000 decode: Failed to read JPEG2000 data\n");
                return false;
            }
        }
        
        // Create code stream from memory
        if (nvimgcodecCodeStreamCreateFromHostMem(manager.get_instance(), &code_stream, 
                                                 jpeg2k_data.data(), jpeg2k_data.size()) != NVIMGCODEC_STATUS_SUCCESS) {
            fmt::print("❌ nvImageCodec JPEG2000 decode: Failed to create code stream\n");
            return false;
        }
        
        // Step 2: Get image information (following official API pattern)
        nvimgcodecImageInfo_t input_image_info{};
        input_image_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
        input_image_info.struct_size = sizeof(nvimgcodecImageInfo_t);
        input_image_info.struct_next = nullptr;
        if (nvimgcodecCodeStreamGetImageInfo(code_stream, &input_image_info) != NVIMGCODEC_STATUS_SUCCESS) {
            fmt::print("❌ nvImageCodec JPEG2000 decode: Failed to get image info\n");
            nvimgcodecCodeStreamDestroy(code_stream);
            return false;
        }
        
        fmt::print("✅ nvImageCodec JPEG2000 decode: Image info - {}x{}, {} planes, codec: {}\n",
                  input_image_info.plane_info[0].width, input_image_info.plane_info[0].height,
                  input_image_info.num_planes, input_image_info.codec_name);
        
        // Step 3: Prepare output image info (following official API pattern)
        nvimgcodecImageInfo_t output_image_info(input_image_info);
        // FIX: Use interleaved RGB format instead of planar
        output_image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
        
        // Map color_space to nvImageCodec color spec
        // Caller convention (from ifd.cpp): 0=RGB, 1=YCbCr
        switch (color_space) {
            case 0: // RGB (Aperio JPEG2000 RGB format - 33005)
                output_image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
                fmt::print("📋 nvImageCodec JPEG2000 decode: Using sRGB color space\n");
                break;
            case 1: // YCbCr (Aperio JPEG2000 YCbCr format - 33003)
                output_image_info.color_spec = NVIMGCODEC_COLORSPEC_SYCC;
                fmt::print("📋 nvImageCodec JPEG2000 decode: Using YCbCr color space\n");
                break;
            default: // Unknown or other
                output_image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
                fmt::print("⚠️  nvImageCodec JPEG2000 decode: Unknown color space {}, defaulting to sRGB\n", color_space);
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
        auto sample_type = output_image_info.plane_info[0].sample_type;
        int bytes_per_element = static_cast<unsigned int>(sample_type) >> (8+3);
        uint32_t width = input_image_info.plane_info[0].width;
        uint32_t height = input_image_info.plane_info[0].height;
        uint32_t num_channels = 3;  // RGB
        
        // For interleaved RGB: row_stride = width * channels * bytes_per_element
        size_t row_stride = width * num_channels * bytes_per_element;
        
        // Set plane info for single interleaved plane
        output_image_info.plane_info[0].height = height;
        output_image_info.plane_info[0].width = width;
        output_image_info.plane_info[0].num_channels = num_channels;
        output_image_info.plane_info[0].row_stride = row_stride;
        
        // Total buffer size for interleaved RGB
        output_image_info.buffer_size = row_stride * height;
        output_image_info.cuda_stream = 0; // Default stream
        
        // Use pre-allocated buffer if provided, otherwise allocate new buffer
        void* output_buffer = *dest;  // Check if caller provided a pre-allocated buffer
        bool buffer_was_preallocated = (output_buffer != nullptr);
        
        if (!buffer_was_preallocated) {
            // Allocate output buffer only if not pre-allocated
            if (output_image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
                if (cudaMalloc(&output_buffer, output_image_info.buffer_size) != cudaSuccess) {
                    fmt::print("❌ nvImageCodec JPEG2000 decode: Failed to allocate GPU memory\n");
                    nvimgcodecCodeStreamDestroy(code_stream);
                    return false;
                }
            } else {
                output_buffer = malloc(output_image_info.buffer_size);
                if (!output_buffer) {
                    fmt::print("❌ nvImageCodec JPEG2000 decode: Failed to allocate host memory\n");
                    nvimgcodecCodeStreamDestroy(code_stream);
                    return false;
                }
            }
        }
        
        output_image_info.buffer = output_buffer;
        
        // Step 4: Create image object (following official API pattern)
        nvimgcodecImage_t image;
        if (nvimgcodecImageCreate(manager.get_instance(), &image, &output_image_info) != NVIMGCODEC_STATUS_SUCCESS) {
            fmt::print("❌ nvImageCodec JPEG2000 decode: Failed to create image object\n");
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
                fmt::print("❌ nvImageCodec JPEG2000 decode: Failed to schedule decoding\n");
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
        size_t status_size;
        nvimgcodecProcessingStatus_t decode_status;
        nvimgcodecFutureGetProcessingStatus(decode_future, &decode_status, &status_size);
        cudaDeviceSynchronize(); // Wait for GPU operations to complete
        
        if (decode_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
            fmt::print("❌ nvImageCodec JPEG2000 decode: Processing failed with status: {}\n", decode_status);
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
        
        fmt::print("✅ nvImageCodec JPEG2000 decode: Successfully decoded {}x{} image\n",
                  output_image_info.plane_info[0].width, output_image_info.plane_info[0].height);
        
        // Cleanup (but keep the output buffer for caller)
        nvimgcodecFutureDestroy(decode_future);
        nvimgcodecImageDestroy(image);
        nvimgcodecCodeStreamDestroy(code_stream);
        
        return true; // Success!
        
    } catch (const std::exception& e) {
        fmt::print("❌ nvImageCodec JPEG2000 decode: Exception - {}\n", e.what());
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
        fmt::print("❌ IFD info has no sub_code_stream\n");
        return false;
    }
    
    fmt::print("🚀 Decoding IFD[{}]: {}x{}, codec: {}\n",
              ifd_info.index, ifd_info.width, ifd_info.height, ifd_info.codec);
    
    try
    {
        // Get decoder from manager
        auto& manager = NvImageCodecManager::instance();
        if (!manager.is_initialized())
        {
            fmt::print("❌ nvImageCodec decoder not initialized\n");
            return false;
        }
        
        nvimgcodecDecoder_t decoder = manager.get_decoder();
        
        // Step 1: Prepare output image info
        nvimgcodecImageInfo_t output_image_info{};
        output_image_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
        output_image_info.struct_size = sizeof(nvimgcodecImageInfo_t);
        output_image_info.struct_next = nullptr;
        
        // Use interleaved RGB format
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
        size_t row_stride = ifd_info.width * num_channels;
        size_t buffer_size = row_stride * ifd_info.height;
        
        output_image_info.plane_info[0].height = ifd_info.height;
        output_image_info.plane_info[0].width = ifd_info.width;
        output_image_info.plane_info[0].num_channels = num_channels;
        output_image_info.plane_info[0].row_stride = row_stride;
        output_image_info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        output_image_info.buffer_size = buffer_size;
        output_image_info.cuda_stream = 0;  // Default stream
        
        fmt::print("  Buffer: {}x{} RGB, stride={}, size={} bytes\n",
                  ifd_info.width, ifd_info.height, row_stride, buffer_size);
        
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
            manager.get_instance(),
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
        nvimgcodecCodeStream_t stream = ifd_info.sub_code_stream;
        status = nvimgcodecDecoderDecode(decoder,
                                        &stream,
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
        
        fmt::print("✅ Successfully decoded IFD[{}]\n", ifd_info.index);
        return true;
    }
    catch (const std::exception& e)
    {
        fmt::print("❌ Exception during decode: {}\n", e.what());
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
        fmt::print("❌ Invalid main_code_stream\n");
        return false;
    }
    
    fmt::print("🚀 Decoding IFD[{}] region: [{},{}] {}x{}, codec: {}\n",
              ifd_info.index, x, y, width, height, ifd_info.codec);
    
    try
    {
        // Get decoder from manager
        auto& manager = NvImageCodecManager::instance();
        if (!manager.is_initialized())
        {
            fmt::print("❌ nvImageCodec decoder not initialized\n");
            return false;
        }
        
        nvimgcodecDecoder_t decoder = manager.get_decoder();
        
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
            fmt::print("❌ Failed to create ROI sub-stream (status: {})\n",
                      static_cast<int>(status));
            return false;
        }
        
        // Step 2: Prepare output image info for the region
        nvimgcodecImageInfo_t output_image_info{};
        output_image_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
        output_image_info.struct_size = sizeof(nvimgcodecImageInfo_t);
        output_image_info.struct_next = nullptr;
        
        // Use interleaved RGB format
        output_image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
        output_image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        output_image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
        output_image_info.num_planes = 1;
        
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
        
        fmt::print("  Buffer: {}x{} RGB, stride={}, size={} bytes\n",
                  width, height, row_stride, buffer_size);
        
        // Step 3: Allocate output buffer
        void* buffer = nullptr;
        if (output_image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE)
        {
            cudaError_t cuda_status = cudaMalloc(&buffer, buffer_size);
            if (cuda_status != cudaSuccess)
            {
                fmt::print("❌ Failed to allocate GPU memory: {}\n", 
                          cudaGetErrorString(cuda_status));
                nvimgcodecCodeStreamDestroy(roi_stream);
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
                nvimgcodecCodeStreamDestroy(roi_stream);
                return false;
            }
            fmt::print("  Allocated CPU buffer\n");
        }
        
        output_image_info.buffer = buffer;
        
        // Step 4: Create image object
        nvimgcodecImage_t image;
        status = nvimgcodecImageCreate(
            manager.get_instance(),
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
            nvimgcodecCodeStreamDestroy(roi_stream);
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
        status = nvimgcodecDecoderDecode(decoder,
                                        &roi_stream,
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
            nvimgcodecCodeStreamDestroy(roi_stream);
            return false;
        }
        
        // Step 7: Wait for completion
        nvimgcodecProcessingStatus_t decode_status;
        size_t status_size;
        nvimgcodecFutureGetProcessingStatus(decode_future, &decode_status, &status_size);
        
        if (output_image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE)
        {
            cudaDeviceSynchronize();
        }
        
        // Cleanup
        nvimgcodecFutureDestroy(decode_future);
        nvimgcodecImageDestroy(image);
        nvimgcodecCodeStreamDestroy(roi_stream);
        
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
        
        fmt::print("✅ Successfully decoded IFD[{}] region\n", ifd_info.index);
        return true;
    }
    catch (const std::exception& e)
    {
        fmt::print("❌ Exception during ROI decode: {}\n", e.what());
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
    
    fmt::print(stderr, "nvImageCodec not available - falling back to original decoder\n");
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
    
    fmt::print(stderr, "nvImageCodec not available - falling back to original decoder\n");
    return false;
}

#endif // CUCIM_HAS_NVIMGCODEC

} // namespace cuslide2::nvimgcodec
