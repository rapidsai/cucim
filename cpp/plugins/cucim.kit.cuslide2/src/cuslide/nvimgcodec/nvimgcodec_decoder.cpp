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
        if (decoder_) nvimgcodecDecoderDestroy(decoder_);
        if (instance_) nvimgcodecInstanceDestroy(instance_);
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
        output_image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_RGB;
        output_image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        output_image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
        output_image_info.num_planes = 3;
        
        // Set buffer kind based on output device
        std::string device_str = std::string(out_device);
        if (device_str.find("cuda") != std::string::npos) {
            output_image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
        } else {
            output_image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
        }
        
        // Calculate buffer requirements
        auto sample_type = output_image_info.plane_info[0].sample_type;
        int bytes_per_element = static_cast<unsigned int>(sample_type) >> (8+3);
        size_t device_pitch_in_bytes = input_image_info.plane_info[0].width * bytes_per_element;
        
        for (uint32_t c = 0; c < output_image_info.num_planes; ++c) {
            output_image_info.plane_info[c].height = input_image_info.plane_info[0].height;
            output_image_info.plane_info[c].width = input_image_info.plane_info[0].width;
            output_image_info.plane_info[c].row_stride = device_pitch_in_bytes;
        }
        
        output_image_info.buffer_size = output_image_info.plane_info[0].row_stride * 
                                       output_image_info.plane_info[0].height * 
                                       output_image_info.num_planes;
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
    
    // Suppress unused parameter warnings for JPEG table parameters (not used in this implementation)
    (void)jpegtable_data; (void)jpegtable_count; (void)jpeg_color_space;
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
        output_image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_RGB;
        output_image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        output_image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
        output_image_info.num_planes = 3;
        
        // Set buffer kind based on output device
        std::string device_str = std::string(out_device);
        if (device_str.find("cuda") != std::string::npos) {
            output_image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
        } else {
            output_image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
        }
        
        // Calculate buffer requirements
        auto sample_type = output_image_info.plane_info[0].sample_type;
        int bytes_per_element = static_cast<unsigned int>(sample_type) >> (8+3);
        size_t device_pitch_in_bytes = input_image_info.plane_info[0].width * bytes_per_element;
        
        for (uint32_t c = 0; c < output_image_info.num_planes; ++c) {
            output_image_info.plane_info[c].height = input_image_info.plane_info[0].height;
            output_image_info.plane_info[c].width = input_image_info.plane_info[0].width;
            output_image_info.plane_info[c].row_stride = device_pitch_in_bytes;
        }
        
        output_image_info.buffer_size = output_image_info.plane_info[0].row_stride * 
                                       output_image_info.plane_info[0].height * 
                                       output_image_info.num_planes;
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
    
    // Suppress unused parameter warnings
    (void)dest_size; (void)color_space;
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
