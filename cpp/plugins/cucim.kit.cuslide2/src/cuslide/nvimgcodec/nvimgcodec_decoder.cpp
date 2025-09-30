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
#include <unistd.h>
#include <fmt/format.h>

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

private:
    NvImageCodecManager()
    {
        // Create nvImageCodec instance
        nvimgcodecInstanceCreateInfo_t create_info{};
        create_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.struct_size = sizeof(nvimgcodecInstanceCreateInfo_t);
        create_info.struct_next = nullptr;
        
        if (nvimgcodecInstanceCreate(&instance_, &create_info) != NVIMGCODEC_STATUS_SUCCESS)
        {
            throw std::runtime_error("Failed to create nvImageCodec instance");
        }

        // Create decoder
        nvimgcodecDecoderCreateInfo_t decoder_info{};
        decoder_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_DECODER_CREATE_INFO;
        decoder_info.struct_size = sizeof(nvimgcodecDecoderCreateInfo_t);
        decoder_info.struct_next = nullptr;
        
        if (nvimgcodecDecoderCreate(instance_, &decoder_, &decoder_info) != NVIMGCODEC_STATUS_SUCCESS)
        {
            nvimgcodecInstanceDestroy(instance_);
            throw std::runtime_error("Failed to create nvImageCodec decoder");
        }
    }

    ~NvImageCodecManager()
    {
        if (decoder_) nvimgcodecDecoderDestroy(decoder_);
        if (instance_) nvimgcodecInstanceDestroy(instance_);
    }

    nvimgcodecInstance_t instance_{nullptr};
    nvimgcodecDecoder_t decoder_{nullptr};
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
    // For now, just log that we tried nvImageCodec and return false to fall back
    // This is a placeholder implementation that will be expanded
    (void)fd; (void)jpeg_buf; (void)offset; (void)size;
    (void)jpegtable_data; (void)jpegtable_count; (void)dest;
    (void)out_device; (void)jpeg_color_space;
    
    fmt::print("DEBUG: nvImageCodec JPEG decode attempted (placeholder implementation)\n");
    return false; // Always fallback to libjpeg-turbo for now
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
    // Similar implementation to JPEG decode but for JPEG2000
    // For now, we'll use a simplified version
    (void)fd; (void)jpeg2k_buf; (void)offset; (void)size;
    (void)dest; (void)dest_size; (void)out_device; (void)color_space;
    
    // TODO: Implement JPEG2000 decoding
    fmt::print(stderr, "nvImageCodec JPEG2000 decode: Not yet implemented\n");
    return false;
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
