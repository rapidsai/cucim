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

/**
 * @file nvimgcodec_tiff_parser_example.cpp
 * @brief Example usage of the nvImageCodec TIFF parser
 * 
 * This file demonstrates how to use the TiffFileParser class to parse and
 * decode TIFF files using nvImageCodec's file-level API.
 * 
 * Compile this example as a standalone program or integrate the parser
 * into your existing codebase.
 */

#include "nvimgcodec_tiff_parser.h"
#include <fmt/format.h>
#include <memory>

#ifdef CUCIM_HAS_NVIMGCODEC

namespace cuslide2::nvimgcodec::examples
{

/**
 * @brief Example 1: Parse TIFF structure and print information
 * 
 * This example shows how to open a TIFF file and query its structure
 * without decoding any images.
 */
void example_parse_tiff_structure(const std::string& tiff_path)
{
    fmt::print("\n=== Example 1: Parse TIFF Structure ===\n\n");
    
    try
    {
        // Open and parse TIFF file
        auto tiff = std::make_unique<TiffFileParser>(tiff_path);
        
        if (!tiff->is_valid())
        {
            fmt::print("❌ Failed to open TIFF file: {}\n", tiff_path);
            return;
        }
        
        // Print TIFF information
        tiff->print_info();
        
        // Access individual IFD information
        fmt::print("\nAccessing IFD information:\n");
        for (uint32_t i = 0; i < tiff->get_ifd_count(); ++i)
        {
            const auto& ifd = tiff->get_ifd(i);
            fmt::print("  Level {}: {}x{} ({})\n", 
                      i, ifd.width, ifd.height, ifd.codec);
        }
    }
    catch (const std::exception& e)
    {
        fmt::print("❌ Exception: {}\n", e.what());
    }
}

/**
 * @brief Example 2: Decode highest resolution IFD to CPU memory
 * 
 * This example shows how to decode an entire resolution level to CPU memory.
 */
void example_decode_ifd_to_cpu(const std::string& tiff_path)
{
    fmt::print("\n=== Example 2: Decode IFD to CPU ===\n\n");
    
    try
    {
        auto tiff = std::make_unique<TiffFileParser>(tiff_path);
        
        if (!tiff->is_valid())
        {
            fmt::print("❌ Failed to open TIFF file\n");
            return;
        }
        
        // Decode highest resolution (IFD 0) to CPU
        uint8_t* image_data = nullptr;
        cucim::io::Device device("cpu");
        
        if (tiff->decode_ifd(0, &image_data, device))
        {
            const auto& ifd = tiff->get_ifd(0);
            
            fmt::print("✅ Successfully decoded IFD 0\n");
            fmt::print("  Image dimensions: {}x{}\n", ifd.width, ifd.height);
            fmt::print("  Buffer size: {} bytes\n", ifd.width * ifd.height * 3);
            fmt::print("  First pixel RGB: [{}, {}, {}]\n",
                      image_data[0], image_data[1], image_data[2]);
            
            // Use image_data for processing...
            // For example, save to file, display, analyze, etc.
            
            // Free buffer when done
            free(image_data);
            fmt::print("  Buffer freed\n");
        }
        else
        {
            fmt::print("❌ Failed to decode IFD 0\n");
        }
    }
    catch (const std::exception& e)
    {
        fmt::print("❌ Exception: {}\n", e.what());
    }
}

/**
 * @brief Example 3: Decode thumbnail to GPU memory
 * 
 * This example shows how to decode a lower resolution IFD to GPU memory.
 */
void example_decode_thumbnail_to_gpu(const std::string& tiff_path)
{
    fmt::print("\n=== Example 3: Decode Thumbnail to GPU ===\n\n");
    
    try
    {
        auto tiff = std::make_unique<TiffFileParser>(tiff_path);
        
        if (!tiff->is_valid())
        {
            fmt::print("❌ Failed to open TIFF file\n");
            return;
        }
        
        if (tiff->get_ifd_count() < 2)
        {
            fmt::print("⚠️  TIFF has only {} IFD(s), need at least 2 for thumbnail\n",
                      tiff->get_ifd_count());
            return;
        }
        
        // Decode lowest resolution (last IFD) to GPU
        uint32_t thumbnail_idx = tiff->get_ifd_count() - 1;
        uint8_t* gpu_image_data = nullptr;
        cucim::io::Device device("cuda");
        
        if (tiff->decode_ifd(thumbnail_idx, &gpu_image_data, device))
        {
            const auto& ifd = tiff->get_ifd(thumbnail_idx);
            
            fmt::print("✅ Successfully decoded IFD {} to GPU\n", thumbnail_idx);
            fmt::print("  Image dimensions: {}x{}\n", ifd.width, ifd.height);
            fmt::print("  GPU buffer size: {} bytes\n", ifd.width * ifd.height * 3);
            fmt::print("  GPU pointer: {}\n", static_cast<void*>(gpu_image_data));
            
            // Use GPU buffer for processing...
            // For example, pass to CUDA kernels, OpenGL textures, etc.
            
            // Free GPU buffer when done
            cudaFree(gpu_image_data);
            fmt::print("  GPU buffer freed\n");
        }
        else
        {
            fmt::print("❌ Failed to decode IFD {} to GPU\n", thumbnail_idx);
        }
    }
    catch (const std::exception& e)
    {
        fmt::print("❌ Exception: {}\n", e.what());
    }
}

/**
 * @brief Example 4: Decode all resolution levels
 * 
 * This example shows how to decode all IFDs in a multi-resolution pyramid.
 */
void example_decode_all_levels(const std::string& tiff_path)
{
    fmt::print("\n=== Example 4: Decode All Levels ===\n\n");
    
    try
    {
        auto tiff = std::make_unique<TiffFileParser>(tiff_path);
        
        if (!tiff->is_valid())
        {
            fmt::print("❌ Failed to open TIFF file\n");
            return;
        }
        
        cucim::io::Device device("cpu");
        
        for (uint32_t i = 0; i < tiff->get_ifd_count(); ++i)
        {
            fmt::print("\nDecoding IFD {}...\n", i);
            
            uint8_t* image_data = nullptr;
            if (tiff->decode_ifd(i, &image_data, device))
            {
                const auto& ifd = tiff->get_ifd(i);
                fmt::print("  ✅ Level {}: {}x{}\n", i, ifd.width, ifd.height);
                
                // Process this resolution level...
                
                free(image_data);
            }
            else
            {
                fmt::print("  ❌ Failed to decode level {}\n", i);
            }
        }
    }
    catch (const std::exception& e)
    {
        fmt::print("❌ Exception: {}\n", e.what());
    }
}

/**
 * @brief Example 5: Error handling
 * 
 * This example demonstrates proper error handling.
 */
void example_error_handling(const std::string& tiff_path)
{
    fmt::print("\n=== Example 5: Error Handling ===\n\n");
    
    // Check if nvImageCodec is available
    auto& manager = NvImageCodecTiffParserManager::instance();
    if (!manager.is_available())
    {
        fmt::print("❌ nvImageCodec not available: {}\n", manager.get_status());
        return;
    }
    
    try
    {
        // Try to open file
        auto tiff = std::make_unique<TiffFileParser>(tiff_path);
        
        if (!tiff->is_valid())
        {
            fmt::print("❌ TIFF file not valid\n");
            return;
        }
        
        // Try to access invalid IFD
        try
        {
            const auto& ifd = tiff->get_ifd(999);
            (void)ifd;  // Suppress warning
        }
        catch (const std::out_of_range& e)
        {
            fmt::print("✅ Caught expected exception: {}\n", e.what());
        }
        
        // Try to decode with pre-allocated buffer (not supported in this API)
        uint8_t* buffer = nullptr;
        cucim::io::Device device("cpu");
        
        if (tiff->decode_ifd(0, &buffer, device))
        {
            fmt::print("✅ Decode succeeded\n");
            free(buffer);
        }
        else
        {
            fmt::print("⚠️  Decode failed (expected if file doesn't exist)\n");
        }
    }
    catch (const std::runtime_error& e)
    {
        fmt::print("✅ Caught runtime error: {}\n", e.what());
    }
    catch (const std::exception& e)
    {
        fmt::print("❌ Unexpected exception: {}\n", e.what());
    }
}

} // namespace cuslide2::nvimgcodec::examples

/**
 * @brief Main function - runs all examples
 * 
 * Usage: ./nvimgcodec_tiff_parser_example <path_to_tiff_file>
 */
int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        fmt::print("Usage: {} <tiff_file_path>\n", argv[0]);
        fmt::print("\nExamples:\n");
        fmt::print("  {} image.tif\n", argv[0]);
        fmt::print("  {} /path/to/slide.svs\n", argv[0]);
        return 1;
    }
    
    std::string tiff_path = argv[1];
    
    fmt::print("nvImageCodec TIFF Parser Examples\n");
    fmt::print("==================================\n");
    fmt::print("File: {}\n", tiff_path);
    
    using namespace cuslide2::nvimgcodec::examples;
    
    // Run examples
    example_parse_tiff_structure(tiff_path);
    example_decode_ifd_to_cpu(tiff_path);
    example_decode_thumbnail_to_gpu(tiff_path);
    example_decode_all_levels(tiff_path);
    example_error_handling(tiff_path);
    
    fmt::print("\n=== All Examples Complete ===\n\n");
    
    return 0;
}

#else // !CUCIM_HAS_NVIMGCODEC

int main()
{
    fmt::print("nvImageCodec not available - examples cannot run\n");
    return 1;
}

#endif // CUCIM_HAS_NVIMGCODEC

