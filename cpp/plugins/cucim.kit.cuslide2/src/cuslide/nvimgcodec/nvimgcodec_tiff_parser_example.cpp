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
#include "nvimgcodec_decoder.h"
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
 * This example shows how to decode an entire resolution level to CPU memory
 * using the separated parser and decoder.
 */
void example_decode_ifd_to_cpu(const std::string& tiff_path)
{
    fmt::print("\n=== Example 2: Decode IFD to CPU ===\n\n");
    
    try
    {
        // Step 1: Parse TIFF structure
        auto tiff = std::make_unique<TiffFileParser>(tiff_path);
        
        if (!tiff->is_valid())
        {
            fmt::print("❌ Failed to open TIFF file\n");
            return;
        }
        
        // Step 2: Get IFD info from parser
        const auto& ifd = tiff->get_ifd(0);
        
        // Step 3: Decode using separate decoder function
        uint8_t* image_data = nullptr;
        cucim::io::Device device("cpu");
        
        if (decode_ifd_nvimgcodec(ifd, &image_data, device))
        {
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
 * This example shows how to decode a lower resolution IFD to GPU memory
 * using the separated parser and decoder.
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
        const auto& ifd = tiff->get_ifd(thumbnail_idx);
        
        uint8_t* gpu_image_data = nullptr;
        cucim::io::Device device("cuda");
        
        if (decode_ifd_nvimgcodec(ifd, &gpu_image_data, device))
        {
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
 * This example shows how to decode all IFDs in a multi-resolution pyramid
 * using the separated parser and decoder.
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
            
            const auto& ifd = tiff->get_ifd(i);
            uint8_t* image_data = nullptr;
            
            if (decode_ifd_nvimgcodec(ifd, &image_data, device))
            {
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
        const auto& ifd = tiff->get_ifd(0);
        uint8_t* buffer = nullptr;
        cucim::io::Device device("cpu");
        
        if (decode_ifd_nvimgcodec(ifd, &buffer, device))
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

/**
 * @brief Example 6: Test Aperio JPEG table handling
 * 
 * This example specifically tests whether nvTiff can handle Aperio SVS files
 * with abbreviated JPEG encoding (TIFFTAG_JPEGTABLES).
 * 
 * Aperio SVS files store quantization/Huffman tables in the TIFF header,
 * and each tile is "abbreviated" (missing DQT/DHT markers). This test
 * verifies if nvTiff's file-level API can properly combine these tables
 * with tile data during decoding.
 */
void example_test_aperio_jpeg_tables(const std::string& tiff_path)
{
    fmt::print("\n=== Example 6: Test Aperio JPEG Table Handling ===\n\n");
    fmt::print("📂 File: {}\n", tiff_path);
    fmt::print("🎯 Testing if nvTiff handles TIFFTAG_JPEGTABLES (abbreviated JPEG)\n\n");
    
    try
    {
        // Step 1: Parse TIFF structure
        auto tiff = std::make_unique<TiffFileParser>(tiff_path);
        
        if (!tiff->is_valid())
        {
            fmt::print("❌ Failed to open TIFF file\n");
            return;
        }
        
        // Step 2: Find first JPEG-compressed IFD
        uint32_t jpeg_ifd_index = UINT32_MAX;
        bool found_jpeg = false;
        
        fmt::print("Searching for JPEG-compressed IFD:\n");
        for (uint32_t i = 0; i < tiff->get_ifd_count(); ++i)
        {
            const auto& ifd = tiff->get_ifd(i);
            fmt::print("  IFD {}: {}x{}, codec: {}\n", 
                      i, ifd.width, ifd.height, ifd.codec);
            
            if (ifd.codec == "jpeg" && !found_jpeg)
            {
                jpeg_ifd_index = i;
                found_jpeg = true;
                fmt::print("  ⭐ Found JPEG-compressed IFD at index {}\n", i);
            }
        }
        
        if (!found_jpeg)
        {
            fmt::print("\n⚠️  No JPEG-compressed IFD found\n");
            fmt::print("ℹ️  This test requires JPEG-compressed tiles (e.g., CMU-1-Small-Region.svs)\n");
            fmt::print("ℹ️  File may use JPEG2000 or other codecs which don't have this issue\n");
            return;
        }
        
        fmt::print("\n{'='*60}\n", fmt::arg("", ""));
        fmt::print("CRITICAL TEST: Decoding Abbreviated JPEG\n");
        fmt::print("{'='*60}\n", fmt::arg("", ""));
        fmt::print("🔬 This reveals if nvTiff handles TIFFTAG_JPEGTABLES correctly\n");
        fmt::print("⏳ Attempting decode...\n\n");
        
        // Step 3: Try to decode the JPEG IFD
        const auto& ifd = tiff->get_ifd(jpeg_ifd_index);
        uint8_t* buffer = nullptr;
        cucim::io::Device device("cpu");
        
        bool success = decode_ifd_nvimgcodec(ifd, &buffer, device);
        
        fmt::print("\n{'='*60}\n", fmt::arg("", ""));
        fmt::print("TEST RESULTS\n");
        fmt::print("{'='*60}\n", fmt::arg("", ""));
        
        if (success && buffer != nullptr)
        {
            fmt::print("✅ DECODE SUCCESS!\n\n");
            fmt::print("🎉 Result: nvTiff DOES handle Aperio JPEG tables correctly!\n");
            fmt::print("📊 Decoded {}x{} image ({} channels)\n", 
                      ifd.width, ifd.height, ifd.num_channels);
            fmt::print("\n💡 Implications:\n");
            fmt::print("   ✅ You CAN use nvImageCodec file-level API for Aperio SVS\n");
            fmt::print("   ✅ ROI-based decoding WILL work with JPEG-compressed tiles\n");
            fmt::print("   ✅ No need for libjpeg-turbo fallback with file-level API\n");
            fmt::print("   ✅ Simplified architecture possible\n");
            
            free(buffer);
        }
        else
        {
            fmt::print("❌ DECODE FAILED!\n\n");
            fmt::print("🔍 Result: nvTiff does NOT handle Aperio JPEG tables in file-level API\n");
            fmt::print("\n💡 Implications:\n");
            fmt::print("   ⚠️  Continue using hybrid approach (libtiff + buffer-level nvImageCodec)\n");
            fmt::print("   ⚠️  File-level API not suitable for Aperio JPEG tiles\n");
            fmt::print("   ⚠️  Keep libjpeg-turbo fallback mechanism\n");
            fmt::print("   ⚠️  Use buffer-level API with manual JPEG table handling\n");
            
            fmt::print("\n📝 Technical Details:\n");
            fmt::print("   - Aperio stores tables in TIFFTAG_JPEGTABLES (TIFF header)\n");
            fmt::print("   - Each tile is 'abbreviated' (missing DQT/DHT markers)\n");
            fmt::print("   - nvTiff file-level API doesn't combine tables with tiles\n");
            fmt::print("   - Buffer-level API with libjpeg-turbo handles this correctly\n");
        }
        
        fmt::print("\n📚 See Also:\n");
        fmt::print("   - APERIO_SVS_JPEG_TABLES_EXPLANATION.md\n");
        fmt::print("   - NVIMAGECODEC_ROI_FEEDBACK_ANALYSIS.md\n");
    }
    catch (const std::exception& e)
    {
        fmt::print("❌ Exception during test: {}\n", e.what());
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
        fmt::print("Usage: {} <tiff_file_path> [--test-jpeg-tables]\n", argv[0]);
        fmt::print("\nExamples:\n");
        fmt::print("  {} image.tif                    # Run all examples\n", argv[0]);
        fmt::print("  {} /path/to/slide.svs           # Run all examples\n", argv[0]);
        fmt::print("  {} aperio.svs --test-jpeg-tables  # Only test JPEG tables\n", argv[0]);
        fmt::print("\nOptions:\n");
        fmt::print("  --test-jpeg-tables    Only run Aperio JPEG table test (Example 6)\n");
        return 1;
    }
    
    std::string tiff_path = argv[1];
    bool test_jpeg_tables_only = false;
    
    // Check for --test-jpeg-tables flag
    if (argc >= 3 && std::string(argv[2]) == "--test-jpeg-tables")
    {
        test_jpeg_tables_only = true;
    }
    
    using namespace cuslide2::nvimgcodec::examples;
    
    if (test_jpeg_tables_only)
    {
        // Run only the JPEG table test
        fmt::print("\n");
        fmt::print("╔{'═'*78}╗\n", fmt::arg("", ""));
        fmt::print("║{:^78}║\n", " nvTiff JPEG Table Handling Test ");
        fmt::print("║{:^78}║\n", " ");
        fmt::print("║{:^78}║\n", " Testing if nvImageCodec/nvTiff can decode Aperio SVS files with ");
        fmt::print("║{:^78}║\n", " abbreviated JPEG encoding (TIFFTAG_JPEGTABLES) ");
        fmt::print("╚{'═'*78}╝\n", fmt::arg("", ""));
        
        example_test_aperio_jpeg_tables(tiff_path);
        
        fmt::print("\n{'═'*78}\n", fmt::arg("", ""));
        fmt::print("Test Complete\n");
        fmt::print("{'═'*78}\n\n", fmt::arg("", ""));
    }
    else
    {
        // Run all examples
        fmt::print("nvImageCodec TIFF Parser Examples\n");
        fmt::print("==================================\n");
        fmt::print("File: {}\n", tiff_path);
        
        example_parse_tiff_structure(tiff_path);
        example_decode_ifd_to_cpu(tiff_path);
        example_decode_thumbnail_to_gpu(tiff_path);
        example_decode_all_levels(tiff_path);
        example_error_handling(tiff_path);
        example_test_aperio_jpeg_tables(tiff_path);  // Test JPEG table handling
        
        fmt::print("\n=== All Examples Complete ===\n\n");
    }
    
    return 0;
}

#else // !CUCIM_HAS_NVIMGCODEC

int main()
{
    fmt::print("nvImageCodec not available - examples cannot run\n");
    return 1;
}

#endif // CUCIM_HAS_NVIMGCODEC

