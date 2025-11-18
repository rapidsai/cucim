# nvImageCodec Decoder Code Documentation

This document provides a detailed line-by-line explanation of the nvImageCodec decoder implementation for the cuCIM project.

## Table of Contents
- [nvimgcodec_decoder.h - Header File](#nvimgcodec_decoderh---header-file)
- [nvimgcodec_decoder.cpp - Implementation File](#nvimgcodec_decodercpp---implementation-file)

---

## nvimgcodec_decoder.h - Header File

### Lines 1-15: Copyright and License Header
```cpp
/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 * ...
 */
```
Standard Apache 2.0 license header indicating NVIDIA copyright.

### Lines 17-18: Include Guards
```cpp
#ifndef CUSLIDE2_NVIMGCODEC_DECODER_H
#define CUSLIDE2_NVIMGCODEC_DECODER_H
```
Standard C++ header guard to prevent multiple inclusions of this header file.

### Lines 20-22: Conditional nvImageCodec Include
```cpp
#ifdef CUCIM_HAS_NVIMGCODEC
#include <nvimgcodec.h>
#endif
```
Conditionally includes the nvImageCodec library header only if the `CUCIM_HAS_NVIMGCODEC` macro is defined. This allows the code to compile even without nvImageCodec available.

### Lines 24-25: Standard Includes
```cpp
#include <cucim/io/device.h>
#include <cstdint>
```
- `cucim/io/device.h`: Provides the Device class for specifying CPU/GPU targets
- `cstdint`: Standard integer types (uint8_t, uint32_t, etc.)

### Lines 27-28: Namespace Declaration
```cpp
namespace cuslide2::nvimgcodec
{
```
Opens a nested namespace for nvImageCodec-related functionality within the cuslide2 plugin.

### Lines 30-52: decode_jpeg_nvimgcodec Function Declaration
```cpp
/**
 * Decode JPEG using nvImageCodec
 * ...
 */
bool decode_jpeg_nvimgcodec(int fd,
                            unsigned char* jpeg_buf,
                            uint64_t offset,
                            uint64_t size,
                            const void* jpegtable_data,
                            uint32_t jpegtable_count,
                            uint8_t** dest,
                            const cucim::io::Device& out_device,
                            int jpeg_color_space = 0);
```
**Purpose**: Decode JPEG compressed tiles using NVIDIA's nvImageCodec library.

**Parameters**:
- `fd`: File descriptor for reading JPEG data from file
- `jpeg_buf`: Pre-loaded JPEG buffer (if nullptr, read from fd)
- `offset`: Byte offset in file to start reading
- `size`: Size of compressed JPEG data in bytes
- `jpegtable_data`: Pointer to JPEG tables (TIFFTAG_JPEGTABLES for abbreviated JPEG streams)
- `jpegtable_count`: Size of JPEG tables in bytes
- `dest`: Output pointer that will receive the decoded image buffer
- `out_device`: Target device for output (CPU or CUDA GPU)
- `jpeg_color_space`: Color space hint (grayscale, RGB, YCbCr, etc.)

**Returns**: `true` if decoding succeeds, `false` to fallback to libjpeg-turbo.

### Lines 54-74: decode_jpeg2k_nvimgcodec Function Declaration
```cpp
/**
 * Decode JPEG2000 using nvImageCodec
 * ...
 */
bool decode_jpeg2k_nvimgcodec(int fd,
                              unsigned char* jpeg2k_buf,
                              uint64_t offset,
                              uint64_t size,
                              uint8_t** dest,
                              size_t dest_size,
                              const cucim::io::Device& out_device,
                              int color_space = 0);
```
**Purpose**: Decode JPEG2000 compressed tiles using nvImageCodec.

**Parameters**:
- `fd`: File descriptor for reading JPEG2000 data
- `jpeg2k_buf`: Pre-loaded JPEG2000 buffer (if nullptr, read from fd)
- `offset`: Byte offset in file
- `size`: Size of compressed data
- `dest`: Output pointer for decoded buffer
- `dest_size`: Expected output buffer size
- `out_device`: Target device (CPU/GPU)
- `color_space`: Color space hint (0=RGB, 1=YCbCr for Aperio formats)

**Returns**: `true` on success, `false` to fallback.

### Lines 76-97: decode_tile_nvtiff_roi Function Declaration
```cpp
/**
 * Decode tile using nvTiff file-level API with ROI
 * ...
 */
bool decode_tile_nvtiff_roi(const char* file_path,
                            uint32_t ifd_index,
                            uint32_t tile_x, uint32_t tile_y,
                            uint32_t tile_width, uint32_t tile_height,
                            uint8_t** dest,
                            const cucim::io::Device& out_device);
```
**Purpose**: High-level tile decoding using nvTiff's file-level API that automatically handles JPEG tables.

**Parameters**:
- `file_path`: Path to the TIFF file
- `ifd_index`: IFD (Image File Directory) index for resolution level
- `tile_x`, `tile_y`: Tile coordinates in pixels
- `tile_width`, `tile_height`: Tile dimensions
- `dest`: Output buffer pointer
- `out_device`: Target device

**Key Feature**: This API automatically handles TIFFTAG_JPEGTABLES without manual merging, making it more robust than tile-level decoding.

### Lines 99-116: decode_ifd_nvimgcodec Function Declaration
```cpp
#ifdef CUCIM_HAS_NVIMGCODEC
// Forward declaration
struct IfdInfo;

bool decode_ifd_nvimgcodec(const IfdInfo& ifd_info,
                           uint8_t** output_buffer,
                           const cucim::io::Device& out_device);
```
**Purpose**: Decode an entire IFD (resolution level) using pre-parsed metadata.

**Design Pattern**: Separates parsing from decoding - uses `IfdInfo` structure that contains parsed TIFF metadata and a `sub_code_stream` handle.

**Parameters**:
- `ifd_info`: Parsed IFD metadata structure
- `output_buffer`: Receives allocated output buffer (caller must free)
- `out_device`: Target device

### Lines 118-139: decode_ifd_region_nvimgcodec Function Declaration
```cpp
bool decode_ifd_region_nvimgcodec(const IfdInfo& ifd_info,
                                  nvimgcodecCodeStream_t main_code_stream,
                                  uint32_t x, uint32_t y,
                                  uint32_t width, uint32_t height,
                                  uint8_t** output_buffer,
                                  const cucim::io::Device& out_device);
```
**Purpose**: Memory-efficient ROI (Region of Interest) decoding using nvImageCodec's CodeStreamView.

**Key Feature**: Uses nvImageCodec's native ROI API to decode only a specific region without loading the entire image.

**Parameters**:
- `ifd_info`: Parsed IFD metadata
- `main_code_stream`: Main TIFF code stream for creating ROI sub-streams
- `x`, `y`: Starting coordinates
- `width`, `height`: Region dimensions
- `output_buffer`: Output buffer pointer
- `out_device`: Target device

### Lines 140-145: Closing Declarations
```cpp
#endif // CUCIM_HAS_NVIMGCODEC

} // namespace cuslide2::nvimgcodec

#endif // CUSLIDE2_NVIMGCODEC_DECODER_H
```
Closes the conditional compilation block, namespace, and include guard.

---

## nvimgcodec_decoder.cpp - Implementation File

### Lines 1-38: Headers and Includes
```cpp
/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 * ...
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
```
**Includes**:
- Own header file and related nvImageCodec components
- Standard C++ containers and utilities
- `unistd.h`: For POSIX file operations (lseek, read)
- `mutex`: For thread-safe operations
- `fmt/format.h`: Modern C++ formatting library for debug output
- `cuda_runtime.h`: CUDA memory management

### Lines 39-49: Namespace and Global Parser Cache
```cpp
namespace cuslide2::nvimgcodec
{

#ifdef CUCIM_HAS_NVIMGCODEC

// NvImageCodecManager is now defined in nvimgcodec_manager.h

// Global TiffFileParser cache for nvTiff file-level API
// This avoids re-parsing the same TIFF file for every tile
static std::mutex parser_cache_mutex;
static std::map<std::string, std::shared_ptr<TiffFileParser>> parser_cache;
```
**Global Parser Cache Design**:
- `parser_cache`: Maps file paths to parsed TIFF file structures
- `parser_cache_mutex`: Protects cache from concurrent access
- **Optimization**: Parsing TIFF headers is expensive; cache reuses parsed structures across multiple tile decodes

### Lines 51-107: decode_tile_nvtiff_roi Implementation

#### Lines 51-61: Function Signature and Input Validation
```cpp
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
```
Validates input pointers before processing.

#### Lines 63-85: Parser Cache Lookup/Creation
```cpp
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
                    fmt::print("⚠️  nvTiff ROI: Failed to parse TIFF file: {}\n", file_path);
                    return false;
                }
                parser_cache[file_path] = parser;
                fmt::print("✅ nvTiff ROI: Cached TIFF parser for {}\n", file_path);
            }
        }
```
**Thread-Safe Cache Pattern**:
1. Lock the cache mutex
2. Check if parser exists for this file
3. If not, create new `TiffFileParser` and validate
4. Add to cache for future reuse
5. Release lock (RAII pattern with lock_guard)

#### Lines 87-99: IFD Validation and Tile Decoding
```cpp
        // Check if IFD index is valid
        if (ifd_index >= parser->get_ifd_count())
        {
            fmt::print("⚠️  nvTiff ROI: Invalid IFD index {} (max: {})\n", 
                      ifd_index, parser->get_ifd_count() - 1);
            return false;
        }
        
        // Decode the tile region using nvTiff file-level API
        *dest = parser->decode_region(ifd_index, tile_x, tile_y, 
                                      tile_width, tile_height, 
                                      nullptr, out_device);
        
        return (*dest != nullptr);
```
- Validates IFD index is within bounds
- Calls `TiffFileParser::decode_region()` which uses nvTiff's high-level API
- Returns success based on whether buffer was allocated

#### Lines 100-107: Exception Handling
```cpp
    }
    catch (const std::exception& e)
    {
        fmt::print("❌ nvTiff ROI decode failed: {}\n", e.what());
        return false;
    }
}
```
Catches any exceptions and returns false to trigger fallback to other decoders.

### Lines 109-433: decode_jpeg_nvimgcodec Implementation

#### Lines 109-126: Function Start and Manager Initialization
```cpp
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
```
**Singleton Pattern**: Gets the global nvImageCodec manager instance which handles library initialization and decoder lifecycle.

#### Lines 128-135: JPEG Tables Limitation Check
```cpp
    // IMPORTANT: nvImageCodec 0.7.0 doesn't reliably handle abbreviated JPEG streams
    // (JPEG with separate tables stored in TIFFTAG_JPEGTABLES).
    // Disable nvImageCodec for JPEG decoding when tables are present.
    if (jpegtable_data && jpegtable_count > 0) {
        fmt::print("⚠️  nvImageCodec: Abbreviated JPEG with separate tables detected\n");
        fmt::print("💡 Using libjpeg-turbo decoder (nvImageCodec doesn't support TIFFTAG_JPEGTABLES)\n");
        return false; // Fallback to libjpeg-turbo
    }
```
**Critical Design Decision**: nvImageCodec 0.7.0 has issues with abbreviated JPEG streams (common in TIFF files). When JPEG tables are present, explicitly fall back to libjpeg-turbo which handles them correctly.

#### Lines 137-159: Reading JPEG Data
```cpp
    fmt::print("🚀 nvImageCodec JPEG decode: Starting, size={} bytes, device={}\n", 
              size, std::string(out_device));
    
    try {
        // Step 1: Create code stream from memory buffer
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
```
**Data Loading Strategy**:
- If `jpeg_buf` is provided, copy from memory
- Otherwise, seek to file offset and read data
- Uses `std::vector<uint8_t>` for automatic memory management

#### Lines 161-210: JPEG Tables Merging (Currently Disabled)
```cpp
        // Handle JPEG tables (common in Aperio SVS files)
        // nvImageCodec 0.7.0: Use safer JPEG table merging with proper validation
        if (jpegtable_data && jpegtable_count > 0) {
            // ... detailed validation and merging logic ...
        }
```
**Note**: This code path is never executed due to the earlier return statement (line 134), but is kept for reference or future API improvements.

**Merging Algorithm** (if enabled):
1. Validate input sizes
2. Remove trailing EOI (0xFFD9) marker from tables
3. Skip leading SOI (0xFFD8) marker from tile data
4. Concatenate: tables + tile data
5. Validate final size (max 10MB safety check)

#### Lines 212-227: Create nvImageCodec Code Stream
```cpp
        // Validate JPEG data before creating code stream
        if (jpeg_data.size() < 4 || jpeg_data.empty()) {
            fmt::print("❌ nvImageCodec JPEG decode: Invalid JPEG data size: {} bytes\n", jpeg_data.size());
            return false;
        }
        
        // Create code stream from memory
        nvimgcodecStatus_t status = nvimgcodecCodeStreamCreateFromHostMem(
            manager.get_instance(), &code_stream, jpeg_data.data(), jpeg_data.size());
            
        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            fmt::print("❌ nvImageCodec JPEG decode: Failed to create code stream (status: {})\n", 
                      static_cast<int>(status));
            fmt::print("💡 Falling back to libjpeg-turbo decoder\n");
            return false; // Fallback to libjpeg-turbo
        }
```
**nvImageCodec API Step 1**: Create a code stream object from the compressed JPEG data in host memory.

#### Lines 229-243: Get Image Information
```cpp
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
```
**nvImageCodec API Step 2**: Query the code stream for image metadata (dimensions, planes, codec type).

**Struct Initialization Pattern**: nvImageCodec uses explicit struct versioning (`struct_type`, `struct_size`) for API stability.

#### Lines 244-268: Configure Output Image Format
```cpp
        // Step 3: Prepare output image info
        nvimgcodecImageInfo_t output_image_info(input_image_info);
        // FIX: Use interleaved RGB format instead of planar
        output_image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
        
        // Map jpeg_color_space to nvImageCodec color spec
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
```
**Key Decisions**:
- **Interleaved RGB**: Uses `NVIMGCODEC_SAMPLEFORMAT_I_RGB` (RGBRGBRGB...) instead of planar (RRR...GGG...BBB...)
- **Color Space Mapping**: Converts JPEG color space enum to nvImageCodec color spec
- **Single Plane**: Interleaved format uses one memory plane

#### Lines 270-298: Buffer Configuration and Calculation
```cpp
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
```
**Buffer Calculations**:
- Detects target device from string (e.g., "cuda:0")
- Calculates `bytes_per_element` from sample type via bit shifting
- **Row stride**: Number of bytes per image row
- **Buffer size**: Total memory needed = stride × height

#### Lines 300-322: Output Buffer Allocation
```cpp
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
```
**Flexible Memory Management**:
- Supports pre-allocated buffers (if `*dest != nullptr`)
- Otherwise allocates new buffer:
  - GPU: `cudaMalloc()`
  - CPU: `malloc()`
- Tracks allocation status for proper cleanup on error

#### Lines 324-337: Create Image Object
```cpp
        // Step 4: Create image object
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
```
**nvImageCodec API Step 4**: Creates an image object that wraps the output buffer with format metadata.

**Error Handling**: Cleans up allocated resources on failure (only if we allocated them).

#### Lines 339-364: Schedule Decoding
```cpp
        // Step 5: Prepare decode parameters
        nvimgcodecDecodeParams_t decode_params{};
        decode_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS;
        decode_params.struct_size = sizeof(nvimgcodecDecodeParams_t);
        decode_params.struct_next = nullptr;
        decode_params.apply_exif_orientation = 1;
        
        // Step 6: Schedule decoding
        // THREAD-SAFETY: Lock the decoder to prevent concurrent access
        nvimgcodecFuture_t decode_future;
        {
            std::lock_guard<std::mutex> lock(manager.get_mutex());
            if (nvimgcodecDecoderDecode(manager.get_decoder(), &code_stream, &image, 1, &decode_params, &decode_future) != NVIMGCODEC_STATUS_SUCCESS) {
                fmt::print("❌ nvImageCodec JPEG decode: Failed to schedule decoding\n");
                nvimgcodecImageDestroy(image);
                // ... cleanup ...
                return false;
            }
        }
```
**nvImageCodec API Steps 5-6**:
- Configures decode parameters (EXIF orientation handling)
- **Thread Safety**: Locks manager mutex because nvImageCodec decoder is not thread-safe
- Schedules asynchronous decode operation
- Returns a `future` object for checking completion

#### Lines 366-397: Wait for Completion and Validate
```cpp
        // Step 7: Wait for decoding to finish
        size_t status_size = 1;
        nvimgcodecProcessingStatus_t decode_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
        
        // Safely get processing status with validation
        nvimgcodecStatus_t future_status = nvimgcodecFutureGetProcessingStatus(
            decode_future, &decode_status, &status_size);
            
        if (future_status != NVIMGCODEC_STATUS_SUCCESS) {
            fmt::print("❌ nvImageCodec JPEG decode: Failed to get future status (code: {})\n", 
                      static_cast<int>(future_status));
            // ... cleanup ...
            fmt::print("💡 Falling back to libjpeg-turbo decoder\n");
            return false;
        }
        
        // Synchronize only if we're on GPU
        if (output_image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
            cudaError_t cuda_err = cudaDeviceSynchronize();
            if (cuda_err != cudaSuccess) {
                fmt::print("⚠️  CUDA synchronization warning: {}\n", cudaGetErrorString(cuda_err));
            }
        }
```
**Synchronization Strategy**:
1. Query future for decode status
2. If decoding to GPU, call `cudaDeviceSynchronize()` to ensure completion
3. Validate decode status

#### Lines 399-427: Success Handling and Cleanup
```cpp
        if (decode_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
            fmt::print("❌ nvImageCodec JPEG decode: Processing failed with status: {}\n", 
                      static_cast<int>(decode_status));
            // ... cleanup and fallback ...
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
```
**Success Path**:
1. Set output pointer to decoded buffer
2. Clean up nvImageCodec objects (but NOT the output buffer - caller owns it)
3. Return true

#### Lines 429-433: Exception Handler
```cpp
    } catch (const std::exception& e) {
        fmt::print("❌ nvImageCodec JPEG decode: Exception - {}\n", e.what());
        return false;
    }
}
```
Top-level exception safety net.

### Lines 435-671: decode_jpeg2k_nvimgcodec Implementation

This function follows nearly identical structure to `decode_jpeg_nvimgcodec` with JPEG2000-specific differences:

#### Key Differences from JPEG:

**Line 453**: Different debug message prefix ("JPEG2000" instead of "JPEG")

**Lines 457-475**: Reads JPEG2000 compressed data (same pattern as JPEG)

**Lines 478-482**: Creates code stream without JPEG table merging logic (JPEG2000 doesn't use separate tables)

**Lines 504-519**: Different color space mapping:
```cpp
switch (color_space) {
    case 0: // RGB (Aperio JPEG2000 RGB format - 33005)
        output_image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        break;
    case 1: // YCbCr (Aperio JPEG2000 YCbCr format - 33003)
        output_image_info.color_spec = NVIMGCODEC_COLORSPEC_SYCC;
        break;
    default:
        output_image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        break;
}
```
**Aperio-Specific**: Aperio SVS files use proprietary JPEG2000 compression tags (33003=YCbCr, 33005=RGB)

**Lines 621-634**: Additional debug logging for processing status:
```cpp
fmt::print("📍 Getting processing status...\n");
size_t status_size;
nvimgcodecProcessingStatus_t decode_status;
nvimgcodecFutureGetProcessingStatus(decode_future, &decode_status, &status_size);
fmt::print("📍 Got processing status: {}\n", static_cast<int>(decode_status));
```
Extra diagnostics for debugging JPEG2000 decoding issues.

**Line 670**: Suppresses unused parameter warning for `dest_size`.

### Lines 673-862: decode_ifd_nvimgcodec Implementation

This function decodes an entire IFD (resolution level) using pre-parsed metadata.

#### Lines 677-688: Input Validation
```cpp
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
```
**Key Input**: `IfdInfo` struct must contain a `sub_code_stream` handle created by `TiffFileParser`.

#### Lines 690-723: Setup Output Image Configuration
```cpp
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
        // ... initialization ...
        
        // Use interleaved RGB format
        output_image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
        output_image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        // ...
```
Similar setup to tile-level decode but using full IFD dimensions from `ifd_info`.

#### Lines 725-763: Buffer Allocation
```cpp
        // Calculate buffer requirements for interleaved RGB
        uint32_t num_channels = 3;  // RGB
        size_t row_stride = ifd_info.width * num_channels;
        size_t buffer_size = row_stride * ifd_info.height;
        
        // ... set plane info ...
        
        // Step 2: Allocate output buffer
        void* buffer = nullptr;
        if (output_image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE)
        {
            cudaError_t cuda_status = cudaMalloc(&buffer, buffer_size);
            // ...
        }
        else
        {
            buffer = malloc(buffer_size);
            // ...
        }
```
Allocates buffer for entire IFD (can be very large for high-resolution levels).

#### Lines 768-821: Create Image and Schedule Decode
```cpp
        // Step 3: Create image object
        nvimgcodecImage_t image;
        nvimgcodecStatus_t status = nvimgcodecImageCreate(
            manager.get_instance(),
            &image,
            &output_image_info
        );
        
        // ... error handling ...
        
        // Step 4: Prepare decode parameters
        nvimgcodecDecodeParams_t decode_params{};
        // ... initialization ...
        
        // Step 5: Schedule decoding
        nvimgcodecFuture_t decode_future;
        nvimgcodecCodeStream_t stream = ifd_info.sub_code_stream;
        status = nvimgcodecDecoderDecode(decoder,
                                        &stream,
                                        &image,
                                        1,
                                        &decode_params,
                                        &decode_future);
```
**Note**: No mutex lock here (unlike tile-level decode) - assumes caller handles thread safety.

#### Lines 823-855: Wait and Validate
```cpp
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
            // ... cleanup buffer ...
            return false;
        }
        
        // Success! Return buffer to caller
        *output_buffer = static_cast<uint8_t*>(buffer);
```
Standard completion check and cleanup pattern.

### Lines 864-1089: decode_ifd_region_nvimgcodec Implementation

This function demonstrates nvImageCodec's ROI (Region of Interest) decoding capability.

#### Lines 864-878: Function Signature and Validation
```cpp
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
```
**Key Difference**: Takes `main_code_stream` parameter to create ROI sub-streams.

#### Lines 892-923: Create ROI Sub-Stream
```cpp
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
```
**ROI API Usage**:
1. Define `nvimgcodecRegion_t` with start/end coordinates
2. Create `nvimgcodecCodeStreamView_t` linking region to IFD index
3. Call `nvimgcodecCodeStreamGetSubCodeStream()` to create ROI-specific stream
4. **Key Benefit**: Decoder only processes tiles overlapping the ROI, saving memory and time

#### Lines 925-962: Configure Output for Region
```cpp
        // Step 2: Prepare output image info for the region
        nvimgcodecImageInfo_t output_image_info{};
        // ... initialization ...
        
        // Calculate buffer requirements for the region
        uint32_t num_channels = 3;  // RGB
        size_t row_stride = width * num_channels;  // Region width, not full IFD width
        size_t buffer_size = row_stride * height;   // Region height
        
        output_image_info.plane_info[0].height = height;
        output_image_info.plane_info[0].width = width;
        // ...
```
**Important**: Buffer dimensions match the ROI size, not the full IFD size.

#### Lines 964-1082: Allocate, Decode, and Cleanup
```cpp
        // Step 3: Allocate output buffer
        // ... same pattern as full IFD decode ...
        
        // Step 4: Create image object
        // ...
        
        // Step 5: Prepare decode parameters
        // ...
        
        // Step 6: Schedule decoding
        status = nvimgcodecDecoderDecode(decoder,
                                        &roi_stream,  // Use ROI stream
                                        &image,
                                        1,
                                        &decode_params,
                                        &decode_future);
        
        // ... error handling ...
        
        // Step 7: Wait for completion
        // ...
        
        // Cleanup
        nvimgcodecFutureDestroy(decode_future);
        nvimgcodecImageDestroy(image);
        nvimgcodecCodeStreamDestroy(roi_stream);  // Destroy ROI stream
```
Follows same decode pattern but uses `roi_stream` instead of full IFD stream.

### Lines 1091-1130: Fallback Implementations (No nvImageCodec)

#### Lines 1091-1110: Fallback decode_jpeg_nvimgcodec
```cpp
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
```
**Conditional Compilation**: When `CUCIM_HAS_NVIMGCODEC` is not defined, provides stub functions that always return `false`.

**Purpose**: Allows code to compile and link without nvImageCodec library, gracefully falling back to libjpeg-turbo/OpenJPEG.

#### Lines 1112-1128: Fallback decode_jpeg2k_nvimgcodec
```cpp
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
```
Same stub pattern for JPEG2000.

### Lines 1129-1131: Namespace Closing
```cpp
} // namespace cuslide2::nvimgcodec
```
Closes the implementation namespace.

---

## Architecture Summary

### Design Patterns Used

1. **Singleton Pattern**: `NvImageCodecManager` provides global decoder instance
2. **Cache Pattern**: Global parser cache avoids re-parsing TIFF files
3. **RAII**: Smart pointers and lock guards for automatic cleanup
4. **Fallback Strategy**: Always returns `false` on error to trigger fallback decoders
5. **Conditional Compilation**: `#ifdef CUCIM_HAS_NVIMGCODEC` for optional dependency

### Key API Concepts

1. **Code Stream**: Represents compressed image data
2. **Image Info**: Metadata about image format and buffer layout
3. **Decoder**: Stateless object that performs decoding
4. **Future**: Handle for asynchronous decode operations
5. **ROI/View**: Efficient sub-region decoding without loading full image

### Thread Safety

- **Parser Cache**: Protected by `parser_cache_mutex`
- **Decoder Access**: Protected by `manager.get_mutex()` in tile-level decode
- **CUDA Operations**: Requires `cudaDeviceSynchronize()` for GPU buffers

### Memory Management

- **Caller-Owned Buffers**: Functions allocate buffers but caller must free them
- **Pre-allocated Buffer Support**: Functions can use pre-allocated buffers
- **Error Cleanup**: Always frees allocated resources on error paths
- **CUDA vs Host**: Separate allocation paths (`cudaMalloc` vs `malloc`)

### Error Handling Strategy

- **Early Return on Error**: Returns `false` to trigger fallback decoders
- **Resource Cleanup**: Destroys nvImageCodec objects before returning
- **Exception Safety**: Top-level try-catch for unexpected errors
- **Diagnostic Logging**: Extensive `fmt::print()` statements for debugging

---

## Integration Points

### Called By
- `ifd.cpp`: Tile decoding functions for TIFF image loading
- `tiff.cpp`: High-level TIFF file reading operations

### Calls To
- `nvimgcodec_manager.h`: Singleton decoder manager
- `nvimgcodec_tiff_parser.h`: TIFF file parsing with nvTiff
- `nvimgcodec.h`: NVIDIA nvImageCodec library API
- `cuda_runtime.h`: CUDA memory operations

### Fallback Path
When nvImageCodec decode returns `false`:
1. Caller tries next decoder in chain
2. Typically falls back to:
   - **JPEG**: libjpeg-turbo
   - **JPEG2000**: OpenJPEG
   - **TIFF**: libtiff

---

## Performance Considerations

### Optimizations
- **Parser caching**: Avoids re-parsing TIFF headers
- **ROI decoding**: Only decodes needed tiles for region requests
- **GPU acceleration**: Direct decode to GPU memory when possible
- **Buffer reuse**: Supports pre-allocated output buffers

### Bottlenecks
- **Thread-safety mutex**: Global decoder mutex serializes decode operations
- **CUDA synchronization**: `cudaDeviceSynchronize()` blocks CPU thread
- **Memory allocation**: Large buffer allocations for high-resolution images

### Scalability
- **Multi-threaded**: Cache and decoder access are thread-safe
- **Multi-GPU**: Could extend to support multiple CUDA devices
- **Large files**: ROI decoding prevents loading entire images into memory

---

## End of Documentation

