# nvImageCodec Implementation - Line-by-Line Code Documentation

**Date:** November 17, 2025  
**Author:** cuCIM Development Team  
**Purpose:** Detailed documentation of nvImageCodec integration for GPU-accelerated TIFF decoding

---

## Table of Contents

1. [Overview](#overview)
2. [File: nvimgcodec_manager.h](#file-nvimgcodec_managerh)
3. [File: nvimgcodec_decoder.h](#file-nvimgcodec_decoderh)
4. [File: nvimgcodec_decoder.cpp](#file-nvimgcodec_decodercpp)
5. [Key Concepts](#key-concepts)
6. [Thread Safety](#thread-safety)
7. [Memory Management](#memory-management)

---

## Overview

The nvImageCodec implementation provides GPU-accelerated image decoding for JPEG, JPEG2000, and other compression formats commonly found in medical imaging TIFF files (Aperio SVS, Philips TIFF, etc.).

### Architecture

```
┌─────────────────────────────────────────────────────┐
│  cuslide2 Plugin (cuslide.cpp)                      │
│  - Initializes TIFF parsing                         │
│  - Manages file handles                             │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│  TIFF Layer (tiff.cpp, ifd.cpp)                     │
│  - Reads TIFF structure                             │
│  - Identifies compression formats                   │
│  - Manages tile/strip layout                        │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│  nvImageCodec Parsing (nvimgcodec_tiff_parser.cpp)  │
│  - Parses TIFF metadata using nvImageCodec API      │
│  - Creates code streams for each IFD                │
│  - Extracts codec information                       │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│  nvImageCodec Decoding (nvimgcodec_decoder.cpp)     │
│  - Decodes compressed data to RGB buffers           │
│  - Handles GPU/CPU output                           │
│  - Manages ROI (region of interest) decoding        │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│  nvImageCodec Manager (nvimgcodec_manager.h)        │
│  - Singleton instance management                    │
│  - Thread-safe decoder access                       │
│  - Lifecycle management                             │
└─────────────────────────────────────────────────────┘
```

---

## File: nvimgcodec_manager.h

**Location:** `cpp/plugins/cucim.kit.cuslide2/src/cuslide/nvimgcodec/nvimgcodec_manager.h`  
**Lines:** 177  
**Purpose:** Singleton manager for nvImageCodec instance and decoder lifecycle

### Header and License (Lines 1-17)

```cpp
/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#ifdef CUCIM_HAS_NVIMGCODEC
#include <nvimgcodec.h>
#endif

#include <string>
#include <mutex>
#include <fmt/format.h>
```

**Lines 1-15:** Standard Apache 2.0 license header  
**Line 17:** `#pragma once` - Modern header guard (compiler-specific but widely supported)  
**Lines 19-21:** Conditional compilation - only include nvImageCodec API if available  
**Lines 23-25:** Standard library includes for string, thread safety, and logging

### Namespace Declaration (Lines 27-29)

```cpp
namespace cuslide2::nvimgcodec
{
```

**Purpose:** All nvImageCodec-related code is in `cuslide2::nvimgcodec` namespace to avoid naming conflicts

### NvImageCodecManager Class (Lines 37-171)

#### Class Overview (Lines 37-38)

```cpp
/**
 * @brief Singleton manager for nvImageCodec instance and decoder
 * 
 * Provides centralized access to nvImageCodec resources with thread-safe initialization.
 */
class NvImageCodecManager
```

**Design Pattern:** Singleton - ensures only one nvImageCodec instance exists per process  
**Thread Safety:** Uses C++11 "magic statics" for lazy initialization (thread-safe since C++11)

#### Public Interface - Singleton Access (Lines 40-44)

```cpp
static NvImageCodecManager& instance()
{
    static NvImageCodecManager instance;
    return instance;
}
```

**Line 40:** Static method to get the singleton instance  
**Line 42:** C++11 "magic static" - thread-safe lazy initialization  
**Line 43:** Returns reference (not pointer) - guarantees valid object  
**Why Singleton?** nvImageCodec instance is expensive to create and should be shared across all decoding operations

#### Accessors (Lines 46-50)

```cpp
nvimgcodecInstance_t get_instance() const { return instance_; }
nvimgcodecDecoder_t get_decoder() const { return decoder_; }
std::mutex& get_mutex() { return decoder_mutex_; }
bool is_initialized() const { return initialized_; }
const std::string& get_status() const { return status_message_; }
```

**Line 46:** Returns raw nvImageCodec instance handle (opaque pointer type)  
**Line 47:** Returns decoder handle - used for all decode operations  
**Line 48:** Returns mutex reference - callers must lock before using decoder (thread safety)  
**Line 49:** Checks if initialization succeeded  
**Line 50:** Returns human-readable status/error message

#### API Test Function (Lines 52-87)

```cpp
bool test_nvimagecodec_api()
{
    if (!initialized_) return false;
    
    try {
        // Test 1: Get nvImageCodec properties
        nvimgcodecProperties_t props{};
        props.struct_type = NVIMGCODEC_STRUCTURE_TYPE_PROPERTIES;
        props.struct_size = sizeof(nvimgcodecProperties_t);
        props.struct_next = nullptr;
```

**Lines 53-87:** Validation test to ensure nvImageCodec API is working  
**Line 59-62:** Initialize properties structure with required fields:
- `struct_type`: Identifies the structure type (API convention)
- `struct_size`: Size verification (API versioning safety)
- `struct_next`: Extension chain pointer (future compatibility)

**Lines 64-72:** Extract version from packed integer:
```cpp
uint32_t version = props.version;
uint32_t major = (version >> 16) & 0xFF;  // Bits 16-23
uint32_t minor = (version >> 8) & 0xFF;   // Bits 8-15
uint32_t patch = version & 0xFF;          // Bits 0-7
```

**Purpose:** Verifies nvImageCodec library is loaded and functional at startup

#### Delete Copy/Move Constructors (Lines 89-93)

```cpp
NvImageCodecManager(const NvImageCodecManager&) = delete;
NvImageCodecManager& operator=(const NvImageCodecManager&) = delete;
NvImageCodecManager(NvImageCodecManager&&) = delete;
NvImageCodecManager& operator=(NvImageCodecManager&&) = delete;
```

**Purpose:** Enforce singleton pattern - prevent copying or moving the manager  
**C++11 Feature:** `= delete` explicitly deletes these operations (compile-time error if attempted)

#### Private Constructor (Lines 95-156)

```cpp
private:
    NvImageCodecManager() : initialized_(false)
    {
        try {
            // Create nvImageCodec instance following official API pattern
            nvimgcodecInstanceCreateInfo_t create_info{};
```

**Line 96:** Constructor is private - only `instance()` can create the object  
**Line 96:** Member initializer list sets `initialized_` to `false`  
**Line 98:** Exception handling to catch any initialization errors

##### Instance Creation (Lines 100-117)

```cpp
nvimgcodecInstanceCreateInfo_t create_info{};
create_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
create_info.struct_size = sizeof(nvimgcodecInstanceCreateInfo_t);
create_info.struct_next = nullptr;
create_info.load_builtin_modules = 1;  // Load JPEG, JPEG2000, etc.
create_info.load_extension_modules = 1; // Load any extensions
create_info.extension_modules_path = nullptr; // Use default path
create_info.create_debug_messenger = 1; // Enable debug logging
create_info.debug_messenger_desc = nullptr; // Use default messenger
create_info.message_severity = 0;
create_info.message_category = 0;

if (nvimgcodecInstanceCreate(&instance_, &create_info) != NVIMGCODEC_STATUS_SUCCESS)
{
    status_message_ = "Failed to create nvImageCodec instance";
    fmt::print("❌ {}\n", status_message_);
    return;
}
```

**Lines 100-111:** Configure instance creation parameters:
- **load_builtin_modules:** Enables built-in codecs (JPEG, JPEG2000, PNG, TIFF, etc.)
- **load_extension_modules:** Allows loading additional codec plugins
- **create_debug_messenger:** Enables diagnostic messages (useful for debugging)

**Line 112:** Creates the nvImageCodec instance (main library initialization)  
**Lines 114-117:** Error handling - sets status and returns early if creation fails

##### Decoder Creation (Lines 119-141)

```cpp
nvimgcodecExecutionParams_t exec_params{};
exec_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS;
exec_params.struct_size = sizeof(nvimgcodecExecutionParams_t);
exec_params.struct_next = nullptr;
exec_params.device_allocator = nullptr;     // Use default GPU allocator
exec_params.pinned_allocator = nullptr;     // Use default pinned allocator
exec_params.max_num_cpu_threads = 0;       // Use default (all cores)
exec_params.executor = nullptr;             // Use default executor
exec_params.device_id = NVIMGCODEC_DEVICE_CURRENT; // Use current CUDA device
exec_params.pre_init = 0;                   // Don't pre-initialize
exec_params.skip_pre_sync = 0;              // Don't skip synchronization
exec_params.num_backends = 0;               // Use all available backends
exec_params.backends = nullptr;             // (GPU, CPU, hybrid)
```

**Lines 120-133:** Configure decoder execution parameters:
- **device_allocator/pinned_allocator:** Custom memory allocators (nullptr = use defaults)
- **max_num_cpu_threads:** CPU thread pool size (0 = auto-detect)
- **device_id:** Which GPU to use (NVIMGCODEC_DEVICE_CURRENT = current context)
- **num_backends/backends:** Which decoding backends to enable (0 = all)

**Line 134:** Creates the decoder object:
```cpp
nvimgcodecDecoderCreate(instance_, &decoder_, &exec_params, nullptr)
```

**Lines 136-141:** Error handling - cleanup instance if decoder creation fails

##### Success Path (Lines 143-148)

```cpp
initialized_ = true;
status_message_ = "nvImageCodec initialized successfully";
fmt::print("✅ {}\n", status_message_);

// Run quick API test
test_nvimagecodec_api();
```

**Line 143:** Mark as initialized  
**Line 145:** Log success message with ✅ emoji (UTF-8)  
**Line 148:** Run validation test to verify API is working

##### Exception Handling (Lines 150-155)

```cpp
catch (const std::exception& e)
{
    status_message_ = fmt::format("nvImageCodec initialization exception: {}", e.what());
    fmt::print("❌ {}\n", status_message_);
    initialized_ = false;
}
```

**Purpose:** Catch any unexpected exceptions during initialization  
**Line 152:** Format error message with exception details  
**Line 154:** Ensure `initialized_` is false on error

#### Destructor (Lines 158-164)

```cpp
~NvImageCodecManager()
{
    // Intentionally NOT destroying resources to avoid crashes during Python interpreter shutdown
    // The OS will reclaim these resources when the process exits.
    // This is a workaround for nvJPEG2000 cleanup issues during static destruction.
    // Resources are only held in a singleton that lives for the entire program lifetime anyway.
}
```

**CRITICAL DESIGN DECISION:** Resources are intentionally leaked!

**Why?**
1. **Python Shutdown Order:** Python interpreter may destroy CUDA context before C++ statics
2. **nvJPEG2000 Bug:** Cleanup during static destruction can cause crashes
3. **Singleton Lifetime:** Object lives for entire program anyway
4. **OS Cleanup:** Operating system will reclaim all resources when process exits

**Alternative Approaches:**
- Could use `std::atexit()` for explicit cleanup
- Could destroy resources in `parser_close()` (but multiple parsers share instance)
- Could use reference counting (complex for singleton)

**Tradeoff:** Small memory "leak" (freed by OS) vs. potential crash

#### Member Variables (Lines 166-170)

```cpp
nvimgcodecInstance_t instance_{nullptr};
nvimgcodecDecoder_t decoder_{nullptr};
bool initialized_{false};
std::string status_message_;
std::mutex decoder_mutex_;
```

**Line 166:** Opaque handle to nvImageCodec instance (C API)  
**Line 167:** Opaque handle to decoder object  
**Line 168:** Initialization success flag  
**Line 169:** Status/error message for diagnostics  
**Line 170:** Mutex to protect decoder operations from concurrent threads

**Why Mutex?** nvImageCodec decoder is **not thread-safe** for concurrent decode calls on the same decoder object. The mutex ensures only one thread decodes at a time.

---

## File: nvimgcodec_decoder.h

**Location:** `cpp/plugins/cucim.kit.cuslide2/src/cuslide/nvimgcodec/nvimgcodec_decoder.h`  
**Lines:** 145  
**Purpose:** Header file declaring decoding functions for JPEG, JPEG2000, and TIFF ROI decoding

### Header Guard and Includes (Lines 17-25)

```cpp
#ifndef CUSLIDE2_NVIMGCODEC_DECODER_H
#define CUSLIDE2_NVIMGCODEC_DECODER_H

#ifdef CUCIM_HAS_NVIMGCODEC
#include <nvimgcodec.h>
#endif

#include <cucim/io/device.h>
#include <cstdint>
```

**Lines 17-18:** Traditional header guard (alternative to `#pragma once`)  
**Lines 20-22:** Conditional include of nvImageCodec API  
**Line 24:** cuCIM device abstraction (CPU vs GPU)  
**Line 25:** Standard integer types (uint8_t, uint32_t, etc.)

### Namespace and Function Declarations (Lines 27-142)

#### decode_jpeg_nvimgcodec (Lines 30-52)

```cpp
/**
 * Decode JPEG using nvImageCodec
 * 
 * @param fd File descriptor
 * @param jpeg_buf JPEG buffer (if nullptr, read from fd at offset)
 * @param offset File offset to read from
 * @param size Size of compressed data
 * @param jpegtable_data JPEG tables data (for TIFF JPEG)
 * @param jpegtable_count Size of JPEG tables
 * @param dest Output buffer pointer
 * @param out_device Output device ("cpu" or "cuda")
 * @param jpeg_color_space JPEG color space hint
 * @return true if successful
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

**Purpose:** Decode JPEG compressed data (common in TIFF files)

**Parameters:**
- **fd:** File descriptor (POSIX file handle)
- **jpeg_buf:** Pre-read buffer (optimization - avoids file I/O if data already in memory)
- **offset/size:** Location of compressed data in file
- **jpegtable_data/jpegtable_count:** TIFF-specific JPEG tables (stored separately in TIFFTAG_JPEGTABLES)
- **dest:** Output buffer pointer (allocated by function, caller must free)
- **out_device:** "cpu" or "cuda:0" etc. (determines CPU vs GPU output)
- **jpeg_color_space:** JPEG colorspace hint (RGB, YCbCr, Grayscale)

**Return:** `true` if successful, `false` to fallback to libjpeg-turbo

**TIFF JPEG Tables:**
In TIFF files, JPEG data is often stored as "abbreviated" JPEG streams:
- Quantization tables → stored in TIFFTAG_JPEGTABLES
- Huffman tables → stored in TIFFTAG_JPEGTABLES  
- Image data → stored in tile/strip

This function merges the tables with tile data to create a complete JPEG stream.

#### decode_jpeg2k_nvimgcodec (Lines 54-74)

```cpp
/**
 * Decode JPEG2000 using nvImageCodec
 * 
 * @param fd File descriptor
 * @param jpeg2k_buf JPEG2000 buffer (if nullptr, read from fd at offset)
 * @param offset File offset to read from
 * @param size Size of compressed data
 * @param dest Output buffer pointer
 * @param dest_size Expected output size
 * @param out_device Output device ("cpu" or "cuda")
 * @param color_space Color space hint (RGB, YCbCr, etc.)
 * @return true if successful
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

**Purpose:** Decode JPEG2000 compressed data (common in Aperio SVS files)

**Color Space Encoding:**
- **0:** RGB (Aperio compression 33005)
- **1:** YCbCr (Aperio compression 33003)

**JPEG2000 in Medical Imaging:**
- Aperio scanners use JPEG2000 for high-quality compression
- Supports lossy and lossless compression
- Often stored with Aperio-specific compression codes

#### decode_tile_nvtiff_roi (Lines 76-97)

```cpp
/**
 * Decode tile using nvTiff file-level API with ROI
 * 
 * This function uses nvTiff's file-level API which automatically handles
 * JPEG tables (TIFFTAG_JPEGTABLES) without manual merging.
 * 
 * @param file_path Path to TIFF file
 * @param ifd_index IFD index (resolution level)
 * @param tile_x Tile X coordinate in pixels
 * @param tile_y Tile Y coordinate in pixels
 * @param tile_width Tile width in pixels
 * @param tile_height Tile height in pixels
 * @param dest Output buffer pointer (will be allocated)
 * @param out_device Output device ("cpu" or "cuda")
 * @return true if successful, false to fallback to other decoders
 */
bool decode_tile_nvtiff_roi(const char* file_path,
                            uint32_t ifd_index,
                            uint32_t tile_x, uint32_t tile_y,
                            uint32_t tile_width, uint32_t tile_height,
                            uint8_t** dest,
                            const cucim::io::Device& out_device);
```

**Purpose:** High-level TIFF-aware decoding with ROI support

**Advantages over decode_jpeg_nvimgcodec:**
1. No manual JPEG table merging (nvTiff handles it internally)
2. File-level caching (parser reused across tiles)
3. ROI support (decode only requested region)

**Use Case:** Preferred for TIFF files with complex JPEG encoding

#### Forward Declaration and IFD Functions (Lines 99-140)

```cpp
#ifdef CUCIM_HAS_NVIMGCODEC
// Forward declaration
struct IfdInfo;

/**
 * Decode an entire IFD using nvImageCodec
 * ...
 */
bool decode_ifd_nvimgcodec(const IfdInfo& ifd_info,
                           uint8_t** output_buffer,
                           const cucim::io::Device& out_device);

/**
 * Decode a region of interest (ROI) from an IFD using nvImageCodec
 * ...
 */
bool decode_ifd_region_nvimgcodec(const IfdInfo& ifd_info,
                                  nvimgcodecCodeStream_t main_code_stream,
                                  uint32_t x, uint32_t y,
                                  uint32_t width, uint32_t height,
                                  uint8_t** output_buffer,
                                  const cucim::io::Device& out_device);
#endif
```

**Line 101:** Forward declare `IfdInfo` struct (defined in nvimgcodec_tiff_parser.h)  
**Lines 114-116:** Decode entire IFD (full resolution level)  
**Lines 134-139:** Decode ROI from IFD (memory-efficient partial decode)

**IFD (Image File Directory):** TIFF structure representing one resolution level  
**Code Stream:** nvImageCodec abstraction for compressed image data

---

## File: nvimgcodec_decoder.cpp

**Location:** `cpp/plugins/cucim.kit.cuslide2/src/cuslide/nvimgcodec/nvimgcodec_decoder.cpp`  
**Lines:** 1137  
**Purpose:** Implementation of all decoding functions

### Includes and Namespace (Lines 1-40)

```cpp
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
```

**Lines 17-19:** Include decoder header, parser, and manager  
**Lines 25-33:** Standard library includes  
**Line 31:** `<unistd.h>` for POSIX file I/O (`lseek`, `read`)  
**Line 36:** CUDA runtime API for GPU operations  
**Line 39:** Namespace declaration

### Global Parser Cache (Lines 44-49)

```cpp
#ifdef CUCIM_HAS_NVIMGCODEC

// Global TiffFileParser cache for nvTiff file-level API
static std::mutex parser_cache_mutex;
static std::map<std::string, std::shared_ptr<TiffFileParser>> parser_cache;
```

**Line 47:** Mutex protects concurrent access to cache  
**Line 48:** Map from file path to parser instance

**Purpose:** Cache `TiffFileParser` objects to avoid re-parsing the same TIFF file for every tile

**Example:** Reading 1000 tiles from same file:
- **Without cache:** Parse TIFF structure 1000 times (slow)
- **With cache:** Parse once, reuse 999 times (fast)

**Thread Safety:** Mutex ensures cache is safe to access from multiple threads

### Function: decode_tile_nvtiff_roi (Lines 51-107)

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

**Lines 58-61:** Validate input parameters (nullptr check)

#### Get or Create Parser (Lines 63-85)

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

**Line 68:** Lock mutex for thread-safe cache access  
**Line 69:** Look up file_path in cache  
**Lines 70-73:** Cache hit - reuse existing parser  
**Lines 74-83:** Cache miss - create new parser and cache it  
**Line 84:** Mutex lock automatically released (RAII)

**RAII (Resource Acquisition Is Initialization):**  
The `std::lock_guard` automatically unlocks when it goes out of scope (line 84), even if an exception is thrown.

#### Validate IFD Index (Lines 87-93)

```cpp
// Check if IFD index is valid
if (ifd_index >= parser->get_ifd_count())
{
    fmt::print("⚠️  nvTiff ROI: Invalid IFD index {} (max: {})\n", 
              ifd_index, parser->get_ifd_count() - 1);
    return false;
}
```

**Purpose:** Ensure requested IFD exists in the TIFF file  
**Example:** File with 5 IFDs (0-4), requesting IFD 10 → error

#### Decode Region (Lines 95-100)

```cpp
// Decode the tile region using nvTiff file-level API
*dest = parser->decode_region(ifd_index, tile_x, tile_y, 
                              tile_width, tile_height, 
                              nullptr, out_device);

return (*dest != nullptr);
```

**Line 96:** Call `TiffFileParser::decode_region()` which internally uses nvImageCodec  
**Line 100:** Return `true` if buffer is allocated, `false` otherwise

#### Exception Handling (Lines 102-106)

```cpp
catch (const std::exception& e)
{
    fmt::print("❌ nvTiff ROI decode failed: {}\n", e.what());
    return false;
}
```

**Purpose:** Catch any exceptions and return `false` for fallback to other decoders

---

### Function: decode_jpeg_nvimgcodec (Lines 109-433)

This is the main JPEG decoding function. It's complex because it handles:
1. Abbreviated JPEG streams (TIFF-specific)
2. JPEG table merging
3. GPU/CPU output
4. Thread safety
5. Memory management

#### Function Signature and Validation (Lines 109-139)

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

**Line 120:** Get singleton manager instance  
**Lines 122-127:** Check if nvImageCodec is initialized, return `false` if not (fallback)

#### JPEG Tables Workaround (Lines 129-135)

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

**CRITICAL DECISION:** nvImageCodec 0.7.0 has issues with abbreviated JPEG streams

**Abbreviated JPEG:**
```
Normal JPEG:     [SOI][Tables][Image Data][EOI]
Abbreviated:     [SOI][Image Data][EOI]
Tables stored separately in TIFF tag
```

**Workaround:** Return `false` to use libjpeg-turbo instead

#### Read JPEG Data (Lines 140-159)

```cpp
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
```

**Lines 145-147:** If data already in memory (`jpeg_buf != nullptr`), use it directly  
**Lines 148-158:** Otherwise, read from file at specified offset

**POSIX File I/O:**
- `lseek(fd, offset, SEEK_SET)` - Move file pointer to offset
- `read(fd, buffer, size)` - Read size bytes into buffer

#### JPEG Tables Merging (Lines 161-210)

This section handles merging JPEG tables with tile data (for future nvImageCodec versions).

```cpp
// Handle JPEG tables (common in Aperio SVS files)
if (jpegtable_data && jpegtable_count > 0) {
    fmt::print("📋 nvImageCodec JPEG decode: Processing JPEG tables ({} bytes) with tile data ({} bytes)\n", 
              jpegtable_count, jpeg_data.size());
    
    // Validate inputs
    if (jpegtable_count < 2 || jpeg_data.size() < 2) {
        fmt::print("⚠️  nvImageCodec: Invalid JPEG data sizes, skipping table merge\n");
    } else {
        // Create properly sized buffer
        std::vector<uint8_t> jpeg_with_tables;
        jpeg_with_tables.reserve(jpegtable_count + jpeg_data.size() + 4);
```

**Line 172:** Reserve space for merged stream (tables + data + safety margin)

##### Remove EOI from Tables (Lines 175-183)

```cpp
const uint8_t* table_ptr = static_cast<const uint8_t*>(jpegtable_data);
size_t table_copy_size = jpegtable_count;

// Remove trailing EOI (0xFFD9) from tables if present
if (table_copy_size >= 2 && table_ptr[table_copy_size - 2] == 0xFF && 
    table_ptr[table_copy_size - 1] == 0xD9) {
    table_copy_size -= 2;
    fmt::print("📋 Removed EOI from tables\n");
}
```

**JPEG Markers:**
- **SOI (Start of Image):** `0xFFD8`
- **EOI (End of Image):** `0xFFD9`

**Why remove EOI from tables?**  
Tables structure: `[SOI][Tables][EOI]`  
Merged stream should be: `[SOI][Tables][Image Data][EOI]`  
Not: `[SOI][Tables][EOI][SOI][Image Data][EOI]` ← invalid!

##### Skip SOI from Tile Data (Lines 188-193)

```cpp
// Skip SOI (0xFFD8) from tile data if present
size_t tile_offset = 0;
if (jpeg_data.size() >= 2 && jpeg_data[0] == 0xFF && jpeg_data[1] == 0xD8) {
    tile_offset = 2;
    fmt::print("📋 Skipped SOI from tile data\n");
}
```

**Why skip SOI from tile data?**  
Tile data: `[SOI][Image Data][EOI]`  
We already have SOI from tables, don't need another one.

##### Merge and Validate (Lines 195-209)

```cpp
// Append tile data
if (tile_offset < jpeg_data.size()) {
    jpeg_with_tables.insert(jpeg_with_tables.end(), 
                          jpeg_data.begin() + tile_offset, 
                          jpeg_data.end());
}

// Validate final size
if (jpeg_with_tables.size() > 0 && jpeg_with_tables.size() < 1024 * 1024 * 10) {
    jpeg_data = std::move(jpeg_with_tables);
    fmt::print("✅ Merged JPEG stream: {} bytes\n", jpeg_data.size());
} else {
    fmt::print("⚠️  Invalid merged size: {} bytes, using original\n", jpeg_with_tables.size());
}
```

**Line 203:** Sanity check - reject if > 10MB (likely corrupted)  
**Line 204:** Move semantics - efficient transfer of vector data

#### Create Code Stream (Lines 212-227)

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
    return false;
}
```

**Line 219:** Create nvImageCodec code stream from memory buffer  
**Key Point:** Code stream is an abstraction that can come from memory, file, or network

#### Get Image Info (Lines 229-242)

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

**Line 234:** Query image dimensions and format from code stream  
**Purpose:** Understand input format before allocating output buffer

#### Prepare Output Format (Lines 244-268)

```cpp
// Step 3: Prepare output image info (following official API pattern)
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
    default:
        output_image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        break;
}
```

**Line 247:** `NVIMGCODEC_SAMPLEFORMAT_I_RGB` = Interleaved RGB (RGBRGBRGB...)  
**Alternative:** `NVIMGCODEC_SAMPLEFORMAT_P_RGB` = Planar RGB (RRR...GGG...BBB...)

**Interleaved vs Planar:**
```
Interleaved: [R0 G0 B0][R1 G1 B1][R2 G2 B2]  ← cuCIM expects this
Planar:      [R0 R1 R2...][G0 G1 G2...][B0 B1 B2...]
```

#### Set Buffer Type (Lines 272-278)

```cpp
// Set buffer kind based on output device
std::string device_str = std::string(out_device);
if (device_str.find("cuda") != std::string::npos) {
    output_image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
} else {
    output_image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
}
```

**Line 274:** Check if device string contains "cuda"  
**Line 275:** `STRIDED_DEVICE` = GPU memory (requires `cudaMalloc`)  
**Line 277:** `STRIDED_HOST` = CPU memory (requires `malloc`)

#### Calculate Buffer Size (Lines 280-298)

```cpp
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

**Line 282:** Extract bytes per pixel from sample type enum  
**Line 288:** Row stride = bytes per row (width × channels × bytes_per_element)  
**Line 297:** Total size = row_stride × height

**Example:**
- Image: 512×512 pixels
- Format: RGB, 8-bit per channel
- Row stride: 512 × 3 × 1 = 1536 bytes
- Total size: 1536 × 512 = 786,432 bytes

#### Allocate Output Buffer (Lines 300-322)

```cpp
// Use pre-allocated buffer if provided, otherwise allocate new buffer
void* output_buffer = *dest;
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
```

**Line 301:** Check if caller pre-allocated buffer  
**Line 307:** GPU allocation with `cudaMalloc`  
**Line 313:** CPU allocation with standard `malloc`

**Why support pre-allocated buffers?**  
Optimization - caller can allocate once and reuse for multiple decodes

#### Create Image Object (Lines 324-337)

```cpp
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
```

**Line 326:** Create image object from output info  
**Lines 328-335:** Cleanup on error (only if we allocated the buffer)

#### Prepare Decode Parameters (Lines 339-344)

```cpp
// Step 5: Prepare decode parameters (following official API pattern)
nvimgcodecDecodeParams_t decode_params{};
decode_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS;
decode_params.struct_size = sizeof(nvimgcodecDecodeParams_t);
decode_params.struct_next = nullptr;
decode_params.apply_exif_orientation = 1;
```

**Line 344:** `apply_exif_orientation = 1` - Auto-rotate based on EXIF orientation tag

#### Schedule Decoding (Lines 346-364)

```cpp
// Step 6: Schedule decoding (following official API pattern)
// THREAD-SAFETY: Lock the decoder to prevent concurrent access from multiple threads
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

**Line 350:** **CRITICAL:** Lock mutex before calling decode  
**Line 351:** Schedule decode operation (asynchronous on GPU)  
**Line 364:** Mutex automatically released (end of scope)

**Why lock?** nvImageCodec decoder is not thread-safe for concurrent calls

#### Wait for Completion (Lines 366-414)

```cpp
// Step 7: Wait for decoding to finish
size_t status_size = 1;
nvimgcodecProcessingStatus_t decode_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;

nvimgcodecStatus_t future_status = nvimgcodecFutureGetProcessingStatus(
    decode_future, &decode_status, &status_size);
    
if (future_status != NVIMGCODEC_STATUS_SUCCESS) {
    fmt::print("❌ nvImageCodec JPEG decode: Failed to get future status\n");
    // ... cleanup ...
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

**Line 371:** Get processing status (blocks until decode completes)  
**Line 392:** `cudaDeviceSynchronize()` ensures GPU operations are finished  
**Why sync?** GPU operations are asynchronous, must wait before accessing result

#### Success Path (Lines 416-426)

```cpp
// Success! Set output pointer
*dest = static_cast<uint8_t*>(output_buffer);

fmt::print("✅ nvImageCodec JPEG decode: Successfully decoded {}x{} image\n",
          output_image_info.plane_info[0].width, output_image_info.plane_info[0].height);

// Cleanup (but keep the output buffer for caller)
nvimgcodecFutureDestroy(decode_future);
nvimgcodecImageDestroy(image);
nvimgcodecCodeStreamDestroy(code_stream);

return true;
```

**Line 417:** Set output pointer to allocated buffer  
**Lines 423-425:** Destroy nvImageCodec objects (but NOT the output buffer - caller owns it)

---

### Function: decode_jpeg2k_nvimgcodec (Lines 435-675)

**Purpose:** Decode JPEG2000 data (similar structure to `decode_jpeg_nvimgcodec`)

**Key Differences from JPEG:**
1. No JPEG tables to merge
2. Different color space mapping (lines 510-523)
3. More debug logging (for troubleshooting)

**Lines 444-456:** Initialize manager and validate  
**Lines 461-479:** Read JPEG2000 data from file or buffer  
**Lines 482-486:** Create code stream  
**Lines 489-501:** Get image info  
**Lines 504-534:** Prepare output format with color space mapping  
**Lines 536-575:** Allocate output buffer  
**Lines 580-593:** Create image object  
**Lines 595-600:** Prepare decode parameters  
**Lines 603-622:** Schedule decoding (with mutex lock)  
**Lines 624-653:** Wait for completion and check status  
**Lines 655-665:** Success path and cleanup

**Color Space Mapping (Lines 510-523):**
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

**Aperio JPEG2000 Formats:**
- **33003:** YCbCr JPEG2000 (for lossy compression)
- **33005:** RGB JPEG2000 (for lossless compression)

---

### Function: decode_ifd_nvimgcodec (Lines 681-867)

**Purpose:** Decode entire IFD (Image File Directory) using parsed metadata

**Key Concept:** This function separates parsing from decoding:
- **Parsing:** Done once by `TiffFileParser` (creates `IfdInfo`)
- **Decoding:** Done many times using `IfdInfo.sub_code_stream`

#### Entry Point (Lines 681-703)

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
    
    try
    {
        // CRITICAL: Must use the same manager that created the sub_code_stream
        auto& manager = NvImageCodecTiffParserManager::instance();
```

**Line 685:** `IfdInfo` contains pre-parsed metadata (width, height, codec, etc.)  
**Line 685:** `sub_code_stream` is nvImageCodec handle to this IFD's compressed data  
**Line 698:** **CRITICAL:** Use `NvImageCodecTiffParserManager` (same instance that created sub_code_stream)

**Why critical?**  
nvImageCodec objects (code streams) are tied to the instance that created them.  
Using a decoder from a different instance → **SEGFAULT**

#### Prepare Output Info (Lines 707-744)

```cpp
// Step 1: Prepare output image info
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

// Calculate buffer requirements
uint32_t num_channels = 3;
size_t row_stride = ifd_info.width * num_channels;
size_t buffer_size = row_stride * ifd_info.height;

output_image_info.plane_info[0].height = ifd_info.height;
output_image_info.plane_info[0].width = ifd_info.width;
output_image_info.plane_info[0].num_channels = num_channels;
output_image_info.plane_info[0].row_stride = row_stride;
output_image_info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
output_image_info.buffer_size = buffer_size;
output_image_info.cuda_stream = 0;
```

**Lines 714-717:** Set output format to interleaved RGB  
**Lines 719-728:** Choose GPU or CPU output  
**Lines 730-741:** Calculate buffer size from IFD dimensions

#### Allocate Buffer (Lines 746-768)

```cpp
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
```

**Lines 750-757:** GPU allocation  
**Lines 758-768:** CPU allocation

#### Create Image and Decode (Lines 772-855)

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

// Step 6: Wait for completion
nvimgcodecProcessingStatus_t decode_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
size_t status_size = 1;
nvimgcodecFutureGetProcessingStatus(decode_future, &decode_status, &status_size);

if (output_image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE)
{
    cudaDeviceSynchronize();  // Wait for GPU operations
}
```

**Line 804:** Use `ifd_info.sub_code_stream` (pre-parsed)  
**Line 829:** Get status (blocks until done)  
**Line 833:** Sync GPU if needed

#### Success and Cleanup (Lines 838-866)

```cpp
// Cleanup
nvimgcodecFutureDestroy(decode_future);
nvimgcodecImageDestroy(image);

if (decode_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
{
    // ... error handling and buffer cleanup ...
    return false;
}

// Success! Return buffer to caller
*output_buffer = static_cast<uint8_t*>(buffer);

fmt::print("✅ Successfully decoded IFD[{}]\n", ifd_info.index);
return true;
```

**Lines 839-840:** Destroy nvImageCodec objects  
**Line 857:** Return allocated buffer to caller

---

### Function: decode_ifd_region_nvimgcodec (Lines 869-1095)

**Purpose:** Decode only a specific region (ROI) from an IFD

**Key Feature:** Uses `nvimgcodecCodeStreamGetSubCodeStream` with ROI specification

#### Entry and Validation (Lines 869-894)

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
    
    try
    {
        // CRITICAL: Must use the same manager that created main_code_stream!
        auto& manager = NvImageCodecTiffParserManager::instance();
```

**Line 870:** `main_code_stream` is the full TIFF file code stream  
**Lines 871-873:** ROI coordinates (x, y, width, height)  
**Line 889:** Use same manager instance (prevent segfault)

#### Create ROI View (Lines 898-929)

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

**Lines 899-908:** Define ROI as a 2D region (start/end coordinates)  
**Line 913:** `image_idx` specifies which IFD to decode from  
**Line 918:** Create sub-code stream for just this region

**How ROI Decoding Works:**
```
Full IFD:     [0,0] ──────────────► [width, height]
              │                                  │
              │    ROI: [x,y]──►[x+w, y+h]      │
              │         │            │           │
              │         └────────────┘           │
              └──────────────────────────────────┘

Only decode the shaded ROI area, not the full IFD
```

**Memory Savings:**
- Full IFD: 40000×40000 pixels = 4.8 GB RGB
- ROI 512×512: 512×512 pixels = 768 KB RGB
- **Savings: 6250×** less memory!

#### Prepare Output for ROI (Lines 932-965)

```cpp
// Step 2: Prepare output image info for the region
nvimgcodecImageInfo_t output_image_info{};
// ... (similar to decode_ifd_nvimgcodec but with region dimensions) ...

uint32_t num_channels = 3;
size_t row_stride = width * num_channels;  // Use ROI width, not full IFD
size_t buffer_size = row_stride * height;  // Use ROI height

output_image_info.plane_info[0].height = height;  // ROI height
output_image_info.plane_info[0].width = width;    // ROI width
```

**Key Difference:** Buffer size based on ROI dimensions, not full IFD

#### Allocate, Decode, and Return (Lines 970-1088)

```cpp
// Step 3: Allocate output buffer (for ROI only)
void* buffer = nullptr;
if (output_image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE)
{
    cudaMalloc(&buffer, buffer_size);  // Much smaller allocation!
}
else
{
    buffer = malloc(buffer_size);
}

// ... (steps 4-7 identical to decode_ifd_nvimgcodec) ...

// Cleanup
nvimgcodecFutureDestroy(decode_future);
nvimgcodecImageDestroy(image);
nvimgcodecCodeStreamDestroy(roi_stream);  // Destroy ROI stream

// Success
*output_buffer = static_cast<uint8_t*>(buffer);
return true;
```

**Line 1068:** Destroy ROI-specific code stream  
**Line 1085:** Return ROI buffer to caller

---

### Fallback Implementations (Lines 1097-1134)

```cpp
#else // !CUCIM_HAS_NVIMGCODEC

// Fallback implementations when nvImageCodec is not available
bool decode_jpeg_nvimgcodec(/* ... */)
{
    (void)fd; (void)jpeg_buf; /* ... suppress unused warnings ... */
    
    fmt::print(stderr, "nvImageCodec not available - falling back to original decoder\n");
    return false;
}

bool decode_jpeg2k_nvimgcodec(/* ... */)
{
    /* ... same pattern ... */
    return false;
}

#endif // CUCIM_HAS_NVIMGCODEC
```

**Purpose:** Provide stub implementations when nvImageCodec is not compiled in  
**Return:** Always `false` → fallback to libjpeg-turbo/libopenjpeg

---

## Key Concepts

### 1. Code Stream Hierarchy

```
TIFF File
  ├─ Main Code Stream (nvimgcodecCodeStreamCreateFromFile)
  │    ├─ Sub-Code Stream: IFD 0 (full resolution)
  │    ├─ Sub-Code Stream: IFD 1 (2× downsample)
  │    ├─ Sub-Code Stream: IFD 2 (4× downsample)
  │    └─ ...
  │
  └─ ROI Code Stream (nvimgcodecCodeStreamGetSubCodeStream)
       └─ View into Sub-Code Stream with region specification
```

### 2. Memory Ownership

```
Caller                   nvImageCodec Function              Responsibility
──────                   ─────────────────────              ──────────────
Input Buffer             → decode_*_nvimgcodec()            Caller allocates and manages
Output Buffer (*dest)    ← allocated by function            Caller must free()
nvImageCodec Objects     Created & destroyed internally     Function manages
```

### 3. Thread Safety Model

```
Thread 1: decode()      ┐
                        ├─► [Mutex Lock] → nvImageCodec Decoder → [Unlock]
Thread 2: decode()      ┘

Only one thread can use the decoder at a time.
Other threads block at mutex until decoder is free.
```

### 4. Error Handling Strategy

```
┌─────────────────────────────────┐
│ nvImageCodec decode attempt     │
└────────────┬────────────────────┘
             │
             ▼
        Success? ──Yes──► Return buffer
             │
             No
             │
             ▼
        Return false
             │
             ▼
┌─────────────────────────────────┐
│ Caller fallback to CPU decoder  │
│ (libjpeg-turbo / libopenjpeg)   │
└─────────────────────────────────┘
```

**Design:** nvImageCodec is an **optimization**, not a requirement  
If nvImageCodec fails → graceful fallback to CPU decoders

---

## Thread Safety

### Mechanisms

1. **Singleton Manager:**
   - C++11 magic statics ensure thread-safe initialization
   - Single instance shared across all threads

2. **Decoder Mutex:**
   - `std::mutex decoder_mutex_` protects decoder operations
   - Lock held during `nvimgcodecDecoderDecode()` call
   - Released after scheduling (decode runs asynchronously)

3. **Parser Cache Mutex:**
   - `std::mutex parser_cache_mutex` protects parser cache
   - Short lock duration (only during cache lookup/insert)

### Lock Ordering

```
1. parser_cache_mutex (if using nvTiff ROI)
2. decoder_mutex (during decode scheduling)

Never lock in reverse order → prevents deadlock
```

### RAII Lock Guards

```cpp
{
    std::lock_guard<std::mutex> lock(mutex);
    // Critical section
    // ...
} // Automatic unlock (even if exception thrown)
```

---

## Memory Management

### Buffer Lifecycle

```
1. Allocation:   decode_*_nvimgcodec()
                 ├─ GPU: cudaMalloc()
                 └─ CPU: malloc()

2. Use:          Caller reads decoded image data

3. Deallocation: Caller's responsibility
                 ├─ GPU: cudaFree()
                 └─ CPU: free()
```

### nvImageCodec Objects

```
Object               Creation                    Destruction
──────               ────────                    ───────────
Instance             NvImageCodecManager()       ~NvImageCodecManager() [intentionally skipped]
Decoder              NvImageCodecManager()       ~NvImageCodecManager() [intentionally skipped]
CodeStream           nvimgcodecCodeStreamCreate  nvimgcodecCodeStreamDestroy
Image                nvimgcodecImageCreate       nvimgcodecImageDestroy
Future               nvimgcodecDecoderDecode     nvimgcodecFutureDestroy
```

### Resource Cleanup Pattern

```cpp
// Create resources
nvimgcodecCodeStream_t stream;
nvimgcodecImage_t image;
nvimgcodecFuture_t future;

// Use resources
nvimgcodecDecoderDecode(decoder, &stream, &image, 1, &params, &future);

// Always cleanup (even on error)
nvimgcodecFutureDestroy(future);
nvimgcodecImageDestroy(image);
nvimgcodecCodeStreamDestroy(stream);
// Buffer is NOT destroyed (caller owns it)
```

---

## Summary

### nvimgcodec_manager.h
- **Purpose:** Singleton lifecycle management for nvImageCodec instance and decoder
- **Key Feature:** Thread-safe initialization with C++11 magic statics
- **Design Decision:** Intentionally leak resources to avoid Python shutdown crashes

### nvimgcodec_decoder.h
- **Purpose:** Public API declarations for JPEG, JPEG2000, and TIFF ROI decoding
- **Key Functions:**
  - `decode_jpeg_nvimgcodec()` - JPEG tile decoding
  - `decode_jpeg2k_nvimgcodec()` - JPEG2000 tile decoding
  - `decode_tile_nvtiff_roi()` - TIFF-aware ROI decoding
  - `decode_ifd_nvimgcodec()` - Full IFD decoding
  - `decode_ifd_region_nvimgcodec()` - IFD ROI decoding

### nvimgcodec_decoder.cpp
- **Purpose:** Implementation of all decoding functions
- **Key Patterns:**
  1. Graceful fallback on nvImageCodec errors
  2. Thread-safe decoder access with mutex
  3. Caller-owned output buffers
  4. Careful resource cleanup
  5. GPU/CPU output support

### Design Philosophy
- **Optimization, not requirement:** nvImageCodec accelerates decoding, but CPU fallback always available
- **Thread safety:** Mutex-protected decoder, safe singleton initialization
- **Memory efficiency:** ROI decoding for large images
- **Robustness:** Extensive error handling and logging

---

**Document Version:** 1.0  
**Last Updated:** November 17, 2025

