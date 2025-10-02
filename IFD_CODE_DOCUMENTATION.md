# IFD Class - Complete Code Documentation

## Overview

The `IFD` (Image File Directory) class represents a single image directory within a TIFF file. In the nvImageCodec implementation, each IFD corresponds to one resolution level in a multi-resolution whole-slide image (WSI).

---

## Table of Contents

1. [Header File (`ifd.h`)](#header-file-ifdh)
2. [Implementation File (`ifd.cpp`)](#implementation-file-ifdcpp)
   - [Constructors](#constructors)
   - [Destructor](#destructor)
   - [Read Methods](#read-methods)
   - [Accessors](#accessors)
   - [Helper Methods](#helper-methods)
   - [Tile Reading Logic](#tile-reading-logic)

---

# Header File (`ifd.h`)

## Lines 1-24: Header Guards and Includes

```cpp
/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CUSLIDE_IFD_H
#define CUSLIDE_IFD_H

#include "types.h"
#include "tiff_constants.h"

#include <memory>
#include <vector>
#include <string>

#include <cucim/concurrent/threadpool.h>
#include <cucim/io/format/image_format.h>
#include <cucim/io/device.h>
#include <cucim/loader/thread_batch_data_loader.h>

#ifdef CUCIM_HAS_NVIMGCODEC
#include <nvimgcodec.h>
#include "cuslide/nvimgcodec/nvimgcodec_tiff_parser.h"
#endif
```

**Purpose:**
- Standard copyright and license headers
- Include guard prevents multiple inclusion
- Imports local types, constants, and nvImageCodec headers
- Conditional compilation for nvImageCodec support

---

## Lines 26-30: Namespace and Forward Declaration

```cpp
namespace cuslide::tiff
{

// Forward declaration.
class TIFF;
```

**Purpose:**
- All code is in `cuslide::tiff` namespace
- Forward declaration of `TIFF` class avoids circular dependency (IFD needs TIFF pointer, TIFF contains IFD vector)

---

## Lines 32-39: Class Declaration and Constructors

```cpp
class EXPORT_VISIBLE IFD : public std::enable_shared_from_this<IFD>
{
public:
    IFD(TIFF* tiff, uint16_t index, ifd_offset_t offset);
#ifdef CUCIM_HAS_NVIMGCODEC
    IFD(TIFF* tiff, uint16_t index, const cuslide2::nvimgcodec::IfdInfo& ifd_info);
#endif
    ~IFD();
```

**Purpose:**
- `EXPORT_VISIBLE`: Makes class visible across shared library boundaries
- `std::enable_shared_from_this<IFD>`: Allows creating shared_ptr from `this` pointer
- **Two constructors**:
  1. Legacy constructor (offset-based, deprecated)
  2. **Primary nvImageCodec constructor** (uses `IfdInfo` metadata)

---

## Lines 41-64: Static Read Methods

```cpp
static bool read_region_tiles(const TIFF* tiff,
                              const IFD* ifd,
                              const int64_t* location,
                              const int64_t location_index,
                              const int64_t w,
                              const int64_t h,
                              void* raster,
                              const cucim::io::Device& out_device,
                              cucim::loader::ThreadBatchDataLoader* loader);

static bool read_region_tiles_boundary(const TIFF* tiff,
                                       const IFD* ifd,
                                       const int64_t* location,
                                       const int64_t location_index,
                                       const int64_t w,
                                       const int64_t h,
                                       void* raster,
                                       const cucim::io::Device& out_device,
                                       cucim::loader::ThreadBatchDataLoader* loader);

bool read(const TIFF* tiff,
          const cucim::io::format::ImageMetadataDesc* metadata,
          const cucim::io::format::ImageReaderRegionRequestDesc* request,
          cucim::io::format::ImageDataDesc* out_image_data);
```

**Purpose:**
- `read_region_tiles()`: Main tile-based reading for in-bounds regions
- `read_region_tiles_boundary()`: Handles out-of-bounds regions with boundary checks
- `read()`: High-level read interface (calls nvImageCodec ROI decoder)

---

## Lines 67-97: Public Accessor Methods

```cpp
uint32_t index() const;
ifd_offset_t offset() const;

std::string& software();
std::string& model();
std::string& image_description();
uint16_t resolution_unit() const;
float x_resolution() const;
float y_resolution() const;
uint32_t width() const;
uint32_t height() const;
uint32_t tile_width() const;
uint32_t tile_height() const;
uint32_t rows_per_strip() const;
uint32_t bits_per_sample() const;
uint32_t samples_per_pixel() const;
uint64_t subfile_type() const;
uint16_t planar_config() const;
uint16_t photometric() const;
uint16_t compression() const;
uint16_t predictor() const;

uint16_t subifd_count() const;
std::vector<uint64_t>& subifd_offsets();

uint32_t image_piece_count() const;
const std::vector<uint64_t>& image_piece_offsets() const;
const std::vector<uint64_t>& image_piece_bytecounts() const;

size_t pixel_size_nbytes() const;
size_t tile_raster_size_nbytes() const;
```

**Purpose:**
- Getters for all IFD metadata fields
- TIFF standard tags (width, height, compression, etc.)
- Tile/strip information
- Calculated sizes (pixel_size_nbytes, tile_raster_size_nbytes)

---

## Lines 99-144: Private Members

```cpp
private:
    TIFF* tiff_; // cannot use shared_ptr as IFD is created during the construction of TIFF using 'new'
    uint32_t ifd_index_ = 0;
    ifd_offset_t ifd_offset_ = 0;

    std::string software_;
    std::string model_;
    std::string image_description_;
    uint16_t resolution_unit_ = 1; // 1 = No absolute unit of measurement, 2 = Inch, 3 = Centimeter
    float x_resolution_ = 1.0f;
    float y_resolution_ = 1.0f;

    uint32_t flags_ = 0;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    uint32_t tile_width_ = 0;
    uint32_t tile_height_ = 0;
    uint32_t rows_per_strip_ = 0;
    uint32_t bits_per_sample_ = 0;
    uint32_t samples_per_pixel_ = 0;
    uint64_t subfile_type_ = 0;
    uint16_t planar_config_ = 0;
    uint16_t photometric_ = 0;
    uint16_t compression_ = 0;
    uint16_t predictor_ = 1; // 1: none, 2: horizontal differencing, 3: floating point predictor

    uint16_t subifd_count_ = 0;
    std::vector<uint64_t> subifd_offsets_;

    std::vector<uint8_t> jpegtable_;
    int32_t jpeg_color_space_ = 0; /// 0: JCS_UNKNOWN, 2: JCS_RGB, 3: JCS_YCbCr

    uint32_t image_piece_count_ = 0;
    std::vector<uint64_t> image_piece_offsets_;
    std::vector<uint64_t> image_piece_bytecounts_;

    uint64_t hash_value_ = 0; /// file hash including ifd index.

#ifdef CUCIM_HAS_NVIMGCODEC
    // nvImageCodec-specific members
    nvimgcodecCodeStream_t nvimgcodec_sub_stream_ = nullptr;
    std::string codec_name_;  // codec name from nvImageCodec (jpeg, jpeg2k, deflate, etc.)
#endif
```

**Key Fields:**
- `tiff_`: Raw pointer to parent TIFF object (can't use shared_ptr due to construction order)
- **Metadata fields**: Standard TIFF tags (software, model, dimensions, etc.)
- **Tile/strip info**: Offsets and byte counts for each tile/strip
- **nvImageCodec specific**:
  - `nvimgcodec_sub_stream_`: Code stream handle for this IFD's image data
  - `codec_name_`: Compression codec (jpeg, jpeg2k, deflate, etc.)

---

## Lines 146-170: Private Helper Methods

```cpp
/**
 * @brief Check if the current compression method is supported or not.
 */
bool is_compression_supported() const;

/**
 *
 * Note: This method is called by the constructor of IFD and read() method so it is possible that the output of
 *       'is_read_optimizable()' could be changed during read() method if user set read configuration
 *       after opening TIFF file.
 * @return
 */
bool is_read_optimizable() const;

/**
 * @brief Check if the specified image format is supported or not.
 */
bool is_format_supported() const;

#ifdef CUCIM_HAS_NVIMGCODEC
/**
 * @brief Parse codec string to TIFF compression code
 */
static uint16_t parse_codec_to_compression(const std::string& codec);
#endif
```

**Purpose:**
- `is_compression_supported()`: Validates compression type (JPEG, JPEG2000, deflate, LZW, etc.)
- `is_read_optimizable()`: Checks if fast path can be used
- `is_format_supported()`: Overall format validation
- `parse_codec_to_compression()`: Converts nvImageCodec codec string to TIFF compression enum

---

# Implementation File (`ifd.cpp`)

## Lines 1-33: Includes and Namespace

```cpp
/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ifd.h"

#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <thread>

#include <fmt/format.h>

#include <cucim/codec/hash_function.h>
#include <cucim/cuimage.h>
#include <cucim/logger/timer.h>
#include <cucim/memory/memory_manager.h>
#include <cucim/profiler/nvtx3.h>
#include <cucim/util/cuda.h>

// nvImageCodec handles ALL decoding (JPEG, JPEG2000, deflate, LZW, raw)
#include "cuslide/nvimgcodec/nvimgcodec_decoder.h"
#include "cuslide/nvimgcodec/nvimgcodec_tiff_parser.h"
#include "tiff.h"
#include "tiff_constants.h"


namespace cuslide::tiff
{
```

**Imports:**
- System headers for file operations
- `fmt` for string formatting
- cuCIM utilities (hashing, profiling, memory management)
- nvImageCodec decoder and parser
- TIFF constants and parent TIFF class

---

## Lines 35-84: Legacy Constructor (Deprecated)

```cpp
// OLD CONSTRUCTOR: libtiff-based (DEPRECATED - use nvImageCodec constructor instead)
// This constructor is kept for API compatibility but is not functional in pure nvImageCodec build
IFD::IFD(TIFF* tiff, uint16_t index, ifd_offset_t offset) : tiff_(tiff), ifd_index_(index), ifd_offset_(offset)
{
#ifdef CUCIM_HAS_NVIMGCODEC
    // Pure nvImageCodec path: try to use IfdInfo instead
    if (tiff->nvimgcodec_parser_ && tiff->nvimgcodec_parser_->is_valid())
    {
        if (static_cast<uint32_t>(index) < tiff->nvimgcodec_parser_->get_ifd_count())
        {
            const auto& ifd_info = tiff->nvimgcodec_parser_->get_ifd(static_cast<uint32_t>(index));

            // Initialize from IfdInfo
            width_ = ifd_info.width;
            height_ = ifd_info.height;
            samples_per_pixel_ = ifd_info.num_channels;
            bits_per_sample_ = ifd_info.bits_per_sample;

            // Parse codec to compression
            compression_ = parse_codec_to_compression(ifd_info.codec);
            codec_name_ = ifd_info.codec;

            // Assume tiled if tile dimensions are provided in IfdInfo (check nvImageCodec metadata)
            // For now, use a heuristic: most whole-slide images are tiled
            tile_width_ = 256;  // Default tile size (can be overridden from IfdInfo metadata)
            tile_height_ = 256;

            // nvImageCodec members
            nvimgcodec_sub_stream_ = ifd_info.sub_code_stream;

            // Calculate hash value
            hash_value_ = tiff->file_handle_shared_.get()->hash_value ^ cucim::codec::splitmix64(index);

            #ifdef DEBUG
            fmt::print("  IFD[{}]: Initialized from nvImageCodec ({}x{}, codec: {})\n",
                      index, width_, height_, codec_name_);
            #endif
            return;
        }
    }

    // Fallback: throw error if nvImageCodec parser not available
    throw std::runtime_error(fmt::format(
        "IFD constructor (offset-based) requires libtiff, which is not available in pure nvImageCodec build. "
        "Use IFD(TIFF*, uint16_t, IfdInfo&) constructor instead."));
#else
    // If nvImageCodec not available, this should never be called
    throw std::runtime_error("Pure nvImageCodec build requires CUCIM_HAS_NVIMGCODEC");
#endif
}
```

**Purpose:**
- **Deprecated constructor** for backward compatibility
- Attempts to redirect to nvImageCodec path if parser available
- Throws error if libtiff mode is required (not supported in pure nvImageCodec build)

**Flow:**
1. Check if nvImageCodec parser exists and is valid
2. If yes, extract `IfdInfo` and initialize from it
3. If no, throw error (libtiff not available)

---

## Lines 86-198: Primary nvImageCodec Constructor

### Lines 91-109: Basic Initialization

```cpp
#ifdef CUCIM_HAS_NVIMGCODEC
IFD::IFD(TIFF* tiff, uint16_t index, const cuslide2::nvimgcodec::IfdInfo& ifd_info)
    : tiff_(tiff), ifd_index_(index), ifd_offset_(index)
{
    PROF_SCOPED_RANGE(PROF_EVENT(ifd_ifd));  // Use standard ifd_ifd profiler event

    #ifdef DEBUG
    fmt::print("🔧 Creating IFD[{}] from nvImageCodec metadata\n", index);
    #endif

    // Extract basic image properties from IfdInfo
    width_ = ifd_info.width;
    height_ = ifd_info.height;
    samples_per_pixel_ = ifd_info.num_channels;
    bits_per_sample_ = ifd_info.bits_per_sample;

    #ifdef DEBUG
    fmt::print("   Dimensions: {}x{}, {} channels, {} bits/sample\n",
              width_, height_, samples_per_pixel_, bits_per_sample_);
    #endif
```

**Purpose:**
- **Primary constructor** for nvImageCodec-based builds
- Initializer list sets basic fields (`tiff_`, `ifd_index_`, `ifd_offset_`)
- Note: `ifd_offset_` set to `index` (not a real file offset in nvImageCodec mode)
- Profiler event for performance tracking
- Extract dimensions and color info from `IfdInfo`

---

### Lines 111-120: Codec Parsing

```cpp
    // Parse codec string to compression enum
    codec_name_ = ifd_info.codec;
    compression_ = parse_codec_to_compression(codec_name_);
    #ifdef DEBUG
    fmt::print("   Codec: {} (compression={})\n", codec_name_, compression_);
    #endif

    // Get ImageDescription from nvImageCodec
    image_description_ = ifd_info.image_description;
```

**Purpose:**
- Store codec name (e.g., "jpeg", "jpeg2k")
- Convert to TIFF compression constant (e.g., 7 for JPEG)
- Extract ImageDescription tag (may contain vendor metadata)

---

### Lines 122-170: TIFF Tag Extraction

```cpp
    // Extract TIFF tags from TiffFileParser
    if (tiff->nvimgcodec_parser_) {
        // Software and Model tags
        software_ = tiff->nvimgcodec_parser_->get_tiff_tag(index, "Software");
        model_ = tiff->nvimgcodec_parser_->get_tiff_tag(index, "Model");

        // SUBFILETYPE for IFD classification
        int subfile_type = tiff->nvimgcodec_parser_->get_subfile_type(index);
        if (subfile_type >= 0) {
            subfile_type_ = static_cast<uint64_t>(subfile_type);
            #ifdef DEBUG
            fmt::print("   SUBFILETYPE: {}\n", subfile_type_);
            #endif
        }

        // Check for JPEGTables (abbreviated JPEG indicator)
        std::string jpeg_tables = tiff->nvimgcodec_parser_->get_tiff_tag(index, "JPEGTables");
        if (!jpeg_tables.empty()) {
            #ifdef DEBUG
            fmt::print("   ✅ JPEGTables detected (abbreviated JPEG)\n");
            #endif
        }

        // Tile dimensions (if available from TIFF tags)
        std::string tile_w_str = tiff->nvimgcodec_parser_->get_tiff_tag(index, "TileWidth");
        std::string tile_h_str = tiff->nvimgcodec_parser_->get_tiff_tag(index, "TileLength");

        if (!tile_w_str.empty() && !tile_h_str.empty()) {
            try {
                tile_width_ = std::stoul(tile_w_str);
                tile_height_ = std::stoul(tile_h_str);
                #ifdef DEBUG
                fmt::print("   Tiles: {}x{}\n", tile_width_, tile_height_);
                #endif
            } catch (...) {
                #ifdef DEBUG
                fmt::print("   ⚠️  Failed to parse tile dimensions\n");
                #endif
                tile_width_ = 0;
                tile_height_ = 0;
            }
        } else {
            // Not tiled - treat as single strip
            tile_width_ = 0;
            tile_height_ = 0;
            #ifdef DEBUG
            fmt::print("   Not tiled (strip-based or whole image)\n");
            #endif
        }
    }
```

**Purpose: Extract metadata using variant-based TIFF tag system**

1. **Software/Model tags**: Vendor identification (e.g., "Aperio Image Library")
2. **SUBFILETYPE**: Classifies IFD (0 = full resolution, 1 = thumbnail/label/macro)
3. **JPEGTables**: Detects abbreviated JPEG (shared tables across tiles)
4. **Tile dimensions**: Get TileWidth/TileLength for tiled images
   - If tags missing or invalid → set to 0 (strip-based image)
   - Uses `get_tiff_tag()` which returns strings (converted from variant storage)

---

### Lines 172-197: Format Defaults and Finalization

```cpp
    // Set format defaults
    planar_config_ = PLANARCONFIG_CONTIG;  // nvImageCodec outputs interleaved
    photometric_ = PHOTOMETRIC_RGB;
    predictor_ = 1;  // No predictor

    // Resolution info (defaults - may not be available from nvImageCodec)
    resolution_unit_ = 1;  // No absolute unit
    x_resolution_ = 1.0f;
    y_resolution_ = 1.0f;

    // Calculate hash for caching
    hash_value_ = cucim::codec::splitmix64(index);

#ifdef CUCIM_HAS_NVIMGCODEC
    // Store reference to nvImageCodec sub-stream
    nvimgcodec_sub_stream_ = ifd_info.sub_code_stream;
    #ifdef DEBUG
    fmt::print("   ✅ nvImageCodec sub-stream: {}\n",
              static_cast<void*>(nvimgcodec_sub_stream_));
    #endif
#endif

    #ifdef DEBUG
    fmt::print("✅ IFD[{}] initialization complete\n", index);
    #endif
}
#endif // CUCIM_HAS_NVIMGCODEC
```

**Purpose:**
- **Set format defaults**:
  - `PLANARCONFIG_CONTIG` (1): Interleaved RGB (not separate planes)
  - `PHOTOMETRIC_RGB` (2): RGB color space
  - `predictor_ = 1`: No prediction (raw values)
  
- **Resolution defaults**: 1.0 (no unit) - may not be available from nvImageCodec

- **Hash calculation**: Used for tile caching (unique per IFD)

- **Store sub-stream pointer**: **CRITICAL** - this is the nvImageCodec code stream for decoding this IFD's data
  - **NOT owned by IFD** - borrowed pointer from `TiffFileParser`
  - Will be set to `nullptr` in destructor (no cleanup here)

---

## Lines 200-209: Destructor

```cpp
IFD::~IFD()
{
#ifdef CUCIM_HAS_NVIMGCODEC
    // NOTE: nvimgcodec_sub_stream_ is NOT owned by IFD - it's a borrowed pointer
    // from TiffFileParser's ifd_infos_ vector. TiffFileParser::~TiffFileParser()
    // destroys all sub-code streams before IFD destructors run.
    // DO NOT call nvimgcodecCodeStreamDestroy here - just clear the pointer.
    nvimgcodec_sub_stream_ = nullptr;
#endif
}
```

**Purpose:**
- **CRITICAL**: `nvimgcodec_sub_stream_` is **NOT owned** by IFD
- It's a **borrowed pointer** from `TiffFileParser::ifd_infos_`
- `TiffFileParser` destructor handles cleanup (destroys all sub-streams)
- Only set to `nullptr` here (no `nvimgcodecCodeStreamDestroy` call)

**Why this matters:**
- Prevents double-free crashes
- Relies on correct destruction order (IFDs destroyed before TiffFileParser)
- Documented in `tiff.h` where `nvimgcodec_parser_` is declared AFTER `ifds_` vector

---

## Lines 211-330: Main Read Method

### Lines 211-225: Entry Point

```cpp
bool IFD::read([[maybe_unused]] const TIFF* tiff,
               [[maybe_unused]] const cucim::io::format::ImageMetadataDesc* metadata,
               [[maybe_unused]] const cucim::io::format::ImageReaderRegionRequestDesc* request,
               [[maybe_unused]] cucim::io::format::ImageDataDesc* out_image_data)
{
    PROF_SCOPED_RANGE(PROF_EVENT(ifd_read));

    #ifdef DEBUG
    fmt::print("🎯 IFD::read() ENTRY: IFD[{}], location=({}, {}), size={}x{}, device={}\n",
              ifd_index_,
              request->location[0], request->location[1],
              request->size[0], request->size[1],
              request->device);
    #endif
```

**Purpose:**
- Main entry point for reading image data from this IFD
- `[[maybe_unused]]` attributes suppress warnings in non-nvImageCodec builds
- Profiler event for performance tracking
- Debug output shows request parameters

**Parameters:**
- `request->location`: (x, y) top-left corner of region
- `request->size`: (width, height) of region
- `request->device`: Output device (CPU or CUDA)

---

### Lines 226-313: nvImageCodec ROI Decoding Path

```cpp
#ifdef CUCIM_HAS_NVIMGCODEC
    // Fast path: Use nvImageCodec ROI decoding when available
    // ROI decoding is supported in nvImageCodec v0.6.0+ for JPEG2000
    // Falls back to tile-based decoding if ROI decode fails
    if (nvimgcodec_sub_stream_ && tiff->nvimgcodec_parser_ &&
        request->location_len == 1 && request->batch_size == 1)
    {
        std::string device_name(request->device);
        if (request->shm_name)
        {
            device_name = device_name + fmt::format("[{}]", request->shm_name);
        }
        cucim::io::Device out_device(device_name);

        int64_t sx = request->location[0];
        int64_t sy = request->location[1];
        int64_t w = request->size[0];
        int64_t h = request->size[1];

        // Output buffer - decode function will allocate (uses pinned memory for CPU)
        uint8_t* output_buffer = nullptr;
        DLTensor* out_buf = request->buf;
        bool is_buf_available = out_buf && out_buf->data;

        if (is_buf_available)
        {
            // User provided pre-allocated buffer
            output_buffer = static_cast<uint8_t*>(out_buf->data);
        }
        // Note: decode_ifd_region_nvimgcodec will allocate buffer if output_buffer is nullptr

        // Get IFD info from TiffFileParser
        const auto& ifd_info = tiff->nvimgcodec_parser_->get_ifd(static_cast<uint32_t>(ifd_index_));

        // Call nvImageCodec ROI decoder
        bool success = cuslide2::nvimgcodec::decode_ifd_region_nvimgcodec(
            ifd_info,
            tiff->nvimgcodec_parser_->get_main_code_stream(),
            sx, sy, w, h,
            &output_buffer,
            out_device);

        if (success)
        {
            #ifdef DEBUG
            fmt::print("✅ nvImageCodec ROI decode successful: {}x{} at ({}, {})\n", w, h, sx, sy);
            #endif

            // Set up output metadata
            out_image_data->container.data = output_buffer;
            out_image_data->container.device = DLDevice{ static_cast<DLDeviceType>(out_device.type()), out_device.index() };
            out_image_data->container.dtype = DLDataType{ kDLUInt, 8, 1 };
            out_image_data->container.ndim = 3;
            out_image_data->container.shape = static_cast<int64_t*>(cucim_malloc(3 * sizeof(int64_t)));
            out_image_data->container.shape[0] = h;
            out_image_data->container.shape[1] = w;
            out_image_data->container.shape[2] = samples_per_pixel_;
            out_image_data->container.strides = nullptr;
            out_image_data->container.byte_offset = 0;

            return true;
        }
        else
        {

            #ifdef DEBUG
            fmt::print("❌ nvImageCodec ROI decode failed for IFD[{}]\n", ifd_index_);
            #endif

            // Free allocated buffer on failure
            // Note: decode function uses cudaMallocHost for CPU (pinned memory)
            if (!is_buf_available && output_buffer)
            {
                if (out_device.type() == cucim::io::DeviceType::kCUDA)
                {
                    cudaFree(output_buffer);
                }
                else
                {
                    cudaFreeHost(output_buffer);  // Pinned memory
                }
            }

            throw std::runtime_error(fmt::format(
                "Failed to decode IFD[{}] with nvImageCodec. ROI: ({},{}) {}x{}",
                ifd_index_, sx, sy, w, h));
        }
    }
#endif
```

**Purpose: Primary decoding path using nvImageCodec ROI decoder**

**Conditions for this path:**
1. `nvimgcodec_sub_stream_` exists (IFD has code stream)
2. `tiff->nvimgcodec_parser_` exists (parser available)
3. `location_len == 1` (single location, not batch)
4. `batch_size == 1` (single region request)

**Flow:**
1. **Parse device**: Create `Device` object from string (e.g., "cuda:0", "cpu")
2. **Extract ROI params**: (sx, sy, w, h) - top-left corner and size
3. **Buffer handling**:
   - Check if user provided buffer (`out_buf->data`)
   - If not, decoder will allocate (uses `cudaMallocHost` for CPU = pinned memory)
4. **Call decoder**: `decode_ifd_region_nvimgcodec()` from `nvimgcodec_decoder.cpp`
   - Passes IFD info, main code stream, ROI params
   - Writes to `output_buffer` pointer
5. **On success**:
   - Fill `out_image_data` DLPack tensor descriptor
   - Shape: `[h, w, samples_per_pixel]` (HWC format)
   - Data type: uint8
   - Return `true`
6. **On failure**:
   - Free allocated buffer (if any)
   - Throw exception with error details

**Memory management:**
- CPU buffers use `cudaMallocHost()` (pinned memory for faster transfers)
- GPU buffers use `cudaMalloc()`
- Caller responsible for freeing buffer (or passing pre-allocated)

---

### Lines 316-330: Fallback Error Path

```cpp
    // If we reach here, nvImageCodec is not available or request doesn't match fast path
    #ifdef DEBUG
    fmt::print("❌ Cannot decode: nvImageCodec not available or unsupported request type\n");
#ifdef CUCIM_HAS_NVIMGCODEC
    fmt::print("   nvimgcodec_sub_stream_={}, location_len={}, batch_size={}\n",
              static_cast<void*>(nvimgcodec_sub_stream_), request->location_len, request->batch_size);
#else
    fmt::print("   location_len={}, batch_size={}\n",
              request->location_len, request->batch_size);
#endif
    #endif
    throw std::runtime_error(fmt::format(
        "IFD[{}]: This library requires nvImageCodec for image decoding. "
        "Multi-location/batch requests not yet supported.", ifd_index_));
}
```

**Purpose:**
- Error handling if nvImageCodec path not taken
- Debug output shows why (sub-stream missing, batch request, etc.)
- Throws exception with clear message

**Reasons for reaching here:**
1. nvImageCodec not compiled in (`!CUCIM_HAS_NVIMGCODEC`)
2. Sub-stream is null (IFD not properly initialized)
3. Batch request (`location_len > 1` or `batch_size > 1`)
4. Parser not available

---

## Lines 332-448: Accessor Methods

```cpp
uint32_t IFD::index() const
{
    return ifd_index_;
}
ifd_offset_t IFD::offset() const
{
    return ifd_offset_;
}

std::string& IFD::software()
{
    return software_;
}
std::string& IFD::model()
{
    return model_;
}
std::string& IFD::image_description()
{
    return image_description_;
}
uint16_t IFD::resolution_unit() const
{
    return resolution_unit_;
}
float IFD::x_resolution() const
{
    return x_resolution_;
}
float IFD::y_resolution() const
{
    return y_resolution_;
}
uint32_t IFD::width() const
{
    return width_;
}
uint32_t IFD::height() const
{
    return height_;
}
uint32_t IFD::tile_width() const
{
    return tile_width_;
}
uint32_t IFD::tile_height() const
{
    return tile_height_;
}
uint32_t IFD::rows_per_strip() const
{
    return rows_per_strip_;
}
uint32_t IFD::bits_per_sample() const
{
    return bits_per_sample_;
}
uint32_t IFD::samples_per_pixel() const
{
    return samples_per_pixel_;
}
uint64_t IFD::subfile_type() const
{
    return subfile_type_;
}
uint16_t IFD::planar_config() const
{
    return planar_config_;
}
uint16_t IFD::photometric() const
{
    return photometric_;
}
uint16_t IFD::compression() const
{
    return compression_;
}
uint16_t IFD::predictor() const
{
    return predictor_;
}

uint16_t IFD::subifd_count() const
{
    return subifd_count_;
}
std::vector<uint64_t>& IFD::subifd_offsets()
{
    return subifd_offsets_;
}
uint32_t IFD::image_piece_count() const
{
    return image_piece_count_;
}
const std::vector<uint64_t>& IFD::image_piece_offsets() const
{
    return image_piece_offsets_;
}
const std::vector<uint64_t>& IFD::image_piece_bytecounts() const
{
    return image_piece_bytecounts_;
}

size_t IFD::pixel_size_nbytes() const
{
    // Calculate pixel size based on bits_per_sample and samples_per_pixel
    // Most whole-slide images are 8-bit RGB (3 bytes per pixel)
    const size_t bytes_per_sample = (bits_per_sample_ + 7) / 8;  // Round up to nearest byte
    const size_t nbytes = bytes_per_sample * samples_per_pixel_;
    return nbytes;
}

size_t IFD::tile_raster_size_nbytes() const
{
    const size_t nbytes = tile_width_ * tile_height_ * pixel_size_nbytes();
    return nbytes;
}
```

**Purpose: Simple getters for all IFD fields**

**Notable calculations:**
- `pixel_size_nbytes()`: Bytes per pixel
  - Example: 8 bits/sample × 3 samples (RGB) = 24 bits = 3 bytes
  - Uses `(bits_per_sample_ + 7) / 8` to round up (handles non-byte-aligned cases)
  
- `tile_raster_size_nbytes()`: Total bytes for one decoded tile
  - Example: 256 × 256 × 3 = 196,608 bytes (for 256×256 RGB tile)

---

## Lines 450-494: Codec Parser Helper

```cpp
// ============================================================================
// Helper: Parse nvImageCodec Codec String to TIFF Compression Enum
// ============================================================================

#ifdef CUCIM_HAS_NVIMGCODEC
uint16_t IFD::parse_codec_to_compression(const std::string& codec)
{
    // Map nvImageCodec codec strings to TIFF compression constants
    if (codec == "jpeg") {
        return COMPRESSION_JPEG;  // 7
    }
    if (codec == "jpeg2000" || codec == "jpeg2k" || codec == "j2k") {
        // Default to YCbCr JPEG2000 (most common in whole-slide imaging)
        return COMPRESSION_APERIO_JP2K_YCBCR;  // 33003
    }
    if (codec == "lzw") {
        return COMPRESSION_LZW;  // 5
    }
    if (codec == "deflate" || codec == "zip") {
        return COMPRESSION_DEFLATE;  // 8
    }
    if (codec == "adobe-deflate") {
        return COMPRESSION_ADOBE_DEFLATE;  // 32946
    }
    if (codec == "none" || codec == "uncompressed" || codec.empty()) {
        return COMPRESSION_NONE;  // 1
    }

    // Handle generic 'tiff' codec from nvImageCodec 0.6.0
    // This is a known limitation where nvImageCodec doesn't expose the actual compression
    // For now, default to JPEG which is most common in whole-slide imaging
    if (codec == "tiff") {
        #ifdef DEBUG
        fmt::print("ℹ️  nvImageCodec returned generic 'tiff' codec, assuming JPEG compression\n");
        #endif
        return COMPRESSION_JPEG;  // 7 - Most common for WSI (Aperio, Philips, etc.)
    }

    // Unknown codec - log warning and default to JPEG (safer than NONE for WSI)
    #ifdef DEBUG
    fmt::print("⚠️  Unknown codec '{}', defaulting to COMPRESSION_JPEG\n", codec);
    #endif
    return COMPRESSION_JPEG;  // 7 - WSI files rarely use uncompressed
}
#endif // CUCIM_HAS_NVIMGCODEC
```

**Purpose: Convert nvImageCodec codec strings to TIFF compression constants**

**Mapping:**
| nvImageCodec String | TIFF Constant | Value | Notes |
|---------------------|---------------|-------|-------|
| `"jpeg"` | `COMPRESSION_JPEG` | 7 | Standard JPEG |
| `"jpeg2000"`, `"jpeg2k"`, `"j2k"` | `COMPRESSION_APERIO_JP2K_YCBCR` | 33003 | Aperio JPEG2000 |
| `"lzw"` | `COMPRESSION_LZW` | 5 | LZW compression |
| `"deflate"`, `"zip"` | `COMPRESSION_DEFLATE` | 8 | ZIP/deflate |
| `"adobe-deflate"` | `COMPRESSION_ADOBE_DEFLATE` | 32946 | Adobe deflate |
| `"none"`, `"uncompressed"` | `COMPRESSION_NONE` | 1 | Uncompressed |
| `"tiff"` (generic) | `COMPRESSION_JPEG` | 7 | **Fallback for nvImageCodec 0.6.0** |
| Unknown | `COMPRESSION_JPEG` | 7 | **Safe default for WSI** |

**Design decisions:**
- **JPEG default**: Whole-slide images rarely uncompressed (too large)
- **Generic "tiff"**: nvImageCodec 0.6.0 limitation - doesn't expose actual compression
- **JPEG2000 → YCbCr**: Most common variant in medical imaging (Aperio)

---

## Lines 496-524: Format Validation Helpers

```cpp
bool IFD::is_compression_supported() const
{
    switch (compression_)
    {
    case COMPRESSION_NONE:
    case COMPRESSION_JPEG:
    case COMPRESSION_ADOBE_DEFLATE:
    case COMPRESSION_DEFLATE:
    case COMPRESSION_APERIO_JP2K_YCBCR: // 33003: Jpeg 2000 with YCbCr format
    case COMPRESSION_APERIO_JP2K_RGB:   // 33005: Jpeg 2000 with RGB
    case COMPRESSION_LZW:
        return true;
    default:
        return false;
    }
}

bool IFD::is_read_optimizable() const
{
    return is_compression_supported() && bits_per_sample_ == 8 && samples_per_pixel_ == 3 &&
           (tile_width_ != 0 && tile_height_ != 0) && planar_config_ == PLANARCONFIG_CONTIG &&
           (photometric_ == PHOTOMETRIC_RGB || photometric_ == PHOTOMETRIC_YCBCR) &&
           !tiff_->is_in_read_config(TIFF::kUseLibTiff);
}

bool IFD::is_format_supported() const
{
    return is_compression_supported();
}
```

**Purpose: Validation checks for decoding compatibility**

### `is_compression_supported()`
Returns `true` if compression method can be decoded by nvImageCodec:
- Uncompressed (1)
- JPEG (7)
- Deflate/ZIP (8, 32946)
- LZW (5)
- Aperio JPEG2000 (33003, 33005)

### `is_read_optimizable()`
Returns `true` if **fast path** can be used - all conditions must be met:
1. ✅ Compression supported
2. ✅ 8-bit samples (`bits_per_sample_ == 8`)
3. ✅ 3-channel RGB (`samples_per_pixel_ == 3`)
4. ✅ Tiled image (`tile_width_ != 0 && tile_height_ != 0`)
5. ✅ Interleaved format (`planar_config_ == PLANARCONFIG_CONTIG`)
6. ✅ RGB or YCbCr color (`photometric_` check)
7. ✅ Not forcing libtiff mode (`!kUseLibTiff` config)

**Typical WSI matches all these** (8-bit RGB, JPEG-compressed, tiled)

### `is_format_supported()`
Currently just wraps `is_compression_supported()` - placeholder for future format checks

---

## Lines 526-912: Tile-Based Reading (Legacy Path)

This is the **legacy tile-by-tile decoding path**, mostly deprecated in favor of nvImageCodec ROI decoding. However, it's still present for:
- Boundary handling
- Fallback if ROI decode fails
- Cached tile access

### Lines 526-558: `read_region_tiles()` Entry

```cpp
bool IFD::read_region_tiles(const TIFF* tiff,
                            const IFD* ifd,
                            const int64_t* location,
                            const int64_t location_index,
                            const int64_t w,
                            const int64_t h,
                            void* raster,
                            const cucim::io::Device& out_device,
                            cucim::loader::ThreadBatchDataLoader* loader)
{
    #ifdef DEBUG
    fmt::print("🔍 read_region_tiles: ENTRY - location_index={}, w={}, h={}, loader={}\n",
              location_index, w, h, static_cast<void*>(loader));
    #endif
    PROF_SCOPED_RANGE(PROF_EVENT(ifd_read_region_tiles));
    // Reference code: https://github.com/libjpeg-turbo/libjpeg-turbo/blob/master/tjexample.c

    int64_t sx = location[location_index * 2];
    int64_t sy = location[location_index * 2 + 1];
    int64_t ex = sx + w - 1;
    int64_t ey = sy + h - 1;
    #ifdef DEBUG
    fmt::print("🔍 read_region_tiles: Region bounds - sx={}, sy={}, ex={}, ey={}\n", sx, sy, ex, ey);
    #endif

    uint32_t width = ifd->width_;
    uint32_t height = ifd->height_;

    // Handle out-of-boundary case
    if (sx < 0 || sy < 0 || sx >= width || sy >= height || ex < 0 || ey < 0 || ex >= width || ey >= height)
    {
        return read_region_tiles_boundary(tiff, ifd, location, location_index, w, h, raster, out_device, loader);
    }
```

**Purpose: Read image region by decoding individual tiles**

**Parameters:**
- `location`: Array of (x, y) coordinates (may have multiple locations for batch)
- `location_index`: Which location in the array to use
- `w, h`: Width/height of region to read
- `raster`: Output buffer (pre-allocated by caller)
- `out_device`: CPU or CUDA
- `loader`: Thread pool for parallel tile decoding (optional)

**Boundary check:**
- Calculate `(sx, sy)` = start, `(ex, ey)` = end
- If any coordinate out of bounds → delegate to `read_region_tiles_boundary()`

---

### Lines 559-605: Setup and Tile Grid Calculation

```cpp
    cucim::cache::ImageCache& image_cache = cucim::CuImage::cache_manager().cache();
    cucim::cache::CacheType cache_type = image_cache.type();

    uint8_t background_value = tiff->background_value_;
    uint16_t compression_method = ifd->compression_;
    int jpeg_color_space = ifd->jpeg_color_space_;
    int predictor = ifd->predictor_;

    // TODO: revert this once we can get RGB data instead of RGBA
    uint32_t samples_per_pixel = 3; // ifd->samples_per_pixel();

    const void* jpegtable_data = ifd->jpegtable_.data();
    uint32_t jpegtable_count = ifd->jpegtable_.size();

    uint32_t tw = ifd->tile_width_;
    uint32_t th = ifd->tile_height_;

    uint32_t offset_sx = static_cast<uint32_t>(sx / tw); // x-axis start offset for the requested region in the ifd tile
                                                         // array as grid
    uint32_t offset_ex = static_cast<uint32_t>(ex / tw); // x-axis end  offset for the requested region in the ifd tile
                                                         // array as grid
    uint32_t offset_sy = static_cast<uint32_t>(sy / th); // y-axis start offset for the requested region in the ifd tile
                                                         // array as grid
    uint32_t offset_ey = static_cast<uint32_t>(ey / th); // y-axis end offset for the requested region in the ifd tile
                                                         // array as grid

    uint32_t pixel_offset_sx = static_cast<uint32_t>(sx % tw);
    uint32_t pixel_offset_ex = static_cast<uint32_t>(ex % tw);
    uint32_t pixel_offset_sy = static_cast<uint32_t>(sy % th);
    uint32_t pixel_offset_ey = static_cast<uint32_t>(ey % th);

    uint32_t stride_y = width / tw + !!(width % tw); // # of tiles in a row(y) in the ifd tile array as grid

    uint32_t start_index_y = offset_sy * stride_y;
    uint32_t end_index_y = offset_ey * stride_y;

    const size_t tile_raster_nbytes = ifd->tile_raster_size_nbytes();

    int tiff_file = tiff->file_handle_shared_.get()->fd;
    uint64_t ifd_hash_value = ifd->hash_value_;
    uint32_t dest_pixel_step_y = w * samples_per_pixel;

    uint32_t nbytes_tw = tw * samples_per_pixel;
    auto dest_start_ptr = static_cast<uint8_t*>(raster);
```

**Purpose: Calculate tile grid coordinates**

**Tile grid mapping:**
```
Image coordinates: (sx, sy) to (ex, ey)
       ↓
Tile coordinates: (offset_sx, offset_sy) to (offset_ex, offset_ey)
       ↓
Pixel offsets within tiles: (pixel_offset_sx, pixel_offset_sy), etc.
```

**Example:**
- Image region: (300, 450) to (600, 700)
- Tile size: 256×256
- Tile grid:
  - `offset_sx = 300/256 = 1` (tile column 1)
  - `offset_sy = 450/256 = 1` (tile row 1)
  - `offset_ex = 600/256 = 2` (tile column 2)
  - `offset_ey = 700/256 = 2` (tile row 2)
- Pixel offsets:
  - `pixel_offset_sx = 300%256 = 44` (44 pixels into tile)
  - etc.

**Stride calculation:**
- `stride_y = width/tw + !!(width%tw)`: Number of tiles per row
- Example: 46000px / 256 = 179 tiles (+ 1 partial = 180 tiles per row)

---

### Lines 606-873: Tile Processing Loop

This section contains the **main tile iteration loop** with complex lambda captures for thread-safe decoding. Due to its length and complexity, here's a high-level breakdown:

```cpp
for (uint32_t index_y = start_index_y; index_y <= end_index_y; index_y += stride_y)
{
    // For each row of tiles...
    for (uint32_t offset_x = offset_sx; offset_x <= offset_ex; ++offset_x, ++index)
    {
        // For each tile in this row...
        
        // 1. Get tile offset and size from IFD
        auto tiledata_offset = ifd->image_piece_offsets_[index];
        auto tiledata_size = ifd->image_piece_bytecounts_[index];
        
        // 2. Create TileDecodeData struct with all parameters
        auto data = std::make_shared<TileDecodeData>();
        // ... populate data fields ...
        
        // 3. Create decode lambda
        auto decode_func = [data]() {
            // Check cache for decoded tile
            auto value = image_cache.find(key);
            if (value) {
                // Cache hit - use cached data
            } else {
                // Cache miss - decode tile
                // ERROR: Legacy CPU decoder path removed
                throw std::runtime_error("Tile-based CPU decoder fallback reached");
            }
            
            // Copy tile data to output buffer
            memcpy(dest_start_ptr + dest_pixel_index, 
                   tile_data + nbytes_tile_index,
                   nbytes_tile_pixel_size_x);
        };
        
        // 4. Execute decode (single-threaded for now)
        if (force_single_threaded || !loader) {
            decode_func();  // Execute immediately
        } else {
            loader->enqueue(decode_func, TileInfo{...});  // Queue for thread pool
        }
    }
}
```

**Key points:**

1. **TileDecodeData struct** (lines 644-667):
   - Packages all parameters to avoid large lambda captures
   - Shared pointer enables safe copying across threads

2. **Decode lambda** (lines 700-873):
   - Captures only `data` shared_ptr (small, cheap to copy)
   - Checks image cache first (tile may already be decoded)
   - **Legacy decoder code REMOVED** (lines 818-828):
     - Used to decode with libjpeg-turbo/CPU codecs
     - Now throws error if cache miss (should use ROI decode instead)
   - Copies tile data to output buffer using `memcpy`

3. **Single-threaded execution** (lines 880-893):
   - `force_single_threaded = true` (line 880) - **hardcoded for testing**
   - Executes decode immediately instead of enqueueing
   - Simplifies debugging (no threading issues)

**Current state:**
- This tile-based path is **mostly deprecated**
- Should use nvImageCodec ROI decode (lines 226-313) instead
- Only reaches here if ROI decode not available
- Will throw error on cache miss (CPU decoder removed)

---

## Lines 914-1364: Boundary Tile Reading

Similar structure to `read_region_tiles()` but with additional boundary handling:

```cpp
bool IFD::read_region_tiles_boundary(const TIFF* tiff,
                                     const IFD* ifd,
                                     const int64_t* location,
                                     const int64_t location_index,
                                     const int64_t w,
                                     const int64_t h,
                                     void* raster,
                                     const cucim::io::Device& out_device,
                                     cucim::loader::ThreadBatchDataLoader* loader)
```

**Additional complexity:**
- Handles regions that extend beyond image boundaries
- Fills out-of-bounds areas with background color (typically white)
- Clips tile data to valid regions
- More complex pixel offset calculations

**Key differences from `read_region_tiles()`:**

1. **Boundary checking** (lines 947-953):
   ```cpp
   bool is_out_of_image = (ex < 0 || width <= sx || ey < 0 || height <= sy);
   if (is_out_of_image)
   {
       // Fill background color(255,255,255) and return
       memset(dest_start_ptr, background_value, w * h * pixel_size_nbytes);
       return true;
   }
   ```

2. **Range clipping** (lines 968-1017):
   - Calculates valid tile range
   - Handles negative coordinates (Python-style modulo)

3. **Partial tile copying** (lines 1118-1198):
   - `copy_partial` flag indicates boundary tiles
   - Copies valid portion, fills rest with background
   - Handles both X and Y boundary conditions

**When used:**
- User requests region extending beyond image
- Example: Request (46000, 32000) to (46512, 32512) on 46000×32914 image
  - Partially out of bounds on right edge
  - Need to fill out-of-bounds pixels with white

---

## Lines 1366-1379: Namespace Closing and Benchmarking Stubs

```cpp
} // namespace cuslide::tiff


// Hidden methods for benchmarking.

#include <fmt/format.h>
#include <langinfo.h>
#include <iostream>
#include <fstream>

namespace cuslide::tiff
{
} // namespace cuslide::tiff
```

**Purpose:**
- Closes main namespace
- Empty namespace for potential benchmarking code (currently unused)
- Includes are for future use

---

# Summary

## Key Design Patterns

### 1. **Dual Constructor Pattern**
- Legacy constructor (deprecated) for API compatibility
- Primary nvImageCodec constructor for new code
- Legacy falls back to nvImageCodec if available

### 2. **Two-Level Decoding**
- **Fast path**: nvImageCodec ROI decode (lines 226-313)
  - Decodes entire region in one call
  - Supports JPEG, JPEG2000, deflate, etc.
  - Used for most operations
  
- **Legacy path**: Tile-by-tile decode (lines 526-912)
  - Deprecated, mostly removed
  - Throws error on cache miss
  - Only for boundary cases

### 3. **Borrowed Pointer Pattern**
- `nvimgcodec_sub_stream_` is **NOT owned** by IFD
- Borrowed from `TiffFileParser::ifd_infos_`
- Set to `nullptr` in destructor (no cleanup)
- Prevents double-free crashes

### 4. **Variant-Based Metadata**
- Uses `TiffFileParser::get_tiff_tag()` for metadata extraction
- Returns strings (converted from typed variants)
- Type-safe storage, simple API

### 5. **Caching Strategy**
- Tile cache using `ImageCache`
- Hash-based lookup (`ifd_hash_value ^ tile_index`)
- Cache miss → throw error (was: decode with CPU)

## Critical Sections

1. **Constructor lines 122-170**: TIFF tag extraction using variant system
2. **Read method lines 226-313**: nvImageCodec ROI decoding (primary path)
3. **Destructor lines 200-209**: Borrowed pointer cleanup (prevents double-free)
4. **Codec parser lines 455-494**: String-to-enum conversion with fallbacks

## Common Pitfalls

1. **Don't call `nvimgcodecCodeStreamDestroy()` on `nvimgcodec_sub_stream_`**
   - It's a borrowed pointer, not owned by IFD
   - TiffFileParser handles cleanup

2. **Tile-based path is deprecated**
   - Don't try to extend it
   - Use nvImageCodec ROI decode instead
   - Legacy CPU decoder code has been removed

3. **"tiff" generic codec**
   - nvImageCodec 0.6.0 limitation
   - Defaults to JPEG (safest for WSI)
   - May need refinement for non-WSI TIFFs

4. **Boundary handling is complex**
   - Use `read_region_tiles_boundary()` for out-of-bounds
   - Fills background color (white) for invalid areas
   - Handles partial tiles at image edges

---

## File Organization

```
ifd.h (175 lines)
├── Class declaration
├── Public interface (constructors, read methods, accessors)
└── Private members (metadata fields, nvImageCodec handles)

ifd.cpp (1379 lines)
├── Legacy constructor (35-84): Deprecated, redirects to nvImageCodec
├── Primary constructor (91-198): nvImageCodec-based, extracts TIFF tags
├── Destructor (200-209): Clears borrowed pointer
├── Read method (211-330): nvImageCodec ROI decode (fast path)
├── Accessors (332-448): Simple getters
├── Codec parser (455-494): String → enum conversion
├── Validation helpers (496-524): Format/compression checks
├── Tile reading (526-912): Legacy tile-by-tile (deprecated)
└── Boundary reading (914-1364): Out-of-bounds handling
```

This documentation should help you understand every aspect of the IFD implementation!

