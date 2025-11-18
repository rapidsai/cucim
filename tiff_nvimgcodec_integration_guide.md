# TIFF/IFD Classes and nvImageCodec Integration Guide

## ⚠️ Important Notice: Architecture Has Changed

**This document now describes TWO architectures:**

1. **OLD (DEPRECATED)**: Uses libtiff for metadata + nvImageCodec for decoding
2. **NEW (RECOMMENDED)**: Uses nvImageCodec EXCLUSIVELY (no libtiff)

See the detailed refactoring plan in: `REFACTORING_PLAN_ELIMINATE_LIBTIFF.md`

## Overview

This document explains how the `TIFF` and `IFD` classes integrate with nvImageCodec for GPU-accelerated image decoding.

### Quick Comparison

| Aspect | OLD Architecture | NEW Architecture |
|--------|-----------------|------------------|
| **Metadata Source** | libtiff (TIFFGetField) | nvImageCodec TiffFileParser |
| **IFD Enumeration** | libtiff (TIFFReadDirectory) | nvImageCodec (get_ifd_count) |
| **Tile Offsets** | libtiff (td_stripoffset_p) | Not needed (nvImageCodec handles internally) |
| **JPEG Tables** | libtiff (TIFFTAG_JPEGTABLES) | nvImageCodec (automatic handling) |
| **Decoding** | Multi-path (nvImageCodec/nvJpeg/libjpeg) | Single path (nvImageCodec only) |
| **Dependencies** | libtiff + nvImageCodec | nvImageCodec only |
| **Code Complexity** | High (dual paths) | Low (single path) |
| **Performance** | Good | Better (no redundant parsing) |
| **Maintenance** | Complex | Simple |

**Recommendation**: Migrate to NEW architecture for simpler, faster, GPU-first design.

---

## Architecture Overview

### OLD Architecture (Before Refactoring - DEPRECATED)

```
User Application
    ↓
TIFF Class (tiff.h/tiff.cpp)
    ├── libtiff → Metadata extraction (PRIMARY)
    └── TiffFileParser (nvImageCodec) → GPU decoding only (SECONDARY)
        ↓
IFD Class (ifd.h/ifd.cpp)
    ├── Tile-based decoding (CPU/GPU)
    └── ROI-based decoding (nvImageCodec GPU acceleration)
        ↓
Decoders:
    ├── nvImageCodec (GPU): JPEG2000, JPEG (with JPEGTables)
    ├── nvJpeg (GPU): Standard JPEG tiles
    ├── libjpeg-turbo (CPU): JPEG fallback
    └── OpenJPEG (CPU): JPEG2000 fallback
```

### NEW Architecture (After Refactoring - RECOMMENDED)

```
User Application
    ↓
TIFF Class (tiff.h/tiff.cpp)
    └── TiffFileParser (nvImageCodec) → EXCLUSIVE parser for metadata & decoding
        ↓
IFD Class (ifd.h/ifd.cpp)
    └── ROI-based decoding ONLY (nvImageCodec GPU acceleration)
        ↓
Decoder:
    └── nvImageCodec (GPU): All formats (JPEG2000, JPEG, JPEG+Tables, etc.)
        └── Automatic fallback to CPU if GPU unavailable

No libtiff dependency ✅
No tile-based fallback ✅
Pure nvImageCodec API ✅
```

**Key Changes:**
- ❌ **Removed**: libtiff completely (no TIFFGetField calls)
- ❌ **Removed**: Tile-based decoding with file offsets
- ❌ **Removed**: CPU decoder fallbacks (libjpeg-turbo, OpenJPEG)
- ✅ **Added**: nvImageCodec as MANDATORY dependency
- ✅ **Simplified**: Single code path for all operations
- ✅ **Improved**: Faster initialization, lower memory usage

---

## Key Components

### 1. TIFF Class (`tiff.h` / `tiff.cpp`)

**Purpose**: High-level TIFF file management, format detection, and metadata extraction.

#### Key Member Variables

```cpp
class TIFF {
private:
    ::TIFF* tiff_client_;  // libtiff handle for metadata
    std::unique_ptr<cuslide2::nvimgcodec::TiffFileParser> nvimgcodec_parser_;  // nvImageCodec parser
    std::vector<std::shared_ptr<IFD>> ifds_;  // IFD objects for each resolution level
    std::vector<size_t> level_to_ifd_idx_;  // Mapping: resolution level → IFD index
    std::map<std::string, AssociatedImageBufferDesc> associated_images_;  // Label, macro, thumbnail
    TiffType tiff_type_;  // Generic, Philips, Aperio
};
```

#### Constructor Integration (Lines 253-290)

```cpp
TIFF::TIFF(const cucim::filesystem::Path& file_path, int mode) : file_path_(file_path)
{
    // 1. Open file with libtiff for metadata
    int fd = ::open(file_path_cstr, mode, 0666);
    tiff_client_ = ::TIFFFdOpen(fd, file_path_cstr, "rm");
    
    // 2. Initialize nvImageCodec TIFF parser for GPU-accelerated decoding
    try {
        nvimgcodec_parser_ = std::make_unique<cuslide2::nvimgcodec::TiffFileParser>(file_path.c_str());
        fmt::print("✅ nvImageCodec TiffFileParser initialized for: {}\n", file_path);
    } catch (const std::exception& e) {
        fmt::print("⚠️  nvImageCodec TiffFileParser init failed: {}\n", e.what());
        fmt::print("   Falling back to libtiff-only mode\n");
        nvimgcodec_parser_ = nullptr;  // Graceful degradation
    }
}
```

**Key Design Decision**: Dual initialization
- **libtiff**: Always used for metadata extraction (TIFF tags, dimensions, compression)
- **nvImageCodec**: Optionally initialized for GPU decoding (gracefully fails if unavailable)

#### Format Detection: `resolve_vendor_format()` (Lines 376-438)

Detects TIFF vendor format (Aperio SVS, Philips TIFF, Generic) and populates metadata:

```cpp
void TIFF::resolve_vendor_format()
{
    auto& first_ifd = ifds_[0];
    
    // Detect Aperio SVS format by ImageDescription prefix
    if (first_ifd->image_description().starts_with("Aperio ")) {
        _populate_aperio_svs_metadata(...);
        tiff_type_ = TiffType::Aperio;
    }
    
    // Detect Philips TIFF by Software tag
    if (first_ifd->software().starts_with("Philips")) {
        _populate_philips_tiff_metadata(...);
        tiff_type_ = TiffType::Philips;
    }
}
```

**Vendor-Specific Handling**:

1. **Aperio SVS** (Lines 635-692)
   - Classifies IFDs as resolution levels or associated images using `subfile_type`
   - Parses ImageDescription metadata (pipe-separated key-value pairs)
   - Extracts thumbnail/label/macro based on `subfile_type` tag:
     - `0` at index 1 → thumbnail
     - `1` → label
     - `9` → macro

2. **Philips TIFF** (Lines 440-633)
   - Parses XML metadata from ImageDescription
   - Extracts pixel spacing for each resolution level
   - Corrects IFD dimensions (Philips reports tile-aligned sizes, not actual image size)
   - Extracts macro/label from XML or IFD data

#### IFD Construction: `construct_ifds()` (Lines 327-375)

```cpp
void TIFF::construct_ifds()
{
    // Step 1: Enumerate all IFDs using libtiff
    uint16_t ifd_index = 0;
    do {
        uint64_t offset = TIFFCurrentDirOffset(tiff_client_);
        ifd_offsets_.push_back(offset);
        
        // Create IFD object (passes this TIFF* to IFD constructor)
        auto ifd = std::make_shared<cuslide::tiff::IFD>(this, ifd_index, offset);
        ifds_.emplace_back(std::move(ifd));
        ++ifd_index;
    } while (TIFFReadDirectory(tiff_client_));
    
    // Step 2: Classify IFDs as resolution levels or associated images
    resolve_vendor_format();
    
    // Step 3: Sort resolution levels by size (largest first)
    std::sort(level_to_ifd_idx_.begin(), level_to_ifd_idx_.end(), ...);
}
```

**Critical Flow**:
1. libtiff enumerates IFDs → Extract basic metadata
2. IFD constructor initializes each IFD (including nvImageCodec sub-stream)
3. Vendor format detection classifies IFDs
4. Resolution levels sorted by dimension

---

### 2. IFD Class (`ifd.h` / `ifd.cpp`)

**Purpose**: Represents a single IFD (resolution level) and handles tile/ROI decoding.

#### Key Member Variables

```cpp
class IFD {
private:
    TIFF* tiff_;  // Parent TIFF object (not owned)
    uint32_t ifd_index_;  // IFD index in TIFF file
    
    // Image properties (from libtiff)
    uint32_t width_, height_;
    uint32_t tile_width_, tile_height_;
    uint16_t compression_;
    uint32_t samples_per_pixel_;
    std::vector<uint64_t> image_piece_offsets_;  // Tile/strip offsets
    std::vector<uint64_t> image_piece_bytecounts_;  // Tile/strip sizes
    std::vector<uint8_t> jpegtable_;  // JPEG tables (if abbreviated JPEG)
    
#ifdef CUCIM_HAS_NVIMGCODEC
    // nvImageCodec integration
    nvimgcodecCodeStream_t nvimgcodec_sub_stream_;  // Sub-stream for this IFD
    std::string codec_name_;  // Codec name from nvImageCodec
#endif
};
```

#### IFD Constructor with nvImageCodec Integration (Lines 42-142)

```cpp
IFD::IFD(TIFF* tiff, uint16_t index, ifd_offset_t offset) 
    : tiff_(tiff), ifd_index_(index), ifd_offset_(offset)
{
    auto tif = tiff->client();  // Get libtiff handle
    
    // Step 1: Extract TIFF metadata with libtiff
    TIFFGetField(tif, TIFFTAG_SOFTWARE, &software_char_ptr);
    TIFFGetField(tif, TIFFTAG_IMAGEDESCRIPTION, &model_char_ptr);
    // ... extract width, height, compression, tile dimensions, etc.
    
    // Step 2: Extract JPEG tables if abbreviated JPEG
    if (compression_ == COMPRESSION_JPEG) {
        uint8_t* jpegtable_data = nullptr;
        uint32_t jpegtable_count = 0;
        TIFFGetField(tif, TIFFTAG_JPEGTABLES, &jpegtable_count, &jpegtable_data);
        jpegtable_.insert(jpegtable_.end(), jpegtable_data, jpegtable_data + jpegtable_count);
    }
    
    // Step 3: Copy tile/strip offsets and byte counts
    image_piece_offsets_.insert(..., td_stripoffset_p, ...);
    image_piece_bytecounts_.insert(..., td_stripbytecount_p, ...);
    
#ifdef CUCIM_HAS_NVIMGCODEC
    // Step 4: Initialize nvImageCodec streams if TiffFileParser is available
    if (tiff->nvimgcodec_parser_ && tiff->nvimgcodec_parser_->is_valid()) {
        try {
            if (static_cast<uint32_t>(index) < tiff->nvimgcodec_parser_->get_ifd_count()) {
                const auto& ifd_info = tiff->nvimgcodec_parser_->get_ifd(static_cast<uint32_t>(index));
                nvimgcodec_sub_stream_ = ifd_info.sub_code_stream;  // Reference (not owned)
                codec_name_ = ifd_info.codec;
                fmt::print("  IFD[{}]: nvImageCodec stream initialized (codec: {})\n", index, codec_name_);
            }
        } catch (const std::exception& e) {
            fmt::print("⚠️  Failed to initialize nvImageCodec for IFD[{}]: {}\n", index, e.what());
        }
    }
#endif
}
```

**Critical Design**: The IFD stores a **reference** to the nvImageCodec sub-code stream (owned by `TiffFileParser`), not a copy.

#### IFD Destructor (Lines 144-155)

```cpp
IFD::~IFD()
{
#ifdef CUCIM_HAS_NVIMGCODEC
    // Clean up nvImageCodec sub-stream if we own it
    if (nvimgcodec_sub_stream_) {
        nvimgcodecCodeStreamDestroy(nvimgcodec_sub_stream_);
        nvimgcodec_sub_stream_ = nullptr;
    }
    // Note: nvimgcodec_main_stream_ is not owned by us, don't destroy it
#endif
}
```

**Memory Management Note**: Currently the destructor destroys the sub-stream, but the comment suggests this might be managed by the parent. This is a potential area for review.

---

## Image Reading Flow

### Entry Point: `IFD::read()` (Lines 157-561)

This is the main entry point for reading image regions. It implements a **fast path** using nvImageCodec ROI decoding and falls back to tile-based decoding.

```cpp
bool IFD::read(const TIFF* tiff,
               const cucim::io::format::ImageMetadataDesc* metadata,
               const cucim::io::format::ImageReaderRegionRequestDesc* request,
               cucim::io::format::ImageDataDesc* out_image_data)
{
    fmt::print("🎯 IFD::read() ENTRY: IFD[{}], location=({}, {}), size={}x{}, device={}\n",
              ifd_index_, request->location[0], request->location[1],
              request->size[0], request->size[1], request->device);
```

### Fast Path: nvImageCodec ROI Decoding (Lines 170-240)

**Conditions for Fast Path**:
1. nvImageCodec sub-stream is available (`nvimgcodec_sub_stream_` != nullptr)
2. TIFF has nvImageCodec parser (`tiff->nvimgcodec_parser_`)
3. Single location request (`request->location_len == 1`)
4. Single batch (`request->batch_size == 1`)

```cpp
#ifdef CUCIM_HAS_NVIMGCODEC
    if (nvimgcodec_sub_stream_ && tiff->nvimgcodec_parser_ && 
        request->location_len == 1 && request->batch_size == 1)
    {
        // Extract ROI parameters
        int64_t sx = request->location[0];
        int64_t sy = request->location[1];
        int64_t w = request->size[0];
        int64_t h = request->size[1];
        
        // Allocate or use pre-allocated buffer
        uint8_t* output_buffer = nullptr;
        if (request->buf && request->buf->data) {
            output_buffer = static_cast<uint8_t*>(request->buf->data);
        } else {
            size_t buffer_size = w * h * samples_per_pixel_;
            output_buffer = static_cast<uint8_t*>(cucim_malloc(buffer_size));
        }
        
        // Get IFD info from TiffFileParser
        const auto& ifd_info = tiff->nvimgcodec_parser_->get_ifd(static_cast<uint32_t>(ifd_index_));
        
        // Call nvImageCodec ROI decoder
        bool success = cuslide2::nvimgcodec::decode_ifd_region_nvimgcodec(
            ifd_info, 
            tiff->nvimgcodec_parser_->get_main_code_stream(),
            sx, sy, w, h,
            &output_buffer,
            out_device);
        
        if (success) {
            fmt::print("✅ nvImageCodec ROI decode successful: {}x{} at ({}, {})\n", w, h, sx, sy);
            // Set up output metadata and return
            return true;
        } else {
            fmt::print("⚠️  nvImageCodec ROI decode failed, falling back to tile-based decode\n");
            // Fall through to tile-based approach below
        }
    }
#endif
```

**Benefits of Fast Path**:
- **No tile iteration**: Decodes ROI directly without loading entire tiles
- **GPU acceleration**: nvImageCodec uses GPU for JPEG2000/JPEG decoding
- **Memory efficiency**: Only allocates memory for requested region

**When Fast Path is Used**:
- JPEG2000 (Aperio SVS with J2K compression)
- Abbreviated JPEG (Aperio SVS with TIFFTAG_JPEGTABLES)
- Standard JPEG/PNG/TIFF formats supported by nvImageCodec

### Fallback: Tile-Based Decoding (Lines 277-441)

If fast path conditions aren't met or ROI decoding fails, fall back to tile-based approach:

```cpp
if (is_read_optimizable()) {
    // Optimized tile-based decoding
    if (location_len > 1 || batch_size > 1 || num_workers > 0) {
        // Multi-threaded batch loading with thread pool
        auto load_func = [tiff, ifd, location, w, h, out_device](...) {
            read_region_tiles(tiff, ifd, location, location_index, w, h, 
                            raster_ptr, out_device, loader_ptr);
        };
        
        // For GPU + JPEG: Use nvJpeg processor
        if (out_device.type() == kCUDA && !is_jpeg2000) {
            batch_processor = std::make_unique<NvJpegProcessor>(...);
        }
        
        auto loader = std::make_unique<ThreadBatchDataLoader>(...);
        loader->request(load_size);
    } else {
        // Single-threaded tile reading
        read_region_tiles(tiff, ifd, location, 0, w, h, raster, out_device, nullptr);
    }
}
```

**Optimization Conditions** (`is_read_optimizable()` - Lines 693-703):
```cpp
bool IFD::is_read_optimizable() const
{
    return is_compression_supported() &&
           (tile_width_ != 0 && tile_height_ != 0) &&
           planar_config_ == PLANARCONFIG_CONTIG &&
           (photometric_ == PHOTOMETRIC_RGB || photometric_ == PHOTOMETRIC_YCBCR) &&
           !tiff_->is_in_read_config(TIFF::kUseLibTiff);
}
```

**Conditions**:
- Supported compression (JPEG, JPEG2000, LZW, DEFLATE, NONE)
- Tiled image (not stripped)
- Contiguous planar configuration
- RGB or YCbCr photometric interpretation
- Not forced to use libtiff

---

## Tile-Based Decoding: `read_region_tiles()` (Lines 710-983)

This method reads a region by decoding overlapping tiles.

### Algorithm Overview

```
1. Calculate tile grid coordinates for ROI
2. For each tile overlapping the ROI:
   a. Check cache for decoded tile
   b. If not cached, decode tile:
      - JPEG → libjpeg-turbo or nvJpeg
      - JPEG2000 → nvImageCodec or OpenJPEG
      - LZW/DEFLATE → custom decoders
   c. Copy relevant portion to output buffer
3. Handle empty tiles (fill with background color)
```

### Key Code Sections

#### Tile Grid Calculation (Lines 753-769)

```cpp
uint32_t tw = ifd->tile_width_;
uint32_t th = ifd->tile_height_;

// Calculate tile grid offsets
uint32_t offset_sx = static_cast<uint32_t>(sx / tw);  // Start tile X
uint32_t offset_ex = static_cast<uint32_t>(ex / tw);  // End tile X
uint32_t offset_sy = static_cast<uint32_t>(sy / th);  // Start tile Y
uint32_t offset_ey = static_cast<uint32_t>(ey / th);  // End tile Y

// Pixel offsets within tiles
uint32_t pixel_offset_sx = static_cast<uint32_t>(sx % tw);
uint32_t pixel_offset_ex = static_cast<uint32_t>(ex % tw);
// ...

uint32_t stride_y = width / tw + !!(width % tw);  // Tiles per row
```

#### Tile Decoding with Cache (Lines 805-965)

```cpp
auto decode_func = [=, &image_cache]() {
    // Check cache
    auto key = image_cache.create_key(ifd_hash_value, index);
    image_cache.lock(index_hash);
    auto value = image_cache.find(key);
    
    if (value) {
        // Cache hit: use cached tile
        tile_data = static_cast<uint8_t*>(value->data);
    } else {
        // Cache miss: decode tile
        if (cache_type != kNoCache) {
            tile_data = static_cast<uint8_t*>(image_cache.allocate(tile_raster_nbytes));
        } else {
            tile_data = cucim_malloc(tile_raster_nbytes);
        }
        
        // Decode based on compression
        switch (compression_method) {
            case COMPRESSION_JPEG:
                cuslide::jpeg::decode_libjpeg(...);
                break;
            case cuslide::jpeg2k::kAperioJpeg2kYCbCr:  // 33003
                // Try nvImageCodec first
                if (!cuslide2::nvimgcodec::decode_jpeg2k_nvimgcodec(...)) {
                    // Fallback to OpenJPEG
                    cuslide::jpeg2k::decode_libopenjpeg(...);
                }
                break;
            case COMPRESSION_DEFLATE:
                cuslide::deflate::decode_deflate(...);
                break;
            // ... other codecs
        }
        
        // Insert into cache
        value = image_cache.create_value(tile_data, tile_raster_nbytes);
        image_cache.insert(key, value);
    }
    
    // Copy relevant portion of tile to output buffer
    for (uint32_t ty = tile_pixel_offset_sy; ty <= tile_pixel_offset_ey; ++ty) {
        memcpy(dest_start_ptr + dest_pixel_index, 
               tile_data + nbytes_tile_index,
               nbytes_tile_pixel_size_x);
    }
};

// Execute immediately or enqueue for thread pool
if (loader && *loader) {
    loader->enqueue(std::move(decode_func), TileInfo{...});
} else {
    decode_func();
}
```

### JPEG2000 Tile Decoding with nvImageCodec (Lines 883-916)

**Priority**: nvImageCodec (GPU) → OpenJPEG (CPU fallback)

```cpp
case cuslide::jpeg2k::kAperioJpeg2kYCbCr:  // 33003
    fmt::print("🔍 Decoding JPEG2000 tile (YCbCr) at offset {}, size {}\n", 
              tiledata_offset, tiledata_size);
    
    cucim::io::Device cpu_device("cpu");  // Decode to CPU (loader expects CPU)
    
    // Try nvImageCodec first (GPU-accelerated)
    if (!cuslide2::nvimgcodec::decode_jpeg2k_nvimgcodec(
            tiff_file, nullptr, tiledata_offset, tiledata_size,
            &tile_data, tile_raster_nbytes, cpu_device, 0)) {
        
        fmt::print("⚠️  nvImageCodec failed, falling back to OpenJPEG\n");
        
        // Fallback to CPU OpenJPEG decoder
        cuslide::jpeg2k::decode_libopenjpeg(
            tiff_file, nullptr, tiledata_offset, tiledata_size, 
            &tile_data, tile_raster_nbytes, out_device, 
            cuslide::jpeg2k::ColorSpace::kSYCC);
    }
    
    fmt::print("✅ JPEG2000 tile decoded successfully\n");
    break;
```

**Why Two-Stage Decode**:
1. **nvImageCodec (GPU)**: Fast but may not support all JPEG2000 variants
2. **OpenJPEG (CPU)**: Slower but more compatible

**Device Handling**: Tiles decoded to CPU when using thread pool loader (nvJpeg handles GPU transfer separately for batched JPEG).

---

## nvImageCodec Decoder Integration

The actual nvImageCodec decoding functions are in separate files (not shown in the attached code):
- `cuslide/nvimgcodec/nvimgcodec_decoder.h`
- `cuslide/nvimgcodec/nvimgcodec_decoder.cpp`

### ROI Decoding Function (Referenced in IFD::read)

```cpp
bool cuslide2::nvimgcodec::decode_ifd_region_nvimgcodec(
    const cuslide2::nvimgcodec::IfdInfo& ifd_info,
    nvimgcodecCodeStream_t main_stream,
    int64_t x, int64_t y, int64_t width, int64_t height,
    uint8_t** output_buffer,
    const cucim::io::Device& device);
```

This function uses the `TiffFileParser::decode_region()` method documented in the TIFF parser documentation.

### Tile Decoding Function (Referenced in read_region_tiles)

```cpp
bool cuslide2::nvimgcodec::decode_jpeg2k_nvimgcodec(
    int fd,
    const uint8_t* buffer,
    uint64_t offset,
    uint64_t size,
    uint8_t** output_buffer,
    size_t output_size,
    const cucim::io::Device& device,
    int color_space);
```

This function decodes individual JPEG2000 tiles from file offset/size.

---

## Threading and Batch Processing

### NvJpegProcessor (Lines 391-400)

For GPU JPEG decoding with batching:

```cpp
if (out_device.type() == cucim::io::DeviceType::kCUDA && !is_jpeg2000) {
    raster_type = cucim::io::DeviceType::kCUDA;
    
    // Calculate maximum tile count for memory allocation
    uint32_t tile_across_count = /* ... */;
    uint32_t tile_down_count = /* ... */;
    maximum_tile_count = tile_across_count * tile_down_count * batch_size;
    
    // Create NvJpegProcessor
    auto& jpegtable = ifd->jpegtable_;
    auto nvjpeg_processor = std::make_unique<cuslide::loader::NvJpegProcessor>(
        tiff->file_handle_, ifd, request_location->data(), request_size->data(), 
        location_len, batch_size, maximum_tile_count,
        jpegtable_data, jpegtable_size);
    
    batch_processor = std::move(nvjpeg_processor);
}
```

**NvJpegProcessor Features**:
- GPU batch JPEG decoding using nvJpeg library
- Handles abbreviated JPEG (JPEGTables) automatically
- Pre-allocates GPU memory for tile batch
- Not used for JPEG2000 (nvImageCodec handles that instead)

### ThreadBatchDataLoader (Lines 408-427)

```cpp
auto loader = std::make_unique<cucim::loader::ThreadBatchDataLoader>(
    load_func,              // Tile loading function
    std::move(batch_processor),  // Optional: nvJpeg or nullptr
    out_device,             // Target device (CPU/GPU)
    std::move(request_location),
    std::move(request_size),
    location_len,           // Number of locations
    one_raster_size,        // Size per location
    batch_size,             // Batch size
    prefetch_factor,        // Prefetch count
    num_workers);           // Thread pool size

loader->request(load_size);  // Start loading
raster = loader->next_data();  // Get decoded data
```

**ThreadBatchDataLoader Features**:
- Thread pool for parallel tile decoding
- Prefetching for data loading pipeline
- Works with NvJpegProcessor for GPU JPEG or CPU decoders for other formats
- Batch support for training workloads

---

## Compression Support Matrix

| Codec | TIFF Constant | nvImageCodec ROI | nvImageCodec Tile | CPU Fallback | Notes |
|-------|---------------|------------------|-------------------|--------------|-------|
| JPEG | `COMPRESSION_JPEG` (7) | ✅ Yes | ✅ Yes (nvJpeg) | libjpeg-turbo | Standard JPEG |
| JPEG with Tables | `COMPRESSION_JPEG` + JPEGTables | ✅ Yes | ✅ Yes (nvJpeg) | libjpeg-turbo | Abbreviated JPEG (Aperio SVS) |
| JPEG2000 YCbCr | `33003` | ✅ Yes | ✅ Yes | OpenJPEG | Aperio SVS common |
| JPEG2000 RGB | `33005` | ✅ Yes | ✅ Yes | OpenJPEG | Aperio SVS less common |
| LZW | `COMPRESSION_LZW` (5) | ❌ No | ❌ No | Custom LZW decoder | Philips TIFF |
| DEFLATE | `COMPRESSION_DEFLATE` (8) | ❌ No | ❌ No | Custom DEFLATE decoder | Generic TIFF |
| Uncompressed | `COMPRESSION_NONE` (1) | ❌ No | ❌ No | memcpy | Generic TIFF |

**nvImageCodec ROI Support**: Only JPEG and JPEG2000 codecs support ROI decoding in nvImageCodec.

---

## Data Flow Examples

### Example 1: Reading Aperio SVS with JPEG2000

```
User Request: Read 512x512 ROI at (1000, 2000) from level 0
    ↓
TIFF::read()
    ├── Get IFD for level 0
    └── Call IFD::read()
        ↓
IFD::read() - Fast Path Check
    ├── ✅ nvimgcodec_sub_stream_ exists
    ├── ✅ Single location, single batch
    ├── Call decode_ifd_region_nvimgcodec()
    │   ↓
    │   TiffFileParser::decode_region()
    │       ├── Create ROI view (IFD=0, region=(1000,2000,512,512))
    │       ├── nvimgcodecCodeStreamGetSubCodeStream() → roi_stream
    │       ├── nvimgcodecImageCreate() → output image
    │       ├── nvimgcodecDecoderDecode(roi_stream, output_image)
    │       │   ↓
    │       │   nvTIFF reads TIFF structure
    │       │   nvJPEG2000 decodes JPEG2000 ROI on GPU
    │       │   ↓
    │       └── Return decoded RGB buffer
    └── ✅ Return success

Result: 512x512x3 RGB buffer on GPU, decoded in single operation
```

### Example 2: Reading Philips TIFF with LZW (No nvImageCodec Support)

```
User Request: Read 1024x1024 ROI at (500, 500) from level 0
    ↓
TIFF::read()
    └── Call IFD::read()
        ↓
IFD::read() - Fast Path Check
    ├── ❌ LZW not supported by nvImageCodec
    └── Fall through to tile-based approach
        ↓
        is_read_optimizable() → true (LZW supported by custom decoder)
        ↓
        read_region_tiles()
            ├── Calculate tiles: (0,0) to (3,3) [4x4 tile grid for 1024x1024]
            └── For each tile:
                ├── Check cache → miss
                ├── Allocate tile buffer
                ├── decode_lzw() → decompress tile
                ├── horAcc8() → apply horizontal predictor
                ├── Insert into cache
                └── memcpy() → copy relevant portion to output

Result: 1024x1024x3 RGB buffer on CPU, decoded tile-by-tile
```

### Example 3: Batch Reading for Training (JPEG, nvJpeg)

```
User Request: Read 100 patches (256x256) with batch_size=32, num_workers=4
    ↓
IFD::read()
    ├── is_read_optimizable() → true
    ├── Detect: batch_size > 1, num_workers > 0
    └── Multi-threaded batch loading path
        ↓
        Create NvJpegProcessor (GPU JPEG decoder)
        ↓
        Create ThreadBatchDataLoader
            ├── 4 worker threads
            ├── Prefetch factor = 2
            └── Batch processor = NvJpegProcessor
        ↓
        loader->request(64)  // Request first 2 batches (32 * 2)
        ↓
        Background thread pool:
            ├── Worker 1: read_region_tiles() → Tiles 0-24
            ├── Worker 2: read_region_tiles() → Tiles 25-49
            ├── Worker 3: read_region_tiles() → Tiles 50-74
            └── Worker 4: read_region_tiles() → Tiles 75-99
                ↓
                For each tile:
                    ├── Enqueue JPEG decode task to NvJpegProcessor
                    └── NvJpegProcessor:
                        ├── Batch decode on GPU (nvJpeg)
                        ├── Handle JPEGTables if present
                        └── Return GPU buffer
        ↓
        User calls loader->next_data() → Get next batch

Result: 32x256x256x3 batch on GPU, pipelined with prefetch
```

---

## Performance Optimization Strategies

### 1. **Fast Path Selection (nvImageCodec ROI)**

**When Used**:
- Single ROI request (not batch)
- JPEG2000 or JPEG with JPEGTables
- nvImageCodec available

**Benefits**:
- 🚀 **Up to 10x faster** than tile-based for JPEG2000
- 🧠 **Lower memory**: Only allocates for ROI, not full tiles
- 🎮 **GPU accelerated**: nvTIFF + nvJPEG2000

**Example**: Reading 512x512 from 100K×100K Aperio SVS
- Tile-based: Load 4 tiles (1024x1024 each) = 4MB → decode → crop
- ROI-based: Decode 512x512 directly = 768KB

### 2. **Tile Caching**

**Strategy**: Cache decoded tiles to avoid redundant decompression

```cpp
auto key = image_cache.create_key(ifd_hash_value, tile_index);
auto value = image_cache.find(key);
if (value) {
    // Cache hit: reuse decoded tile
} else {
    // Cache miss: decode and insert
}
```

**Benefits**:
- Adjacent ROI requests share tiles
- Training loops benefit from spatial locality
- Configurable cache size (LRU eviction)

### 3. **Batch Processing with nvJpeg**

**When Used**:
- JPEG compression (not JPEG2000)
- GPU output device
- Multiple patches (batch_size > 1)

**Benefits**:
- **Batch GPU decode**: nvJpeg processes multiple tiles in parallel
- **Asynchronous pipeline**: Prefetch next batch while processing current
- **Zero-copy**: Decode directly to GPU memory

### 4. **Thread Pool for CPU Workloads**

**When Used**:
- Multiple patches (location_len > 1)
- CPU-intensive codecs (LZW, DEFLATE, OpenJPEG)

**Benefits**:
- **Parallel tile decoding**: Utilize all CPU cores
- **Overlapped I/O**: Read file while decoding
- **Prefetch pipeline**: Load next batch during processing

---

## Configuration Flags

### Read Configuration Options

```cpp
class TIFF {
    static constexpr uint64_t kUseDirectJpegTurbo = 1;      // Use libjpeg-turbo directly
    static constexpr uint64_t kUseLibTiff = 1 << 1;        // Force libtiff (slow path)
};

// Usage:
tiff->add_read_config(TIFF::kUseLibTiff);  // Force slow path
if (tiff->is_in_read_config(TIFF::kUseDirectJpegTurbo)) { /* ... */ }
```

**Impact**:
- `kUseLibTiff`: Disables `is_read_optimizable()` → Forces slow path (RGBA output)
- `kUseDirectJpegTurbo`: Prefer direct libjpeg-turbo over other JPEG decoders

---

## Error Handling and Fallback Strategy

### Multi-Level Fallback Chain

```
1. nvImageCodec ROI decode (GPU, fast)
   ↓ (fails)
2. Tile-based with nvJpeg batch (GPU, medium)
   ↓ (fails or not applicable)
3. Tile-based with nvImageCodec tiles (GPU, medium)
   ↓ (fails)
4. Tile-based with CPU decoders (CPU, slow)
   ↓ (fails)
5. libtiff slow path (CPU, very slow, RGBA output)
```

**Example Failure Scenarios**:
- nvImageCodec not installed → Skip to step 3
- Unsupported JPEG2000 variant → nvImageCodec fails → OpenJPEG fallback
- Corrupted tile → Fill with background color (255,255,255)
- Invalid ROI bounds → Throw exception

### Graceful Degradation

```cpp
try {
    nvimgcodec_parser_ = std::make_unique<TiffFileParser>(file_path);
} catch (const std::exception& e) {
    fmt::print("⚠️  nvImageCodec init failed: {}\n", e.what());
    nvimgcodec_parser_ = nullptr;  // Continue without GPU acceleration
}
```

**Philosophy**: Always provide a working path, even if slower.

---

## Memory Management

### Buffer Ownership Patterns

1. **User-Provided Buffer** (Pre-allocated)
```cpp
if (request->buf && request->buf->data) {
    raster = request->buf->data;  // Use existing buffer
}
```

2. **Auto-Allocated Buffer** (Owned by this function)
```cpp
if (!raster) {
    raster = cucim_malloc(raster_size);  // Allocate new buffer
    // ... decode into raster
    // Ownership transferred to out_image_data
}
```

3. **Cached Tiles** (Owned by cache)
```cpp
tile_data = static_cast<uint8_t*>(image_cache.allocate(tile_raster_nbytes));
// Cache manages lifetime
```

4. **Temporary Tiles** (RAII)
```cpp
std::unique_ptr<uint8_t, decltype(cucim_free)*> tile_raster(
    reinterpret_cast<uint8_t*>(cucim_malloc(tile_raster_nbytes)), 
    cucim_free);
// Automatically freed when out of scope
```

### nvImageCodec Stream Ownership

```
TiffFileParser (owns main_code_stream)
    ↓
IfdInfo (owns sub_code_stream for each IFD)
    ↓
IFD (references sub_code_stream, does NOT own)
```

**Critical**: IFD must not outlive TIFF, which owns the TiffFileParser.

---

## Integration Points Summary

### TIFF Class → nvImageCodec

```cpp
// Constructor
nvimgcodec_parser_ = std::make_unique<TiffFileParser>(file_path);

// Access in IFD
if (tiff->nvimgcodec_parser_ && tiff->nvimgcodec_parser_->is_valid()) {
    // Use parser for fast path
}
```

### IFD Class → nvImageCodec

```cpp
// Constructor: Store reference to sub-stream
nvimgcodec_sub_stream_ = ifd_info.sub_code_stream;

// read(): Fast path ROI decoding
if (nvimgcodec_sub_stream_) {
    decode_ifd_region_nvimgcodec(ifd_info, main_stream, ...);
}

// read_region_tiles(): Tile-level JPEG2000 decoding
decode_jpeg2k_nvimgcodec(fd, nullptr, offset, size, ...);
```

### Decoder Functions (External)

```cpp
// ROI decoding (uses TiffFileParser::decode_region internally)
bool decode_ifd_region_nvimgcodec(const IfdInfo&, nvimgcodecCodeStream_t, ...);

// Tile decoding (creates temporary code stream from file offset)
bool decode_jpeg2k_nvimgcodec(int fd, const uint8_t*, uint64_t offset, ...);
```

---

## Best Practices and Recommendations

### For Library Users

1. **Enable nvImageCodec for Performance**
   - Ensure nvImageCodec is installed and available at runtime
   - Check `tiff->nvimgcodec_parser_` is non-null after opening file

2. **Use Single ROI Requests for GPU Acceleration**
   ```cpp
   // Good: Single ROI → nvImageCodec fast path
   read_region(x, y, width, height, level, device="cuda");
   
   // Slower: Multiple ROIs → tile-based fallback
   read_regions(locations, sizes, level, device="cuda");
   ```

3. **Enable Caching for Repeated Access**
   ```cpp
   cache_manager.configure(cache_type=kMemory, cache_size_mb=1024);
   ```

4. **Use Batch Processing for Training**
   ```cpp
   loader = create_loader(
       locations, sizes, 
       batch_size=32,       // GPU batch decode
       num_workers=4,       // Parallel tile loading
       prefetch_factor=2    // Pipeline prefetch
   );
   ```

### For Library Developers

1. **Maintain Fallback Paths**
   - Always check nvImageCodec availability
   - Provide CPU fallbacks for all codecs
   - Test graceful degradation

2. **Memory Management**
   - Use RAII for temporary buffers
   - Clear ownership semantics (who frees what)
   - Avoid leaks in error paths

3. **Performance Monitoring**
   - Add profiling markers (PROF_SCOPED_RANGE)
   - Log fast path vs. slow path decisions
   - Monitor cache hit rates

4. **Error Handling**
   - Validate ROI bounds before decoding
   - Handle corrupted tiles gracefully
   - Provide informative error messages

---

## Troubleshooting Guide

### Problem: nvImageCodec not being used (slow performance)

**Symptoms**: No "nvImageCodec" log messages, slow JPEG2000 decoding

**Checks**:
1. Is nvImageCodec installed? Check `nvimgcodec_parser_` initialization
2. Is request multi-location or multi-batch? (Fast path requires single location/batch)
3. Is codec supported? (Only JPEG, JPEG2000 have fast path)

**Solution**: Use single-location requests, ensure nvImageCodec is available

### Problem: RGBA output instead of RGB

**Symptoms**: 4 channels instead of 3, slow performance

**Cause**: Slow path (libtiff) is being used

**Checks**:
1. Is `TIFF::kUseLibTiff` flag set? Remove it.
2. Does `is_read_optimizable()` return false? Check compression/format support.

**Solution**: Use supported formats (tiled, RGB/YCbCr, supported codecs)

### Problem: Out of memory errors

**Symptoms**: cudaMalloc or malloc failures

**Causes**:
1. Large ROI without pre-allocated buffer
2. Cache size too large
3. Batch size too large for GPU memory

**Solutions**:
1. Pre-allocate output buffer: `request->buf->data = pre_allocated_buffer`
2. Reduce cache size in configuration
3. Reduce batch size or use CPU device

### Problem: Segmentation fault

**Common Causes**:
1. IFD outlives TIFF (dangling `nvimgcodec_sub_stream_` reference)
2. Double-free of nvImageCodec streams
3. Null pointer access when nvImageCodec unavailable

**Prevention**:
1. Ensure TIFF object lifetime exceeds all IFD objects
2. Review destructor logic for nvImageCodec resources
3. Always null-check `nvimgcodec_parser_` before use

---

## Future Enhancements

### Potential Improvements

1. **ROI Decoding for All Codecs**
   - Extend ROI support to LZW, DEFLATE
   - Implement tile-intersection optimization

2. **Better Multi-Location Fast Path**
   - Batch multiple ROIs in single nvImageCodec call
   - Reduce overhead for small patches

3. **Adaptive Fast Path Selection**
   - Profile ROI size vs. tile overhead
   - Automatically choose best path based on request

4. **Unified Decoder Interface**
   - Abstract codec-specific logic
   - Pluggable decoder architecture

5. **Better Error Recovery**
   - Partial decode on tile corruption
   - Retry logic for transient GPU errors

---

## Conclusion

The TIFF/IFD integration with nvImageCodec provides:

✅ **Dual-mode operation**: GPU-accelerated fast path + CPU fallback  
✅ **Flexible decoding**: ROI, tile-based, batch processing  
✅ **Multi-codec support**: JPEG, JPEG2000, LZW, DEFLATE, etc.  
✅ **Performance optimization**: Caching, threading, GPU batch decode  
✅ **Graceful degradation**: Works even without nvImageCodec  

**Key Takeaway**: The architecture prioritizes **performance** (GPU fast path) while ensuring **reliability** (CPU fallbacks), making it suitable for both interactive viewing and high-throughput training workloads.

