# nvImageCodec TIFF Parser - Line-by-Line Documentation

## Overview

The TIFF Parser provides a high-level interface for parsing and decoding TIFF files using nvImageCodec's file-level API. It consists of two files:

- **nvimgcodec_tiff_parser.h**: Header file with class declarations and interfaces
- **nvimgcodec_tiff_parser.cpp**: Implementation file with parsing and decoding logic

### Key Responsibilities

1. **TIFF Structure Parsing**: Extract IFD (Image File Directory) information including dimensions, codecs, and metadata
2. **Metadata Extraction**: Retrieve vendor-specific metadata (Aperio, Philips, etc.) and TIFF tags
3. **IFD Classification**: Distinguish between resolution levels and associated images (thumbnail, label, macro)
4. **ROI Decoding**: Decode specific regions of interest without loading entire images
5. **Format Detection**: Automatically detect file format (Aperio SVS, Philips TIFF, Generic TIFF, etc.)

---

## Header File: nvimgcodec_tiff_parser.h

### Lines 1-16: Copyright and License

Standard Apache 2.0 license header for NVIDIA CORPORATION.

### Lines 17-21: Include Guards and Dependencies

```cpp
#pragma once

#ifdef CUCIM_HAS_NVIMGCODEC
#include <nvimgcodec.h>
#endif
```

- **Line 17**: `#pragma once` ensures the header is included only once per compilation unit
- **Lines 19-21**: Conditionally include nvImageCodec headers only if the library is available at build time

### Lines 23-29: Standard Library Includes

```cpp
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <mutex>
#include <stdexcept>
#include <cucim/io/device.h>
```

Essential C++ standard library components:
- `string`, `vector`, `map`: Container types for metadata and IFD lists
- `memory`: Smart pointer support
- `mutex`: Thread-safety for decoder operations
- `stdexcept`: Exception handling
- `cucim/io/device.h`: CuCIM device abstraction (CPU/GPU)

### Lines 31-33: Namespace Declaration

```cpp
namespace cuslide2::nvimgcodec
{
```

All classes are in the `cuslide2::nvimgcodec` namespace to avoid naming conflicts.

### Lines 36-48: ImageType Enumeration

```cpp
enum class ImageType {
    RESOLUTION_LEVEL,  // Full or reduced resolution image
    THUMBNAIL,         // Thumbnail image
    LABEL,             // Slide label image
    MACRO,             // Macro/overview image
    UNKNOWN            // Unclassified
};
```

**Purpose**: Classify IFDs based on their content type.

- **RESOLUTION_LEVEL**: Main pyramid levels (full resolution and downsampled versions)
- **THUMBNAIL**: Small preview image
- **LABEL**: Slide label (text/barcode)
- **MACRO**: Overview/macro photograph
- **UNKNOWN**: Cannot be classified

**Usage**: Essential for formats like Aperio SVS that mix resolution levels with associated images in a single TIFF file.

### Lines 50-98: IfdInfo Structure

This structure holds all information about a single IFD (resolution level).

#### Lines 57-63: Basic Image Properties

```cpp
uint32_t index;                          // IFD index (0, 1, 2, ...)
uint32_t width;                          // Image width in pixels
uint32_t height;                         // Image height in pixels
uint32_t num_channels;                   // Number of channels (typically 3 for RGB)
uint32_t bits_per_sample;                // Bits per channel (8, 16, etc.)
std::string codec;                       // Compression codec (jpeg, jpeg2k, deflate, etc.)
nvimgcodecCodeStream_t sub_code_stream;  // nvImageCodec code stream for this IFD
```

- **index**: 0-based IFD index (0 = highest resolution)
- **width, height**: Image dimensions in pixels
- **num_channels**: Usually 3 for RGB, 4 for RGBA
- **bits_per_sample**: Bit depth (8 for standard RGB, 16 for high-bit-depth)
- **codec**: Compression type detected by nvImageCodec (e.g., "jpeg", "jpeg2k")
- **sub_code_stream**: nvImageCodec handle for this specific IFD (used for decoding)

#### Lines 65-66: ImageDescription Metadata

```cpp
std::string image_description;           // ImageDescription TIFF tag (270)
```

The ImageDescription TIFF tag (tag 270) contains vendor-specific metadata:
- **Aperio SVS**: Contains keywords like "label", "macro", pyramid dimensions
- **Philips TIFF**: Contains XML metadata
- **Generic TIFF**: May be empty or contain simple description

#### Lines 68-76: Format-Specific Metadata

```cpp
struct MetadataBlob {
    int format;  // nvimgcodecMetadataFormat_t
    std::vector<uint8_t> data;
};
std::map<int, MetadataBlob> metadata_blobs;
```

**Purpose**: Store vendor-specific metadata extracted by nvImageCodec.

- **key**: `nvimgcodecMetadataKind_t` enumeration value
  - `0`: TIFF_TAG (individual TIFF tags)
  - `1`: MED_APERIO (Aperio SVS metadata)
  - `2`: MED_PHILIPS (Philips TIFF metadata)
  - `3`: MED_LEICA (Leica SCN metadata)
  - etc.
- **value**: MetadataBlob containing format type and raw binary data

#### Lines 78-80: TIFF Tag Storage

```cpp
std::map<std::string, std::string> tiff_tags;
```

**nvImageCodec 0.7.0+ feature**: Individual TIFF tag retrieval by name.

Examples:
- `"SUBFILETYPE"` → `"0"` (main image) or `"1"` (reduced resolution)
- `"Compression"` → `"7"` (JPEG) or `"33005"` (JPEG2000)
- `"ImageDescription"` → Full text content
- `"JPEGTables"` → `"<detected by libtiff>"` (abbreviated JPEG marker)

#### Lines 82-97: Constructor, Destructor, and Move Semantics

```cpp
IfdInfo() : index(0), width(0), height(0), num_channels(0), 
            bits_per_sample(0), sub_code_stream(nullptr) {}

~IfdInfo()
{
    // NOTE: sub_code_stream is managed by TiffFileParser and should NOT be destroyed here
}

// Disable copy, enable move
IfdInfo(const IfdInfo&) = delete;
IfdInfo& operator=(const IfdInfo&) = delete;
IfdInfo(IfdInfo&&) = default;
IfdInfo& operator=(IfdInfo&&) = default;
```

**Critical Design Decision**: The destructor does NOT destroy `sub_code_stream`.

**Reason**: Sub-code streams are hierarchical and owned by the parent `main_code_stream_`. They are automatically destroyed when the main stream is destroyed. Attempting to manually destroy them can cause double-free errors.

**Move-only semantics**: IfdInfo cannot be copied (to prevent accidental duplication of nvImageCodec handles), but can be moved efficiently.

### Lines 100-119: TiffFileParser Class Overview

```cpp
/**
 * @brief TIFF file parser using nvImageCodec file-level API
 * 
 * This class provides TIFF parsing capabilities using nvImageCodec's native
 * TIFF support. It can query TIFF structure (IFD count, dimensions, codecs)
 * and decode entire resolution levels.
 */
class TiffFileParser
{
```

**Design Philosophy**: This parser uses nvImageCodec's high-level API, which is simpler than libtiff but provides less granular control (no individual tile access).

**Trade-offs**:
- ✅ Simpler code
- ✅ Automatic format detection
- ✅ GPU-accelerated decoding
- ❌ No tile-level access (only full IFD or ROI decoding)
- ❌ Less metadata control

### Lines 123-140: Constructor and Basic Lifecycle

```cpp
explicit TiffFileParser(const std::string& file_path);
~TiffFileParser();

// Disable copy, enable move
TiffFileParser(const TiffFileParser&) = delete;
TiffFileParser& operator=(const TiffFileParser&) = delete;
TiffFileParser(TiffFileParser&&) = default;
TiffFileParser& operator=(TiffFileParser&&) = default;
```

- **Constructor**: Opens TIFF file and parses all IFD metadata
- **Destructor**: Cleans up nvImageCodec resources (code streams)
- **Move-only**: Prevents accidental copying of nvImageCodec handles

### Lines 142-161: Basic Query Methods

```cpp
bool is_valid() const { return initialized_; }
const std::string& get_file_path() const { return file_path_; }
uint32_t get_ifd_count() const { return static_cast<uint32_t>(ifd_infos_.size()); }
const IfdInfo& get_ifd(uint32_t index) const;
```

Simple accessors for file status and IFD information.

### Lines 172-199: IFD Classification Methods

```cpp
ImageType classify_ifd(uint32_t ifd_index) const;
std::vector<uint32_t> get_resolution_levels() const;
std::map<std::string, uint32_t> get_associated_images() const;
```

**classify_ifd()**: Determines if an IFD is a resolution level or associated image.

**Algorithm**:
1. Parse ImageDescription for keywords ("label", "macro", "thumbnail")
2. Check dimension heuristics (small images likely associated)
3. Apply format-specific rules

**get_resolution_levels()**: Returns indices of all pyramid levels.

**get_associated_images()**: Returns map like `{"thumbnail": 1, "label": 2, "macro": 3}`.

### Lines 201-223: Metadata Accessors

```cpp
void override_ifd_dimensions(uint32_t ifd_index, uint32_t width, uint32_t height);
std::string get_image_description(uint32_t ifd_index) const;
const std::map<int, IfdInfo::MetadataBlob>& get_metadata_blobs(uint32_t ifd_index) const;
const IfdInfo::MetadataBlob* get_metadata_blob(uint32_t ifd_index, int kind) const;
```

- **override_ifd_dimensions()**: Useful for Philips TIFF where reported dimensions include padding
- **get_image_description()**: Returns TIFF tag 270 content
- **get_metadata_blobs()**: Returns all vendor-specific metadata
- **get_metadata_blob()**: Returns specific metadata by kind (e.g., kind=1 for Aperio)

### Lines 260-314: TIFF Tag Methods (nvImageCodec 0.7.0+)

```cpp
std::string get_tiff_tag(uint32_t ifd_index, const std::string& tag_name) const;
int get_subfile_type(uint32_t ifd_index) const;
std::vector<int> query_metadata_kinds(uint32_t ifd_index = 0) const;
std::string get_detected_format() const;
void print_info() const;
```

**New in nvImageCodec 0.7.0**: Individual TIFF tag retrieval without manual parsing.

**get_tiff_tag()**: Retrieve any TIFF tag by name (e.g., "Compression", "DateTime").

**get_subfile_type()**: Returns SUBFILETYPE tag value:
- `0` = main image
- `1` = reduced resolution (thumbnail/label/macro)

**query_metadata_kinds()**: Returns list of available metadata types in the file.

**get_detected_format()**: Automatically detects file format:
- "Aperio SVS"
- "Philips TIFF"
- "Leica SCN"
- "Generic TIFF (jpeg)" / "Generic TIFF (jpeg2k)"

### Lines 316-366: ROI Decoding Methods

```cpp
uint8_t* decode_region(
    uint32_t ifd_index,
    uint32_t x, uint32_t y,
    uint32_t width, uint32_t height,
    uint8_t* output_buffer = nullptr,
    const cucim::io::Device& device = cucim::io::Device("cpu")
);

uint8_t* decode_ifd(
    uint32_t ifd_index,
    uint8_t* output_buffer = nullptr,
    const cucim::io::Device& device = cucim::io::Device("cpu")
);

bool has_roi_decode_support() const;
```

**decode_region()**: Core ROI decoding method.

**Parameters**:
- `ifd_index`: Which resolution level to decode
- `x, y`: Top-left corner of ROI in pixels
- `width, height`: ROI dimensions
- `output_buffer`: Pre-allocated buffer (or nullptr for auto-allocation)
- `device`: CPU or GPU decoding

**Returns**: Pointer to decoded RGB data (interleaved format: RGBRGBRGB...)

**decode_ifd()**: Convenience wrapper that decodes the entire IFD.

**has_roi_decode_support()**: Checks if nvImageCodec is available.

### Lines 368-407: Private Methods and Member Variables

```cpp
void parse_tiff_structure();
void extract_ifd_metadata(IfdInfo& ifd_info);
void extract_tiff_tags(IfdInfo& ifd_info);

std::string file_path_;
bool initialized_;
nvimgcodecCodeStream_t main_code_stream_;
std::vector<IfdInfo> ifd_infos_;
```

**parse_tiff_structure()**: Called by constructor to enumerate IFDs.

**extract_ifd_metadata()**: Uses `nvimgcodecDecoderGetMetadata()` to get vendor metadata.

**extract_tiff_tags()**: Uses libtiff directly (nvTIFF 0.6.0.77 compatibility).

**Member variables**:
- `main_code_stream_`: Root code stream representing entire TIFF file
- `ifd_infos_`: Vector of all parsed IFD information

### Lines 409-479: NvImageCodecTiffParserManager Class

```cpp
class NvImageCodecTiffParserManager
{
public:
    static NvImageCodecTiffParserManager& instance();
    nvimgcodecInstance_t get_instance() const { return instance_; }
    nvimgcodecDecoder_t get_decoder() const { return decoder_; }
    std::mutex& get_mutex() { return decoder_mutex_; }
    bool is_available() const { return initialized_; }
    const std::string& get_status() const { return status_message_; }
```

**Singleton Design Pattern**: Manages global nvImageCodec instance for TIFF parsing.

**Why Separate from Main Decoder?**
- The main decoder (for tile decoding) may have different settings
- Parser only needs CPU-only metadata extraction
- Prevents conflicts between parsing and decoding operations

**Thread Safety**: Provides mutex for protecting decoder operations.

### Lines 481-536: Stub Implementations (No nvImageCodec)

When nvImageCodec is not available at build time, provide stub implementations that throw runtime errors. This allows code to compile but fail gracefully at runtime if nvImageCodec features are attempted.

---

## Implementation File: nvimgcodec_tiff_parser.cpp

### Lines 1-35: Headers and Includes

```cpp
#include "nvimgcodec_tiff_parser.h"
#include "nvimgcodec_manager.h"

#include <tiffio.h>
#include <cstring>

#ifdef CUCIM_HAS_NVIMGCODEC
#include <nvimgcodec.h>
#include <cuda_runtime.h>
#endif

#include <fmt/format.h>
#include <stdexcept>
#include <algorithm>
#include <mutex>
```

**Key Dependencies**:
- `tiffio.h`: libtiff library for direct TIFF tag access (needed for JPEGTables detection)
- `nvimgcodec.h`: nvImageCodec C API
- `cuda_runtime.h`: CUDA memory management for GPU decoding
- `fmt/format.h`: Modern C++ string formatting

### Lines 43-47: IfdInfo::print() Implementation

```cpp
void IfdInfo::print() const
{
    fmt::print("  IFD[{}]: {}x{}, {} channels, {} bits/sample, codec: {}\n",
               index, width, height, num_channels, bits_per_sample, codec);
}
```

Simple diagnostic output for an IFD. Called during TIFF parsing to show structure.

---

## NvImageCodecTiffParserManager Implementation

### Lines 53-119: Constructor

This is the initialization sequence for the TIFF parser's nvImageCodec instance.

#### Lines 56-69: Create Instance Configuration

```cpp
nvimgcodecInstanceCreateInfo_t create_info{};
create_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
create_info.struct_size = sizeof(nvimgcodecInstanceCreateInfo_t);
create_info.struct_next = nullptr;
create_info.load_builtin_modules = 1;       // Load JPEG, PNG, etc.
create_info.load_extension_modules = 1;     // Load JPEG2K, TIFF, etc.
create_info.extension_modules_path = nullptr;
create_info.debug_messenger = 0;            // Disable debug for TIFF parser
create_info.debug_messenger_desc = nullptr;
create_info.message_severity = 0;
create_info.message_category = 0;
```

**Purpose**: Configure nvImageCodec instance for metadata extraction.

**Key Settings**:
- `load_builtin_modules = 1`: Enable JPEG, PNG decoders
- `load_extension_modules = 1`: Enable JPEG2000, TIFF extensions
- `create_debug_messenger = 0`: Disable verbose logging (parser is background operation)

#### Lines 71-79: Create Instance

```cpp
nvimgcodecStatus_t status = nvimgcodecInstanceCreate(&instance_, &create_info);

if (status != NVIMGCODEC_STATUS_SUCCESS)
{
    status_message_ = fmt::format("Failed to create nvImageCodec instance for TIFF parsing (status: {})", 
                                 static_cast<int>(status));
    fmt::print("⚠️  {}\n", status_message_);
    return;
}
```

**Error Handling**: If instance creation fails, log error but don't throw exception. Manager remains in "unavailable" state.

#### Lines 81-107: Create Decoder for Metadata Extraction

```cpp
nvimgcodecExecutionParams_t exec_params{};
exec_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS;
exec_params.struct_size = sizeof(nvimgcodecExecutionParams_t);
exec_params.struct_next = nullptr;
exec_params.device_allocator = nullptr;
exec_params.pinned_allocator = nullptr;
exec_params.max_num_cpu_threads = 0;
exec_params.executor = nullptr;
exec_params.device_id = NVIMGCODEC_DEVICE_CPU_ONLY;  // CPU-only for metadata
exec_params.pre_init = 0;
exec_params.skip_pre_sync = 0;
exec_params.num_backends = 0;
exec_params.backends = nullptr;

status = nvimgcodecDecoderCreate(instance_, &decoder_, &exec_params, nullptr);
```

**Critical Setting**: `device_id = NVIMGCODEC_DEVICE_CPU_ONLY`

**Why CPU-only?** The decoder is ONLY used for `nvimgcodecDecoderGetMetadata()` calls during parsing. It doesn't decode actual images. CPU-only saves GPU resources.

**Decoder Usage**:
- ✅ Used for: `nvimgcodecDecoderGetMetadata()` (extracts Aperio/Philips metadata)
- ❌ NOT used for: Actual image decoding (separate decoder handles that)

### Lines 121-134: Destructor

```cpp
NvImageCodecTiffParserManager::~NvImageCodecTiffParserManager()
{
    if (decoder_)
    {
        nvimgcodecDecoderDestroy(decoder_);
        decoder_ = nullptr;
    }
    
    if (instance_)
    {
        nvimgcodecInstanceDestroy(instance_);
        instance_ = nullptr;
    }
}
```

**Resource Cleanup Order**:
1. Destroy decoder first (depends on instance)
2. Then destroy instance

**Thread Safety**: Destructor is called during program exit. Singleton ensures only one instance exists.

---

## TiffFileParser Implementation

### Lines 140-186: Constructor

The constructor performs complete TIFF parsing in a single call.

#### Lines 140-150: Initialization and Manager Check

```cpp
TiffFileParser::TiffFileParser(const std::string& file_path)
    : file_path_(file_path), initialized_(false), 
      main_code_stream_(nullptr)
{
    auto& manager = NvImageCodecTiffParserManager::instance();
    
    if (!manager.is_available())
    {
        throw std::runtime_error(fmt::format("nvImageCodec not available: {}", 
                                            manager.get_status()));
    }
```

**Early Validation**: Check if nvImageCodec is available before attempting to parse. If not available (e.g., library not installed), throw immediately.

#### Lines 154-165: Create Code Stream

```cpp
nvimgcodecStatus_t status = nvimgcodecCodeStreamCreateFromFile(
    manager.get_instance(),
    &main_code_stream_,
    file_path.c_str()
);

if (status != NVIMGCODEC_STATUS_SUCCESS)
{
    throw std::runtime_error(fmt::format("Failed to create code stream from file: {} (status: {})",
                                        file_path, static_cast<int>(status)));
}

fmt::print("✅ Opened TIFF file: {}\n", file_path);
```

**Code Stream**: nvImageCodec abstraction representing a file or memory buffer containing encoded image data.

**What happens here?**
- nvImageCodec opens the TIFF file
- Validates it's a valid TIFF format
- Creates handle for accessing the file

**No decoding yet**: This only opens the file structure, doesn't decode any pixels.

#### Lines 169-185: Parse Structure and Error Handling

```cpp
parse_tiff_structure();

initialized_ = true;
fmt::print("✅ TIFF parser initialized with {} IFDs\n", ifd_infos_.size());
```

**parse_tiff_structure()**: Does the heavy lifting of enumerating IFDs and extracting metadata.

**Exception Safety**: If parsing fails, the constructor cleans up `main_code_stream_` before re-throwing the exception (RAII pattern).

### Lines 188-208: Destructor

```cpp
TiffFileParser::~TiffFileParser()
{
    // Destroy sub-code streams first
    for (auto& ifd_info : ifd_infos_)
    {
        if (ifd_info.sub_code_stream)
        {
            nvimgcodecCodeStreamDestroy(ifd_info.sub_code_stream);
            ifd_info.sub_code_stream = nullptr;
        }
    }
    
    // Then destroy main code stream
    if (main_code_stream_)
    {
        nvimgcodecCodeStreamDestroy(main_code_stream_);
        main_code_stream_ = nullptr;
    }
    
    ifd_infos_.clear();
}
```

**Cleanup Order is CRITICAL**:
1. First destroy all sub-code streams (IFD-specific streams)
2. Then destroy main code stream (parent stream)

**Why this order?** Sub-streams may have internal references to the main stream. Destroying main first could cause segfaults.

### Lines 210-325: parse_tiff_structure() - Core Parsing Logic

This is the heart of the TIFF parser. It enumerates all IFDs and extracts their metadata.

#### Lines 212-225: Get TIFF Structure Info

```cpp
nvimgcodecCodeStreamInfo_t stream_info{};
stream_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO;
stream_info.struct_size = sizeof(nvimgcodecCodeStreamInfo_t);
stream_info.struct_next = nullptr;

nvimgcodecStatus_t status = nvimgcodecCodeStreamGetCodeStreamInfo(
    main_code_stream_, &stream_info);

if (status != NVIMGCODEC_STATUS_SUCCESS)
{
    throw std::runtime_error(fmt::format("Failed to get code stream info (status: {})",
                                        static_cast<int>(status)));
}
```

**nvimgcodecCodeStreamGetCodeStreamInfo()**: Queries the TIFF file structure.

**Returns**:
- `num_images`: Number of IFDs in the file
- `codec_name`: Overall codec (often "tiff" for multi-IFD files)

#### Lines 227-234: IFD Count and Codec

```cpp
uint32_t num_ifds = stream_info.num_images;
fmt::print("  TIFF has {} IFDs (resolution levels)\n", num_ifds);

if (stream_info.codec_name[0] != '\0')
{
    fmt::print("  Codec: {}\n", stream_info.codec_name);
}
```

**Example Output**:
```
  TIFF has 4 IFDs (resolution levels)
  Codec: tiff
```

#### Lines 236-261: Per-IFD Parsing Loop

```cpp
for (uint32_t i = 0; i < num_ifds; ++i)
{
    IfdInfo ifd_info;
    ifd_info.index = i;
    
    // Create view for this IFD
    nvimgcodecCodeStreamView_t view{};
    view.struct_type = NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_VIEW;
    view.struct_size = sizeof(nvimgcodecCodeStreamView_t);
    view.struct_next = nullptr;
    view.image_idx = i;  // Note: nvImageCodec uses 'image_idx' not 'image_index'
    
    // Get sub-code stream for this IFD
    status = nvimgcodecCodeStreamGetSubCodeStream(main_code_stream_,
                                                  &ifd_info.sub_code_stream,
                                                  &view);
```

**Code Stream View**: Specification for creating a sub-stream.

**Critical Field**: `view.image_idx = i` selects which IFD in the TIFF file.

**nvimgcodecCodeStreamGetSubCodeStream()**: Creates a new code stream representing just one IFD.

**Result**: `ifd_info.sub_code_stream` is a handle to IFD #i that can be decoded independently.

#### Lines 252-261: Error Handling for Failed IFDs

```cpp
if (status != NVIMGCODEC_STATUS_SUCCESS)
{
    fmt::print("❌ Failed to get sub-code stream for IFD {} (status: {})\n", 
              i, static_cast<int>(status));
    fmt::print("   This IFD will be SKIPPED and cannot be decoded.\n");
    ifd_info.sub_code_stream = nullptr;
    continue;
}
```

**Graceful Degradation**: If one IFD fails to parse, skip it but continue with others.

**Why might this fail?**
- Unsupported compression codec
- Corrupted IFD structure
- nvImageCodec version doesn't support this format variant

#### Lines 263-283: Extract Image Information

```cpp
nvimgcodecImageInfo_t image_info{};
image_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
image_info.struct_size = sizeof(nvimgcodecImageInfo_t);
image_info.struct_next = nullptr;

status = nvimgcodecCodeStreamGetImageInfo(ifd_info.sub_code_stream, &image_info);

if (status != NVIMGCODEC_STATUS_SUCCESS)
{
    fmt::print("❌ Failed to get image info for IFD {} (status: {})\n",
              i, static_cast<int>(status));
    fmt::print("   This IFD will be SKIPPED and cannot be decoded.\n");
    if (ifd_info.sub_code_stream)
    {
        nvimgcodecCodeStreamDestroy(ifd_info.sub_code_stream);
        ifd_info.sub_code_stream = nullptr;
    }
    continue;
}
```

**nvimgcodecCodeStreamGetImageInfo()**: Extracts detailed image properties from the IFD.

**Returns** (in `image_info`):
- Dimensions (width, height)
- Number of planes/channels
- Sample type (data type: uint8, uint16, etc.)
- Codec name (specific to this IFD: "jpeg", "jpeg2k", "deflate")

**Error Cleanup**: If getting image info fails, properly destroy the sub-code stream before continuing.

#### Lines 285-300: Extract Dimensions and Format

```cpp
ifd_info.width = image_info.plane_info[0].width;
ifd_info.height = image_info.plane_info[0].height;
ifd_info.num_channels = image_info.num_planes;

// Extract bits per sample from sample type
// sample_type encoding: bytes_per_element = (type >> 11) & 0xFF
auto sample_type = image_info.plane_info[0].sample_type;
int bytes_per_element = (static_cast<unsigned int>(sample_type) >> 11) & 0xFF;
ifd_info.bits_per_sample = bytes_per_element * 8;  // Convert bytes to bits

if (image_info.codec_name[0] != '\0')
{
    ifd_info.codec = image_info.codec_name;
}
```

**Bit Manipulation**: nvImageCodec encodes sample type as a bitfield. Bits 11-18 encode the bytes per element.

**Example**:
- `NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8` → 1 byte → 8 bits/sample
- `NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16` → 2 bytes → 16 bits/sample

#### Lines 302-312: Extract Metadata and TIFF Tags

```cpp
extract_ifd_metadata(ifd_info);
extract_tiff_tags(ifd_info);

ifd_info.print();

ifd_infos_.push_back(std::move(ifd_info));
```

**extract_ifd_metadata()**: Gets vendor-specific metadata (Aperio, Philips, etc.)

**extract_tiff_tags()**: Gets individual TIFF tags (SUBFILETYPE, JPEGTables, etc.)

**Move Semantics**: `std::move(ifd_info)` efficiently transfers ownership to the vector without copying nvImageCodec handles.

#### Lines 314-324: Parsing Summary

```cpp
if (ifd_infos_.size() == num_ifds)
{
    fmt::print("✅ TIFF parser initialized with {} IFDs (all successful)\n", ifd_infos_.size());
}
else
{
    fmt::print("⚠️  TIFF parser initialized with {} IFDs ({} out of {} total)\n", 
              ifd_infos_.size(), ifd_infos_.size(), num_ifds);
    fmt::print("   {} IFDs were skipped due to parsing errors\n", num_ifds - ifd_infos_.size());
}
```

**Diagnostic Output**: Reports success or partial failure.

**Example**:
```
✅ TIFF parser initialized with 4 IFDs (all successful)
```
or
```
⚠️  TIFF parser initialized with 3 IFDs (3 out of 4 total)
   1 IFDs were skipped due to parsing errors
```

### Lines 327-410: extract_ifd_metadata() - Vendor Metadata Extraction

This method extracts vendor-specific metadata using nvImageCodec's metadata API.

#### Lines 329-334: Validation

```cpp
auto& manager = NvImageCodecTiffParserManager::instance();

if (!manager.get_decoder() || !ifd_info.sub_code_stream)
{
    return;  // No decoder or stream available
}
```

**Prerequisites**: Requires both a decoder (for metadata extraction) and a valid sub-code stream.

#### Lines 336-348: Step 1 - Get Metadata Count

```cpp
int metadata_count = 0;
nvimgcodecStatus_t status = nvimgcodecDecoderGetMetadata(
    manager.get_decoder(),
    ifd_info.sub_code_stream,
    nullptr,  // First call: get count only
    &metadata_count
);

if (status != NVIMGCODEC_STATUS_SUCCESS || metadata_count == 0)
{
    return;  // No metadata or error
}

fmt::print("  Found {} metadata entries for IFD[{}]\n", metadata_count, ifd_info.index);
```

**Two-Step API Pattern**:
1. **First call** with `nullptr`: Returns count of metadata entries
2. **Second call** with allocated array: Returns actual metadata

**Why two calls?** Allows caller to allocate exact amount of memory needed.

#### Lines 352-368: Step 2 - Get Actual Metadata

```cpp
std::vector<nvimgcodecMetadata_t*> metadata_ptrs(metadata_count, nullptr);

status = nvimgcodecDecoderGetMetadata(
    manager.get_decoder(),
    ifd_info.sub_code_stream,
    metadata_ptrs.data(),
    &metadata_count
);

if (status != NVIMGCODEC_STATUS_SUCCESS)
{
    fmt::print("⚠️  Failed to retrieve metadata for IFD[{}] (status: {})\n",
              ifd_info.index, static_cast<int>(status));
    return;
}
```

**Second Call**: Fills `metadata_ptrs` array with pointers to metadata structures.

**Memory Ownership**: nvImageCodec manages the metadata memory. Pointers are valid until decoder is destroyed.

#### Lines 370-409: Step 3 - Process Each Metadata Entry

```cpp
for (int j = 0; j < metadata_count; ++j)
{
    if (!metadata_ptrs[j])
        continue;
    
    nvimgcodecMetadata_t* metadata = metadata_ptrs[j];
    
    int kind = metadata->kind;
    int format = metadata->format;
    size_t buffer_size = metadata->buffer_size;
    const uint8_t* buffer = static_cast<const uint8_t*>(metadata->buffer);
    
    fmt::print("    Metadata[{}]: kind={}, format={}, size={}\n",
              j, kind, format, buffer_size);
```

**Metadata Structure Fields**:
- `kind`: Metadata category (see below)
- `format`: Data format (RAW=0, XML=1, JSON=2)
- `buffer_size`: Size in bytes
- `buffer`: Raw data pointer

**Metadata Kinds** (nvimgcodecMetadataKind_t):
- `0`: TIFF_TAG (individual tags)
- `1`: MED_APERIO (Aperio SVS metadata)
- `2`: MED_PHILIPS (Philips TIFF XML metadata)
- `3`: MED_LEICA (Leica SCN metadata)
- `4`: MED_VENTANA (Ventana metadata)
- `5`: MED_TRESTLE (Trestle metadata)

#### Lines 388-408: Store Metadata and Extract ImageDescription

```cpp
if (buffer && buffer_size > 0)
{
    IfdInfo::MetadataBlob blob;
    blob.format = format;
    blob.data.assign(buffer, buffer + buffer_size);
    ifd_info.metadata_blobs[kind] = std::move(blob);
    
    // Special handling: extract ImageDescription if it's a text format
    if (kind == 1 && ifd_info.image_description.empty())  // MED_APERIO = 1
    {
        ifd_info.image_description.assign(buffer, buffer + buffer_size);
    }
    else if (kind == 2)  // MED_PHILIPS = 2
    {
        ifd_info.image_description.assign(buffer, buffer + buffer_size);
    }
}
```

**Storage Strategy**: Copy all metadata to `metadata_blobs` map for later access.

**ImageDescription Extraction**: For Aperio and Philips formats, extract the primary metadata string to `image_description` field for convenient access.

**Example Aperio Metadata**:
```
Aperio Image Library v10.0.50
16000x17597 [0,100 15374x17497] (256x256) J2K/YUV16 Q=30
```

### Lines 412-420: get_ifd() - IFD Accessor

```cpp
const IfdInfo& TiffFileParser::get_ifd(uint32_t index) const
{
    if (index >= ifd_infos_.size())
    {
        throw std::out_of_range(fmt::format("IFD index {} out of range (have {} IFDs)",
                                           index, ifd_infos_.size()));
    }
    return ifd_infos_[index];
}
```

Simple bounds-checked accessor with helpful error message.

### Lines 422-498: classify_ifd() - Image Type Classification

This method determines whether an IFD is a resolution level or associated image.

#### Lines 422-431: Validation and Setup

```cpp
ImageType TiffFileParser::classify_ifd(uint32_t ifd_index) const
{
    if (ifd_index >= ifd_infos_.size())
    {
        return ImageType::UNKNOWN;
    }
    
    const auto& ifd = ifd_infos_[ifd_index];
    const std::string& desc = ifd.image_description;
```

**Input**: IFD index
**Output**: ImageType enumeration
**Primary Data Source**: ImageDescription string

#### Lines 433-465: Aperio SVS Classification

```cpp
if (!desc.empty())
{
    std::string desc_lower = desc;
    std::transform(desc_lower.begin(), desc_lower.end(), desc_lower.begin(),
                  [](unsigned char c){ return std::tolower(c); });
    
    // Check for explicit keywords
    if (desc_lower.find("label ") != std::string::npos || 
        desc_lower.find("\nlabel ") != std::string::npos)
    {
        return ImageType::LABEL;
    }
    
    if (desc_lower.find("macro ") != std::string::npos || 
        desc_lower.find("\nmacro ") != std::string::npos)
    {
        return ImageType::MACRO;
    }
    
    // Aperio thumbnail has dimension transformation: "WxH -> WxH"
    if (desc.find(" -> ") != std::string::npos && desc.find(" - ") != std::string::npos)
    {
        return ImageType::THUMBNAIL;
    }
}
```

**Aperio SVS Keywords**:
- **Label**: Contains `"label "` or `"\nlabel "` in ImageDescription
  - Example: `"Aperio Image Library v10.0.50\nlabel 415x422"`
- **Macro**: Contains `"macro "` or `"\nmacro "`
  - Example: `"Aperio Image Library v10.0.50\nmacro 1280x421"`
- **Thumbnail**: Contains dimension transformation `" -> "` and `" - "`
  - Example: `"Aperio Image Library v10.0.50\n15374x17497 -> 674x768 - |..."`

**Case Insensitive**: Converts to lowercase for robust matching.

#### Lines 467-483: Fallback Heuristics

```cpp
// Fallback heuristics for formats without clear keywords
if (ifd.width < 2000 && ifd.height < 2000)
{
    // Convention: Second IFD (index 1) is often thumbnail
    if (ifd_index == 1)
    {
        return ImageType::THUMBNAIL;
    }
    
    if (!desc.empty())
    {
        return ImageType::UNKNOWN;  // Has description but can't classify
    }
}
```

**Heuristic Rules** (when keywords not found):
1. Small images (< 2000x2000) with index 1 → THUMBNAIL
2. Small images with description but no keywords → UNKNOWN

#### Lines 485-497: Resolution Level Classification

```cpp
// IFD 0 is always main resolution level
if (ifd_index == 0)
{
    return ImageType::RESOLUTION_LEVEL;
}

// Large images are resolution levels
if (ifd.width >= 2000 || ifd.height >= 2000)
{
    return ImageType::RESOLUTION_LEVEL;
}

return ImageType::UNKNOWN;
```

**Resolution Level Rules**:
1. IFD 0 is ALWAYS main level (standard TIFF convention)
2. Large images (≥ 2000 pixels on any dimension) are levels
3. Otherwise UNKNOWN

### Lines 500-539: Helper Methods for IFD Organization

#### get_resolution_levels()

```cpp
std::vector<uint32_t> TiffFileParser::get_resolution_levels() const
{
    std::vector<uint32_t> levels;
    
    for (const auto& ifd : ifd_infos_)
    {
        if (classify_ifd(ifd.index) == ImageType::RESOLUTION_LEVEL)
        {
            levels.push_back(ifd.index);
        }
    }
    
    return levels;
}
```

**Returns**: Vector of IFD indices that are resolution levels.

**Example**: `[0, 3, 5, 7]` (IFDs 1, 2, 4, 6 might be associated images)

#### get_associated_images()

```cpp
std::map<std::string, uint32_t> TiffFileParser::get_associated_images() const
{
    std::map<std::string, uint32_t> associated;
    
    for (const auto& ifd : ifd_infos_)
    {
        auto type = classify_ifd(ifd.index);
        switch (type)
        {
            case ImageType::THUMBNAIL:
                associated["thumbnail"] = ifd.index;
                break;
            case ImageType::LABEL:
                associated["label"] = ifd.index;
                break;
            case ImageType::MACRO:
                associated["macro"] = ifd.index;
                break;
            default:
                break;
        }
    }
    
    return associated;
}
```

**Returns**: Map of name → IFD index for associated images.

**Example**: `{"thumbnail": 1, "label": 2, "macro": 4}`

**Usage**: OpenSlide-compatible API for accessing non-pyramid images.

### Lines 541-557: override_ifd_dimensions()

```cpp
void TiffFileParser::override_ifd_dimensions(uint32_t ifd_index, 
                                             uint32_t width, 
                                             uint32_t height)
{
    if (ifd_index >= ifd_infos_.size())
    {
        throw std::out_of_range(fmt::format("IFD index {} out of range (have {} IFDs)",
                                           ifd_index, ifd_infos_.size()));
    }
    
    auto& ifd = ifd_infos_[ifd_index];
    fmt::print("⚙️  Overriding IFD[{}] dimensions: {}x{} -> {}x{}\n",
              ifd_index, ifd.width, ifd.height, width, height);
    
    ifd.width = width;
    ifd.height = height;
}
```

**Purpose**: Correct dimensions for formats where reported size includes padding.

**Use Case**: Philips TIFF files report tile-aligned dimensions, but actual image is smaller. XML metadata contains true dimensions.

**Example**:
```
Reported: 71680x51968 (tile-aligned)
Actual:   71412x51761 (from XML)
```

### Lines 559-581: Utility Methods

#### get_image_description()

```cpp
std::string TiffFileParser::get_image_description(uint32_t ifd_index) const
{
    if (ifd_index >= ifd_infos_.size())
    {
        return "";
    }
    
    const auto& ifd = ifd_infos_[ifd_index];
    return ifd.image_description;
}
```

Returns ImageDescription for the specified IFD.

#### print_info()

```cpp
void TiffFileParser::print_info() const
{
    fmt::print("\nTIFF File Information:\n");
    fmt::print("  File: {}\n", file_path_);
    fmt::print("  Number of IFDs: {}\n", ifd_infos_.size());
    fmt::print("\nIFD Details:\n");
    
    for (const auto& ifd : ifd_infos_)
    {
        ifd.print();
    }
}
```

Diagnostic output showing complete TIFF structure.

---

## TIFF Tag Extraction (nvImageCodec 0.7.0+)

### Lines 588-701: extract_tiff_tags() - Individual Tag Retrieval

This method extracts individual TIFF tags using libtiff (for nvTIFF 0.6.0.77 compatibility).

#### Lines 589-601: Setup and Validation

```cpp
void TiffFileParser::extract_tiff_tags(IfdInfo& ifd_info)
{
    auto& manager = NvImageCodecTiffParserManager::instance();
    
    if (!manager.get_decoder())
    {
        fmt::print("  ⚠️  Cannot extract TIFF tags: decoder not available\n");
        return;
    }
    
    if (!ifd_info.sub_code_stream)
    {
        fmt::print("  ⚠️  Cannot extract TIFF tags: sub_code_stream is null\n");
        return;
    }
```

**Compatibility Note**: nvTIFF 0.6.0.77 metadata API has compatibility issues. We use libtiff directly instead.

#### Lines 604-620: TIFF Tag Names Map

```cpp
std::map<uint32_t, std::string> tiff_tag_names = {
    {254, "SUBFILETYPE"},
    {256, "ImageWidth"},
    {257, "ImageLength"},
    {258, "BitsPerSample"},
    {259, "Compression"},
    {262, "PhotometricInterpretation"},
    {270, "ImageDescription"},
    {271, "Make"},
    {272, "Model"},
    {305, "Software"},
    {306, "DateTime"},
    {322, "TileWidth"},
    {323, "TileLength"},
    {339, "SampleFormat"},
    {347, "JPEGTables"}
};
```

**Standard TIFF Tags**: Map of tag ID to human-readable name.

**Most Important**:
- **254 (SUBFILETYPE)**: Image type (0=main, 1=reduced/associated)
- **259 (Compression)**: Codec (1=uncompressed, 7=JPEG, 33005=JPEG2000)
- **270 (ImageDescription)**: Vendor metadata
- **347 (JPEGTables)**: Shared JPEG tables (abbreviated JPEG)

#### Lines 633-682: Extract Tags with libtiff

```cpp
TIFF* tif = TIFFOpen(file_path_.c_str(), "r");
if (tif)
{
    if (TIFFSetDirectory(tif, ifd_info.index))
    {
        // Check for TIFFTAG_JPEGTABLES (tag 347)
        uint32_t jpegtables_count = 0;
        const void* jpegtables_data = nullptr;
        
        if (TIFFGetField(tif, TIFFTAG_JPEGTABLES, &jpegtables_count, &jpegtables_data))
        {
            has_jpeg_tables = true;
            ifd_info.tiff_tags["JPEGTables"] = "<detected by libtiff>";
            tiff_tag_count++;
            fmt::print("    🔍 Tag 347 (JPEGTables): [binary data, {} bytes] - ABBREVIATED JPEG DETECTED!\n", 
                      jpegtables_count);
        }
```

**Why libtiff?** nvTIFF 0.6.0.77 metadata API has compatibility issues. Direct libtiff access is more reliable.

**JPEGTables Tag (347)**: Critical for Aperio SVS files.

**Abbreviated JPEG**: JPEG compression where quantization and Huffman tables are stored once in TIFFTAG_JPEGTABLES, then referenced by all tiles. Saves space and maintains consistency.

**nvTIFF Support**: nvTIFF 0.6.0.77 handles JPEGTables automatically with GPU acceleration!

#### Lines 654-678: Extract Other Common Tags

```cpp
char* image_desc = nullptr;
if (TIFFGetField(tif, TIFFTAG_IMAGEDESCRIPTION, &image_desc))
{
    if (image_desc && strlen(image_desc) > 0)
    {
        ifd_info.tiff_tags["ImageDescription"] = std::string(image_desc);
        tiff_tag_count++;
    }
}

char* software = nullptr;
if (TIFFGetField(tif, TIFFTAG_SOFTWARE, &software))
{
    if (software && strlen(software) > 0)
    {
        ifd_info.tiff_tags["Software"] = std::string(software);
        tiff_tag_count++;
    }
}

uint16_t compression = 0;
if (TIFFGetField(tif, TIFFTAG_COMPRESSION, &compression))
{
    ifd_info.tiff_tags["Compression"] = std::to_string(compression);
    tiff_tag_count++;
}
```

**Additional Tags**:
- **ImageDescription**: Vendor metadata string
- **Software**: Scanner/software version
- **Compression**: Codec enumeration

#### Lines 688-700: Summary and JPEGTables Notice

```cpp
if (tiff_tag_count > 0)
{
    fmt::print("  ✅ Extracted {} TIFF tags for IFD[{}]\n", tiff_tag_count, ifd_info.index);
    if (has_jpeg_tables)
    {
        fmt::print("  ℹ️  IFD[{}] uses abbreviated JPEG (JPEGTables present)\n", ifd_info.index);
        fmt::print("  ✅ nvTIFF 0.6.0.77 will handle JPEGTables automatically with GPU acceleration\n");
    }
}
```

**Important Notice**: When JPEGTables is detected, inform user that nvTIFF will handle it with GPU acceleration (no CPU fallback needed).

### Lines 703-748: Metadata Query Methods

These methods provide access to extracted TIFF tags and metadata kinds.

#### get_tiff_tag()

```cpp
std::string TiffFileParser::get_tiff_tag(uint32_t ifd_index, const std::string& tag_name) const
{
    if (ifd_index >= ifd_infos_.size())
        return "";
    
    auto it = ifd_infos_[ifd_index].tiff_tags.find(tag_name);
    if (it != ifd_infos_[ifd_index].tiff_tags.end())
        return it->second;
    
    return "";
}
```

Retrieve any extracted TIFF tag by name.

#### get_subfile_type()

```cpp
int TiffFileParser::get_subfile_type(uint32_t ifd_index) const
{
    std::string subfile_str = get_tiff_tag(ifd_index, "SUBFILETYPE");
    if (subfile_str.empty())
        return -1;
    
    try {
        return std::stoi(subfile_str);
    } catch (...) {
        return -1;
    }
}
```

**Returns**:
- `0` = Full resolution image
- `1` = Reduced resolution (thumbnail/label/macro)
- `-1` = Tag not present

#### query_metadata_kinds()

```cpp
std::vector<int> TiffFileParser::query_metadata_kinds(uint32_t ifd_index) const
{
    std::vector<int> kinds;
    
    if (ifd_index >= ifd_infos_.size())
        return kinds;
    
    // Return all metadata kinds found in this IFD
    for (const auto& [kind, blob] : ifd_infos_[ifd_index].metadata_blobs)
    {
        kinds.push_back(kind);
    }
    
    // Also add TIFF_TAG kind (0) if any tags were extracted
    if (!ifd_infos_[ifd_index].tiff_tags.empty())
    {
        kinds.insert(kinds.begin(), 0);
    }
    
    return kinds;
}
```

**Returns**: List of all metadata kind values present in the IFD.

**Example**: `[0, 1]` means TIFF_TAG (0) and MED_APERIO (1) metadata available.

#### get_detected_format()

```cpp
std::string TiffFileParser::get_detected_format() const
{
    if (ifd_infos_.empty())
        return "Unknown";
    
    const auto& kinds = query_metadata_kinds(0);
    
    for (int kind : kinds)
    {
        switch (kind)
        {
            case 1:  // NVIMGCODEC_METADATA_KIND_MED_APERIO
                return "Aperio SVS";
            case 2:  // NVIMGCODEC_METADATA_KIND_MED_PHILIPS
                return "Philips TIFF";
            case 3:  // NVIMGCODEC_METADATA_KIND_MED_LEICA
                return "Leica SCN";
            case 4:  // NVIMGCODEC_METADATA_KIND_MED_VENTANA
                return "Ventana";
            case 5:  // NVIMGCODEC_METADATA_KIND_MED_TRESTLE
                return "Trestle";
        }
    }
    
    // Fallback: Generic TIFF with codec
    if (!ifd_infos_.empty() && !ifd_infos_[0].codec.empty())
    {
        return fmt::format("Generic TIFF ({})", ifd_infos_[0].codec);
    }
    
    return "Generic TIFF";
}
```

**Format Detection**: Checks IFD 0 for vendor-specific metadata.

**Returns**:
- "Aperio SVS"
- "Philips TIFF"
- "Leica SCN"
- "Generic TIFF (jpeg)"
- "Generic TIFF"

---

## ROI-Based Decoding Implementation

### Lines 790-1161: decode_region() - Core ROI Decoding Method

This is the most complex method in the parser. It decodes a specific region of an IFD using nvImageCodec.

#### Lines 790-806: Initial Validation

```cpp
uint8_t* TiffFileParser::decode_region(
    uint32_t ifd_index,
    uint32_t x, uint32_t y,
    uint32_t width, uint32_t height,
    uint8_t* output_buffer,
    const cucim::io::Device& device)
{
    if (!initialized_)
    {
        throw std::runtime_error("TIFF parser not initialized");
    }
    
    if (ifd_index >= ifd_infos_.size())
    {
        throw std::out_of_range(fmt::format("IFD index {} out of range (max: {})",
                                            ifd_index, ifd_infos_.size() - 1));
    }
```

**Parameters**:
- `ifd_index`: Which resolution level
- `x, y`: Top-left corner of ROI
- `width, height`: ROI dimensions
- `output_buffer`: Pre-allocated buffer (or nullptr for auto-allocation)
- `device`: "cpu" or "cuda"

#### Lines 808-824: Validate Sub-Code Stream and ROI Bounds

```cpp
const auto& ifd = ifd_infos_[ifd_index];

if (!ifd.sub_code_stream)
{
    throw std::runtime_error(fmt::format(
        "IFD[{}] has invalid sub_code_stream - TIFF parsing may have failed during initialization. "
        "This IFD cannot be decoded.", ifd_index));
}

if (x + width > ifd.width || y + height > ifd.height)
{
    throw std::invalid_argument(fmt::format(
        "ROI ({},{} {}x{}) exceeds IFD dimensions ({}x{})",
        x, y, width, height, ifd.width, ifd.height));
}
```

**Critical Check**: Verify sub_code_stream is valid. If parsing failed for this IFD during initialization, sub_code_stream will be nullptr.

**Bounds Validation**: Ensure ROI is within IFD dimensions.

#### Lines 826-841: JPEGTables Handling Notice

```cpp
// NOTE: nvTIFF 0.6.0.77 CAN handle JPEGTables (TIFFTAG_JPEGTABLES)!
// Previous documentation suggested nvImageCodec couldn't handle abbreviated JPEG,
// but testing confirms nvTIFF 0.6.0.77 successfully decodes with automatic JPEG table handling.

if (ifd.tiff_tags.find("JPEGTables") != ifd.tiff_tags.end())
{
    fmt::print("ℹ️  JPEG with JPEGTables detected - nvTIFF 0.6.0.77 will handle automatically\n");
}

fmt::print("✓ Proceeding with nvTIFF/nvImageCodec decode (codec='{}')\n", ifd.codec);

fmt::print("🎯 nvTiff ROI Decode: IFD[{}] region ({},{}) {}x{}, device={}\n",
          ifd_index, x, y, width, height, std::string(device));
```

**Important Discovery**: nvTIFF 0.6.0.77 DOES support abbreviated JPEG (JPEGTables) with GPU acceleration!

This was tested and confirmed to work correctly for Aperio SVS files.

#### Lines 843-854: Get Manager and Decoder

```cpp
// CRITICAL: Must use the same manager that created main_code_stream_!
auto& manager = NvImageCodecTiffParserManager::instance();
if (!manager.is_available())
{
    throw std::runtime_error("nvImageCodec not available for ROI decoding");
}

try
{
    nvimgcodecDecoder_t decoder = manager.get_decoder();
```

**Critical Rule**: Must use decoder from the SAME nvImageCodec instance that created the code stream.

**Why?** nvImageCodec maintains internal state mapping code streams to their instance. Using a decoder from a different instance causes segfaults.

#### Lines 856-873: Prepare Decode Parameters and ROI Region

```cpp
nvimgcodecDecodeParams_t decode_params{};
decode_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS;
decode_params.struct_size = sizeof(nvimgcodecDecodeParams_t);
decode_params.struct_next = nullptr;
decode_params.apply_exif_orientation = 0;

nvimgcodecRegion_t region{};
region.struct_type = NVIMGCODEC_STRUCTURE_TYPE_REGION;
region.struct_size = sizeof(nvimgcodecRegion_t);
region.struct_next = nullptr;
region.ndim = 2;
region.start[0] = y;  // Height dimension
region.start[1] = x;  // Width dimension
region.end[0] = y + height;
region.end[1] = x + width;
```

**ROI Region Structure**: Specifies rectangular region to decode.

**Coordinate Order**:
- `start[0]`, `end[0]` = Y dimension (height)
- `start[1]`, `end[1]` = X dimension (width)

**Example**: ROI at (100, 50) with size 256x128:
- `start = [50, 100]`
- `end = [178, 356]`

#### Lines 876-901: Create ROI Code Stream View

```cpp
// CRITICAL: Must create ROI stream from main_code_stream, not from ifd.sub_code_stream!
// Nested sub-streams don't properly handle JPEG tables in TIFF files.
nvimgcodecCodeStreamView_t view{};
view.struct_type = NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_VIEW;
view.struct_size = sizeof(nvimgcodecCodeStreamView_t);
view.struct_next = nullptr;
view.image_idx = ifd_index;  // Specify which IFD in the main stream
view.region = region;         // AND the ROI region within that IFD

// Get ROI-specific code stream directly from main stream
nvimgcodecCodeStream_t roi_stream = nullptr;
fmt::print("📍 Creating ROI sub-stream: IFD[{}] ROI=[{},{}:{}x{}] from main stream\n",
          ifd_index, x, y, width, height);

nvimgcodecStatus_t status = nvimgcodecCodeStreamGetSubCodeStream(
    main_code_stream_, &roi_stream, &view);

if (status != NVIMGCODEC_STATUS_SUCCESS)
{
    throw std::runtime_error(fmt::format(
        "Failed to create ROI code stream for IFD[{}] ROI=[{},{}:{}x{}]: status={}\n"
        "  IFD dimensions: {}x{}, codec: {}\n",
        ifd_index, x, y, width, height, static_cast<int>(status),
        ifd.width, ifd.height, ifd.codec));
}
```

**CRITICAL DESIGN DECISION**: Create ROI stream from `main_code_stream_`, NOT from `ifd.sub_code_stream`.

**Why?** Nested sub-streams (sub-stream of a sub-stream) don't properly handle JPEG tables. Must go directly from main stream to ROI stream.

**View Specifies**:
1. `image_idx`: Which IFD in the TIFF file
2. `region`: Which rectangle within that IFD

#### Lines 906-937: Get Image Info from ROI Stream

```cpp
fmt::print("🔍 Getting image info from ROI stream...\n");
nvimgcodecImageInfo_t input_image_info{};
input_image_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
input_image_info.struct_size = sizeof(nvimgcodecImageInfo_t);
input_image_info.struct_next = nullptr;

status = nvimgcodecCodeStreamGetImageInfo(roi_stream, &input_image_info);

if (status != NVIMGCODEC_STATUS_SUCCESS)
{
    nvimgcodecCodeStreamDestroy(roi_stream);
    throw std::runtime_error(fmt::format(
        "Failed to get image info for IFD[{}]: status={}", ifd_index, static_cast<int>(status)));
}

if (input_image_info.num_planes == 0)
{
    nvimgcodecCodeStreamDestroy(roi_stream);
    throw std::runtime_error(fmt::format(
        "IFD[{}] ROI image info has 0 planes", ifd_index));
}

fmt::print("✅ Got image info: {}x{}, {} channels, sample_format={}, color_spec={}\n", 
          input_image_info.plane_info[0].width,
          input_image_info.plane_info[0].height,
          input_image_info.num_planes,
          static_cast<int>(input_image_info.sample_format),
          static_cast<int>(input_image_info.color_spec));

fmt::print("⚠️  Note: ROI stream returns full image dimensions, will use requested ROI: {}x{}\n",
          width, height);
```

**Quirk**: ROI stream still reports full IFD dimensions in image info, not ROI dimensions.

**Workaround**: We use the requested ROI dimensions for output buffer sizing, not the image info dimensions.

#### Lines 939-968: Prepare Output Image Info

```cpp
fmt::print("📝 Preparing output image info...\n");

// CRITICAL: Use zero-initialization to avoid copying codec-specific fields
nvimgcodecImageInfo_t output_image_info{};

output_image_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
output_image_info.struct_size = sizeof(nvimgcodecImageInfo_t);
output_image_info.struct_next = nullptr;

// Set output format - IMPORTANT: For interleaved RGB, num_planes = 1
output_image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
output_image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
output_image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
output_image_info.num_planes = 1;  // Interleaved RGB is single plane

// Set plane info
output_image_info.plane_info[0].width = width;
output_image_info.plane_info[0].height = height;
output_image_info.plane_info[0].num_channels = ifd.num_channels;
output_image_info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
output_image_info.plane_info[0].precision = 0;
```

**CRITICAL**: Zero-initialize output_image_info. Do NOT copy from input_image_info!

**Why?** input_image_info contains codec-specific internal fields and pointers that are only valid for the input stream. Copying them to output causes segfaults.

**Output Format**:
- `NVIMGCODEC_SAMPLEFORMAT_I_RGB`: Interleaved RGB (RGBRGBRGB...)
- `num_planes = 1`: Interleaved format is treated as single plane with 3 channels
- `NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8`: 8-bit unsigned integer per channel

#### Lines 970-1023: Allocate Output Buffer

```cpp
bool use_gpu = (device.type() == cucim::io::DeviceType::kCUDA);
output_image_info.buffer_kind = use_gpu ? 
    NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE :
    NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;

int bytes_per_element = 1;  // UINT8
size_t row_stride = width * ifd.num_channels * bytes_per_element;
size_t output_size = row_stride * height;

fmt::print("💾 Allocating output buffer: {} bytes on {} ({}x{}x{}x{} bytes/element)\n", 
          output_size, use_gpu ? "GPU" : "CPU",
          width, height, ifd.num_channels, bytes_per_element);

bool buffer_was_preallocated = (output_buffer != nullptr);

if (!buffer_was_preallocated)
{
    if (use_gpu)
    {
        cudaError_t cuda_err = cudaMalloc(&output_buffer, output_size);
        if (cuda_err != cudaSuccess)
        {
            throw std::runtime_error(fmt::format(
                "Failed to allocate {} bytes on GPU: {}",
                output_size, cudaGetErrorString(cuda_err)));
        }
    }
    else
    {
        output_buffer = static_cast<uint8_t*>(malloc(output_size));
        if (!output_buffer)
        {
            throw std::runtime_error(fmt::format(
                "Failed to allocate {} bytes on host", output_size));
        }
    }
    fmt::print("✅ Buffer allocated successfully\n");
}
```

**Buffer Sizing**:
- **Row stride**: `width × channels × bytes_per_element`
- **Total size**: `row_stride × height`

**Example**: 256×128 RGB image
- Row stride: `256 × 3 × 1 = 768 bytes`
- Total: `768 × 128 = 98,304 bytes`

**Allocation Strategy**:
- If `output_buffer` is provided, use it (caller manages memory)
- If nullptr, allocate buffer (this function manages memory)

#### Lines 1020-1054: Create nvImageCodec Image Object

```cpp
output_image_info.buffer = output_buffer;
output_image_info.buffer_size = output_size;
output_image_info.plane_info[0].row_stride = row_stride;
output_image_info.cuda_stream = 0;  // CRITICAL: Default CUDA stream

fmt::print("🖼️  Creating nvImageCodec image object...\n");

nvimgcodecImage_t image;
status = nvimgcodecImageCreate(manager.get_instance(), &image, &output_image_info);

if (status != NVIMGCODEC_STATUS_SUCCESS)
{
    nvimgcodecCodeStreamDestroy(roi_stream);
    if (!buffer_was_preallocated)
    {
        if (use_gpu)
            cudaFree(output_buffer);
        else
            free(output_buffer);
    }
    throw std::runtime_error(fmt::format(
        "Failed to create nvImageCodec image: status={}", static_cast<int>(status)));
}
```

**nvimgcodecImageCreate()**: Creates image object representing the output buffer.

**Purpose**: nvImageCodec needs an image handle to write decoded data to. This associates the buffer with metadata (dimensions, format, stride).

**Critical Field**: `cuda_stream = 0` must be set (default CUDA stream for GPU operations).

#### Lines 1058-1092: Perform Decode Operation

```cpp
fmt::print("📋 nvTiff: Decoding with automatic JPEG table handling...\n");

nvimgcodecFuture_t decode_future;
{
    std::lock_guard<std::mutex> lock(manager.get_mutex());
    fmt::print("   Calling nvimgcodecDecoderDecode()...\n");
    status = nvimgcodecDecoderDecode(
        decoder,
        &roi_stream,
        &image,
        1,
        &decode_params,
        &decode_future);
    fmt::print("   Decode scheduled, status={}\n", static_cast<int>(status));
}

if (status != NVIMGCODEC_STATUS_SUCCESS)
{
    nvimgcodecImageDestroy(image);
    nvimgcodecCodeStreamDestroy(roi_stream);
    if (!buffer_was_preallocated)
    {
        if (use_gpu)
            cudaFree(output_buffer);
        else
            free(output_buffer);
    }
    throw std::runtime_error(fmt::format(
        "Failed to schedule decode: status={}", static_cast<int>(status)));
}
```

**Thread Safety**: Mutex protects decoder from concurrent access.

**nvimgcodecDecoderDecode()**: Schedules asynchronous decode operation.

**Parameters**:
- `decoder`: Decoder handle
- `&roi_stream`: Input code stream (ROI)
- `&image`: Output image
- `1`: Batch size (decoding 1 image)
- `&decode_params`: Decode parameters
- `&decode_future`: Future handle for checking completion

**Asynchronous**: Decode happens on background thread or GPU stream. Must wait for completion.

#### Lines 1094-1140: Wait for Completion and Handle Errors

```cpp
fmt::print("⏳ Waiting for decode to complete...\n");
size_t status_size = 1;
nvimgcodecProcessingStatus_t decode_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
status = nvimgcodecFutureGetProcessingStatus(decode_future, &decode_status, &status_size);

if (use_gpu)
{
    cudaDeviceSynchronize();
    fmt::print("   GPU synchronized\n");
}

bool decode_failed = (status != NVIMGCODEC_STATUS_SUCCESS || 
                     decode_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS);

if (decode_failed)
{
    fmt::print("⚠️  nvImageCodec decode failed (status={}, decode_status={})\n",
              static_cast<int>(status), static_cast<int>(decode_status));
    
    // CRITICAL: Detach buffer ownership before destroying image
    output_image_info.buffer = nullptr;
    
    fmt::print("🧹 Cleaning up after failed decode...\n");
    nvimgcodecFutureDestroy(decode_future);
    nvimgcodecImageDestroy(image);
    nvimgcodecCodeStreamDestroy(roi_stream);
    
    if (!buffer_was_preallocated && output_buffer != nullptr)
    {
        fmt::print("   Freeing allocated buffer...\n");
        if (use_gpu)
            cudaFree(output_buffer);
        else
            free(output_buffer);
        output_buffer = nullptr;
    }
    
    fmt::print("💡 Returning nullptr to trigger libjpeg-turbo fallback\n");
    return nullptr;
}
```

**Completion Check**:
1. Get processing status from future
2. For GPU: Synchronize to ensure completion
3. Check both API status and processing status

**Failure Handling**:
- **Detach buffer**: Set `output_image_info.buffer = nullptr` to prevent nvImageCodec from trying to free it
- **Cleanup resources**: Destroy future, image, and stream
- **Free buffer**: If we allocated it
- **Return nullptr**: Signals caller to use fallback decoder (e.g., libjpeg-turbo)

**Why detach?** nvimgcodecImageDestroy() might try to free the buffer if it thinks it owns it. Setting buffer to nullptr prevents this.

#### Lines 1143-1154: Success Path Cleanup

```cpp
fmt::print("🧹 Cleaning up nvImageCodec objects...\n");
fmt::print("   Destroying future...\n");
nvimgcodecFutureDestroy(decode_future);
fmt::print("   Destroying image...\n");
nvimgcodecImageDestroy(image);
fmt::print("   Destroying ROI stream...\n");
nvimgcodecCodeStreamDestroy(roi_stream);
fmt::print("✅ Cleanup complete\n");

fmt::print("✅ nvTiff ROI Decode: Success! {}x{} decoded\n", width, height);
return output_buffer;
```

**Success Path**: Clean up nvImageCodec objects but keep the output buffer.

**Return Value**: Pointer to decoded RGB data in output_buffer.

**Memory Ownership**: 
- If caller provided buffer: Caller owns it
- If we allocated: Caller must free it (with appropriate method: cudaFree or free)

### Lines 1163-1175: decode_ifd() - Decode Entire IFD

```cpp
uint8_t* TiffFileParser::decode_ifd(
    uint32_t ifd_index,
    uint8_t* output_buffer,
    const cucim::io::Device& device)
{
    if (ifd_index >= ifd_infos_.size())
    {
        throw std::out_of_range(fmt::format("IFD index {} out of range", ifd_index));
    }
    
    const auto& ifd = ifd_infos_[ifd_index];
    return decode_region(ifd_index, 0, 0, ifd.width, ifd.height, output_buffer, device);
}
```

**Convenience Method**: Decodes the entire IFD by calling decode_region() with full dimensions.

**ROI**: `(0, 0)` to `(ifd.width, ifd.height)` (full image).

### Lines 1177-1181: has_roi_decode_support()

```cpp
bool TiffFileParser::has_roi_decode_support() const
{
    auto& manager = NvImageCodecManager::instance();
    return manager.is_initialized();
}
```

**Query Method**: Check if ROI decoding is available (i.e., nvImageCodec is loaded and initialized).

---

## Summary

### Class Hierarchy

```
NvImageCodecTiffParserManager (Singleton)
└── Manages global nvImageCodec instance for parsing
    └── Provides decoder for metadata extraction

TiffFileParser
├── Opens TIFF file using nvImageCodec
├── Parses all IFDs and extracts metadata
├── Classifies IFDs (resolution levels vs. associated images)
└── Decodes ROI or entire IFDs
```

### Key Data Structures

```
main_code_stream (nvimgcodecCodeStream_t)
└── Represents entire TIFF file
    ├── IFD 0: sub_code_stream
    │   ├── ROI (x,y,w,h): roi_stream
    │   └── Metadata: metadata_blobs, tiff_tags
    ├── IFD 1: sub_code_stream
    └── IFD N: sub_code_stream
```

### API Flow for ROI Decoding

```
1. TiffFileParser constructor
   ├── nvimgcodecCodeStreamCreateFromFile() → main_code_stream
   └── For each IFD:
       ├── nvimgcodecCodeStreamGetSubCodeStream() → sub_code_stream
       ├── nvimgcodecCodeStreamGetImageInfo() → dimensions, codec
       ├── nvimgcodecDecoderGetMetadata() → vendor metadata
       └── TIFFGetField() → TIFF tags (JPEGTables, etc.)

2. decode_region(ifd_idx, x, y, w, h)
   ├── Create ROI view (image_idx + region)
   ├── nvimgcodecCodeStreamGetSubCodeStream(main_stream, view) → roi_stream
   ├── nvimgcodecImageCreate() → output image
   ├── nvimgcodecDecoderDecode(roi_stream, output_image) → decode!
   └── nvimgcodecFutureGetProcessingStatus() → wait for completion
```

### Critical Design Patterns

1. **RAII**: Resources (code streams, images) are automatically cleaned up in destructors
2. **Move Semantics**: IfdInfo is move-only to prevent accidental handle duplication
3. **Singleton**: Manager ensures single global nvImageCodec instance
4. **Graceful Degradation**: If IFD parsing fails, skip it but continue with others
5. **Fallback Strategy**: If nvImageCodec decode fails, return nullptr to trigger fallback decoder

### Performance Optimizations

1. **GPU Acceleration**: nvTIFF provides GPU-accelerated JPEG/JPEG2000 decoding
2. **ROI Decoding**: Only decode needed region, not entire image
3. **Metadata Caching**: Parse all metadata once in constructor
4. **Lazy Decoding**: Pixels decoded only when decode_region() is called

### Compatibility Notes

- **nvImageCodec 0.7.0+**: Individual TIFF tag retrieval
- **nvTIFF 0.6.0.77**: Supports JPEGTables (abbreviated JPEG) with GPU acceleration
- **libtiff fallback**: Used for TIFF tag extraction when nvImageCodec API has issues

---

## Usage Example

```cpp
#include "nvimgcodec_tiff_parser.h"

// Open TIFF file
auto parser = std::make_unique<TiffFileParser>("/path/to/image.tif");

if (!parser->is_valid()) {
    std::cerr << "Failed to open TIFF file\n";
    return;
}

// Get TIFF structure
uint32_t num_ifds = parser->get_ifd_count();
std::string format = parser->get_detected_format();
std::cout << "Format: " << format << ", IFDs: " << num_ifds << "\n";

// Get resolution levels
auto levels = parser->get_resolution_levels();
std::cout << "Resolution levels: ";
for (auto idx : levels) {
    const auto& ifd = parser->get_ifd(idx);
    std::cout << idx << " (" << ifd.width << "x" << ifd.height << ") ";
}
std::cout << "\n";

// Get associated images
auto associated = parser->get_associated_images();
for (const auto& [name, idx] : associated) {
    std::cout << "Associated image: " << name << " (IFD " << idx << ")\n";
}

// Decode ROI from highest resolution level
uint32_t level_idx = levels[0];
uint32_t x = 1000, y = 1000;
uint32_t width = 512, height = 512;

cucim::io::Device device("cuda");  // or "cpu"
uint8_t* rgb_data = parser->decode_region(level_idx, x, y, width, height, nullptr, device);

if (rgb_data) {
    std::cout << "Successfully decoded " << width << "x" << height << " ROI\n";
    
    // Process RGB data...
    
    // Free buffer
    if (device.type() == cucim::io::DeviceType::kCUDA) {
        cudaFree(rgb_data);
    } else {
        free(rgb_data);
    }
} else {
    std::cout << "Decode failed, using fallback decoder\n";
}
```

This documentation provides a comprehensive line-by-line explanation of the nvImageCodec TIFF Parser implementation.

