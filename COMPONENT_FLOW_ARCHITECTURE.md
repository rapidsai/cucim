# TIFF Component Flow Architecture

Complete architectural overview of how `tiff.cpp`, `ifd.cpp`, `nvimgcodec_tiff_parser.cpp`, and `nvimgcodec_decoder.cpp` interact in the cuslide2 plugin.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Component Responsibilities](#component-responsibilities)
- [Phase 1: Opening a TIFF File](#phase-1-opening-a-tiff-file)
- [Phase 2: Reading Image Data](#phase-2-reading-image-data)
- [Key Method Connections](#key-method-connections)
- [Data Flow Diagram](#data-flow-diagram)
- [nvImageCodec API Calls](#nvimagecodec-api-calls)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         tiff.cpp                                │
│  Main TIFF file orchestrator - manages IFDs and metadata       │
│  Namespace: cuslide::tiff                                       │
└───────────────┬─────────────────────────────────────────────────┘
                │ creates/manages
                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         ifd.cpp                                 │
│  Individual IFD management - handles region reading logic      │
│  Namespace: cuslide::tiff                                       │
└───────────────┬─────────────────────────────────────────────────┘
                │ uses for parsing       │ uses for decoding
                ▼                        ▼
┌──────────────────────────┐  ┌──────────────────────────────────┐
│ nvimgcodec_tiff_parser.cpp│  │   nvimgcodec_decoder.cpp        │
│ Wraps nvImageCodec for   │  │ Wraps nvImageCodec for          │
│ metadata/structure       │  │ actual image decoding           │
│ Namespace:               │  │ Namespace:                      │
│ cuslide2::nvimgcodec     │  │ cuslide2::nvimgcodec            │
└──────────────────────────┘  └──────────────────────────────────┘
```

---

## Component Responsibilities

### 1. **tiff.cpp** - TIFF File Orchestrator
- **Location:** `cpp/plugins/cucim.kit.cuslide2/src/cuslide/tiff/tiff.cpp`
- **Class:** `TIFF`
- **Responsibilities:**
  - Opens TIFF files and manages file handles
  - Creates and manages multiple `IFD` objects (one per resolution level)
  - Detects vendor format (Aperio SVS, Philips TIFF, etc.)
  - Manages metadata extraction and JSON serialization
  - Routes read requests to appropriate IFD based on level
  - Handles associated images (label, macro, thumbnail)

### 2. **ifd.cpp** - Individual IFD Manager
- **Location:** `cpp/plugins/cucim.kit.cuslide2/src/cuslide/tiff/ifd.cpp`
- **Class:** `IFD`
- **Responsibilities:**
  - Represents a single IFD (Image File Directory / resolution level)
  - Stores IFD properties (width, height, tile size, compression, etc.)
  - Routes read requests to nvImageCodec decoder
  - Manages tile-based reading (for legacy compatibility)
  - Handles boundary conditions in region requests

### 3. **nvimgcodec_tiff_parser.cpp** - Metadata Extraction Layer
- **Location:** `cpp/plugins/cucim.kit.cuslide2/src/cuslide/nvimgcodec/nvimgcodec_tiff_parser.cpp`
- **Classes:** `TiffFileParser`, `NvImageCodecTiffParserManager` (singleton)
- **Responsibilities:**
  - Wraps nvImageCodec for TIFF structure parsing
  - Extracts IFD count and per-IFD metadata
  - Queries TIFF tags (Software, Model, ImageDescription, etc.)
  - Extracts vendor-specific metadata (Aperio, Philips, Leica, etc.)
  - Manages nvImageCodec code streams (main and sub-streams)
  - Provides singleton manager for shared nvImageCodec instance

### 4. **nvimgcodec_decoder.cpp** - Image Decoding Layer
- **Location:** `cpp/plugins/cucim.kit.cuslide2/src/cuslide/nvimgcodec/nvimgcodec_decoder.cpp`
- **Function:** `decode_ifd_region_nvimgcodec()`
- **Responsibilities:**
  - Wraps nvImageCodec for actual image decoding
  - Handles ROI (Region of Interest) decoding
  - Manages CPU vs GPU buffer allocation
  - Handles decode futures and synchronization
  - RAII management of nvImageCodec resources

---

## Phase 1: Opening a TIFF File

### Step 1: User calls `TIFF::open()` → `tiff.cpp`

**File:** `cpp/plugins/cucim.kit.cuslide2/src/cuslide/tiff/tiff.cpp`  
**Lines:** 252-364

```cpp
// NEW CONSTRUCTOR: nvImageCodec-only (no libtiff)
TIFF::TIFF(const cucim::filesystem::Path& file_path) : file_path_(file_path)
{
    #ifdef DEBUG
    fmt::print("📂 Opening TIFF file with nvImageCodec: {}\n", file_path);
    #endif

    // Step 1: Open file descriptor (needed for CuCIMFileHandle)
    char* file_path_cstr = static_cast<char*>(cucim_malloc(file_path.size() + 1));
    memcpy(file_path_cstr, file_path.c_str(), file_path.size());
    file_path_cstr[file_path.size()] = '\0';

    int fd = ::open(file_path_cstr, O_RDONLY, 0666);
    if (fd == -1)
    {
        cucim_free(file_path_cstr);
        throw std::invalid_argument(fmt::format("Cannot open {}!", file_path));
    }

    // Step 2: Create CuCIMFileHandle with 'this' as client_data
    file_handle_shared_ = std::make_shared<CuCIMFileHandle>(
        fd, nullptr, FileHandleType::kPosix, file_path_cstr, this);

    // Step 3: Initialize nvImageCodec TiffFileParser (MANDATORY)
    nvimgcodec_parser_ = std::make_unique<cuslide2::nvimgcodec::TiffFileParser>(
        file_path.c_str());

    if (!nvimgcodec_parser_->is_valid()) {
        throw std::runtime_error("TiffFileParser initialization failed");
    }

    // Initialize metadata container
    metadata_ = new json{};
}

std::shared_ptr<TIFF> TIFF::open(const cucim::filesystem::Path& file_path)
{
    auto tif = std::make_shared<TIFF>(file_path);
    tif->construct_ifds();  // ← Next step
    return tif;
}
```

**Purpose:**
- Opens the TIFF file with a POSIX file descriptor
- Creates the file handle wrapper
- **Initializes TiffFileParser** to extract metadata
- Validates parser initialization

---

### Step 2: `TiffFileParser` constructor → `nvimgcodec_tiff_parser.cpp`

**File:** `cpp/plugins/cucim.kit.cuslide2/src/cuslide/nvimgcodec/nvimgcodec_tiff_parser.cpp`  
**Lines:** 282-329

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
    
    try
    {
        // Step 1: Create code stream from TIFF file
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
        
        #ifdef DEBUG
        fmt::print("✅ Opened TIFF file: {}\n", file_path);
        #endif
        
        // Step 2: Parse TIFF structure (metadata only)
        parse_tiff_structure();  // ← Next step
        
        initialized_ = true;
        #ifdef DEBUG
        fmt::print("✅ TIFF parser initialized with {} IFDs\n", ifd_infos_.size());
        #endif
    }
    catch (const std::exception& e)
    {
        main_code_stream_ = nullptr;
        throw;  // Re-throw
    }
}
```

**nvImageCodec API calls:**
- `nvimgcodecCodeStreamCreateFromFile()` - Creates a code stream from file path

**Purpose:**
- Uses singleton `NvImageCodecTiffParserManager` to get shared nvImageCodec instance
- Creates a **code stream** (nvImageCodec's file handle)
- Triggers TIFF structure parsing

---

### Step 3: `parse_tiff_structure()` extracts IFD info → `nvimgcodec_tiff_parser.cpp`

**File:** `cpp/plugins/cucim.kit.cuslide2/src/cuslide/nvimgcodec/nvimgcodec_tiff_parser.cpp`  
**Lines:** 357-583

```cpp
void TiffFileParser::parse_tiff_structure()
{
    // Get TIFF structure information
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
    
    uint32_t num_ifds = stream_info.num_images;
    #ifdef DEBUG
    fmt::print("  TIFF has {} IFDs (resolution levels)\n", num_ifds);
    #endif
    
    // Get information for each IFD
    for (uint32_t i = 0; i < num_ifds; ++i)
    {
        IfdInfo ifd_info;
        ifd_info.index = i;
        
        // Create view for this IFD
        nvimgcodecCodeStreamView_t view{};
        view.struct_type = NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_VIEW;
        view.struct_size = sizeof(nvimgcodecCodeStreamView_t);
        view.struct_next = nullptr;
        view.image_idx = i;
        
        // Get sub-code stream for this IFD
        status = nvimgcodecCodeStreamGetSubCodeStream(main_code_stream_,
                                                      &ifd_info.sub_code_stream,
                                                      &view);
        
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            #ifdef DEBUG
            fmt::print("❌ Failed to get sub-code stream for IFD {} (status: {})\n", 
                      i, static_cast<int>(status));
            #endif
            ifd_info.sub_code_stream = nullptr;
            continue;
        }
        
        // Get image information for this IFD
        nvimgcodecImageInfo_t image_info{};
        image_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
        image_info.struct_size = sizeof(nvimgcodecImageInfo_t);
        image_info.struct_next = nullptr;
        
        status = nvimgcodecCodeStreamGetImageInfo(ifd_info.sub_code_stream, &image_info);
        
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            #ifdef DEBUG
            fmt::print("❌ Failed to get image info for IFD {} (status: {})\n",
                      i, static_cast<int>(status));
            #endif
            ifd_info.sub_code_stream = nullptr;
            continue;
        }
        
        // Extract IFD metadata
        ifd_info.width = image_info.plane_info[0].width;
        ifd_info.height = image_info.plane_info[0].height;
        ifd_info.num_channels = image_info.num_planes;
        
        // Extract bits per sample from sample type
        auto sample_type = image_info.plane_info[0].sample_type;
        int bytes_per_element = (static_cast<unsigned int>(sample_type) >> 11) & 0xFF;
        ifd_info.bits_per_sample = bytes_per_element * 8;
        
        if (image_info.codec_name[0] != '\0')
        {
            ifd_info.codec = image_info.codec_name;
        }
        
        // Extract vendor-specific metadata (Aperio, Philips, etc.)
        extract_ifd_metadata(ifd_info);
        
        // Extract TIFF tags using nvImageCodec 0.7.0 API
        extract_tiff_tags(ifd_info);
        
        ifd_infos_.push_back(std::move(ifd_info));
    }
}
```

**nvImageCodec API calls:**
- `nvimgcodecCodeStreamGetCodeStreamInfo()` - Gets number of IFDs
- `nvimgcodecCodeStreamGetSubCodeStream()` - Creates view for each IFD
- `nvimgcodecCodeStreamGetImageInfo()` - Gets dimensions, channels, codec

**Purpose:**
- Queries nvImageCodec for number of IFDs (resolution levels)
- For each IFD:
  - Creates a **sub-code stream** (view into specific IFD)
  - Extracts dimensions, channels, bits per sample, codec
  - Extracts vendor metadata (Aperio/Philips/etc.)
  - Extracts TIFF tags (Software, Model, ImageDescription, etc.)

---

### Step 4: `extract_tiff_tags()` queries individual TIFF tags → `nvimgcodec_tiff_parser.cpp`

**File:** `cpp/plugins/cucim.kit.cuslide2/src/cuslide/nvimgcodec/nvimgcodec_tiff_parser.cpp`  
**Lines:** 769-1156

```cpp
void TiffFileParser::extract_tiff_tags(IfdInfo& ifd_info)
{
    auto& manager = NvImageCodecTiffParserManager::instance();
    
    if (!manager.get_decoder())
    {
        return;
    }
    
    // Map of TIFF tag IDs to names for tags we want to extract
    std::vector<std::pair<uint16_t, std::string>> tiff_tags_to_query = {
        {254, "SUBFILETYPE"},      // Image type classification (0=full, 1=reduced, etc.)
        {256, "IMAGEWIDTH"},
        {257, "IMAGELENGTH"},
        {258, "BITSPERSAMPLE"},
        {259, "COMPRESSION"},      // Critical for codec detection!
        {262, "PHOTOMETRIC"},
        {270, "IMAGEDESCRIPTION"}, // Vendor metadata
        {271, "MAKE"},             // Scanner manufacturer
        {272, "MODEL"},            // Scanner model
        {277, "SAMPLESPERPIXEL"},
        {305, "SOFTWARE"},
        {306, "DATETIME"},
        {322, "TILEWIDTH"},
        {323, "TILELENGTH"},
        {330, "SUBIFD"},           // SubIFD offsets (for OME-TIFF, etc.)
        {339, "SAMPLEFORMAT"},
        {347, "JPEGTABLES"}        // Shared JPEG tables
    };
    
    #ifdef DEBUG
    fmt::print("  📋 Extracting TIFF tags (nvImageCodec 0.7.0 - query by ID)...\n");
    #endif
    
    int extracted_count = 0;
    
    // Query each tag individually by ID (nvImageCodec 0.7.0 API)
    for (const auto& [tag_id, tag_name] : tiff_tags_to_query)
    {
        // Set up metadata request for specific tag
        nvimgcodecMetadata_t metadata{};
        metadata.struct_type = NVIMGCODEC_STRUCTURE_TYPE_METADATA;
        metadata.struct_size = sizeof(nvimgcodecMetadata_t);
        metadata.struct_next = nullptr;
        metadata.kind = NVIMGCODEC_METADATA_KIND_TIFF_TAG;
        metadata.id = tag_id;  // Query specific tag by ID
        metadata.buffer = nullptr;
        metadata.buffer_size = 0;
        
        nvimgcodecMetadata_t* metadata_ptr = &metadata;
        int metadata_count = 1;
        
        // First call: query buffer size
        nvimgcodecStatus_t status = nvimgcodecDecoderGetMetadata(
            manager.get_decoder(),
            ifd_info.sub_code_stream,
            &metadata_ptr,
            &metadata_count
        );
        
        if (status != NVIMGCODEC_STATUS_SUCCESS || metadata.buffer_size == 0)
        {
            continue;  // Tag not present
        }
        
        // Allocate buffer for tag value
        std::vector<uint8_t> buffer(metadata.buffer_size);
        metadata.buffer = buffer.data();
        
        // Second call: retrieve actual value
        status = nvimgcodecDecoderGetMetadata(
            manager.get_decoder(),
            ifd_info.sub_code_stream,
            &metadata_ptr,
            &metadata_count
        );
        
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            continue;
        }
        
        // Convert value based on type and store as typed variant
        TiffTagValue tag_value;
        
        switch (metadata.value_type)
        {
            case NVIMGCODEC_METADATA_VALUE_TYPE_ASCII:
            {
                std::string str_val;
                str_val.assign(reinterpret_cast<const char*>(buffer.data()), metadata.buffer_size);
                while (!str_val.empty() && str_val.back() == '\0')
                    str_val.pop_back();
                if (!str_val.empty())
                {
                    tag_value = std::move(str_val);
                }
                break;
            }
            
            case NVIMGCODEC_METADATA_VALUE_TYPE_SHORT:
                extract_single_value<uint16_t>(buffer, metadata.value_count, tag_value) ||
                extract_value_array<uint16_t>(buffer, metadata.value_count, tag_value);
                break;
            
            case NVIMGCODEC_METADATA_VALUE_TYPE_LONG:
                extract_single_value<uint32_t>(buffer, metadata.value_count, tag_value) ||
                extract_value_array<uint32_t>(buffer, metadata.value_count, tag_value);
                break;
            
            // ... (other types: BYTE, FLOAT, DOUBLE, RATIONAL, etc.)
        }
        
        // Store in IFD's tag map
        if (!std::holds_alternative<std::monostate>(tag_value))
        {
            ifd_info.tiff_tags[tag_name] = std::move(tag_value);
            extracted_count++;
        }
    }
    
    #ifdef DEBUG
    fmt::print("  ✅ Extracted {} TIFF tags using nvImageCodec 0.7.0 API\n", extracted_count);
    #endif
}
```

**nvImageCodec API calls:**
- `nvimgcodecDecoderGetMetadata()` - Query specific TIFF tag by ID

**Purpose:**
- Queries specific TIFF tags by ID using nvImageCodec 0.7.0 API
- Stores tags as typed variants (string, uint16_t, vector, etc.)
- Makes tags accessible via `get_tiff_tag(index, "TagName")`
- Handles all TIFF tag value types (ASCII, SHORT, LONG, RATIONAL, etc.)

---

### Step 5: `TIFF::construct_ifds()` creates IFD objects → `tiff.cpp`

**File:** `cpp/plugins/cucim.kit.cuslide2/src/cuslide/tiff/tiff.cpp`  
**Lines:** 402-485

```cpp
void TIFF::construct_ifds()
{
    PROF_SCOPED_RANGE(PROF_EVENT(tiff_construct_ifds));

    if (!nvimgcodec_parser_ || !nvimgcodec_parser_->is_valid()) {
        throw std::runtime_error("Cannot construct IFDs: nvImageCodec parser not available");
    }

    ifd_offsets_.clear();
    ifds_.clear();

    uint32_t ifd_count = nvimgcodec_parser_->get_ifd_count();
    #ifdef DEBUG
    fmt::print("📋 Constructing {} IFDs from nvImageCodec metadata\n", ifd_count);
    #endif

    ifd_offsets_.reserve(ifd_count);
    ifds_.reserve(ifd_count);

    for (uint32_t ifd_index = 0; ifd_index < ifd_count; ++ifd_index) {
        try {
            // Get IFD metadata from TiffFileParser
            const auto& ifd_info = nvimgcodec_parser_->get_ifd(ifd_index);

            // Use IFD index as pseudo-offset
            ifd_offsets_.push_back(ifd_index);

            // Create IFD from nvImageCodec metadata using NEW constructor
            auto ifd = std::make_shared<cuslide::tiff::IFD>(this, ifd_index, ifd_info);
            ifds_.emplace_back(std::move(ifd));

            #ifdef DEBUG
            fmt::print("  ✅ IFD[{}]: {}x{}, {} channels, codec: {}\n",
                      ifd_index, ifd_info.width, ifd_info.height,
                      ifd_info.num_channels, ifd_info.codec);
            #endif

        } catch (const std::exception& e) {
            #ifdef DEBUG
            fmt::print("  ⚠️  Failed to create IFD[{}]: {}\n", ifd_index, e.what());
            #endif
        }
    }

    if (ifds_.empty()) {
        throw std::runtime_error("No valid IFDs found in TIFF file");
    }

    // Initialize level-to-IFD mapping
    level_to_ifd_idx_.clear();
    level_to_ifd_idx_.reserve(ifds_.size());
    for (size_t index = 0; index < ifds_.size(); ++index) {
        level_to_ifd_idx_.emplace_back(index);
    }

    // Detect vendor format and classify IFDs
    resolve_vendor_format();  // ← Detects Aperio/Philips/etc.

    // Sort resolution levels by size (largest first)
    std::sort(level_to_ifd_idx_.begin(), level_to_ifd_idx_.end(),
             [this](const size_t& a, const size_t& b) {
        uint32_t width_a = this->ifds_[a]->width();
        uint32_t width_b = this->ifds_[b]->width();
        if (width_a != width_b) {
            return width_a > width_b;
        }
        return this->ifds_[a]->height() > this->ifds_[b]->height();
    });
}
```

**Purpose:**
- Retrieves IFD metadata from `TiffFileParser`
- Creates one `IFD` object per resolution level
- Detects vendor format (Aperio SVS, Philips TIFF, etc.)
- Sorts levels by size (largest first)

---

### Step 6: `IFD` constructor → `ifd.cpp`

**File:** `cpp/plugins/cucim.kit.cuslide2/src/cuslide/tiff/ifd.cpp`  
**Lines:** 90-198

```cpp
IFD::IFD(TIFF* tiff, uint16_t index, const cuslide2::nvimgcodec::IfdInfo& ifd_info)
    : tiff_(tiff), ifd_index_(index), ifd_offset_(index)
{
    PROF_SCOPED_RANGE(PROF_EVENT(ifd_ifd));

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

    // Parse codec string to compression enum
    codec_name_ = ifd_info.codec;
    compression_ = parse_codec_to_compression(codec_name_);

    // Get ImageDescription from nvImageCodec
    image_description_ = ifd_info.image_description;

    // Extract TIFF tags from TiffFileParser
    if (tiff->nvimgcodec_parser_) {
        // Software and Model tags
        software_ = tiff->nvimgcodec_parser_->get_tiff_tag(index, "Software");
        model_ = tiff->nvimgcodec_parser_->get_tiff_tag(index, "Model");

        // SUBFILETYPE for IFD classification
        int subfile_type = tiff->nvimgcodec_parser_->get_subfile_type(index);
        if (subfile_type >= 0) {
            subfile_type_ = static_cast<uint64_t>(subfile_type);
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
                tile_width_ = 0;
                tile_height_ = 0;
            }
        } else {
            // Not tiled - treat as single strip
            tile_width_ = 0;
            tile_height_ = 0;
        }
    }

    // Set format defaults
    planar_config_ = PLANARCONFIG_CONTIG;  // nvImageCodec outputs interleaved
    photometric_ = PHOTOMETRIC_RGB;
    predictor_ = 1;  // No predictor

    // Calculate hash for caching
    hash_value_ = cucim::codec::splitmix64(index);

    // Store reference to nvImageCodec sub-stream
    nvimgcodec_sub_stream_ = ifd_info.sub_code_stream;

    #ifdef DEBUG
    fmt::print("✅ IFD[{}] initialization complete\n", index);
    #endif
}
```

**Methods called:**
- `tiff->nvimgcodec_parser_->get_tiff_tag()` - Retrieves individual TIFF tags
- `tiff->nvimgcodec_parser_->get_subfile_type()` - Gets SUBFILETYPE tag

**Purpose:**
- Copies metadata from `IfdInfo` struct to IFD member variables
- Queries additional TIFF tags (Software, Model, TileWidth, etc.)
- Stores reference to nvImageCodec sub-code stream for later decoding
- Parses codec string to compression enum

---

## Phase 2: Reading Image Data

### Step 7: User calls `TIFF::read()` → `tiff.cpp`

**File:** `cpp/plugins/cucim.kit.cuslide2/src/cuslide/tiff/tiff.cpp`  
**Lines:** 913-965

```cpp
bool TIFF::read(const cucim::io::format::ImageMetadataDesc* metadata,
                const cucim::io::format::ImageReaderRegionRequestDesc* request,
                cucim::io::format::ImageDataDesc* out_image_data,
                cucim::io::format::ImageMetadataDesc* out_metadata)
{
    PROF_SCOPED_RANGE(PROF_EVENT(tiff_read));
    
    if (request->associated_image_name)
    {
        return read_associated_image(metadata, request, out_image_data, out_metadata);
    }

    const int32_t ndim = request->size_ndim;
    const uint64_t location_len = request->location_len;

    if (request->level >= level_to_ifd_idx_.size())
    {
        throw std::invalid_argument(fmt::format(
            "Invalid level ({}) in the request! (Should be < {})", request->level, level_to_ifd_idx_.size()));
    }
    
    auto main_ifd = ifds_[level_to_ifd_idx_[0]];
    auto ifd = ifds_[level_to_ifd_idx_[request->level]];
    auto original_img_width = main_ifd->width();
    auto original_img_height = main_ifd->height();

    // Validate request size
    for (int32_t i = 0; i < ndim; ++i)
    {
        if (request->size[i] <= 0)
        {
            throw std::invalid_argument(
                fmt::format("Invalid size ({}) in the request! (Should be > 0)", request->size[i]));
        }
    }

    float downsample_factor = metadata->resolution_info.level_downsamples[request->level];

    // Change request based on downsample factor
    // (normalized value at level-0 -> real location at the requested level)
    for (int64_t i = ndim * location_len - 1; i >= 0; --i)
    {
        request->location[i] /= downsample_factor;
    }
    
    // Delegate to IFD
    return ifd->read(this, metadata, request, out_image_data);  // ← Next step
}
```

**Purpose:**
- Routes read request to appropriate IFD based on requested level
- Adjusts coordinates based on downsample factor
- Validates request parameters
- Delegates actual reading to IFD

---

### Step 8: `IFD::read()` processes the request → `ifd.cpp`

**File:** `cpp/plugins/cucim.kit.cuslide2/src/cuslide/tiff/ifd.cpp`  
**Lines:** 211-330

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

    // Fast path: Use nvImageCodec ROI decoding when available
    // ROI decoding is supported in nvImageCodec v0.6.0+ for JPEG2000
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

        // Output buffer - decode function will allocate
        uint8_t* output_buffer = nullptr;
        DLTensor* out_buf = request->buf;
        bool is_buf_available = out_buf && out_buf->data;

        if (is_buf_available)
        {
            // User provided pre-allocated buffer
            output_buffer = static_cast<uint8_t*>(out_buf->data);
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
            if (!is_buf_available && output_buffer)
            {
                if (out_device.type() == cucim::io::DeviceType::kCUDA)
                {
                    cudaFree(output_buffer);
                }
                else
                {
                    cudaFreeHost(output_buffer);
                }
            }

            throw std::runtime_error(fmt::format(
                "Failed to decode IFD[{}] with nvImageCodec. ROI: ({},{}) {}x{}",
                ifd_index_, sx, sy, w, h));
        }
    }

    // If we reach here, nvImageCodec is not available or unsupported request type
    throw std::runtime_error(fmt::format(
        "IFD[{}]: This library requires nvImageCodec for image decoding. "
        "Multi-location/batch requests not yet supported.", ifd_index_));
}
```

**Methods called:**
- `tiff->nvimgcodec_parser_->get_ifd()` - Gets IFD metadata
- `tiff->nvimgcodec_parser_->get_main_code_stream()` - Gets main code stream
- `decode_ifd_region_nvimgcodec()` - **Performs actual decoding** ← Next step

**Purpose:**
- Checks if nvImageCodec ROI decoding is available
- Extracts region coordinates and size from request
- Retrieves IFD info and main code stream from parser
- Delegates to nvImageCodec decoder function
- Sets up output buffer metadata on success

---

### Step 9: `decode_ifd_region_nvimgcodec()` performs actual decode → `nvimgcodec_decoder.cpp`

**File:** `cpp/plugins/cucim.kit.cuslide2/src/cuslide/nvimgcodec/nvimgcodec_decoder.cpp`  
**Lines:** 153-423

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
        #ifdef DEBUG
        fmt::print("❌ Invalid main_code_stream\n");
        #endif
        return false;
    }
    
    #ifdef DEBUG
    fmt::print("🚀 Decoding IFD[{}] region: [{},{}] {}x{}, codec: {}\n",
              ifd_info.index, x, y, width, height, ifd_info.codec);
    #endif
    
    try
    {
        // CRITICAL: Must use the same manager that created main_code_stream!
        auto& manager = NvImageCodecTiffParserManager::instance();
        if (!manager.is_available())
        {
            return false;
        }
        
        // Select decoder based on target device
        std::string device_str = std::string(out_device);
        bool target_is_cpu = (device_str.find("cpu") != std::string::npos);
        
        // Check if ROI is out of bounds
        bool roi_out_of_bounds = (x + width > ifd_info.width) || (y + height > ifd_info.height);
        if (target_is_cpu && roi_out_of_bounds)
        {
            target_is_cpu = false;  // Force hybrid decoder for out-of-bounds ROI
            #ifdef DEBUG
            fmt::print("  ⚠️  ROI out of bounds, using hybrid decoder\n");
            #endif
        }
        
        nvimgcodecDecoder_t decoder;
        if (target_is_cpu && manager.has_cpu_decoder())
        {
            decoder = manager.get_cpu_decoder();
        }
        else
        {
            decoder = manager.get_decoder();
        }
        
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
        
        // Get sub-code stream for this ROI (RAII managed)
        nvimgcodecCodeStream_t roi_stream_raw = nullptr;
        nvimgcodecStatus_t status = nvimgcodecCodeStreamGetSubCodeStream(
            main_code_stream,
            &roi_stream_raw,
            &view
        );
        
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            return false;
        }
        UniqueCodeStream roi_stream(roi_stream_raw);
        
        // Step 2: Determine buffer kind based on target device
        int device_count = 0;
        cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
        bool gpu_available = (cuda_err == cudaSuccess && device_count > 0);
        
        nvimgcodecImageBufferKind_t buffer_kind;
        if (target_is_cpu)
        {
            buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
        }
        else if (gpu_available)
        {
            buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
        }
        else
        {
            buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
        }
        
        // Step 3: Prepare output image info for the region
        nvimgcodecImageInfo_t output_image_info{};
        output_image_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
        output_image_info.struct_size = sizeof(nvimgcodecImageInfo_t);
        output_image_info.struct_next = nullptr;
        
        // Use interleaved RGB format
        output_image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
        output_image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        output_image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
        output_image_info.num_planes = 1;
        output_image_info.buffer_kind = buffer_kind;
        
        // Calculate buffer requirements for the region
        uint32_t num_channels = 3;  // RGB
        size_t row_stride = width * num_channels;
        size_t buffer_size = row_stride * height;
        
        output_image_info.plane_info[0].height = height;
        output_image_info.plane_info[0].width = width;
        output_image_info.plane_info[0].num_channels = num_channels;
        output_image_info.plane_info[0].row_stride = row_stride;
        output_image_info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        output_image_info.cuda_stream = 0;
        
        // Step 4: Allocate output buffer (RAII managed)
        bool use_device_memory = (buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE);
        DecodeBuffer decode_buffer;
        if (!decode_buffer.allocate(buffer_size, use_device_memory))
        {
            return false;
        }
        
        output_image_info.buffer = decode_buffer.get();
        
        // Step 5: Create image object (RAII managed)
        nvimgcodecImage_t image_raw = nullptr;
        status = nvimgcodecImageCreate(
            manager.get_instance(),
            &image_raw,
            &output_image_info
        );
        
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            return false;
        }
        UniqueImage image(image_raw);
        
        // Step 6: Prepare decode parameters
        nvimgcodecDecodeParams_t decode_params{};
        decode_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS;
        decode_params.struct_size = sizeof(nvimgcodecDecodeParams_t);
        decode_params.struct_next = nullptr;
        decode_params.apply_exif_orientation = 1;
        
        // Step 7: Schedule decoding (RAII managed)
        nvimgcodecCodeStream_t roi_stream_ptr = roi_stream.get();
        nvimgcodecImage_t image_ptr = image.get();
        nvimgcodecFuture_t decode_future_raw = nullptr;
        status = nvimgcodecDecoderDecode(decoder,
                                        &roi_stream_ptr,
                                        &image_ptr,
                                        1,
                                        &decode_params,
                                        &decode_future_raw);
        
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            return false;
        }
        UniqueFuture decode_future(decode_future_raw);
        
        // Step 8: Wait for completion
        nvimgcodecProcessingStatus_t decode_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
        size_t status_size = 1;
        status = nvimgcodecFutureGetProcessingStatus(decode_future.get(), &decode_status, &status_size);
        
        if (status != NVIMGCODEC_STATUS_SUCCESS)
        {
            return false;
        }
        
        if (use_device_memory)
        {
            cudaDeviceSynchronize();
        }
        
        // Step 9: Check decode status
        if (decode_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
        {
            return false;
        }
        
        #ifdef DEBUG
        fmt::print("✅ Successfully decoded IFD[{}] region\n", ifd_info.index);
        #endif
        
        // Success: release buffer ownership to caller
        *output_buffer = reinterpret_cast<uint8_t*>(decode_buffer.release());
        return true;
    }
    catch (const std::exception& e)
    {
        #ifdef DEBUG
        fmt::print("❌ Exception in ROI decoding: {}\n", e.what());
        #endif
        return false;
    }
}
```

**nvImageCodec API calls:**
- `nvimgcodecCodeStreamGetSubCodeStream()` - Creates ROI view
- `nvimgcodecImageCreate()` - Creates output image descriptor
- `nvimgcodecDecoderDecode()` - Schedules decode operation
- `nvimgcodecFutureGetProcessingStatus()` - Waits for completion

**Purpose:**
- Creates ROI-specific sub-code stream with region bounds
- Allocates output buffer (CPU or GPU based on request)
- Creates nvImageCodec image descriptor
- Schedules and waits for decode operation
- Returns decoded buffer to caller

---

## Key Method Connections

### **tiff.cpp → nvimgcodec_tiff_parser.cpp**

| `TIFF` method | → | `TiffFileParser` method |
|---------------|---|-------------------------|
| `TIFF::TIFF()` | → | `TiffFileParser::TiffFileParser()` |
| `TIFF::construct_ifds()` | → | `TiffFileParser::get_ifd()` |
| `TIFF::construct_ifds()` | → | `TiffFileParser::get_ifd_count()` |

### **tiff.cpp → ifd.cpp**

| `TIFF` method | → | `IFD` method |
|---------------|---|--------------|
| `TIFF::construct_ifds()` | → | `IFD::IFD(TIFF*, uint16_t, IfdInfo&)` |
| `TIFF::read()` | → | `IFD::read()` |

### **ifd.cpp → nvimgcodec_tiff_parser.cpp**

| `IFD` method | → | `TiffFileParser` method |
|--------------|---|-------------------------|
| `IFD::IFD()` | → | `TiffFileParser::get_tiff_tag()` (multiple tags) |
| `IFD::read()` | → | `TiffFileParser::get_ifd()` |
| `IFD::read()` | → | `TiffFileParser::get_main_code_stream()` |

### **ifd.cpp → nvimgcodec_decoder.cpp**

| `IFD` method | → | Function |
|--------------|---|----------|
| `IFD::read()` | → | `decode_ifd_region_nvimgcodec()` |

### **Singleton Manager Access**

All components use: `NvImageCodecTiffParserManager::instance()`
- Manages global nvImageCodec instance and decoders
- Used by both `TiffFileParser` and `decode_ifd_region_nvimgcodec()`
- Ensures all components use the same nvImageCodec instance

---

## Data Flow Diagram

```
User API Call
     ↓
┌────────────────────────────────────────────────────────────────┐
│ TIFF::open(file_path)                                          │
│ • Opens file descriptor                                        │
│ • Creates file handle                                          │
└────────────┬───────────────────────────────────────────────────┘
             ↓
┌────────────────────────────────────────────────────────────────┐
│ TiffFileParser::TiffFileParser(file_path)                      │
│ • nvimgcodecCodeStreamCreateFromFile()                         │
│ • Creates main_code_stream_                                    │
└────────────┬───────────────────────────────────────────────────┘
             ↓
┌────────────────────────────────────────────────────────────────┐
│ TiffFileParser::parse_tiff_structure()                         │
│ • nvimgcodecCodeStreamGetCodeStreamInfo()                      │
│ • Get number of IFDs                                           │
│ • For each IFD:                                                │
│   - nvimgcodecCodeStreamGetSubCodeStream()                     │
│   - nvimgcodecCodeStreamGetImageInfo()                         │
│   - extract_ifd_metadata()                                     │
│   - extract_tiff_tags()                                        │
│ • Stores IfdInfo[] array                                       │
└────────────┬───────────────────────────────────────────────────┘
             ↓
┌────────────────────────────────────────────────────────────────┐
│ TIFF::construct_ifds()                                         │
│ • For each IFD:                                                │
│   - Get IfdInfo from TiffFileParser                            │
│   - Create IFD object                                          │
│ • resolve_vendor_format()                                      │
│ • Sort levels by size                                          │
└────────────┬───────────────────────────────────────────────────┘
             ↓
┌────────────────────────────────────────────────────────────────┐
│ IFD::IFD(tiff, index, ifd_info)                                │
│ • Copy metadata from IfdInfo                                   │
│ • Query additional tags via TiffFileParser::get_tiff_tag()     │
│ • Store nvimgcodec_sub_stream_ reference                       │
└────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════
         FILE IS NOW OPEN - READY FOR READ REQUESTS
═══════════════════════════════════════════════════════════════════

┌────────────────────────────────────────────────────────────────┐
│ User calls read(level, x, y, width, height)                    │
└────────────┬───────────────────────────────────────────────────┘
             ↓
┌────────────────────────────────────────────────────────────────┐
│ TIFF::read(metadata, request, out_image_data)                  │
│ • Select IFD based on requested level                          │
│ • Adjust coordinates for downsample factor                     │
│ • Validate request parameters                                  │
└────────────┬───────────────────────────────────────────────────┘
             ↓
┌────────────────────────────────────────────────────────────────┐
│ IFD::read(tiff, metadata, request, out_image_data)             │
│ • Extract region coordinates (sx, sy, w, h)                    │
│ • Get IFD info from TiffFileParser                             │
│ • Get main_code_stream from TiffFileParser                     │
└────────────┬───────────────────────────────────────────────────┘
             ↓
┌────────────────────────────────────────────────────────────────┐
│ decode_ifd_region_nvimgcodec(ifd_info, main_code_stream,       │
│                              x, y, width, height)              │
│ • Create ROI view:                                             │
│   - nvimgcodecCodeStreamGetSubCodeStream() with region         │
│ • Allocate buffer (CPU or GPU)                                 │
│ • Create image descriptor:                                     │
│   - nvimgcodecImageCreate()                                    │
│ • Schedule decode:                                             │
│   - nvimgcodecDecoderDecode()                                  │
│ • Wait for completion:                                         │
│   - nvimgcodecFutureGetProcessingStatus()                      │
│ • Return decoded buffer                                        │
└────────────┬───────────────────────────────────────────────────┘
             ↓
┌────────────────────────────────────────────────────────────────┐
│ IFD::read() receives decoded buffer                            │
│ • Set up out_image_data metadata                               │
│ • Return success                                               │
└────────────┬───────────────────────────────────────────────────┘
             ↓
┌────────────────────────────────────────────────────────────────┐
│ TIFF::read() returns success                                   │
└────────────┬───────────────────────────────────────────────────┘
             ↓
       User receives decoded image data
```

---

## nvImageCodec API Calls

### Initialization Phase

| Component | API Call | Purpose |
|-----------|----------|---------|
| `NvImageCodecTiffParserManager` | `nvimgcodecInstanceCreate()` | Create singleton instance |
| `NvImageCodecTiffParserManager` | `nvimgcodecDecoderCreate()` | Create hybrid decoder |
| `NvImageCodecTiffParserManager` | `nvimgcodecDecoderCreate()` | Create CPU-only decoder |

### Parsing Phase

| Component | API Call | Purpose |
|-----------|----------|---------|
| `TiffFileParser` | `nvimgcodecCodeStreamCreateFromFile()` | Open TIFF file |
| `TiffFileParser` | `nvimgcodecCodeStreamGetCodeStreamInfo()` | Get IFD count |
| `TiffFileParser` | `nvimgcodecCodeStreamGetSubCodeStream()` | Create IFD view |
| `TiffFileParser` | `nvimgcodecCodeStreamGetImageInfo()` | Get dimensions/codec |
| `TiffFileParser` | `nvimgcodecDecoderGetMetadata()` | Query TIFF tags |
| `TiffFileParser` | `nvimgcodecDecoderGetMetadata()` | Get vendor metadata |

### Decoding Phase

| Component | API Call | Purpose |
|-----------|----------|---------|
| `decode_ifd_region_nvimgcodec` | `nvimgcodecCodeStreamGetSubCodeStream()` | Create ROI view |
| `decode_ifd_region_nvimgcodec` | `nvimgcodecImageCreate()` | Create output image |
| `decode_ifd_region_nvimgcodec` | `nvimgcodecDecoderDecode()` | Schedule decode |
| `decode_ifd_region_nvimgcodec` | `nvimgcodecFutureGetProcessingStatus()` | Wait for result |

### Cleanup Phase

| Component | API Call | Purpose |
|-----------|----------|---------|
| `TiffFileParser` | `nvimgcodecCodeStreamDestroy()` | Destroy sub-streams |
| `TiffFileParser` | `nvimgcodecCodeStreamDestroy()` | Destroy main stream |
| `decode_ifd_region_nvimgcodec` | `nvimgcodecImageDestroy()` | Cleanup image (RAII) |
| `decode_ifd_region_nvimgcodec` | `nvimgcodecFutureDestroy()` | Cleanup future (RAII) |
| `NvImageCodecTiffParserManager` | `nvimgcodecDecoderDestroy()` | Destroy decoders |
| `NvImageCodecTiffParserManager` | `nvimgcodecInstanceDestroy()` | Destroy instance |

---

## Architecture Principles

### 1. **Layered Design**
- **tiff.cpp**: High-level orchestration layer
- **ifd.cpp**: Per-resolution-level management layer
- **nvimgcodec_tiff_parser.cpp**: Metadata extraction layer
- **nvimgcodec_decoder.cpp**: Image decoding layer

### 2. **Separation of Concerns**
- All nvImageCodec API calls isolated to `nvimgcodec_*` files
- TIFF-level logic (vendor detection, metadata) in `tiff.cpp`
- IFD-level logic (tile management, region reading) in `ifd.cpp`

### 3. **Singleton Pattern**
- `NvImageCodecTiffParserManager::instance()` provides shared nvImageCodec instance
- Ensures all components use the same instance/decoders
- Prevents segfaults from mixed instances

### 4. **RAII Resource Management**
- All nvImageCodec resources use RAII wrappers:
  - `UniqueCodeStream` for code streams
  - `UniqueImage` for image objects
  - `UniqueFuture` for decode futures
  - `DecodeBuffer` for CPU/GPU buffers

### 5. **Data Flow**
- Metadata flows: `nvImageCodec → TiffFileParser → IfdInfo → IFD`
- Decode requests flow: `TIFF → IFD → decode_ifd_region_nvimgcodec`
- Decoded data flows: `nvImageCodec → decode buffer → IFD → TIFF → User`

---

## Summary

This architecture provides:
- **Clean abstraction** - nvImageCodec details isolated from TIFF logic
- **Extensibility** - Easy to add new vendor formats or codecs
- **Performance** - Direct ROI decoding without tile-based fallback
- **Safety** - RAII management prevents resource leaks
- **Maintainability** - Clear separation of responsibilities

The flow ensures that:
1. Files are parsed once during `TIFF::open()`
2. Metadata is cached in `IfdInfo` structs
3. Decoding uses the most efficient path (ROI decode when available)
4. All resources are properly cleaned up via RAII

