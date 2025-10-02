# TIFF Tag Variant System - Usage Examples

This document shows **actual usage** of the variant-based TIFF tag storage system in the cuCIM codebase.

---

## Overview

The variant system provides a clean API where:
- **Storage**: Tags are stored as typed `std::variant` in `IfdInfo::tiff_tags`
- **Retrieval**: `get_tiff_tag(ifd_index, tag_name)` returns `std::string` for easy consumption
- **Type safety**: Internal conversions handle all TIFF types automatically

---

## Usage Location: `IFD` Constructor (`ifd.cpp`)

The primary usage is in the `IFD` class constructor, where TIFF tags are extracted during IFD initialization.

### File: `cpp/plugins/cucim.kit.cuslide2/src/cuslide/tiff/ifd.cpp`

**Lines 122-170**: Extracting various TIFF tags during IFD construction

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

---

## Usage Pattern Breakdown

### 1. **String Tags** (Software, Model, ImageDescription)

These tags are stored as `std::string` in the variant and returned directly:

```cpp
software_ = tiff->nvimgcodec_parser_->get_tiff_tag(index, "Software");
model_ = tiff->nvimgcodec_parser_->get_tiff_tag(index, "Model");
```

**Behind the scenes:**
- Variant stores: `TiffTagValue = std::string("Aperio Image Library v1.0.0")`
- `get_tiff_tag()` returns: `"Aperio Image Library v1.0.0"`

---

### 2. **Numeric Tags** (TileWidth, TileLength)

Numeric tags are stored as `uint32_t` (or appropriate type) but converted to string for the API:

```cpp
std::string tile_w_str = tiff->nvimgcodec_parser_->get_tiff_tag(index, "TileWidth");
std::string tile_h_str = tiff->nvimgcodec_parser_->get_tiff_tag(index, "TileLength");

if (!tile_w_str.empty() && !tile_h_str.empty()) {
    try {
        tile_width_ = std::stoul(tile_w_str);   // Convert back to uint32_t
        tile_height_ = std::stoul(tile_h_str);
    } catch (...) {
        // Handle parse error
    }
}
```

**Behind the scenes:**
- Variant stores: `TiffTagValue = uint32_t(256)`
- `get_tiff_tag()` returns: `"256"` (via `tiff_tag_value_to_string()`)
- User converts: `std::stoul("256")` → `256`

---

### 3. **Binary Tags** (JPEGTables)

Binary data tags are stored as `std::vector<uint8_t>` and summarized as strings:

```cpp
std::string jpeg_tables = tiff->nvimgcodec_parser_->get_tiff_tag(index, "JPEGTables");
if (!jpeg_tables.empty()) {
    // JPEGTables exist (abbreviated JPEG)
}
```

**Behind the scenes:**
- Variant stores: `TiffTagValue = std::vector<uint8_t>{...binary data...}`
- `get_tiff_tag()` returns: `"[514 bytes]"` (summary format)
- User checks: If not empty, binary data exists

---

### 4. **Empty/Missing Tags**

Tags that don't exist return empty strings:

```cpp
std::string tag_value = tiff->nvimgcodec_parser_->get_tiff_tag(index, "NonExistentTag");
// tag_value == "" (empty string)
```

**Behind the scenes:**
- Tag not found in `tiff_tags` map
- `get_tiff_tag()` returns: `""` (empty string)

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ nvImageCodec Metadata API                                   │
│ (Raw bytes + type information)                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ TiffFileParser::extract_tiff_tags()                         │
│ - Queries nvImageCodec for tag metadata                     │
│ - Converts raw bytes to typed variant                       │
│ - Stores in ifd_info.tiff_tags map                          │
│                                                              │
│   Example:                                                   │
│   Tag "TileWidth" (value_type=LONG, value_count=1)          │
│   Raw bytes: [0x00, 0x01, 0x00, 0x00] (256 in little-endian)│
│   ↓                                                          │
│   tiff_tags["TileWidth"] = uint32_t(256)                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ TiffFileParser::get_tiff_tag(index, "TileWidth")            │
│ - Looks up tag in tiff_tags map                             │
│ - Converts variant to string using visitor pattern          │
│                                                              │
│   tiff_tags["TileWidth"] = uint32_t(256)                    │
│   ↓                                                          │
│   tiff_tag_value_to_string() → "256"                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ IFD Constructor (ifd.cpp)                                   │
│ - Receives string: "256"                                    │
│ - Converts to appropriate type: std::stoul("256") → 256     │
│ - Stores in IFD member: tile_width_ = 256                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Tag Types in Use

### Currently Extracted Tags

| Tag Name | TIFF Type | Variant Storage | Use Case |
|----------|-----------|-----------------|----------|
| `Software` | ASCII | `std::string` | Vendor identification |
| `Model` | ASCII | `std::string` | Scanner model |
| `ImageDescription` | ASCII | `std::string` | Metadata (Aperio/Philips) |
| `TileWidth` | LONG | `uint32_t` | Tile dimensions |
| `TileLength` | LONG | `uint32_t` | Tile dimensions |
| `JPEGTables` | UNDEFINED | `std::vector<uint8_t>` | Abbreviated JPEG detection |
| `SUBFILETYPE` | LONG | `uint32_t` | IFD classification |
| `Compression` | SHORT | `uint16_t` | Compression method |
| `BitsPerSample` | SHORT (array) | `std::vector<uint16_t>` | Channel bit depths |

---

## Advanced Usage: Direct Variant Access

If you need type-safe direct access (avoiding string conversion), you can access the variant directly:

### Example: Accessing SubIFD Offsets

```cpp
// Direct access to tiff_tags (requires access to IfdInfo)
const auto& ifd_info = tiff->nvimgcodec_parser_->get_ifd(index);
auto it = ifd_info.tiff_tags.find("SubIFD");

if (it != ifd_info.tiff_tags.end()) {
    // Check if it's an array of uint64_t
    if (std::holds_alternative<std::vector<uint64_t>>(it->second)) {
        const auto& subifd_offsets = std::get<std::vector<uint64_t>>(it->second);
        
        for (uint64_t offset : subifd_offsets) {
            // Process each SubIFD offset directly
            fmt::print("SubIFD at offset: {}\n", offset);
        }
    }
}
```

### Example: Type-Safe Visitor

```cpp
const auto& tag_value = ifd_info.tiff_tags["Compression"];

std::visit([](const auto& v) {
    using T = std::decay_t<decltype(v)>;
    
    if constexpr (std::is_same_v<T, uint16_t>) {
        // Process compression value
        fmt::print("Compression: {}\n", v);
        
        if (v == 7) {
            fmt::print("  → JPEG compression\n");
        } else if (v == 33003 || v == 33005) {
            fmt::print("  → Aperio JPEG2000\n");
        }
    }
}, tag_value);
```

---

## Error Handling Patterns

### Pattern 1: Check for Empty String

```cpp
std::string tile_w = tiff->nvimgcodec_parser_->get_tiff_tag(index, "TileWidth");

if (!tile_w.empty()) {
    // Tag exists, safe to process
    tile_width_ = std::stoul(tile_w);
} else {
    // Tag missing or unset
    tile_width_ = 0;  // Use default
}
```

### Pattern 2: Try-Catch for Conversion

```cpp
try {
    tile_width_ = std::stoul(tile_w_str);
    tile_height_ = std::stoul(tile_h_str);
} catch (...) {
    // Conversion failed (invalid format)
    tile_width_ = 0;
    tile_height_ = 0;
}
```

### Pattern 3: Direct Variant Check

```cpp
const auto& ifd_info = parser->get_ifd(index);
auto it = ifd_info.tiff_tags.find("TileWidth");

if (it != ifd_info.tiff_tags.end() && 
    !std::holds_alternative<std::monostate>(it->second)) {
    // Tag exists and has a value
    std::string value = tiff_tag_value_to_string(it->second);
}
```

---

## Benefits in Practice

### Before (String-Based Parsing)

Hypothetical old approach:
```cpp
// Fragile string parsing
std::string desc = get_image_description();
size_t pos = desc.find("TileWidth=");
if (pos != std::string::npos) {
    std::string width_str = desc.substr(pos + 10, ...);
    tile_width_ = std::stoi(width_str);  // Brittle!
}
```

### After (Variant-Based)

Current clean approach:
```cpp
// Direct tag access with type safety
std::string tile_w_str = tiff->nvimgcodec_parser_->get_tiff_tag(index, "TileWidth");
if (!tile_w_str.empty()) {
    tile_width_ = std::stoul(tile_w_str);  // Clean!
}
```

**Advantages:**
- ✅ No manual string parsing
- ✅ Type-safe storage (compiler-enforced)
- ✅ Clean error handling (empty string = missing tag)
- ✅ Extensible (new tags automatically supported)
- ✅ Efficient (no redundant conversions)

---

## Future Enhancements

### Potential Additional Tags to Extract

```cpp
// Resolution information
std::string x_res = parser->get_tiff_tag(index, "XResolution");  // RATIONAL
std::string y_res = parser->get_tiff_tag(index, "YResolution");  // RATIONAL
std::string res_unit = parser->get_tiff_tag(index, "ResolutionUnit");  // SHORT

// Color space information
std::string photometric = parser->get_tiff_tag(index, "PhotometricInterpretation");
std::string samples_per_pixel = parser->get_tiff_tag(index, "SamplesPerPixel");

// Strip information (for non-tiled images)
std::string rows_per_strip = parser->get_tiff_tag(index, "RowsPerStrip");
```

### Type-Aware Helper Functions

You could add convenience functions for common conversions:

```cpp
// In TiffFileParser class
template<typename T>
std::optional<T> get_tiff_tag_as(uint32_t ifd_index, const std::string& tag_name) const
{
    if (ifd_index >= ifd_infos_.size())
        return std::nullopt;
    
    auto it = ifd_infos_[ifd_index].tiff_tags.find(tag_name);
    if (it != ifd_infos_[ifd_index].tiff_tags.end()) {
        if (std::holds_alternative<T>(it->second)) {
            return std::get<T>(it->second);
        }
    }
    return std::nullopt;
}

// Usage:
auto tile_width = parser->get_tiff_tag_as<uint32_t>(index, "TileWidth");
if (tile_width.has_value()) {
    tile_width_ = tile_width.value();
}
```

---

## Summary

The variant-based TIFF tag system provides a **clean separation of concerns**:

1. **TiffFileParser** handles:
   - nvImageCodec API interaction
   - Type conversion (bytes → variant)
   - Storage in typed map

2. **Consumers (IFD, TIFF)** handle:
   - Simple string-based retrieval via `get_tiff_tag()`
   - Type conversion for specific use cases (string → uint32_t, etc.)
   - Business logic

This design keeps the complexity contained in the parser while providing a simple, robust API for the rest of the codebase.

