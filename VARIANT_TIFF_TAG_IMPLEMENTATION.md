# Variant-Based TIFF Tag Storage Implementation

## Overview

This document describes the implementation of a **fully type-safe, extensible variant system** for storing TIFF tag values extracted from the nvImageCodec metadata API. This design replaces fragile string-based parsing with strongly-typed storage, providing type safety, memory efficiency, and extensibility.

## Architecture

### 1. Type Definition: `TiffTagValue` Variant

A comprehensive variant type supporting **all TIFF data types** defined in `nvimgcodec_tiff_parser.h`:

```cpp
using TiffTagValue = std::variant<
    std::monostate,             // Empty/unset state
    std::string,                // ASCII strings
    int8_t,                     // SBYTE
    uint8_t,                    // BYTE
    int16_t,                    // SSHORT
    uint16_t,                   // SHORT
    int32_t,                    // SLONG
    uint32_t,                   // LONG
    int64_t,                    // SLONG8
    uint64_t,                   // LONG8/IFD8
    float,                      // FLOAT
    double,                     // DOUBLE
    std::vector<uint8_t>,       // Binary data (JPEGTables, UNDEFINED)
    std::vector<uint16_t>,      // Arrays of SHORT (BitsPerSample, etc.)
    std::vector<uint32_t>,      // Arrays of LONG (SubIFD offsets, etc.)
    std::vector<uint64_t>,      // Arrays of LONG8 (BigTIFF offsets)
    std::vector<float>,         // Arrays of FLOAT values
    std::vector<double>         // Arrays of DOUBLE values
>;
```

**Key Features:**
- **17 different types** covering all TIFF tag value types
- **Scalar types**: All signed/unsigned integers (8/16/32/64-bit), floats, doubles
- **Vector types**: For multi-value tags (arrays)
- **Binary data**: `std::vector<uint8_t>` for JPEGTables, UNDEFINED types
- **Empty state**: `std::monostate` for tags not found or extraction failures

---

## 2. Storage Container

Each `IfdInfo` structure stores TIFF tags in an unordered map:

```cpp
struct IfdInfo {
    // ... other fields ...
    
    // nvImageCodec 0.7.0: Individual TIFF tag storage with typed values
    // tag_name -> TiffTagValue (variant with typed storage)
    std::unordered_map<std::string, TiffTagValue> tiff_tags;
};
```

- **Key**: Tag name as string (e.g., `"SUBFILETYPE"`, `"ImageDescription"`, `"TileWidth"`)
- **Value**: `TiffTagValue` variant holding the typed data

---

## 3. Extraction Logic: Type Conversion from nvImageCodec

### Main Extraction Function: `extract_tiff_tags()`

Located in `nvimgcodec_tiff_parser.cpp`, this function:

1. Queries nvImageCodec for available TIFF tags using `NVIMGCODEC_METADATA_KIND_TIFF_TAG`
2. Allocates buffers based on reported sizes
3. Retrieves tag metadata with type information
4. Converts raw bytes to typed variant values

### Type Mapping Switch Statement

The core conversion logic maps nvImageCodec metadata types to C++ variant types:

```cpp
switch (metadata.value_type)
{
    case NVIMGCODEC_METADATA_VALUE_TYPE_ASCII:
        // ASCII string - remove trailing nulls
        std::string str_val;
        str_val.assign(reinterpret_cast<const char*>(buffer.data()), metadata.buffer_size);
        while (!str_val.empty() && str_val.back() == '\0')
            str_val.pop_back();
        if (!str_val.empty())
            tag_value = std::move(str_val);
        break;
        
    case NVIMGCODEC_METADATA_VALUE_TYPE_SHORT:
        // uint16_t: single value or array
        extract_single_value<uint16_t>(buffer, metadata.value_count, tag_value) ||
        extract_value_array<uint16_t>(buffer, metadata.value_count, tag_value);
        break;
        
    case NVIMGCODEC_METADATA_VALUE_TYPE_LONG:
        // uint32_t: single value or array
        extract_single_value<uint32_t>(buffer, metadata.value_count, tag_value) ||
        extract_value_array<uint32_t>(buffer, metadata.value_count, tag_value);
        break;
        
    case NVIMGCODEC_METADATA_VALUE_TYPE_BYTE:
        if (metadata.value_count == 1)
            tag_value = buffer[0];  // Single byte
        else
            tag_value = std::vector<uint8_t>(buffer.begin(), buffer.begin() + metadata.buffer_size);
        break;
        
    case NVIMGCODEC_METADATA_VALUE_TYPE_LONG8:
    case NVIMGCODEC_METADATA_VALUE_TYPE_IFD8:
        // uint64_t: single value or array (BigTIFF support)
        extract_single_value<uint64_t>(buffer, metadata.value_count, tag_value) ||
        extract_value_array<uint64_t>(buffer, metadata.value_count, tag_value);
        break;
        
    case NVIMGCODEC_METADATA_VALUE_TYPE_FLOAT:
        extract_single_value<float>(buffer, metadata.value_count, tag_value) ||
        extract_value_array<float>(buffer, metadata.value_count, tag_value);
        break;
        
    case NVIMGCODEC_METADATA_VALUE_TYPE_DOUBLE:
        extract_single_value<double>(buffer, metadata.value_count, tag_value) ||
        extract_value_array<double>(buffer, metadata.value_count, tag_value);
        break;
        
    case NVIMGCODEC_METADATA_VALUE_TYPE_RATIONAL:
        // Convert to string format "numerator/denominator"
        if (metadata.value_count == 1 && metadata.buffer_size >= 8) {
            uint32_t num = *reinterpret_cast<const uint32_t*>(buffer.data());
            uint32_t den = *reinterpret_cast<const uint32_t*>(buffer.data() + 4);
            if (den != 0)
                tag_value = fmt::format("{}/{}", num, den);
            else
                tag_value = std::to_string(num);
        }
        break;
        
    case NVIMGCODEC_METADATA_VALUE_TYPE_UNDEFINED:
        // Binary/unknown data - store as vector<uint8_t>
        tag_value = std::vector<uint8_t>(buffer.begin(), buffer.begin() + metadata.buffer_size);
        break;
        
    // ... additional cases for SBYTE, SSHORT, SLONG, SLONG8, SRATIONAL, etc.
}

// Store in IFD's tag map if extraction succeeded
if (!std::holds_alternative<std::monostate>(tag_value)) {
    ifd_info.tiff_tags[tag_name] = std::move(tag_value);
}
```

### Key Extraction Strategy

1. **Single vs Array Detection**: Automatically distinguishes based on `metadata.value_count`
   - `value_count == 1` → Store as scalar (e.g., `uint16_t`)
   - `value_count > 1` → Store as vector (e.g., `std::vector<uint16_t>`)

2. **Rational Types**: Converted to human-readable string format (`"num/den"`)

3. **Binary Safety**: Configurable size limit (`max_binary_tag_size_`) prevents memory bloat from large binary blobs

4. **Move Semantics**: Uses `std::move()` to avoid unnecessary copies

---

## 4. Template Helpers for Type-Safe Extraction

Two template functions handle the **single value vs array** pattern elegantly:

### Single Value Extraction

```cpp
template<typename T>
static bool extract_single_value(const std::vector<uint8_t>& buffer, 
                                 int value_count,
                                 TiffTagValue& out_value)
{
    if (value_count == 1)
    {
        T val = *reinterpret_cast<const T*>(buffer.data());
        out_value = val;
        return true;
    }
    return false;
}
```

### Array Extraction

```cpp
template<typename T>
static bool extract_value_array(const std::vector<uint8_t>& buffer,
                                int value_count,
                                TiffTagValue& out_value)
{
    if (value_count > 1)
    {
        const T* vals = reinterpret_cast<const T*>(buffer.data());
        out_value = std::vector<T>(vals, vals + value_count);
        return true;
    }
    return false;
}
```

### Usage Pattern

The templates are used with logical OR for clean fallback:

```cpp
// Try single value first, fall back to array if value_count > 1
extract_single_value<uint16_t>(buffer, value_count, tag_value) ||
extract_value_array<uint16_t>(buffer, value_count, tag_value);
```

This approach:
- Returns `true` and populates `tag_value` on success
- Automatically selects the correct storage type (scalar vs vector)
- Compiles to optimal code via template instantiation

---

## 5. Retrieval and Conversion to String

### Visitor Pattern for Type-Safe Conversion

The `tiff_tag_value_to_string()` helper uses `std::visit` for compile-time type dispatch:

```cpp
static std::string tiff_tag_value_to_string(const TiffTagValue& value)
{
    return std::visit([](const auto& v) -> std::string {
        using T = std::decay_t<decltype(v)>;
        
        if constexpr (std::is_same_v<T, std::monostate>)
        {
            return "";  // Empty/unset
        }
        else if constexpr (std::is_same_v<T, std::string>)
        {
            return v;  // Already a string
        }
        else if constexpr (std::is_same_v<T, std::vector<uint8_t>>)
        {
            return fmt::format("[{} bytes]", v.size());  // Binary data summary
        }
        else if constexpr (std::is_same_v<T, std::vector<uint16_t>>)
        {
            // Array: show first 10 elements, truncate with "..."
            std::string result;
            for (size_t i = 0; i < v.size() && i < 10; ++i)
            {
                if (i > 0) result += ",";
                result += std::to_string(v[i]);
            }
            if (v.size() > 10) result += ",...";
            return result;
        }
        else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
        {
            return fmt::format("{}", v);
        }
        else
        {
            return std::to_string(v);  // Fallback for all integer types
        }
    }, value);
}
```

**Features:**
- **Compile-time dispatch**: `if constexpr` eliminates runtime overhead
- **Array truncation**: Shows first 10 elements for readability
- **Binary summaries**: Large binary data shows `"[1024 bytes]"` instead of dumping raw data
- **Type-safe**: Compiler ensures all variant types are handled

---

## 6. Public API for Tag Retrieval

Simple string-based interface in `TiffFileParser`:

```cpp
std::string TiffFileParser::get_tiff_tag(uint32_t ifd_index, const std::string& tag_name) const
{
    if (ifd_index >= ifd_infos_.size())
        return "";
    
    auto it = ifd_infos_[ifd_index].tiff_tags.find(tag_name);
    if (it != ifd_infos_[ifd_index].tiff_tags.end())
        return tiff_tag_value_to_string(it->second);
    
    return "";
}
```

### Usage Example (from `tiff/ifd.cpp`)

```cpp
// Extract metadata from TIFF tags
software_ = tiff->nvimgcodec_parser_->get_tiff_tag(index, "Software");
model_ = tiff->nvimgcodec_parser_->get_tiff_tag(index, "Model");

std::string tile_w_str = tiff->nvimgcodec_parser_->get_tiff_tag(index, "TileWidth");
std::string tile_h_str = tiff->nvimgcodec_parser_->get_tiff_tag(index, "TileLength");

if (!tile_w_str.empty() && !tile_h_str.empty()) {
    tile_width_ = std::stoul(tile_w_str);
    tile_height_ = std::stoul(tile_h_str);
}
```

---

## Design Benefits

### 1. Type Safety
- Compiler enforces correct type handling at compile time
- No runtime type confusion or casting errors
- `std::variant` provides exhaustive type checking

### 2. Memory Efficiency
- No redundant string parsing or conversions
- Direct binary-to-typed storage
- Move semantics eliminate unnecessary copies
- Configurable limits for large binary data

### 3. Performance
- Template instantiation creates optimal specialized code (zero-overhead abstraction)
- Compile-time `if constexpr` eliminates runtime branching
- Direct memory access via `reinterpret_cast` (validated by nvImageCodec API contract)

### 4. Extensibility
- Easy to add new types to the variant
- Template helpers work with any numeric type
- Visitor pattern scales naturally

### 5. Debugging
- Clear `std::visit` logic shows exactly what's stored
- Type-safe string conversion for logging
- `#ifdef DEBUG` blocks provide detailed extraction traces

### 6. Binary Safety
- Configurable size limits (`max_binary_tag_size_`) prevent memory bloat
- Large binary tags can be truncated without affecting other data
- Clear indication when truncation occurs (debug output)

---

## Evolution History

The implementation evolved through multiple iterations:

### Version 1: Basic Types
```cpp
std::unordered_map<std::string, std::variant<std::string, int8_t, uint8_t, int16_t, uint16_t>>
```
- Initial proof of concept
- Supported only basic scalar types

### Version 2: Array Support
```cpp
std::unordered_map<std::string, std::variant<std::string, int8_t, uint8_t, int16_t, uint16_t, 
                                              std::vector<uint16_t>>>
```
- Added vector type for multi-value tags
- Enabled BitsPerSample, SubIFD arrays

### Version 3: Complete TIFF Coverage (Current)
```cpp
using TiffTagValue = std::variant<
    std::monostate, std::string,
    int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t,
    float, double,
    std::vector<uint8_t>, std::vector<uint16_t>, std::vector<uint32_t>, 
    std::vector<uint64_t>, std::vector<float>, std::vector<double>
>;
```
- **17 types** covering all TIFF specifications
- BigTIFF support (64-bit types)
- Floating-point support (FLOAT, DOUBLE)
- Complete array type coverage

---

## Integration with nvImageCodec API

### Metadata Extraction Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Query Metadata Count                                     │
│    nvimgcodecDecoderGetMetadata(decoder, stream, nullptr,   │
│                                  &metadata_count)           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Query Buffer Sizes                                       │
│    Allocate metadata structs with buffer=nullptr            │
│    nvimgcodecDecoderGetMetadata(...) fills buffer_size      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Allocate Buffers                                         │
│    Create std::vector<uint8_t> for each metadata entry      │
│    Set buffer pointers in metadata structs                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Retrieve Metadata Content                                │
│    nvimgcodecDecoderGetMetadata(...) fills buffers          │
│    Each entry has: kind, format, value_type, value_count    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Type Conversion (switch on value_type)                   │
│    - Parse tag_id and tag_name from buffer                  │
│    - Convert raw bytes to TiffTagValue variant              │
│    - Store in ifd_info.tiff_tags[tag_name]                  │
└─────────────────────────────────────────────────────────────┘
```

### nvImageCodec Metadata Types Supported

| nvImageCodec Type | C++ Variant Type | Notes |
|-------------------|------------------|-------|
| `NVIMGCODEC_METADATA_VALUE_TYPE_ASCII` | `std::string` | Trailing nulls removed |
| `NVIMGCODEC_METADATA_VALUE_TYPE_BYTE` | `uint8_t` or `std::vector<uint8_t>` | Single vs array |
| `NVIMGCODEC_METADATA_VALUE_TYPE_SHORT` | `uint16_t` or `std::vector<uint16_t>` | Most common tag type |
| `NVIMGCODEC_METADATA_VALUE_TYPE_LONG` | `uint32_t` or `std::vector<uint32_t>` | Dimensions, offsets |
| `NVIMGCODEC_METADATA_VALUE_TYPE_LONG8` | `uint64_t` or `std::vector<uint64_t>` | BigTIFF support |
| `NVIMGCODEC_METADATA_VALUE_TYPE_FLOAT` | `float` or `std::vector<float>` | Scientific data |
| `NVIMGCODEC_METADATA_VALUE_TYPE_DOUBLE` | `double` or `std::vector<double>` | High precision |
| `NVIMGCODEC_METADATA_VALUE_TYPE_RATIONAL` | `std::string` | Formatted as "num/den" |
| `NVIMGCODEC_METADATA_VALUE_TYPE_UNDEFINED` | `std::vector<uint8_t>` | Binary data |

---

## Error Handling

### Graceful Degradation

The implementation handles errors without throwing exceptions:

1. **Missing tags**: Return `std::monostate` (empty variant state)
2. **Invalid IFD index**: `get_tiff_tag()` returns empty string
3. **Unsupported types**: Fallback to binary storage or string representation
4. **Oversized data**: Configurable truncation with debug warnings

### Debug Output

When compiled with `DEBUG` defined, the code provides detailed logging:

```cpp
#ifdef DEBUG
fmt::print("    ✅ Tag {} ({}): {}\n", tag_id, tag_name, debug_str);
#endif
```

Example output:
```
    ✅ Tag 256 (ImageWidth): 46000
    ✅ Tag 257 (ImageLength): 32914
    ✅ Tag 258 (BitsPerSample): 8,8,8
    ✅ Tag 259 (Compression): 7
    ✅ Tag 270 (ImageDescription): Aperio Image Library v1.0.0
```

---

## Best Practices for Usage

### Direct Access (Type-Safe)

If you know the expected type, access the variant directly:

```cpp
const auto& tag_value = ifd_info.tiff_tags["Compression"];
if (std::holds_alternative<uint16_t>(tag_value)) {
    uint16_t compression = std::get<uint16_t>(tag_value);
    // Use compression value...
}
```

### String Conversion (Generic)

For display or generic processing:

```cpp
std::string value = parser->get_tiff_tag(ifd_index, "Software");
if (!value.empty()) {
    fmt::print("Software: {}\n", value);
}
```

### Visitor Pattern (Advanced)

For custom processing logic:

```cpp
std::visit([](const auto& v) {
    using T = std::decay_t<decltype(v)>;
    if constexpr (std::is_same_v<T, uint32_t>) {
        // Process as uint32_t
    } else if constexpr (std::is_same_v<T, std::vector<uint32_t>>) {
        // Process as array
    }
}, ifd_info.tiff_tags["SubIFD"]);
```

---

## Conclusion

The variant-based TIFF tag storage system provides a **modern, type-safe, and efficient** approach to handling TIFF metadata from nvImageCodec. By leveraging C++17 features (`std::variant`, `std::visit`, `if constexpr`), the implementation achieves:

- **Zero-overhead type safety** through compile-time specialization
- **Flexible storage** supporting all TIFF data types
- **Clean API** hiding complexity from consumers
- **Extensibility** for future TIFF format enhancements
- **Production-ready robustness** with configurable limits and error handling

This design pattern can serve as a template for other metadata storage needs in the cuCIM project.

