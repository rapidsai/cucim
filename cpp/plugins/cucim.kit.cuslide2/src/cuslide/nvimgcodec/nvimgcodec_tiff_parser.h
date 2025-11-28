/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifdef CUCIM_HAS_NVIMGCODEC
#include <nvimgcodec.h>
#endif

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <unordered_map>
#include <variant>
#include <mutex>
#include <stdexcept>
#include <cucim/io/device.h>

namespace cuslide2::nvimgcodec
{

#ifdef CUCIM_HAS_NVIMGCODEC

/**
 * @brief Variant type for storing typed TIFF tag values
 *
 * Supports all TIFF tag value types as defined in nvimgcodecMetadataValueType_t:
 * - std::monostate: Empty/unset state (tag not found or not extracted)
 * - std::string: ASCII strings (ImageDescription, Software, DateTime, etc.)
 * - int8_t, uint8_t: SBYTE/BYTE values
 * - int16_t, uint16_t: SSHORT/SHORT values (Compression, Photometric, etc.)
 * - int32_t, uint32_t: SLONG/LONG values (ImageWidth, ImageLength, SubfileType, etc.)
 * - int64_t, uint64_t: SLONG8/LONG8/IFD8 values (BigTIFF support)
 * - float, double: FLOAT/DOUBLE values
 * - std::vector<uint8_t>: Binary data (JPEGTables, UNDEFINED type, etc.)
 * - std::vector<uint16_t>: Arrays of SHORT values (BitsPerSample, etc.)
 * - std::vector<uint32_t>: Arrays of LONG values (SubIFD offsets, etc.)
 * - std::vector<uint64_t>: Arrays of LONG8 values (BigTIFF offsets, etc.)
 */
using TiffTagValue = std::variant<
    std::monostate,  // Empty/unset state
    std::string,
    int8_t,
    uint8_t,
    int16_t,
    uint16_t,
    int32_t,
    uint32_t,
    int64_t,
    uint64_t,
    float,
    double,
    std::vector<uint8_t>,
    std::vector<uint16_t>,
    std::vector<uint32_t>,
    std::vector<uint64_t>,
    std::vector<float>,   // Arrays of FLOAT values
    std::vector<double>   // Arrays of DOUBLE values
>;

/**
 * @brief Image type classification for TIFF IFDs
 *
 * Used to categorize IFDs as resolution levels or associated images
 * (particularly for formats like Aperio SVS that use SUBFILETYPE tags)
 */
/**
 * @brief Information about a single IFD (Image File Directory) in a TIFF file
 *
 * Represents one resolution level in a multi-resolution TIFF pyramid.
 */
struct IfdInfo
{
    uint32_t index;                          // IFD index (0, 1, 2, ...)
    uint32_t width;                          // Image width in pixels
    uint32_t height;                         // Image height in pixels
    uint32_t num_channels;                   // Number of channels (typically 3 for RGB)
    uint32_t bits_per_sample;                // Bits per channel (8, 16, etc.)
    std::string codec;                       // Compression codec (jpeg, jpeg2k, deflate, etc.) - replace with int  : 0,1,2, for each codec type
    nvimgcodecCodeStream_t sub_code_stream;  // nvImageCodec code stream for this IFD

    // Metadata fields (extracted from nvImageCodec metadata API)
    std::string image_description;           // ImageDescription TIFF tag (270)

    // Format-specific metadata: kind -> (format, buffer_data)
    // kind: nvimgcodecMetadataKind_t (e.g., MED_APERIO=1, MED_PHILIPS=2, etc.)
    // format: nvimgcodecMetadataFormat_t (e.g., RAW, XML, JSON)
    // buffer_data: raw bytes from metadata buffer
    struct MetadataBlob {
        int format;  // nvimgcodecMetadataFormat_t
        std::vector<uint8_t> data;
    };
    std::map<int, MetadataBlob> metadata_blobs;

    // nvImageCodec 0.7.0: Individual TIFF tag storage with typed values
    // tag_name -> TiffTagValue (variant with typed storage)
    std::unordered_map<std::string, TiffTagValue> tiff_tags;

    IfdInfo() : index(0), width(0), height(0), num_channels(0),
                bits_per_sample(0), sub_code_stream(nullptr) {}

    ~IfdInfo()
    {
        // NOTE: sub_code_stream is managed by TiffFileParser and should NOT be destroyed here
        // The parent TiffFileParser destroys all sub-code streams when destroying main_code_stream
    }

    // Disable copy, enable move
    IfdInfo(const IfdInfo&) = delete;
    IfdInfo& operator=(const IfdInfo&) = delete;
    IfdInfo(IfdInfo&&) = default;
    IfdInfo& operator=(IfdInfo&&) = default;
};

/**
 * @brief TIFF file parser using nvImageCodec file-level API
 *
 * This class provides TIFF parsing capabilities using nvImageCodec's native
 * TIFF support. It can query TIFF structure (IFD count, dimensions, codecs)
 * and decode entire resolution levels.
 *
 * Note: This is an alternative to the libtiff-based approach. It provides
 * simpler code but less metadata access and no tile-level granularity.
 *
 * Usage:
 *   auto tiff = std::make_unique<TiffFileParser>("image.tif");
 *   if (tiff->is_valid()) {
 *       uint32_t num_levels = tiff->get_ifd_count();
 *       const auto& ifd = tiff->get_ifd(0);
 *
 *       // Use IFD information for decoding via separate decoder
 *       // (decoding is handled by IFD::read() or similar)
 *   }
 */
class TiffFileParser
{
public:
    /**
     * @brief Open and parse a TIFF file
     *
     * @param file_path Path to TIFF file
     * @throws std::runtime_error if nvImageCodec is not available or file cannot be opened
     */
    explicit TiffFileParser(const std::string& file_path);

    /**
     * @brief Destructor - cleans up nvImageCodec resources
     */
    ~TiffFileParser();

    // Disable copy, enable move
    TiffFileParser(const TiffFileParser&) = delete;
    TiffFileParser& operator=(const TiffFileParser&) = delete;
    TiffFileParser(TiffFileParser&&) = default;
    TiffFileParser& operator=(TiffFileParser&&) = default;

    /**
     * @brief Check if TIFF file was successfully opened and parsed
     *
     * @return true if file is valid and ready to use
     */
    bool is_valid() const { return initialized_; }

    /**
     * @brief Get the number of IFDs (resolution levels) in the TIFF file
     *
     * @return Number of IFDs
     */
    uint32_t get_ifd_count() const { return static_cast<uint32_t>(ifd_infos_.size()); }

    /**
     * @brief Get information about a specific IFD
     *
     * @param index IFD index (0 = highest resolution)
     * @return Reference to IFD information
     * @throws std::out_of_range if index is invalid
     */
    const IfdInfo& get_ifd(uint32_t index) const;

    /**
     * @brief Get all metadata blobs for an IFD
     *
     * Returns all vendor-specific metadata extracted by nvImageCodec.
     * The map key is nvimgcodecMetadataKind_t (e.g., MED_APERIO=1, MED_PHILIPS=2).
     *
     * @param ifd_index IFD index
     * @return Map of metadata kind to blob (format + data), or empty if no metadata
     */
    const std::map<int, IfdInfo::MetadataBlob>& get_metadata_blobs(uint32_t ifd_index) const
    {
        static const std::map<int, IfdInfo::MetadataBlob> empty_map;
        if (ifd_index >= ifd_infos_.size())
            return empty_map;
        return ifd_infos_[ifd_index].metadata_blobs;
    }

    // ========================================================================
    // nvImageCodec 0.7.0 Features: Individual TIFF Tag Retrieval
    // ========================================================================

    /**
     * @brief Get a specific TIFF tag value as string (nvImageCodec 0.7.0+)
     *
     * Uses NVIMGCODEC_METADATA_KIND_TIFF_TAG to retrieve individual TIFF tags
     * by name (e.g., "SUBFILETYPE", "ImageDescription", "DateTime", etc.)
     *
     * @param ifd_index IFD index
     * @param tag_name TIFF tag name (case-sensitive)
     * @return Tag value as string, or empty if not found
     */
    std::string get_tiff_tag(uint32_t ifd_index, const std::string& tag_name) const;

    /**
     * @brief Get SUBFILETYPE tag for format classification (nvImageCodec 0.7.0+)
     *
     * Returns the SUBFILETYPE value used in formats like Aperio SVS:
     * - 0 = full resolution image
     * - 1 = reduced resolution image (thumbnail/label/macro)
     *
     * @param ifd_index IFD index
     * @return SUBFILETYPE value, or -1 if not present
     */
    int get_subfile_type(uint32_t ifd_index) const;

    /**
     * @brief Query all available metadata kinds in file (nvImageCodec 0.7.0+)
     *
     * Returns a list of metadata kinds present in the file for discovery.
     * Useful for detecting file format (Aperio, Philips, Generic TIFF, etc.)
     *
     * Example kinds: TIFF_TAG=0, MED_APERIO=1, MED_PHILIPS=2, etc.
     *
     * @param ifd_index IFD index (default 0 for file-level metadata)
     * @return Vector of metadata kind values present in the IFD
     */
    std::vector<int> query_metadata_kinds(uint32_t ifd_index = 0) const;

    /**
     * @brief Get detected file format based on metadata (nvImageCodec 0.7.0+)
     *
     * Automatically detects format by checking available metadata kinds.
     * nvImageCodec 0.7.0 handles detection internally.
     *
     * @return Format name: "Aperio SVS", "Philips TIFF", "Leica SCN", "Generic TIFF", etc.
     */
    std::string get_detected_format() const;

    /**
     * @brief Get the main code stream for the TIFF file
     *
     * This is used by decoder functions (in nvimgcodec_decoder.cpp) to create
     * ROI sub-streams for decoding. The parser provides the stream, but does
     * NOT perform decoding itself (separation of concerns).
     *
     * @return nvImageCodec code stream handle
     */
    nvimgcodecCodeStream_t get_main_code_stream() const { return main_code_stream_; }

    /**
     * @brief Set maximum size for binary TIFF tag data storage
     *
     * Controls the maximum size of binary data (UNDEFINED type, JPEGTables, etc.)
     * that will be stored in memory when parsing TIFF tags. This can prevent
     * memory issues when encountering very large binary blobs.
     *
     * @param max_size Maximum size in bytes (0 = unlimited, default)
     *
     * Note: This limit only applies to binary/unknown data types. Known typed
     * arrays (SHORT, LONG, FLOAT, etc.) are not limited for consistency.
     */
    void set_max_binary_tag_size(size_t max_size) { max_binary_tag_size_ = max_size; }

    /**
     * @brief Get the current maximum binary tag size limit
     *
     * @return Maximum binary tag size in bytes (0 = unlimited)
     */
    size_t get_max_binary_tag_size() const { return max_binary_tag_size_; }

private:
    /**
     * @brief Parse TIFF file structure using nvImageCodec
     *
     * Queries the number of IFDs and gets metadata for each one.
     */
    void parse_tiff_structure();

    /**
     * @brief Extract metadata for a specific IFD using nvimgcodecDecoderGetMetadata
     *
     * Retrieves vendor-specific metadata (Aperio, Philips, etc.) for the given IFD.
     * Populates ifd_info.metadata_blobs and ifd_info.image_description.
     *
     * @param ifd_info IFD to extract metadata for (must have valid sub_code_stream)
     */
    void extract_ifd_metadata(IfdInfo& ifd_info);

    /**
     * @brief Extract individual TIFF tags (nvImageCodec 0.7.0+)
     *
     * Uses NVIMGCODEC_METADATA_KIND_TIFF_TAG to query specific TIFF tags by name.
     * Populates ifd_info.tiff_tags map.
     *
     * @param ifd_info IFD to extract TIFF tags for
     */
    void extract_tiff_tags(IfdInfo& ifd_info);

    std::string file_path_;
    bool initialized_;
    nvimgcodecCodeStream_t main_code_stream_;
    std::vector<IfdInfo> ifd_infos_;
    
    // Configuration: Maximum size for binary TIFF tag data (0 = unlimited)
    size_t max_binary_tag_size_ = 0;
};

/**
 * @brief Singleton manager for nvImageCodec TIFF parsing
 *
 * Manages the global nvImageCodec instance for TIFF parsing operations.
 * This is separate from the tile decoder manager to avoid conflicts.
 */
class NvImageCodecTiffParserManager
{
public:
    /**
     * @brief Get the singleton instance
     *
     * @return Reference to the global manager
     */
    static NvImageCodecTiffParserManager& instance()
    {
        static NvImageCodecTiffParserManager manager;
        return manager;
    }

    /**
     * @brief Get the nvImageCodec instance
     *
     * @return nvImageCodec instance handle
     */
    nvimgcodecInstance_t get_instance() const { return instance_; }

    /**
     * @brief Get the nvImageCodec decoder (for metadata extraction)
     *
     * @return nvImageCodec decoder handle
     */
    nvimgcodecDecoder_t get_decoder() const { return decoder_; }

    /**
     * @brief Get the CPU-only decoder (for native CPU decoding)
     *
     * @return nvImageCodec CPU decoder handle
     */
    nvimgcodecDecoder_t get_cpu_decoder() const { return cpu_decoder_; }

    /**
     * @brief Check if CPU-only decoder is available
     *
     * @return true if CPU decoder is available
     */
    bool has_cpu_decoder() const { return cpu_decoder_ != nullptr; }

    /**
     * @brief Get the mutex for thread-safe decoder operations
     *
     * @return Reference to the decoder mutex
     */
    std::mutex& get_mutex() { return decoder_mutex_; }

    /**
     * @brief Check if nvImageCodec is available and initialized
     *
     * @return true if available
     */
    bool is_available() const { return initialized_; }

    /**
     * @brief Get initialization status message
     *
     * @return Status message
     */
    const std::string& get_status() const { return status_message_; }

private:
    NvImageCodecTiffParserManager();
    ~NvImageCodecTiffParserManager();

    // Disable copy and move
    NvImageCodecTiffParserManager(const NvImageCodecTiffParserManager&) = delete;
    NvImageCodecTiffParserManager& operator=(const NvImageCodecTiffParserManager&) = delete;
    NvImageCodecTiffParserManager(NvImageCodecTiffParserManager&&) = delete;
    NvImageCodecTiffParserManager& operator=(NvImageCodecTiffParserManager&&) = delete;

    nvimgcodecInstance_t instance_;
    nvimgcodecDecoder_t decoder_;
    nvimgcodecDecoder_t cpu_decoder_;  // CPU-only decoder (uses libjpeg-turbo, etc.)
    bool initialized_;
    std::string status_message_;
    std::mutex decoder_mutex_;  // Protect decoder operations from concurrent access
};

#else // !CUCIM_HAS_NVIMGCODEC

// Stub implementations when nvImageCodec is not available

// Stub TiffTagValue for API compatibility
using TiffTagValue = std::variant<
    std::monostate,  // Empty/unset state
    std::string,
    int8_t,
    uint8_t,
    int16_t,
    uint16_t,
    int32_t,
    uint32_t,
    int64_t,
    uint64_t,
    float,
    double,
    std::vector<uint8_t>,
    std::vector<uint16_t>,
    std::vector<uint32_t>,
    std::vector<uint64_t>,
    std::vector<float>,   // Arrays of FLOAT values
    std::vector<double>   // Arrays of DOUBLE values
>;

struct IfdInfo {
    struct MetadataBlob {
        int format;
        std::vector<uint8_t> data;
    };
    std::unordered_map<std::string, TiffTagValue> tiff_tags;
};

class TiffFileParser
{
public:
    explicit TiffFileParser(const std::string& file_path) { (void)file_path; }
    bool is_valid() const { return false; }
    uint32_t get_ifd_count() const { return 0; }
    const IfdInfo& get_ifd(uint32_t index) const
    {
        (void)index;
        throw std::runtime_error("nvImageCodec not available");
    }

    // Stub methods for API compatibility
    const std::map<int, IfdInfo::MetadataBlob>& get_metadata_blobs(uint32_t ifd_index) const
    {
        (void)ifd_index;
        static const std::map<int, IfdInfo::MetadataBlob> empty_map;
        return empty_map;
    }

    std::string get_detected_format() const
    {
        return "Unknown";
    }

    std::string get_tiff_tag(uint32_t ifd_index, const std::string& tag_name) const
    {
        (void)ifd_index;
        (void)tag_name;
        return "";
    }

    int get_subfile_type(uint32_t ifd_index) const
    {
        (void)ifd_index;
        return -1;
    }
};

class NvImageCodecTiffParserManager
{
public:
    static NvImageCodecTiffParserManager& instance()
    {
        static NvImageCodecTiffParserManager manager;
        return manager;
    }
    bool is_available() const { return false; }
    const std::string& get_status() const
    {
        static std::string msg = "nvImageCodec not available";
        return msg;
    }
};

#endif // CUCIM_HAS_NVIMGCODEC

} // namespace cuslide2::nvimgcodec
