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

#pragma once

#ifdef CUCIM_HAS_NVIMGCODEC
#include <nvimgcodec.h>
#endif

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <mutex>
#include <stdexcept>
#include <cucim/io/device.h>

namespace cuslide2::nvimgcodec
{

#ifdef CUCIM_HAS_NVIMGCODEC

/**
 * @brief Image type classification for TIFF IFDs
 * 
 * Used to categorize IFDs as resolution levels or associated images
 * (particularly for formats like Aperio SVS that use SUBFILETYPE tags)
 */
enum class ImageType {
    RESOLUTION_LEVEL,  // Full or reduced resolution image
    THUMBNAIL,         // Thumbnail image
    LABEL,             // Slide label image
    MACRO,             // Macro/overview image
    UNKNOWN            // Unclassified
};

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
    
    // nvImageCodec 0.7.0: Individual TIFF tag storage
    // tag_name -> tag_value (e.g., "SUBFILETYPE" -> "0")
    std::map<std::string, std::string> tiff_tags;
    
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
    
    void print() const;
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
     * @brief Get the file path
     * 
     * @return File path
     */
    const std::string& get_file_path() const { return file_path_; }
    
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
     * @brief Classify an IFD by type (resolution level vs. associated image)
     * 
     * Parses ImageDescription metadata to determine image purpose using
     * vendor-specific keywords (e.g., "label", "macro" for Aperio SVS).
     * Falls back to dimension-based heuristics for unknown formats.
     * 
     * @param ifd_index IFD index to classify
     * @return ImageType classification
     */
    ImageType classify_ifd(uint32_t ifd_index) const;
    
    /**
     * @brief Get indices of all resolution level IFDs
     * 
     * @return Vector of IFD indices that represent resolution levels
     */
    std::vector<uint32_t> get_resolution_levels() const;
    
    /**
     * @brief Get associated images (thumbnail, label, macro)
     * 
     * Returns a map of associated image names to their IFD indices.
     * Particularly useful for Aperio SVS files.
     * 
     * @return Map of image name to IFD index
     */
    std::map<std::string, uint32_t> get_associated_images() const;
    
    /**
     * @brief Override IFD dimensions
     * 
     * Useful for formats like Philips TIFF where reported dimensions
     * include tile padding and need to be corrected based on metadata.
     * 
     * @param ifd_index IFD index to modify
     * @param width Corrected width in pixels
     * @param height Corrected height in pixels
     */
    void override_ifd_dimensions(uint32_t ifd_index, uint32_t width, uint32_t height);
    
    /**
     * @brief Get ImageDescription metadata for an IFD
     * 
     * Returns the ImageDescription from nvImageCodec metadata buffers.
     * For Aperio SVS: Contains full image description with keywords.
     * For Philips TIFF: Contains XML metadata or level info.
     * 
     * @param ifd_index IFD index
     * @return ImageDescription string, or empty if not present
     */
    std::string get_image_description(uint32_t ifd_index) const;
    
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
    
    /**
     * @brief Get specific metadata blob by kind
     * 
     * @param ifd_index IFD index
     * @param kind Metadata kind (e.g., 1=MED_APERIO, 2=MED_PHILIPS)
     * @return Pointer to metadata blob, or nullptr if not found
     */
    const IfdInfo::MetadataBlob* get_metadata_blob(uint32_t ifd_index, int kind) const
    {
        if (ifd_index >= ifd_infos_.size())
            return nullptr;
        
        auto it = ifd_infos_[ifd_index].metadata_blobs.find(kind);
        if (it != ifd_infos_[ifd_index].metadata_blobs.end())
            return &it->second;
        return nullptr;
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
     * @brief Print TIFF structure information
     */
    void print_info() const;
    
    // ========================================================================
    // ROI-Based Decoding (nvTiff File-Level API)
    // ========================================================================
    
    /**
     * @brief Decode a region of interest (ROI) from a specific IFD
     * 
     * Uses nvTiff's file-level API with ROI parameters to decode a specific
     * region without loading the entire image. nvTiff automatically handles
     * JPEG tables (TIFFTAG_JPEGTABLES) internally.
     * 
     * @param ifd_index IFD index to decode from
     * @param x X offset in pixels (left edge of region)
     * @param y Y offset in pixels (top edge of region)
     * @param width Width of region in pixels
     * @param height Height of region in pixels
     * @param output_buffer Pre-allocated buffer for decoded pixels (can be nullptr for auto-allocation)
     * @param device Device to decode to ("cpu" or "cuda")
     * @return Decoded pixel data (caller owns the buffer if auto-allocated)
     * @throws std::runtime_error on decode failure
     */
    uint8_t* decode_region(
        uint32_t ifd_index,
        uint32_t x, uint32_t y,
        uint32_t width, uint32_t height,
        uint8_t* output_buffer = nullptr,
        const cucim::io::Device& device = cucim::io::Device("cpu")
    );
    
    /**
     * @brief Decode an entire IFD
     * 
     * Convenience method that decodes the full IFD image.
     * 
     * @param ifd_index IFD index to decode
     * @param output_buffer Pre-allocated buffer (can be nullptr)
     * @param device Device to decode to
     * @return Decoded pixel data
     */
    uint8_t* decode_ifd(
        uint32_t ifd_index,
        uint8_t* output_buffer = nullptr,
        const cucim::io::Device& device = cucim::io::Device("cpu")
    );
    
    /**
     * @brief Check if ROI-based decoding is available
     * 
     * @return true if nvImageCodec decoder is available with ROI support
     */
    bool has_roi_decode_support() const;
    
    /**
     * @brief Get the main code stream for the TIFF file
     * 
     * @return nvImageCodec code stream handle
     */
    nvimgcodecCodeStream_t get_main_code_stream() const { return main_code_stream_; }

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
    bool initialized_;
    std::string status_message_;
    std::mutex decoder_mutex_;  // Protect decoder operations from concurrent access
};

#else // !CUCIM_HAS_NVIMGCODEC

// Stub implementations when nvImageCodec is not available
enum class ImageType {
    RESOLUTION_LEVEL,
    THUMBNAIL,
    LABEL,
    MACRO,
    UNKNOWN
};

struct IfdInfo {};

class TiffFileParser
{
public:
    explicit TiffFileParser(const std::string& file_path) { (void)file_path; }
    bool is_valid() const { return false; }
    const std::string& get_file_path() const { static std::string empty; return empty; }
    uint32_t get_ifd_count() const { return 0; }
    const IfdInfo& get_ifd(uint32_t index) const 
    { 
        (void)index; 
        throw std::runtime_error("nvImageCodec not available"); 
    }
    ImageType classify_ifd(uint32_t index) const { (void)index; return ImageType::UNKNOWN; }
    std::vector<uint32_t> get_resolution_levels() const { return {}; }
    std::map<std::string, uint32_t> get_associated_images() const { return {}; }
    void override_ifd_dimensions(uint32_t ifd_index, uint32_t width, uint32_t height)
    {
        (void)ifd_index; (void)width; (void)height;
    }
    std::string get_image_description(uint32_t ifd_index) const
    {
        (void)ifd_index;
        return "";
    }
    void print_info() const {}
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

