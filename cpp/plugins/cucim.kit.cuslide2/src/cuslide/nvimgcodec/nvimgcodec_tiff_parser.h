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
#include <cucim/io/device.h>

namespace cuslide2::nvimgcodec
{

#ifdef CUCIM_HAS_NVIMGCODEC

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
    std::string codec;                       // Compression codec (jpeg, jpeg2k, deflate, etc.)
    nvimgcodecCodeStream_t sub_code_stream;  // nvImageCodec code stream for this IFD
    
    IfdInfo() : index(0), width(0), height(0), num_channels(0), 
                bits_per_sample(0), sub_code_stream(nullptr) {}
    
    ~IfdInfo()
    {
        if (sub_code_stream)
        {
            nvimgcodecCodeStreamDestroy(sub_code_stream);
            sub_code_stream = nullptr;
        }
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
 *       uint8_t* image_data = nullptr;
 *       if (tiff->decode_ifd(0, &image_data, cucim::io::Device("cpu"))) {
 *           // Use image_data...
 *           free(image_data);
 *       }
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
     * @brief Decode an entire IFD (full resolution image)
     * 
     * Note: This decodes the entire IFD, not individual tiles. For efficient
     * region reading, use the libtiff + buffer-level nvImageCodec approach instead.
     * 
     * @param ifd_index IFD index to decode (0 = highest resolution)
     * @param output_buffer Pointer to receive allocated buffer (caller must free)
     * @param out_device Output device ("cpu" or "cuda")
     * @return true if successful, false otherwise
     */
    bool decode_ifd(uint32_t ifd_index, 
                    uint8_t** output_buffer,
                    const cucim::io::Device& out_device);
    
    /**
     * @brief Print TIFF structure information
     */
    void print_info() const;

private:
    /**
     * @brief Parse TIFF file structure using nvImageCodec
     * 
     * Queries the number of IFDs and gets metadata for each one.
     */
    void parse_tiff_structure();
    
    std::string file_path_;
    bool initialized_;
    nvimgcodecCodeStream_t main_code_stream_;
    nvimgcodecDecoder_t decoder_;
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
    bool initialized_;
    std::string status_message_;
};

#else // !CUCIM_HAS_NVIMGCODEC

// Stub implementations when nvImageCodec is not available
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
    bool decode_ifd(uint32_t ifd_index, uint8_t** output_buffer, const cucim::io::Device& out_device)
    {
        (void)ifd_index; (void)output_buffer; (void)out_device;
        return false;
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

