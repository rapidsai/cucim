/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#ifndef CUSLIDE_IFD_H
#define CUSLIDE_IFD_H

#include "types.h"

#include <cucim/io/format/image_format.h>
#include <cucim/io/device.h>
//#include <tiffio.h>

#include <memory>
#include <vector>

namespace cuslide::tiff
{

// Forward declaration.
class TIFF;

class EXPORT_VISIBLE IFD : public std::enable_shared_from_this<IFD>
{
public:
    IFD(TIFF* tiff, uint16_t index, ifd_offset_t offset);
    ~IFD() = default;

    static bool read_region_tiles(const TIFF* tiff,
                                  const IFD* ifd,
                                  const int64_t sx,
                                  const int64_t sy,
                                  const int64_t w,
                                  const int64_t h,
                                  void* raster,
                                  const cucim::io::Device& out_device);

    static bool read_region_tiles_boundary(const TIFF* tiff,
                                           const IFD* ifd,
                                           const int64_t sx,
                                           const int64_t sy,
                                           const int64_t w,
                                           const int64_t h,
                                           void* raster,
                                           const cucim::io::Device& out_device);

    bool read(const TIFF* tiff,
              const cucim::io::format::ImageMetadataDesc* metadata,
              const cucim::io::format::ImageReaderRegionRequestDesc* request,
              cucim::io::format::ImageDataDesc* out_image_data);


    uint32_t index() const;
    ifd_offset_t offset() const;

    std::string& software();
    std::string& model();
    std::string& image_description();
    uint32_t width() const;
    uint32_t height() const;
    uint32_t tile_width() const;
    uint32_t tile_height() const;
    uint32_t bits_per_sample() const;
    uint32_t samples_per_pixel() const;
    uint64_t subfile_type() const;
    uint16_t planar_config() const;
    uint16_t photometric() const;
    uint16_t compression() const;

    uint16_t subifd_count() const;
    std::vector<uint64_t>& subifd_offsets();

    uint32_t image_piece_count() const;
    const std::vector<uint64_t>& image_piece_offsets() const;
    const std::vector<uint64_t>& image_piece_bytecounts() const;

    // Hidden methods for benchmarking
    void write_offsets_(const char* file_path);

    // Make TIFF available to access private members of IFD
    friend class TIFF;

private:
    TIFF* tiff_; // cannot use shared_ptr as IFD is created during the construction of TIFF using 'new'
    uint32_t ifd_index_ = 0;
    ifd_offset_t ifd_offset_ = 0;

    std::string software_;
    std::string model_;
    std::string image_description_;
    uint32_t flags_ = 0;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    uint32_t tile_width_ = 0;
    uint32_t tile_height_ = 0;
    uint32_t bits_per_sample_ = 0;
    uint32_t samples_per_pixel_ = 0;
    uint64_t subfile_type_ = 0;
    uint16_t planar_config_ = 0;
    uint16_t photometric_ = 0;
    uint16_t compression_ = 0;

    uint16_t subifd_count_ = 0;
    std::vector<uint64_t> subifd_offsets_;

    std::vector<uint8_t> jpegtable_;

    uint32_t image_piece_count_ = 0;
    std::vector<uint64_t> image_piece_offsets_;
    std::vector<uint64_t> image_piece_bytecounts_;

    uint64_t hash_value_ = 0; /// file hash including ifd index.

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
};
} // namespace cuslide::tiff

#endif // CUSLIDE_IFD_H
