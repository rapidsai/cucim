/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#ifndef CUSLIDE_TIFF_H
#define CUSLIDE_TIFF_H

#include <cucim/filesystem/file_path.h>
#include <cucim/filesystem/file_handle.h>
#include <cucim/io/format/image_format.h>
#include <cucim/macros/api_header.h>
#include <cucim/memory/memory_manager.h>

#include <cstdint>
#include <fcntl.h>
#include <map>
#include <memory>
#include <vector>

#include "ifd.h"
#include "types.h"

typedef struct tiff TIFF;

namespace cuslide::tiff
{

/**
 * TIFF file handler class.
 *
 * This class doesn't use PImpl idiom for performance reasons and is not
 * intended to be used for subclassing.
 */
class EXPORT_VISIBLE TIFF : public std::enable_shared_from_this<TIFF>
{
public:
    TIFF(const cucim::filesystem::Path& file_path, int mode);
    TIFF(const cucim::filesystem::Path& file_path, int mode, uint64_t config);
    static std::shared_ptr<TIFF> open(const cucim::filesystem::Path& file_path, int mode);
    static std::shared_ptr<TIFF> open(const cucim::filesystem::Path& file_path, int mode, uint64_t config);
    void close();
    void construct_ifds();

    /**
     * Resolve vendor format and fix values for `associated_image_descs_` and `level_to_ifd_idx_.
     */
    void resolve_vendor_format();
    bool read(const cucim::io::format::ImageMetadataDesc* metadata,
              const cucim::io::format::ImageReaderRegionRequestDesc* request,
              cucim::io::format::ImageDataDesc* out_image_data,
              cucim::io::format::ImageMetadataDesc* out_metadata = nullptr);

    bool read_associated_image(const cucim::io::format::ImageMetadataDesc* metadata,
                               const cucim::io::format::ImageReaderRegionRequestDesc* request,
                               cucim::io::format::ImageDataDesc* out_image_data,
                               cucim::io::format::ImageMetadataDesc* out_metadata);

    cucim::filesystem::Path file_path() const;
    std::shared_ptr<CuCIMFileHandle>& file_handle(); /// used for moving the ownership of the file handle to the caller.
    ::TIFF* client() const;
    const std::vector<ifd_offset_t>& ifd_offsets() const;
    std::shared_ptr<IFD> ifd(size_t index) const;
    std::shared_ptr<IFD> level_ifd(size_t level_index) const;
    size_t ifd_count() const;
    size_t level_count() const;
    const std::map<std::string, AssociatedImageBufferDesc>& associated_images() const;
    size_t associated_image_count() const;
    bool is_big_endian() const;
    uint64_t read_config() const;
    bool is_in_read_config(uint64_t configs) const;
    void add_read_config(uint64_t configs);
    TiffType tiff_type();
    std::string metadata();

    ~TIFF();

    static void* operator new(std::size_t sz);
    static void operator delete(void* ptr);
    //    static void* operator new[](std::size_t sz);
    //    static void operator delete(void* ptr, std::size_t sz);
    //    static void operator delete[](void* ptr, std::size_t sz);

    // const values for read_configs_
    static constexpr uint64_t kUseDirectJpegTurbo = 1;
    static constexpr uint64_t kUseLibTiff = 1 << 1;

    // Make IFD available to access private members of TIFF
    friend class IFD;

private:
    void _populate_philips_tiff_metadata(uint16_t ifd_count, void* metadata, std::shared_ptr<IFD>& first_ifd);
    void _populate_aperio_svs_metadata(uint16_t ifd_count, void* metadata, std::shared_ptr<IFD>& first_ifd);

    cucim::filesystem::Path file_path_;
    /// Temporary shared file handle whose ownership would be transferred to CuImage through parser_open()
    std::shared_ptr<CuCIMFileHandle> file_handle_shared_;
    CuCIMFileHandle* file_handle_ = nullptr;
    ::TIFF* tiff_client_ = nullptr;
    std::vector<ifd_offset_t> ifd_offsets_; /// IFD offset for an index (IFD index)
    std::vector<std::shared_ptr<IFD>> ifds_; /// IFD object for an index (IFD index)
    std::vector<size_t> level_to_ifd_idx_;
    // note: we use std::map instead of std::unordered_map as # of associated_image would be usually less than 10.
    std::map<std::string, AssociatedImageBufferDesc> associated_images_;
    bool is_big_endian_ = false; /// if big endian
    uint8_t background_value_ = 0x00; /// background_value
    uint64_t read_config_ = 0;
    TiffType tiff_type_ = TiffType::Generic;
    void* metadata_ = nullptr;
};
} // namespace cuslide::tiff

#endif // CUSLIDE_TIFF_H
