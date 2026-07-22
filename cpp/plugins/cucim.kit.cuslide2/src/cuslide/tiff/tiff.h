/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include "ifd.h"
#include "types.h"
#include "cuslide/nvimgcodec/nvimgcodec_tiff_parser.h"

// Forward declaration removed - no longer using libtiff
// typedef struct tiff TIFF;  // REMOVED: libtiff forward declaration

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
    // nvImageCodec constructors (primary - no libtiff mode parameter)
    TIFF(const cucim::filesystem::Path& file_path);
    TIFF(const cucim::filesystem::Path& file_path, uint64_t read_config);
    static std::shared_ptr<TIFF> open(const cucim::filesystem::Path& file_path);
    static std::shared_ptr<TIFF> open(const cucim::filesystem::Path& file_path, uint64_t config);

    // Legacy libtiff-style constructors (for compatibility if needed)
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
    std::shared_ptr<CuCIMFileHandle>& file_handle();
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
    static constexpr uint64_t kUseLibTiff = 1 << 1;

    // Make IFD available to access private members of TIFF
    friend class IFD;

private:
    std::vector<std::unique_ptr<cuslide2::nvimgcodec::TiffFileParser>> companion_parsers_;
    std::vector<std::string> companion_parser_paths_;
    std::vector<uint64_t> parser_file_hashes_;
    std::map<std::pair<size_t, uint32_t>, size_t> parser_local_to_global_ifd_;
    std::vector<std::pair<size_t, uint32_t>> global_to_parser_local_ifd_;
    bool has_ome_plane_index_ = false;
    int64_t ome_size_c_ = 1;
    int64_t ome_size_z_ = 1;
    int64_t ome_size_t_ = 1;
    std::vector<std::string> ome_channel_names_;
    std::map<std::string, int64_t> ome_channel_name_to_index_;
    std::map<std::tuple<int64_t, int64_t, int64_t, uint16_t>, size_t> ome_plane_to_ifd_;
    void _populate_ome_tiff_metadata(uint16_t ifd_count, void* metadata, std::shared_ptr<IFD>& first_ifd);
    size_t _ensure_global_ifd(size_t parser_idx, uint32_t local_ifd_idx);
    size_t _find_or_add_parser_context(const std::string& abs_path);
    static uint64_t _hash_path(const std::string& path);
    const cuslide2::nvimgcodec::TiffFileParser& parser_for_global_ifd(size_t global_ifd_idx) const;
    uint32_t local_ifd_for_global_ifd(size_t global_ifd_idx) const;
    uint64_t file_hash_for_global_ifd(size_t global_ifd_idx) const;

    // UPDATED: These now use nvImageCodec TiffFileParser instead of libtiff
    void _populate_philips_tiff_metadata(uint16_t ifd_count, void* metadata, std::shared_ptr<IFD>& first_ifd);
    void _populate_aperio_svs_metadata(uint16_t ifd_count, void* metadata, std::shared_ptr<IFD>& first_ifd);

    cucim::filesystem::Path file_path_;
    std::shared_ptr<CuCIMFileHandle> file_handle_shared_;
    // REMOVED: file_handle_ raw pointer - use file_handle_shared_.get() instead
    // REMOVED: tiff_client_ - no longer using libtiff
    std::vector<ifd_offset_t> ifd_offsets_; /// IFD offset for an index (IFD index)
    std::vector<std::shared_ptr<IFD>> ifds_; /// IFD object for an index (IFD index)
    /// nvImageCodec TIFF parser - MUST be destroyed BEFORE ifds_ to avoid double-free of sub-code streams
    /// Placed AFTER ifds_ so it's destroyed FIRST (reverse declaration order)
    std::unique_ptr<cuslide2::nvimgcodec::TiffFileParser> nvimgcodec_parser_;
    std::vector<size_t> level_to_ifd_idx_;
    // note: we use std::map instead of std::unordered_map as # of associated_image would be usually less than 10.
    std::map<std::string, AssociatedImageBufferDesc> associated_images_;
    bool is_big_endian_ = false; /// if big endian
    uint8_t background_value_ = 0x00; /// background_value
    uint64_t read_config_ = 0;
    TiffType tiff_type_ = TiffType::Generic;
    void* metadata_ = nullptr;

    mutable std::once_flag slow_path_warning_flag_;
};
} // namespace cuslide::tiff

#endif // CUSLIDE_TIFF_H
