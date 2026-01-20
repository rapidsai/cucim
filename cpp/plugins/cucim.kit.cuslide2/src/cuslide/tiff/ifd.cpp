/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ifd.h"

#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <thread>

#include <fmt/format.h>

#include <cucim/codec/hash_function.h>
#include <cucim/cuimage.h>
#include <cucim/logger/timer.h>
#include <cucim/memory/memory_manager.h>
#include <cucim/profiler/nvtx3.h>
#include <cucim/util/cuda.h>

// nvImageCodec handles ALL decoding (JPEG, JPEG2000, deflate, LZW, raw)
#include "cuslide/nvimgcodec/nvimgcodec_decoder.h"
#include "cuslide/nvimgcodec/nvimgcodec_tiff_parser.h"
#include "cuslide/loader/nvimgcodec_processor.h"
#include "tiff.h"
#include "tiff_constants.h"


namespace cuslide::tiff
{

// OLD CONSTRUCTOR: libtiff-based (DEPRECATED - use nvImageCodec constructor instead)
// This constructor is kept for API compatibility but is not functional in pure nvImageCodec build
IFD::IFD(TIFF* tiff, uint16_t index, ifd_offset_t offset) : tiff_(tiff), ifd_index_(index), ifd_offset_(offset)
{
#ifdef CUCIM_HAS_NVIMGCODEC
    // Pure nvImageCodec path: try to use IfdInfo instead
    if (tiff->nvimgcodec_parser_ && tiff->nvimgcodec_parser_->is_valid())
    {
        if (static_cast<uint32_t>(index) < tiff->nvimgcodec_parser_->get_ifd_count())
        {
            const auto& ifd_info = tiff->nvimgcodec_parser_->get_ifd(static_cast<uint32_t>(index));

            // Initialize from IfdInfo
            width_ = ifd_info.width;
            height_ = ifd_info.height;
            samples_per_pixel_ = ifd_info.num_channels;
            bits_per_sample_ = ifd_info.bits_per_sample;

            // Parse codec to compression
            compression_ = parse_codec_to_compression(ifd_info.codec);
            codec_name_ = ifd_info.codec;

            // Assume tiled if tile dimensions are provided in IfdInfo (check nvImageCodec metadata)
            // For now, use a heuristic: most whole-slide images are tiled
            tile_width_ = 256;  // Default tile size (can be overridden from IfdInfo metadata)
            tile_height_ = 256;

            // nvImageCodec members
            nvimgcodec_sub_stream_ = ifd_info.sub_code_stream;

            // Calculate hash value
            hash_value_ = tiff->file_handle_shared_.get()->hash_value ^ cucim::codec::splitmix64(index);

            #ifdef DEBUG
            fmt::print("  IFD[{}]: Initialized from nvImageCodec ({}x{}, codec: {})\n",
                      index, width_, height_, codec_name_);
            #endif
            return;
        }
    }

    // Fallback: throw error if nvImageCodec parser not available
    throw std::runtime_error(fmt::format(
        "IFD constructor (offset-based) requires libtiff, which is not available in pure nvImageCodec build. "
        "Use IFD(TIFF*, uint16_t, IfdInfo&) constructor instead."));
#else
    // If nvImageCodec not available, this should never be called
    throw std::runtime_error("Pure nvImageCodec build requires CUCIM_HAS_NVIMGCODEC");
#endif
}

// ============================================================================
// NEW PRIMARY CONSTRUCTOR: nvImageCodec-Only (No libtiff)
// ============================================================================

#ifdef CUCIM_HAS_NVIMGCODEC
IFD::IFD(TIFF* tiff, uint16_t index, const cuslide2::nvimgcodec::IfdInfo& ifd_info)
    : tiff_(tiff), ifd_index_(index), ifd_offset_(index)
{
    PROF_SCOPED_RANGE(PROF_EVENT(ifd_ifd));  // Use standard ifd_ifd profiler event

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
    #ifdef DEBUG
    fmt::print("   Codec: {} (compression={})\n", codec_name_, compression_);
    #endif

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

    // Set format defaults
    planar_config_ = PLANARCONFIG_CONTIG;  // nvImageCodec outputs interleaved
    photometric_ = PHOTOMETRIC_RGB;
    predictor_ = 1;  // No predictor

    // Resolution info (defaults - may not be available from nvImageCodec)
    resolution_unit_ = 1;  // No absolute unit
    x_resolution_ = 1.0f;
    y_resolution_ = 1.0f;

    // Calculate hash for caching
    hash_value_ = cucim::codec::splitmix64(index);

#ifdef CUCIM_HAS_NVIMGCODEC
    // Store reference to nvImageCodec sub-stream
    nvimgcodec_sub_stream_ = ifd_info.sub_code_stream;
    #ifdef DEBUG
    fmt::print("   ✅ nvImageCodec sub-stream: {}\n",
              static_cast<void*>(nvimgcodec_sub_stream_));
    #endif
#endif

    #ifdef DEBUG
    fmt::print("✅ IFD[{}] initialization complete\n", index);
    #endif
}
#endif // CUCIM_HAS_NVIMGCODEC

IFD::~IFD()
{
#ifdef CUCIM_HAS_NVIMGCODEC
    // NOTE: nvimgcodec_sub_stream_ is NOT owned by IFD - it's owned by TiffFileParser
    // TiffFileParser::~TiffFileParser() will destroy all sub-code streams
    // DO NOT call nvimgcodecCodeStreamDestroy here to avoid double-free or use-after-free
    //
    // The destruction order in TIFF is:
    // 1. nvimgcodec_parser_ destroyed → TiffFileParser destroys sub-code streams
    // 2. ifds_ destroyed → IFD destructors run (we're here)
    //
    // By this point, sub-code streams are already destroyed, so we just clear the pointer
    nvimgcodec_sub_stream_ = nullptr;
#endif
}

bool IFD::read([[maybe_unused]] const TIFF* tiff,
               [[maybe_unused]] const cucim::io::format::ImageMetadataDesc* metadata,
               [[maybe_unused]] const cucim::io::format::ImageReaderRegionRequestDesc* request,
               [[maybe_unused]] cucim::io::format::ImageDataDesc* out_image_data)
{
    PROF_SCOPED_RANGE(PROF_EVENT(ifd_read));

    #ifdef DEBUG
    fmt::print("🎯 IFD::read() ENTRY: IFD[{}], location=({}, {}), size={}x{}, device={}, location_len={}\n",
              ifd_index_,
              request->location[0], request->location[1],
              request->size[0], request->size[1],
              request->device,
              request->location_len);
    #endif

#ifdef CUCIM_HAS_NVIMGCODEC
    if (!nvimgcodec_sub_stream_ || !tiff->nvimgcodec_parser_)
    {
        throw std::runtime_error(fmt::format(
            "IFD[{}]: nvImageCodec parser not available", ifd_index_));
    }

    std::string device_name(request->device);
    if (request->shm_name)
    {
        device_name = device_name + fmt::format("[{}]", request->shm_name);
    }
    cucim::io::Device out_device(device_name);

    int64_t w = request->size[0];
    int64_t h = request->size[1];
    const uint64_t location_len = request->location_len;

    uint32_t batch_size = request->batch_size;
    int32_t n_ch = samples_per_pixel_;
    int ndim = 3;

    size_t one_raster_size = static_cast<size_t>(w) * static_cast<size_t>(h) * samples_per_pixel_;
    size_t raster_size = one_raster_size;

    void* raster = nullptr;
    auto raster_type = cucim::io::DeviceType::kCPU;

    DLTensor* out_buf = request->buf;
    bool is_buf_available = out_buf && out_buf->data;

    if (is_buf_available)
    {
        raster = out_buf->data;
    }

    // ========================================================================
    // BATCH DECODING PATH: Multiple locations or batch_size > 1
    // Uses ThreadBatchDataLoader with NvImageCodecProcessor for parallel ROI decoding
    // ========================================================================
    if (location_len > 1 || batch_size > 1 || request->num_workers > 0)
    {
        if (batch_size > 1)
        {
            ndim = 4;
        }

        int64_t* location = request->location;
        const uint32_t num_workers = request->num_workers;
        const bool drop_last = request->drop_last;
        uint32_t prefetch_factor = request->prefetch_factor;
        const bool shuffle = request->shuffle;
        const uint64_t seed = request->seed;

        #ifdef DEBUG
        fmt::print("🚀 Using ThreadBatchDataLoader for {} locations, batch_size={}, workers={}\n",
                  location_len, batch_size, num_workers);
        #endif

        if (num_workers == 0 && location_len > 1)
        {
            throw std::runtime_error("Cannot read multiple locations with zero workers!");
        }

        // Shuffle data if requested
        if (shuffle)
        {
            auto rng = std::default_random_engine{ seed };
            struct position { int64_t x; int64_t y; };
            std::shuffle(reinterpret_cast<position*>(&location[0]),
                         reinterpret_cast<position*>(&location[location_len * 2]), rng);
        }

        // Adjust location length based on 'drop_last'
        uint64_t adjusted_location_len = location_len;
        const uint32_t remaining_len = adjusted_location_len % batch_size;
        if (drop_last)
        {
            adjusted_location_len -= remaining_len;
        }

        // Do not use prefetch if the image count is too small
        if (1 + prefetch_factor > adjusted_location_len)
        {
            prefetch_factor = adjusted_location_len > 0 ? adjusted_location_len - 1 : 0;
        }

        raster_size *= batch_size;

        const IFD* ifd = this;

        // Reconstruct location unique_ptr
        std::unique_ptr<std::vector<int64_t>>* location_unique =
            reinterpret_cast<std::unique_ptr<std::vector<int64_t>>*>(request->location_unique);
        std::unique_ptr<std::vector<int64_t>> request_location = std::move(*location_unique);
        delete location_unique;

        // Reconstruct size unique_ptr
        std::unique_ptr<std::vector<int64_t>>* size_unique =
            reinterpret_cast<std::unique_ptr<std::vector<int64_t>>*>(request->size_unique);
        std::unique_ptr<std::vector<int64_t>> request_size = std::move(*size_unique);
        delete size_unique;

        // Create load function that decodes ROI using nvImageCodec
        auto load_func = [tiff, ifd, location, w, h, out_device](
                             cucim::loader::ThreadBatchDataLoader* loader_ptr, uint64_t location_index) {
            uint8_t* raster_ptr = loader_ptr->raster_pointer(location_index);

            int64_t sx = location[location_index * 2];
            int64_t sy = location[location_index * 2 + 1];

            // Get IFD info from TiffFileParser
            const auto& ifd_info = tiff->nvimgcodec_parser_->get_ifd(static_cast<uint32_t>(ifd->ifd_index_));

            // Decode ROI directly into the loader's raster buffer
            bool success = cuslide2::nvimgcodec::decode_ifd_region_nvimgcodec(
                ifd_info,
                tiff->nvimgcodec_parser_->get_main_code_stream(),
                static_cast<uint32_t>(sx), static_cast<uint32_t>(sy),
                static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                &raster_ptr,
                out_device);

            if (!success)
            {
                fmt::print(stderr, "[Error] Failed to decode ROI at ({}, {}) {}x{}\n", sx, sy, w, h);
            }
        };

        // Create batch processor using nvImageCodec
        std::unique_ptr<cucim::loader::BatchDataProcessor> batch_processor;

        if (out_device.type() == cucim::io::DeviceType::kCUDA)
        {
            raster_type = cucim::io::DeviceType::kCUDA;

            // Create NvImageCodecProcessor for GPU-accelerated batch decoding
            auto nvimgcodec_processor = std::make_unique<cuslide2::loader::NvImageCodecProcessor>(
                tiff->nvimgcodec_parser_.get(),
                request_location->data(),
                request_size->data(),
                adjusted_location_len,
                batch_size,
                static_cast<uint32_t>(ifd_index_),
                out_device);

            // Update prefetch_factor from processor
            prefetch_factor = nvimgcodec_processor->preferred_loader_prefetch_factor();

            batch_processor = std::move(nvimgcodec_processor);
        }

        // Create ThreadBatchDataLoader
        auto loader = std::make_unique<cucim::loader::ThreadBatchDataLoader>(
            load_func, std::move(batch_processor), out_device,
            std::move(request_location), std::move(request_size),
            adjusted_location_len, one_raster_size, batch_size, prefetch_factor, num_workers);

        const uint32_t load_size = std::min(
            static_cast<uint64_t>(batch_size) * (1 + prefetch_factor),
            adjusted_location_len);

        loader->request(load_size);

        // If reading single location synchronously, fetch immediately
        if (adjusted_location_len == 1 && batch_size == 1)
        {
            raster = loader->next_data();
        }

        // Set loader in output for iterator access
        out_image_data->loader = loader.release();

        #ifdef DEBUG
        fmt::print("✅ ThreadBatchDataLoader created for {} locations\n", adjusted_location_len);
        #endif

        // Set up output metadata
        if (raster)
        {
            out_image_data->container.data = raster;
            out_image_data->container.device = DLDevice{ static_cast<DLDeviceType>(raster_type), out_device.index() };
            out_image_data->container.dtype = DLDataType{ kDLUInt, 8, 1 };
            out_image_data->container.ndim = ndim;
            out_image_data->container.shape = static_cast<int64_t*>(cucim_malloc(ndim * sizeof(int64_t)));
            if (ndim == 4)
            {
                out_image_data->container.shape[0] = batch_size;
                out_image_data->container.shape[1] = h;
                out_image_data->container.shape[2] = w;
                out_image_data->container.shape[3] = n_ch;
            }
            else
            {
                out_image_data->container.shape[0] = h;
                out_image_data->container.shape[1] = w;
                out_image_data->container.shape[2] = n_ch;
            }
            out_image_data->container.strides = nullptr;
            out_image_data->container.byte_offset = 0;
        }

        return true;
    }

    // ========================================================================
    // SINGLE REGION PATH: Standard single-location decoding
    // ========================================================================
    int64_t sx = request->location[0];
    int64_t sy = request->location[1];

    // Output buffer - decoder may allocate if not provided.
    uint8_t* output_buffer = nullptr;

    if (is_buf_available)
    {
        // User provided pre-allocated buffer
        output_buffer = static_cast<uint8_t*>(out_buf->data);
    }
    // Note: decode_ifd_region_nvimgcodec will allocate buffer if output_buffer is nullptr

    // Get IFD info from TiffFileParser
    const auto& ifd_info = tiff->nvimgcodec_parser_->get_ifd(static_cast<uint32_t>(ifd_index_));

    // Call nvImageCodec ROI decoder
    bool success = cuslide2::nvimgcodec::decode_ifd_region_nvimgcodec(
        ifd_info,
        tiff->nvimgcodec_parser_->get_main_code_stream(),
        sx, sy, w, h,
        output_buffer,
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

        // Free allocated buffer on failure (only if we allocated it).
        if (!is_buf_available && output_buffer)
        {
            if (out_device.type() == cucim::io::DeviceType::kCUDA)
            {
                cudaFree(output_buffer);
            }
            else
            {
                free(output_buffer);
            }
        }

        throw std::runtime_error(fmt::format(
            "Failed to decode IFD[{}] with nvImageCodec. ROI: ({},{}) {}x{}",
            ifd_index_, sx, sy, w, h));
    }
#else
    // If nvImageCodec not available, throw error
    throw std::runtime_error(fmt::format(
        "IFD[{}]: This library requires nvImageCodec for image decoding.", ifd_index_));
#endif
}

uint32_t IFD::index() const
{
    return ifd_index_;
}
ifd_offset_t IFD::offset() const
{
    return ifd_offset_;
}

std::string& IFD::software()
{
    return software_;
}
std::string& IFD::model()
{
    return model_;
}
std::string& IFD::image_description()
{
    return image_description_;
}
uint16_t IFD::resolution_unit() const
{
    return resolution_unit_;
}
float IFD::x_resolution() const
{
    return x_resolution_;
}
float IFD::y_resolution() const
{
    return y_resolution_;
}
uint32_t IFD::width() const
{
    return width_;
}
uint32_t IFD::height() const
{
    return height_;
}
uint32_t IFD::tile_width() const
{
    return tile_width_;
}
uint32_t IFD::tile_height() const
{
    return tile_height_;
}
uint32_t IFD::rows_per_strip() const
{
    return rows_per_strip_;
}
uint32_t IFD::bits_per_sample() const
{
    return bits_per_sample_;
}
uint32_t IFD::samples_per_pixel() const
{
    return samples_per_pixel_;
}
uint64_t IFD::subfile_type() const
{
    return subfile_type_;
}
uint16_t IFD::planar_config() const
{
    return planar_config_;
}
uint16_t IFD::photometric() const
{
    return photometric_;
}
uint16_t IFD::compression() const
{
    return compression_;
}
uint16_t IFD::predictor() const
{
    return predictor_;
}

uint16_t IFD::subifd_count() const
{
    return subifd_count_;
}
std::vector<uint64_t>& IFD::subifd_offsets()
{
    return subifd_offsets_;
}
uint32_t IFD::image_piece_count() const
{
    return image_piece_count_;
}
const std::vector<uint64_t>& IFD::image_piece_offsets() const
{
    return image_piece_offsets_;
}
const std::vector<uint64_t>& IFD::image_piece_bytecounts() const
{
    return image_piece_bytecounts_;
}

size_t IFD::pixel_size_nbytes() const
{
    // Calculate pixel size based on bits_per_sample and samples_per_pixel
    // Most whole-slide images are 8-bit RGB (3 bytes per pixel)
    const size_t bytes_per_sample = (bits_per_sample_ + 7) / 8;  // Round up to nearest byte
    const size_t nbytes = bytes_per_sample * samples_per_pixel_;
    return nbytes;
}

size_t IFD::tile_raster_size_nbytes() const
{
    const size_t nbytes = tile_width_ * tile_height_ * pixel_size_nbytes();
    return nbytes;
}

// ============================================================================
// Helper: Parse nvImageCodec Codec String to TIFF Compression Enum
// ============================================================================

#ifdef CUCIM_HAS_NVIMGCODEC
uint16_t IFD::parse_codec_to_compression(const std::string& codec)
{
    // Map nvImageCodec codec strings to TIFF compression constants
    if (codec == "jpeg") {
        return COMPRESSION_JPEG;  // 7
    }
    if (codec == "jpeg2000" || codec == "jpeg2k" || codec == "j2k") {
        // Default to YCbCr JPEG2000 (most common in whole-slide imaging)
        return COMPRESSION_APERIO_JP2K_YCBCR;  // 33003
    }
    if (codec == "lzw") {
        return COMPRESSION_LZW;  // 5
    }
    if (codec == "deflate" || codec == "zip") {
        return COMPRESSION_DEFLATE;  // 8
    }
    if (codec == "adobe-deflate") {
        return COMPRESSION_ADOBE_DEFLATE;  // 32946
    }
    if (codec == "none" || codec == "uncompressed" || codec.empty()) {
        return COMPRESSION_NONE;  // 1
    }

    // Handle generic 'tiff' codec from nvImageCodec 0.6.0
    // This is a known limitation where nvImageCodec doesn't expose the actual compression
    // For now, default to JPEG which is most common in whole-slide imaging
    if (codec == "tiff") {
        #ifdef DEBUG
        fmt::print("ℹ️  nvImageCodec returned generic 'tiff' codec, assuming JPEG compression\n");
        #endif
        return COMPRESSION_JPEG;  // 7 - Most common for WSI (Aperio, Philips, etc.)
    }

    // Unknown codec - log warning and default to JPEG (safer than NONE for WSI)
    #ifdef DEBUG
    fmt::print("⚠️  Unknown codec '{}', defaulting to COMPRESSION_JPEG\n", codec);
    #endif
    return COMPRESSION_JPEG;  // 7 - WSI files rarely use uncompressed
}
#endif // CUCIM_HAS_NVIMGCODEC

bool IFD::is_compression_supported() const
{
    switch (compression_)
    {
    case COMPRESSION_NONE:
    case COMPRESSION_JPEG:
    case COMPRESSION_ADOBE_DEFLATE:
    case COMPRESSION_DEFLATE:
    case COMPRESSION_APERIO_JP2K_YCBCR: // 33003: Jpeg 2000 with YCbCr format
    case COMPRESSION_APERIO_JP2K_RGB:   // 33005: Jpeg 2000 with RGB
    case COMPRESSION_LZW:
        return true;
    default:
        return false;
    }
}

bool IFD::is_read_optimizable() const
{
    return is_compression_supported() && bits_per_sample_ == 8 && samples_per_pixel_ == 3 &&
           (tile_width_ != 0 && tile_height_ != 0) && planar_config_ == PLANARCONFIG_CONTIG &&
           (photometric_ == PHOTOMETRIC_RGB || photometric_ == PHOTOMETRIC_YCBCR) &&
           !tiff_->is_in_read_config(TIFF::kUseLibTiff);
}

bool IFD::is_format_supported() const
{
    return is_compression_supported();
}

} // namespace cuslide::tiff


// Hidden methods for benchmarking.

#include <fmt/format.h>
#include <langinfo.h>
#include <iostream>
#include <fstream>

namespace cuslide::tiff
{
} // namespace cuslide::tiff
