/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ifd.h"

// Standard library includes - MUST be before any headers that open namespaces
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <thread>
#include <fstream>
#include <langinfo.h>

// Third-party includes - MUST be before any headers that open namespaces
#include <fmt/format.h>

// cuCIM includes
#include <cucim/cache/image_cache.h>
#include <cucim/codec/hash_function.h>
#include <cucim/cuimage.h>
#include <cucim/logger/timer.h>
#include <cucim/memory/memory_manager.h>
#include <cucim/profiler/nvtx3.h>
#include <cucim/util/cuda.h>

// nvImageCodec handles ALL decoding (JPEG, JPEG2000, deflate, LZW, raw)
// Include these BEFORE opening namespace to avoid namespace pollution
#include "cuslide/nvimgcodec/nvimgcodec_decoder.h"
#include "cuslide/nvimgcodec/nvimgcodec_tiff_parser.h"
#include "tiff.h"
#include "tiff_constants.h"

#ifdef CUCIM_HAS_NVIMGCODEC
#include "cuslide/loader/nvimgcodec_processor.h"
#endif

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

            // Try to read tile dimensions from extracted TIFF tags
            tile_width_ = 0;
            tile_height_ = 0;
            if (tiff->nvimgcodec_parser_) {
                std::string tw = tiff->nvimgcodec_parser_->get_tiff_tag(index, "TILEWIDTH");
                std::string th = tiff->nvimgcodec_parser_->get_tiff_tag(index, "TILELENGTH");
                if (!tw.empty() && !th.empty()) {
                    try {
                        tile_width_ = std::stoul(tw);
                        tile_height_ = std::stoul(th);
                    } catch (...) {
                        tile_width_ = 0;
                        tile_height_ = 0;
                    }
                }
            }

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
IFD::IFD(TIFF* tiff, uint16_t index, const ::cuslide2::nvimgcodec::IfdInfo& ifd_info)
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
        software_ = tiff->nvimgcodec_parser_->get_tiff_tag(index, "SOFTWARE");
        model_ = tiff->nvimgcodec_parser_->get_tiff_tag(index, "MODEL");

        // SUBFILETYPE for IFD classification
        int subfile_type = tiff->nvimgcodec_parser_->get_subfile_type(index);
        if (subfile_type >= 0) {
            subfile_type_ = static_cast<uint64_t>(subfile_type);
            #ifdef DEBUG
            fmt::print("   SUBFILETYPE: {}\n", subfile_type_);
            #endif
        }

        // Check for JPEGTables (abbreviated JPEG indicator)
        std::string jpeg_tables = tiff->nvimgcodec_parser_->get_tiff_tag(index, "JPEGTABLES");
        if (!jpeg_tables.empty()) {
            #ifdef DEBUG
            fmt::print("   ✅ JPEGTables detected (abbreviated JPEG)\n");
            #endif
        }

        // Tile dimensions (if available from TIFF tags)
        std::string tile_w_str = tiff->nvimgcodec_parser_->get_tiff_tag(index, "TILEWIDTH");
        std::string tile_h_str = tiff->nvimgcodec_parser_->get_tiff_tag(index, "TILELENGTH");

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
    planar_config_ = cuslide::tiff::PLANARCONFIG_CONTIG;  // nvImageCodec outputs interleaved
    photometric_ = cuslide::tiff::PHOTOMETRIC_RGB;
    predictor_ = 1;  // No predictor

    // Resolution info (defaults - may not be available from nvImageCodec)
    resolution_unit_ = 1;  // No absolute unit
    x_resolution_ = 1.0f;
    y_resolution_ = 1.0f;

    // Calculate hash for caching (include file hash for cross-file uniqueness)
    hash_value_ = tiff->file_handle_shared_.get()->hash_value ^ cucim::codec::splitmix64(index);

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
    if (location_len > 1 || batch_size > 1)
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
            auto rng = std::mt19937{ seed };
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

        // Create batch processor using nvImageCodec for both CPU and GPU output.
        // NvImageCodecProcessor handles buffer placement via out_device:
        //   - GPU: STRIDED_DEVICE buffers, custom CUDA stream
        //   - CPU: STRIDED_HOST buffers, nvImageCodec handles GPU decode → host copy
        std::unique_ptr<cucim::loader::BatchDataProcessor> batch_processor;

        if (out_device.type() == cucim::io::DeviceType::kCUDA)
        {
            raster_type = cucim::io::DeviceType::kCUDA;
        }

        auto nvimgcodec_processor = std::make_unique<cuslide2::loader::NvImageCodecProcessor>(
            *tiff->nvimgcodec_parser_,
            request_location->data(),
            request_size->data(),
            adjusted_location_len,
            batch_size,
            static_cast<uint32_t>(ifd_index_),
            out_device);

        prefetch_factor = nvimgcodec_processor->preferred_loader_prefetch_factor();
        batch_processor = std::move(nvimgcodec_processor);

        // load_func is a no-op because NvImageCodecProcessor handles all decoding
        // (both CPU and GPU). The ThreadBatchDataLoader still requires a load_func
        // but it is never called when a batch_processor is active.
        auto load_func = [](cucim::loader::ThreadBatchDataLoader* /*loader_ptr*/,
                            uint64_t /*location_index*/) {
            // No-op: NvImageCodecProcessor handles decoding for all device types.
        };

        // Create ThreadBatchDataLoader
        auto loader = std::make_unique<cucim::loader::ThreadBatchDataLoader>(
            load_func, std::move(batch_processor), out_device,
            std::move(request_location), std::move(request_size),
            adjusted_location_len, one_raster_size, batch_size, prefetch_factor, num_workers);

        // load_func is a no-op (batch processor handles decoding), so load_size
        // only affects bookkeeping in the loader.
        const uint32_t load_size =
            std::min(static_cast<uint64_t>(1), adjusted_location_len);

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

        // Always set container metadata (ndim, shape, dtype, device) even when
        // raster is nullptr.  libcucim's CuImage::read_region() accesses
        // container.shape immediately after the reader returns, so leaving
        // shape==nullptr causes a segfault.  For the multi-location batch path,
        // container.data is nullptr here and will be populated later by
        // loader->next_data() during Python iteration.
        out_image_data->container.data = raster;  // may be nullptr for batch iteration
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

    // Get IFD info from TiffFileParser
    const auto& ifd_info = tiff->nvimgcodec_parser_->get_ifd(static_cast<uint32_t>(ifd_index_));
    auto main_code_stream = tiff->nvimgcodec_parser_->get_main_code_stream();

    // ====================================================================
    // TILE-LEVEL CACHING PATH
    // Decompose the ROI into its constituent TIFF tiles, check the cache
    // for each tile, decode only cache-miss tiles via nvImageCodec, insert
    // decoded tiles into cache, and assemble the output raster.
    // ====================================================================
    int64_t ex = sx + w - 1;
    int64_t ey = sy + h - 1;

    // Tile caching is applicable when:
    //  1. The image is tiled (tile_width_ > 0 && tile_height_ > 0)
    //  2. The ROI is fully within the image bounds (no boundary handling)
    //  3. An image cache is configured (not kNoCache)
    bool use_tile_caching = (tile_width_ > 0 && tile_height_ > 0 &&
                             sx >= 0 && sy >= 0 &&
                             ex < static_cast<int64_t>(width_) &&
                             ey < static_cast<int64_t>(height_));

    if (use_tile_caching)
    {
        cucim::cache::ImageCache& image_cache = cucim::CuImage::cache_manager().cache();
        cucim::cache::CacheType cache_type = image_cache.type();

        if (cache_type == cucim::cache::CacheType::kNoCache)
        {
            use_tile_caching = false; // No cache configured — fall through to direct decode
        }
    }

    if (use_tile_caching)
    {
        cucim::cache::ImageCache& image_cache = cucim::CuImage::cache_manager().cache();

        const uint32_t tw = tile_width_;
        const uint32_t th = tile_height_;
        const uint32_t samples = samples_per_pixel_;
        const uint8_t background_value = tiff->background_value_;

        // Tile grid offsets that the ROI overlaps
        const uint32_t offset_sx = static_cast<uint32_t>(sx / tw);
        const uint32_t offset_ex = static_cast<uint32_t>(ex / tw);
        const uint32_t offset_sy = static_cast<uint32_t>(sy / th);
        const uint32_t offset_ey = static_cast<uint32_t>(ey / th);

        // Pixel offsets within start/end tiles
        const uint32_t pixel_offset_sx = static_cast<uint32_t>(sx % tw);
        const uint32_t pixel_offset_ex = static_cast<uint32_t>(ex % tw);
        const uint32_t pixel_offset_sy = static_cast<uint32_t>(sy % th);
        const uint32_t pixel_offset_ey = static_cast<uint32_t>(ey % th);

        // Tiles per row in the IFD tile grid
        const uint32_t stride_y = (width_ + tw - 1) / tw;

        const uint64_t ifd_hash = hash_value_;

        // Allocate host output buffer for tile assembly
        uint8_t* host_raster = static_cast<uint8_t*>(cucim_malloc(one_raster_size));
        if (!host_raster)
        {
            throw std::runtime_error("Failed to allocate host output buffer for tile assembly");
        }

        const uint32_t dest_row_stride = static_cast<uint32_t>(w) * samples;
        uint8_t* dest_row_ptr = host_raster;

        #ifdef DEBUG
        ::fmt::print("🧩 Tile caching: ROI ({},{})→({},{}) maps to tiles [{},{}]→[{},{}], stride_y={}\n",
                     sx, sy, ex, ey, offset_sx, offset_sy, offset_ex, offset_ey, stride_y);
        #endif

        for (uint32_t tile_row = offset_sy; tile_row <= offset_ey; ++tile_row)
        {
            // Vertical pixel offsets within the current tile row
            const uint32_t tile_pixel_sy = (tile_row == offset_sy) ? pixel_offset_sy : 0;
            const uint32_t tile_pixel_ey = (tile_row == offset_ey) ? pixel_offset_ey : (th - 1);
            const uint32_t copy_rows = tile_pixel_ey - tile_pixel_sy + 1;

            uint32_t dest_col_byte_offset = 0;

            for (uint32_t tile_col = offset_sx; tile_col <= offset_ex; ++tile_col)
            {
                const uint32_t tile_index = tile_row * stride_y + tile_col;

                // Horizontal pixel offsets within the current tile
                const uint32_t tile_pixel_sx = (tile_col == offset_sx) ? pixel_offset_sx : 0;
                const uint32_t tile_pixel_ex = (tile_col == offset_ex) ? pixel_offset_ex : (tw - 1);
                const uint32_t copy_cols = tile_pixel_ex - tile_pixel_sx + 1;
                const uint32_t copy_bytes_per_row = copy_cols * samples;

                // Actual decoded tile dimensions (clipped to image bounds for edge tiles)
                const uint32_t tile_origin_x = tile_col * tw;
                const uint32_t tile_origin_y = tile_row * th;
                const uint32_t actual_tw = std::min(tw, width_ - tile_origin_x);
                const uint32_t actual_th = std::min(th, height_ - tile_origin_y);
                const size_t tile_buf_size = static_cast<size_t>(actual_tw) * actual_th * samples;
                const uint32_t tile_row_stride = actual_tw * samples;

                // Hash for per-tile locking
                const uint64_t index_hash = ifd_hash ^
                    (static_cast<uint64_t>(tile_index) | (static_cast<uint64_t>(tile_index) << 32));

                // --- Cache lookup ---
                auto key = image_cache.create_key(ifd_hash, tile_index);
                image_cache.lock(index_hash);
                auto cached_value = image_cache.find(key);

                uint8_t* tile_data = nullptr;

                if (cached_value)
                {
                    // *** Cache hit ***
                    image_cache.unlock(index_hash);
                    tile_data = static_cast<uint8_t*>(cached_value->data);

                    #ifdef DEBUG
                    ::fmt::print("  ♻️  Cache HIT  tile[{},{}] idx={}\n", tile_col, tile_row, tile_index);
                    #endif
                }
                else
                {
                    // *** Cache miss — decode this tile via nvImageCodec ***
                    uint8_t* tile_buf = static_cast<uint8_t*>(image_cache.allocate(tile_buf_size));

                    // Decode tile-aligned ROI to host memory (for caching)
                    bool decode_ok = ::cuslide2::nvimgcodec::decode_ifd_region_nvimgcodec(
                        ifd_info, main_code_stream,
                        tile_origin_x, tile_origin_y, actual_tw, actual_th,
                        tile_buf,
                        cucim::io::Device("cpu"));

                    if (decode_ok)
                    {
                        tile_data = tile_buf;
                        auto value = image_cache.create_value(tile_data, tile_buf_size);
                        image_cache.insert(key, value);
                        image_cache.unlock(index_hash);

                        #ifdef DEBUG
                        ::fmt::print("  📥 Cache MISS tile[{},{}] idx={} decoded {}x{}\n",
                                     tile_col, tile_row, tile_index, actual_tw, actual_th);
                        #endif
                    }
                    else
                    {
                        image_cache.unlock(index_hash);
                        #ifdef DEBUG
                        ::fmt::print("  ❌ Tile decode FAILED tile[{},{}] — filling background\n",
                                     tile_col, tile_row);
                        #endif

                        // Fill with background colour on decode failure
                        for (uint32_t r = 0; r < copy_rows; ++r)
                        {
                            memset(dest_row_ptr + dest_col_byte_offset + r * dest_row_stride,
                                   background_value, copy_bytes_per_row);
                        }
                        dest_col_byte_offset += copy_bytes_per_row;
                        continue;
                    }
                }

                // --- Assembly: copy relevant sub-rectangle from tile into output ---
                const uint32_t src_byte_offset =
                    (tile_pixel_sy * actual_tw + tile_pixel_sx) * samples;

                for (uint32_t r = 0; r < copy_rows; ++r)
                {
                    memcpy(dest_row_ptr + dest_col_byte_offset + r * dest_row_stride,
                           tile_data + src_byte_offset + r * tile_row_stride,
                           copy_bytes_per_row);
                }

                dest_col_byte_offset += copy_bytes_per_row;
            }

            dest_row_ptr += static_cast<size_t>(copy_rows) * dest_row_stride;
        }

        // --- Move assembled raster to the requested output device ---
        if (out_device.type() == cucim::io::DeviceType::kCUDA)
        {
            uint8_t* gpu_buf = nullptr;
            cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&gpu_buf), one_raster_size);
            if (err != cudaSuccess)
            {
                cucim_free(host_raster);
                throw std::runtime_error("Failed to allocate GPU buffer for tile-cached output");
            }
            cudaMemcpy(gpu_buf, host_raster, one_raster_size, cudaMemcpyHostToDevice);
            cucim_free(host_raster);

            output_buffer = gpu_buf;
            raster_type = cucim::io::DeviceType::kCUDA;
        }
        else
        {
            output_buffer = host_raster;
        }

        // Set up output metadata
        out_image_data->container.data = output_buffer;
        out_image_data->container.device = DLDevice{ static_cast<DLDeviceType>(raster_type), out_device.index() };
        out_image_data->container.dtype = DLDataType{ kDLUInt, 8, 1 };
        out_image_data->container.ndim = 3;
        out_image_data->container.shape = static_cast<int64_t*>(cucim_malloc(3 * sizeof(int64_t)));
        out_image_data->container.shape[0] = h;
        out_image_data->container.shape[1] = w;
        out_image_data->container.shape[2] = samples_per_pixel_;
        out_image_data->container.strides = nullptr;
        out_image_data->container.byte_offset = 0;

        #ifdef DEBUG
        ::fmt::print("✅ Tile-cached read complete: {}x{} at ({},{}) → {}\n",
                     w, h, sx, sy, (raster_type == cucim::io::DeviceType::kCUDA) ? "GPU" : "CPU");
        #endif

        return true;
    }

    // ====================================================================
    // DIRECT ROI DECODE PATH (fallback: no caching)
    // Used when: tile caching is not applicable (strip-based images,
    // boundary ROIs, or no cache configured).
    // ====================================================================

    // Synchronous single-region decode via nvImageCodec.
    // This is intentional: IFD::read() for a single location is the read_region()
    // API, which returns decoded data immediately.  The asynchronous batch path
    // is handled separately by NvImageCodecProcessor.
    bool success = ::cuslide2::nvimgcodec::decode_ifd_region_nvimgcodec(
        ifd_info,
        main_code_stream,
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
                cucim_free(output_buffer);
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
        return cuslide::tiff::COMPRESSION_JPEG;  // 7
    }
    if (codec == "jpeg2000" || codec == "jpeg2k" || codec == "j2k") {
        // Default to YCbCr JPEG2000 (most common in whole-slide imaging)
        return cuslide::tiff::COMPRESSION_APERIO_JP2K_YCBCR;  // 33003
    }
    if (codec == "lzw") {
        return cuslide::tiff::COMPRESSION_LZW;  // 5
    }
    if (codec == "deflate" || codec == "zip") {
        return cuslide::tiff::COMPRESSION_DEFLATE;  // 8
    }
    if (codec == "adobe-deflate") {
        return cuslide::tiff::COMPRESSION_ADOBE_DEFLATE;  // 32946
    }
    if (codec == "none" || codec == "uncompressed" || codec.empty()) {
        return cuslide::tiff::COMPRESSION_NONE;  // 1
    }

    // Handle generic 'tiff' codec from nvImageCodec 0.6.0
    // This is a known limitation where nvImageCodec doesn't expose the actual compression
    // For now, default to JPEG which is most common in whole-slide imaging
    if (codec == "tiff") {
        #ifdef DEBUG
        fmt::print("ℹ️  nvImageCodec returned generic 'tiff' codec, assuming JPEG compression\n");
        #endif
        return cuslide::tiff::COMPRESSION_JPEG;  // 7 - Most common for WSI (Aperio, Philips, etc.)
    }

    // Unknown codec - log warning and default to JPEG (safer than NONE for WSI)
    #ifdef DEBUG
    fmt::print("⚠️  Unknown codec '{}', defaulting to COMPRESSION_JPEG\n", codec);
    #endif
    return cuslide::tiff::COMPRESSION_JPEG;  // 7 - WSI files rarely use uncompressed
}
#endif // CUCIM_HAS_NVIMGCODEC

bool IFD::is_compression_supported() const
{
    switch (compression_)
    {
    case cuslide::tiff::COMPRESSION_NONE:
    case cuslide::tiff::COMPRESSION_JPEG:
    case cuslide::tiff::COMPRESSION_ADOBE_DEFLATE:
    case cuslide::tiff::COMPRESSION_DEFLATE:
    case cuslide::tiff::COMPRESSION_APERIO_JP2K_YCBCR: // 33003: Jpeg 2000 with YCbCr format
    case cuslide::tiff::COMPRESSION_APERIO_JP2K_RGB:   // 33005: Jpeg 2000 with RGB
    case cuslide::tiff::COMPRESSION_LZW:
        return true;
    default:
        return false;
    }
}

bool IFD::is_read_optimizable() const
{
    return is_compression_supported() && bits_per_sample_ == 8 && samples_per_pixel_ == 3 &&
           (tile_width_ != 0 && tile_height_ != 0) && planar_config_ == cuslide::tiff::PLANARCONFIG_CONTIG &&
           (photometric_ == cuslide::tiff::PHOTOMETRIC_RGB || photometric_ == cuslide::tiff::PHOTOMETRIC_YCBCR) &&
           !tiff_->is_in_read_config(TIFF::kUseLibTiff);
}

bool IFD::is_format_supported() const
{
    return is_compression_supported();
}

} // namespace cuslide::tiff
