/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ifd.h"

#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
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
    fmt::print("🎯 IFD::read() ENTRY: IFD[{}], location=({}, {}), size={}x{}, device={}\n",
              ifd_index_,
              request->location[0], request->location[1],
              request->size[0], request->size[1],
              request->device);
    #endif

#ifdef CUCIM_HAS_NVIMGCODEC
    // Fast path: Use nvImageCodec ROI decoding when available
    // ROI decoding is supported in nvImageCodec v0.6.0+ for JPEG2000
    // Falls back to tile-based decoding if ROI decode fails
    if (nvimgcodec_sub_stream_ && tiff->nvimgcodec_parser_ &&
        request->location_len == 1 && request->batch_size == 1)
    {
        std::string device_name(request->device);
        if (request->shm_name)
        {
            device_name = device_name + fmt::format("[{}]", request->shm_name);
        }
        cucim::io::Device out_device(device_name);

        int64_t sx = request->location[0];
        int64_t sy = request->location[1];
        int64_t w = request->size[0];
        int64_t h = request->size[1];

        // Output buffer - decode function will allocate (uses pinned memory for CPU)
        uint8_t* output_buffer = nullptr;
        DLTensor* out_buf = request->buf;
        bool is_buf_available = out_buf && out_buf->data;

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
            &output_buffer,
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

            // Free allocated buffer on failure
            // Note: decode function uses cudaMallocHost for CPU (pinned memory)
            if (!is_buf_available && output_buffer)
            {
                if (out_device.type() == cucim::io::DeviceType::kCUDA)
                {
                    cudaFree(output_buffer);
                }
                else
                {
                    cudaFreeHost(output_buffer);  // Pinned memory
                }
            }

            throw std::runtime_error(fmt::format(
                "Failed to decode IFD[{}] with nvImageCodec. ROI: ({},{}) {}x{}",
                ifd_index_, sx, sy, w, h));
        }
    }
#endif

    // If we reach here, nvImageCodec is not available or request doesn't match fast path
    #ifdef DEBUG
    fmt::print("❌ Cannot decode: nvImageCodec not available or unsupported request type\n");
#ifdef CUCIM_HAS_NVIMGCODEC
    fmt::print("   nvimgcodec_sub_stream_={}, location_len={}, batch_size={}\n",
              static_cast<void*>(nvimgcodec_sub_stream_), request->location_len, request->batch_size);
#else
    fmt::print("   location_len={}, batch_size={}\n",
              request->location_len, request->batch_size);
#endif
    #endif
    throw std::runtime_error(fmt::format(
        "IFD[{}]: This library requires nvImageCodec for image decoding. "
        "Multi-location/batch requests not yet supported.", ifd_index_));

    // OLD TILE-BASED CODE BELOW - REMOVED
    // All the code below this point (lines 335-1573) can be deleted
    // Keeping it commented for reference during transition

#if 0  // DISABLED - Old libtiff tile-based code
    ::TIFF* tif = tiff->tiff_client_;

    uint16_t ifd_index = ifd_index_;

    std::string device_name(request->device);

    if (request->shm_name)
    {
        device_name = device_name + fmt::format("[{}]", request->shm_name); // TODO: check performance
    }
    cucim::io::Device out_device(device_name);

    int64_t sx = request->location[0];
    int64_t sy = request->location[1];
    uint32_t batch_size = request->batch_size;
    int64_t w = request->size[0];
    int64_t h = request->size[1];
    int32_t n_ch = samples_per_pixel_; // number of channels
    int ndim = 3;

    size_t raster_size = w * h * samples_per_pixel_;
    void* raster = nullptr;
    auto raster_type = cucim::io::DeviceType::kCPU;

    DLTensor* out_buf = request->buf;
    bool is_buf_available = out_buf && out_buf->data;

    if (is_buf_available)
    {
        // TODO: memory size check if out_buf->data has high-enough memory (>= tjBufSize())
        raster = out_buf->data;
    }

    fmt::print("🔎 Checking is_read_optimizable(): {}\n", is_read_optimizable());

    if (is_read_optimizable())
    {
        fmt::print("✅ Using optimized read path\n");
        if (batch_size > 1)
        {
            ndim = 4;
        }
        int64_t* location = request->location;
        uint64_t location_len = request->location_len;

        // TEMPORARY: Force synchronous execution to avoid thread pool issues
        const uint32_t num_workers = 0;  // Was: request->num_workers;
        fmt::print("⚠️  FORCED num_workers=0 for synchronous execution (debugging)\n");
        fflush(stdout);

        const bool drop_last = request->drop_last;
        uint32_t prefetch_factor = request->prefetch_factor;
        const bool shuffle = request->shuffle;
        const uint64_t seed = request->seed;

        if (num_workers == 0 && location_len > 1)
        {
            throw std::runtime_error("Cannot read multiple locations with zero workers!");
        }

        // Shuffle data
        if (shuffle)
        {
            auto rng = std::default_random_engine{ seed };
            struct position
            {
                int64_t x;
                int64_t y;
            };
            std::shuffle(reinterpret_cast<position*>(&location[0]),
                         reinterpret_cast<position*>(&location[location_len * 2]), rng);
        }

        // Adjust location length based on 'drop_last'
        const uint32_t remaining_len = location_len % batch_size;
        if (drop_last)
        {
            location_len -= remaining_len;
        }

        // Do not use prefetch if the image is too small
        if (1 + prefetch_factor > location_len)
        {
            prefetch_factor = location_len - 1;
        }

        size_t one_raster_size = raster_size;
        raster_size *= batch_size;

        const IFD* ifd = this;

        fmt::print("📍 location_len={}, batch_size={}, num_workers={}\n", location_len, batch_size, num_workers);
        fmt::print("📍 BUFFER_SIZE: w={}, h={}, samples_per_pixel={}\n", w, h, samples_per_pixel_);
        fmt::print("📍 BUFFER_SIZE: one_raster_size={} bytes ({} KB)\n", one_raster_size, one_raster_size / 1024);
        fmt::print("📍 BUFFER_SIZE: total raster_size={} bytes ({} KB)\n", raster_size, raster_size / 1024);

        if (location_len > 1 || batch_size > 1 || num_workers > 0)
        {
            fmt::print("📍 Entering multi-location/batch/worker path\n");
            // Reconstruct location
            std::unique_ptr<std::vector<int64_t>>* location_unique =
                reinterpret_cast<std::unique_ptr<std::vector<int64_t>>*>(request->location_unique);
            std::unique_ptr<std::vector<int64_t>> request_location = std::move(*location_unique);
            delete location_unique;

            // Reconstruct size
            std::unique_ptr<std::vector<int64_t>>* size_unique =
                reinterpret_cast<std::unique_ptr<std::vector<int64_t>>*>(request->size_unique);
            std::unique_ptr<std::vector<int64_t>> request_size = std::move(*size_unique);
            delete size_unique;

            auto load_func = [tiff, ifd, location, w, h, out_device](
                                 cucim::loader::ThreadBatchDataLoader* loader_ptr, uint64_t location_index) {
                fmt::print("🔍 load_func: ENTRY - location_index={}\n", location_index);
                fflush(stdout);
                uint8_t* raster_ptr = loader_ptr->raster_pointer(location_index);
                fmt::print("🔍 load_func: Got raster_ptr={}\n", static_cast<void*>(raster_ptr));
                fflush(stdout);

                fmt::print("🔍 load_func: Calling read_region_tiles\n");
                fflush(stdout);
                if (!read_region_tiles(tiff, ifd, location, location_index, w, h,
                                       raster_ptr, out_device, loader_ptr))
                {
                    fmt::print(stderr, "[Error] Failed to read region!\n");
                    fflush(stderr);
                }
                fmt::print("🔍 load_func: read_region_tiles completed\n");
                fflush(stdout);
            };

            uint32_t maximum_tile_count = 0;

            std::unique_ptr<cucim::loader::BatchDataProcessor> batch_processor;

            // Set raster_type to CUDA because loader will handle this with nvjpeg
            // BUT: NvJpegProcessor only handles JPEG (not JPEG2000), so check compression
            fmt::print("📍 Checking device type: {} compression: {}\n",
                      static_cast<int>(out_device.type()), compression_);

            bool is_jpeg2000 = (compression_ == COMPRESSION_APERIO_JP2K_YCBCR ||
                               compression_ == COMPRESSION_APERIO_JP2K_RGB);

            if (out_device.type() == cucim::io::DeviceType::kCUDA && !is_jpeg2000)
            {
                fmt::print("📍 Using CUDA device path with nvjpeg loader\n");
                raster_type = cucim::io::DeviceType::kCUDA;

                // The maximal number of tiles (x-axis) overapped with the given patch
                uint32_t tile_across_count = std::min(static_cast<uint64_t>(ifd->width_) + (ifd->tile_width_ - 1),
                                                      static_cast<uint64_t>(w) + (ifd->tile_width_ - 1)) /
                                                 ifd->tile_width_ +
                                             1;
                // The maximal number of tiles (y-axis) overapped with the given patch
                uint32_t tile_down_count = std::min(static_cast<uint64_t>(ifd->height_) + (ifd->tile_height_ - 1),
                                                    static_cast<uint64_t>(h) + (ifd->tile_height_ - 1)) /
                                               ifd->tile_height_ +
                                           1;
                // The maximal number of possible tiles (# of tasks) to load for the given image batch
                maximum_tile_count = tile_across_count * tile_down_count * batch_size;

                // Create NvJpegProcessor
                auto& jpegtable = ifd->jpegtable_;
                const void* jpegtable_data = jpegtable.data();
                uint32_t jpegtable_size = jpegtable.size();

                auto nvjpeg_processor = std::make_unique<cuslide::loader::NvJpegProcessor>(
                    tiff->file_handle_shared_.get(), ifd, request_location->data(), request_size->data(), location_len, batch_size,
                    maximum_tile_count, static_cast<const uint8_t*>(jpegtable_data), jpegtable_size);

                // Update prefetch_factor
                prefetch_factor = nvjpeg_processor->preferred_loader_prefetch_factor();

                batch_processor = std::move(nvjpeg_processor);
                fmt::print("📍 NvJpegProcessor created\n");
            }
            else if (is_jpeg2000)
            {
                fmt::print("⚠️  JPEG2000 detected - skipping NvJpegProcessor (will use nvImageCodec/OpenJPEG)\n");
            }

            fmt::print("📍 Creating ThreadBatchDataLoader (location_len={}, batch_size={}, num_workers={})\n",
                      location_len, batch_size, num_workers);
            auto loader = std::make_unique<cucim::loader::ThreadBatchDataLoader>(
                load_func, std::move(batch_processor), out_device, std::move(request_location), std::move(request_size),
                location_len, one_raster_size, batch_size, prefetch_factor, num_workers);
            fmt::print("📍 ThreadBatchDataLoader created\n");

            const uint32_t load_size = std::min(static_cast<uint64_t>(batch_size) * (1 + prefetch_factor), location_len);

            fmt::print("📍 Calling loader->request({})\n", load_size);
            loader->request(load_size);
            fmt::print("📍 loader->request() completed\n");

            // If it reads entire image with multi threads (using loader), fetch the next item.
            if (location_len == 1 && batch_size == 1)
            {
                fmt::print("📍 Calling loader->next_data()\n");
                raster = loader->next_data();
                fmt::print("📍 loader->next_data() returned\n");
            }

            out_image_data->loader = loader.release(); // set loader to out_image_data
        }
        else
        {
            if (!raster)
            {
                raster = cucim_malloc(one_raster_size);
            }

            if (!read_region_tiles(tiff, ifd, location, 0, w, h, raster, out_device, nullptr))
            {
                fmt::print(stderr, "[Error] Failed to read region!\n");
            }
        }
    }
    else
    {
        PROF_SCOPED_RANGE(PROF_EVENT(ifd_read_slowpath));
        // Print a warning message for the slow path
        std::call_once(
            tiff->slow_path_warning_flag_,
            [](const std::string& file_path) {
                fmt::print(
                    stderr,
                    "[Warning] Loading image('{}') with a slow-path. The pixel format of the loaded image would be RGBA (4 channels) instead of RGB!\n",
                    file_path);
            },
            tiff->file_path());
        // Handle out-of-boundary case
        int64_t ex = sx + w - 1;
        int64_t ey = sy + h - 1;
        if (sx < 0 || sy < 0 || sx >= width_ || sy >= height_ || ex < 0 || ey < 0 || ex >= width_ || ey >= height_)
        {
            throw std::invalid_argument(fmt::format("Cannot handle the out-of-boundary cases."));
        }

        // Check if the image format is supported or not
        if (!is_format_supported())
        {
            throw std::runtime_error(fmt::format(
                "This format (compression: {}, sample_per_pixel: {}, planar_config: {}, photometric: {}) is not supported yet!.",
                compression_, samples_per_pixel_, planar_config_, photometric_));
        }

        if (tif->tif_curdir != ifd_index)
        {
            TIFFSetDirectory(tif, ifd_index);
        }
        // RGBA -> 4 channels
        n_ch = 4;

        char emsg[1024];
        if (TIFFRGBAImageOK(tif, emsg))
        {
            TIFFRGBAImage img;
            if (TIFFRGBAImageBegin(&img, tif, -1, emsg))
            {
                size_t npixels;
                npixels = w * h;
                raster_size = npixels * 4;
                if (!raster)
                {
                    raster = cucim_malloc(raster_size);
                }
                img.col_offset = sx;
                img.row_offset = sy;
                img.req_orientation = ORIENTATION_TOPLEFT;

                if (raster != nullptr)
                {
                    if (!TIFFRGBAImageGet(&img, (uint32_t*)raster, w, h))
                    {
                        memset(raster, 0, raster_size);
                    }
                }
            }
            else
            {
                throw std::runtime_error(fmt::format(
                    "This format (compression: {}, sample_per_pixel: {}, planar_config: {}, photometric: {}) is not supported yet!: {}",
                    compression_, samples_per_pixel_, planar_config_, photometric_, emsg));
            }
            TIFFRGBAImageEnd(&img);
        }
        else
        {
            throw std::runtime_error(fmt::format(
                "This format (compression: {}, sample_per_pixel: {}, planar_config: {}, photometric: {}) is not supported yet!: {}",
                compression_, samples_per_pixel_, planar_config_, photometric_, emsg));
        }
    }

    int64_t* shape = static_cast<int64_t*>(cucim_malloc(sizeof(int64_t) * ndim));
    if (ndim == 3)
    {
        shape[0] = h;
        shape[1] = w;
        shape[2] = n_ch;
    }
    else // ndim == 4
    {
        shape[0] = batch_size;
        shape[1] = h;
        shape[2] = w;
        shape[3] = n_ch;
    }

    // Copy the raster memory and free it if needed.
    if (!is_buf_available && raster && raster_type == cucim::io::DeviceType::kCPU)
    {
        cucim::memory::move_raster_from_host(&raster, raster_size, out_device);
    }

    auto& out_image_container = out_image_data->container;
    out_image_container.data = raster;
    out_image_container.device = DLDevice{ static_cast<DLDeviceType>(out_device.type()), out_device.index() };
    out_image_container.ndim = ndim;
    out_image_container.dtype = metadata->dtype;
    out_image_container.shape = shape;
    out_image_container.strides = nullptr; // Tensor is compact and row-majored
    out_image_container.byte_offset = 0;
    auto& shm_name = out_device.shm_name();
    size_t shm_name_len = shm_name.size();
    if (shm_name_len != 0)
    {
        out_image_data->shm_name = static_cast<char*>(cucim_malloc(shm_name_len + 1));
        memcpy(out_image_data->shm_name, shm_name.c_str(), shm_name_len + 1);
    }
    else
    {
        out_image_data->shm_name = nullptr;
    }

    return true;
#endif  // End of #if 0 - disabled old tile-based code
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

bool IFD::read_region_tiles(const TIFF* tiff,
                            const IFD* ifd,
                            const int64_t* location,
                            const int64_t location_index,
                            const int64_t w,
                            const int64_t h,
                            void* raster,
                            const cucim::io::Device& out_device,
                            cucim::loader::ThreadBatchDataLoader* loader)
{
    #ifdef DEBUG
    fmt::print("🔍 read_region_tiles: ENTRY - location_index={}, w={}, h={}, loader={}\n",
              location_index, w, h, static_cast<void*>(loader));
    #endif
    PROF_SCOPED_RANGE(PROF_EVENT(ifd_read_region_tiles));
    // Reference code: https://github.com/libjpeg-turbo/libjpeg-turbo/blob/master/tjexample.c

    int64_t sx = location[location_index * 2];
    int64_t sy = location[location_index * 2 + 1];
    int64_t ex = sx + w - 1;
    int64_t ey = sy + h - 1;
    #ifdef DEBUG
    fmt::print("🔍 read_region_tiles: Region bounds - sx={}, sy={}, ex={}, ey={}\n", sx, sy, ex, ey);
    #endif

    uint32_t width = ifd->width_;
    uint32_t height = ifd->height_;

    // Handle out-of-boundary case
    if (sx < 0 || sy < 0 || sx >= width || sy >= height || ex < 0 || ey < 0 || ex >= width || ey >= height)
    {
        return read_region_tiles_boundary(tiff, ifd, location, location_index, w, h, raster, out_device, loader);
    }
    cucim::cache::ImageCache& image_cache = cucim::CuImage::cache_manager().cache();
    cucim::cache::CacheType cache_type = image_cache.type();

    uint8_t background_value = tiff->background_value_;
    uint16_t compression_method = ifd->compression_;
    int jpeg_color_space = ifd->jpeg_color_space_;
    int predictor = ifd->predictor_;

    // TODO: revert this once we can get RGB data instead of RGBA
    uint32_t samples_per_pixel = 3; // ifd->samples_per_pixel();

    const void* jpegtable_data = ifd->jpegtable_.data();
    uint32_t jpegtable_count = ifd->jpegtable_.size();

    uint32_t tw = ifd->tile_width_;
    uint32_t th = ifd->tile_height_;

    uint32_t offset_sx = static_cast<uint32_t>(sx / tw); // x-axis start offset for the requested region in the ifd tile
                                                         // array as grid
    uint32_t offset_ex = static_cast<uint32_t>(ex / tw); // x-axis end  offset for the requested region in the ifd tile
                                                         // array as grid
    uint32_t offset_sy = static_cast<uint32_t>(sy / th); // y-axis start offset for the requested region in the ifd tile
                                                         // array as grid
    uint32_t offset_ey = static_cast<uint32_t>(ey / th); // y-axis end offset for the requested region in the ifd tile
                                                         // array as grid

    uint32_t pixel_offset_sx = static_cast<uint32_t>(sx % tw);
    uint32_t pixel_offset_ex = static_cast<uint32_t>(ex % tw);
    uint32_t pixel_offset_sy = static_cast<uint32_t>(sy % th);
    uint32_t pixel_offset_ey = static_cast<uint32_t>(ey % th);

    uint32_t stride_y = width / tw + !!(width % tw); // # of tiles in a row(y) in the ifd tile array as grid

    uint32_t start_index_y = offset_sy * stride_y;
    uint32_t end_index_y = offset_ey * stride_y;

    const size_t tile_raster_nbytes = ifd->tile_raster_size_nbytes();

    int tiff_file = tiff->file_handle_shared_.get()->fd;
    uint64_t ifd_hash_value = ifd->hash_value_;
    uint32_t dest_pixel_step_y = w * samples_per_pixel;

    uint32_t nbytes_tw = tw * samples_per_pixel;
    auto dest_start_ptr = static_cast<uint8_t*>(raster);

    // TODO: Current implementation doesn't consider endianness so need to consider later
    // TODO: Consider tile's depth tag.
    for (uint32_t index_y = start_index_y; index_y <= end_index_y; index_y += stride_y)
    {
        uint32_t tile_pixel_offset_sy = (index_y == start_index_y) ? pixel_offset_sy : 0;
        uint32_t tile_pixel_offset_ey = (index_y == end_index_y) ? pixel_offset_ey : (th - 1);
        uint32_t dest_pixel_offset_len_y = tile_pixel_offset_ey - tile_pixel_offset_sy + 1;

        uint32_t dest_pixel_index_x = 0;

        uint32_t index = index_y + offset_sx;
        for (uint32_t offset_x = offset_sx; offset_x <= offset_ex; ++offset_x, ++index)
        {
            #ifdef DEBUG
            fmt::print("🔍 read_region_tiles: Processing tile index={}, offset_x={}\n", index, offset_x);
            #endif
            PROF_SCOPED_RANGE(PROF_EVENT_P(ifd_read_region_tiles_iter, index));
            auto tiledata_offset = static_cast<uint64_t>(ifd->image_piece_offsets_[index]);
            auto tiledata_size = static_cast<uint64_t>(ifd->image_piece_bytecounts_[index]);
            #ifdef DEBUG
            fmt::print("🔍 read_region_tiles: tile_offset={}, tile_size={}\n", tiledata_offset, tiledata_size);
            #endif

            // Calculate a simple hash value for the tile index
            uint64_t index_hash = ifd_hash_value ^ (static_cast<uint64_t>(index) | (static_cast<uint64_t>(index) << 32));

            uint32_t tile_pixel_offset_x = (offset_x == offset_sx) ? pixel_offset_sx : 0;
            uint32_t nbytes_tile_pixel_size_x = (offset_x == offset_ex) ?
                                                    (pixel_offset_ex - tile_pixel_offset_x + 1) * samples_per_pixel :
                                                    (tw - tile_pixel_offset_x) * samples_per_pixel;
            #ifdef DEBUG
            fmt::print("🔍 read_region_tiles: About to create decode_func lambda\n");
            fflush(stdout);
            #endif

            // Capture device type as integer to avoid copying Device object
            auto device_type_int = static_cast<int>(out_device.type());
            auto device_index = out_device.index();

            // Create a struct to hold all data - avoids large lambda captures
            struct TileDecodeData {
                uint32_t index;
                uint64_t index_hash;
                uint16_t compression_method;
                uint64_t tiledata_offset;
                uint64_t tiledata_size;
                uint32_t tile_pixel_offset_sy, tile_pixel_offset_ey, tile_pixel_offset_x;
                uint32_t tw, th, samples_per_pixel, nbytes_tw, nbytes_tile_pixel_size_x;
                uint32_t dest_pixel_index_x;
                uint8_t* dest_start_ptr;
                uint32_t dest_pixel_step_y;
                int tiff_file;
                uint64_t ifd_hash_value;
                size_t tile_raster_nbytes;
                cucim::cache::CacheType cache_type;
                const void* jpegtable_data;
                uint32_t jpegtable_count;
                int jpeg_color_space;
                uint8_t background_value;
                int predictor;
                int device_type_int;
                int16_t device_index;
                cucim::loader::ThreadBatchDataLoader* loader;
            };

            auto data = std::make_shared<TileDecodeData>();
            data->index = index;
            data->index_hash = index_hash;
            data->compression_method = compression_method;
            data->tiledata_offset = tiledata_offset;
            data->tiledata_size = tiledata_size;
            data->tile_pixel_offset_sy = tile_pixel_offset_sy;
            data->tile_pixel_offset_ey = tile_pixel_offset_ey;
            data->tile_pixel_offset_x = tile_pixel_offset_x;
            data->tw = tw;
            data->th = th;
            data->samples_per_pixel = samples_per_pixel;
            data->nbytes_tw = nbytes_tw;
            data->nbytes_tile_pixel_size_x = nbytes_tile_pixel_size_x;
            data->dest_pixel_index_x = dest_pixel_index_x;
            data->dest_start_ptr = dest_start_ptr;
            data->dest_pixel_step_y = dest_pixel_step_y;
            data->tiff_file = tiff_file;
            data->ifd_hash_value = ifd_hash_value;
            data->tile_raster_nbytes = tile_raster_nbytes;
            data->cache_type = cache_type;
            data->jpegtable_data = jpegtable_data;
            data->jpegtable_count = jpegtable_count;
            data->jpeg_color_space = jpeg_color_space;
            data->background_value = background_value;
            data->predictor = predictor;
            data->device_type_int = device_type_int;
            data->device_index = device_index;
            data->loader = loader;

            // Small lambda that only captures shared_ptr - cheap to copy!
            auto decode_func = [data]() {
                // FIRST THING - print before ANY other code
                #ifdef DEBUG
                fmt::print("🔍🔍🔍 decode_func: LAMBDA INVOKED! index={}\n", data->index);
                fflush(stdout);
                #endif

                // Extract all data to local variables to avoid repeated data-> access
                auto index = data->index;
                auto index_hash = data->index_hash;
                auto compression_method = data->compression_method;
                auto tiledata_offset = data->tiledata_offset;
                auto tiledata_size = data->tiledata_size;
                auto tile_pixel_offset_sy = data->tile_pixel_offset_sy;
                auto tile_pixel_offset_ey = data->tile_pixel_offset_ey;
                auto tile_pixel_offset_x = data->tile_pixel_offset_x;
                auto tw = data->tw;
                [[maybe_unused]] auto th = data->th;
                auto samples_per_pixel = data->samples_per_pixel;
                auto nbytes_tw = data->nbytes_tw;
                auto nbytes_tile_pixel_size_x = data->nbytes_tile_pixel_size_x;
                auto dest_pixel_index_x = data->dest_pixel_index_x;
                auto dest_start_ptr = data->dest_start_ptr;
                auto dest_pixel_step_y = data->dest_pixel_step_y;
                // REMOVED: Legacy CPU decoder variables (unused after removing CPU decoder code)
                // auto tiff_file = data->tiff_file;
                auto ifd_hash_value = data->ifd_hash_value;
                auto tile_raster_nbytes = data->tile_raster_nbytes;
                auto cache_type = data->cache_type;
                // REMOVED: Legacy CPU decoder variables (unused after removing CPU decoder code)
                // auto jpegtable_data = data->jpegtable_data;
                // auto jpegtable_count = data->jpegtable_count;
                // auto jpeg_color_space = data->jpeg_color_space;
                // auto predictor = data->predictor;
                auto background_value = data->background_value;
                auto loader = data->loader;

                // Reconstruct Device object inside lambda to avoid copying issues
                cucim::io::Device out_device(static_cast<cucim::io::DeviceType>(data->device_type_int), data->device_index);
                try {
                    #ifdef DEBUG
                    fmt::print("🔍 decode_func: START - index={}, compression={}, tiledata_offset={}, tiledata_size={}\n",
                              index, compression_method, tiledata_offset, tiledata_size);
                    fflush(stdout);
                    #endif
                    PROF_SCOPED_RANGE(PROF_EVENT_P(ifd_read_region_tiles_task, index_hash));

                    // Get image cache directly instead of capturing by reference
                    #ifdef DEBUG
                    fmt::print("🔍 decode_func: Getting image cache...\n");
                    fflush(stdout);
                    #endif
                    cucim::cache::ImageCache& image_cache = cucim::CuImage::cache_manager().cache();
                    #ifdef DEBUG
                    fmt::print("🔍 decode_func: Got image cache\n");
                    fflush(stdout);
                    #endif

                    uint32_t nbytes_tile_index = (tile_pixel_offset_sy * tw + tile_pixel_offset_x) * samples_per_pixel;
                    uint32_t dest_pixel_index = dest_pixel_index_x;
                    uint8_t* tile_data = nullptr;
                    if (tiledata_size > 0)
                    {
                        #ifdef DEBUG
                        fmt::print("🔍 decode_func: tiledata_size > 0, entering decode path\n");
                        #endif
                        std::unique_ptr<uint8_t, decltype(cucim_free)*> tile_raster =
                            std::unique_ptr<uint8_t, decltype(cucim_free)*>(nullptr, cucim_free);

                    if (loader && loader->batch_data_processor())
                    {
                        switch (compression_method)
                        {
                        case COMPRESSION_JPEG:
                        case COMPRESSION_APERIO_JP2K_YCBCR: // 33003
                        case COMPRESSION_APERIO_JP2K_RGB:   // 33005
                            break;
                        default:
                            throw std::runtime_error("Unsupported compression method");
                        }
                        auto value = loader->wait_for_processing(index);
                        if (!value) // if shutdown
                        {
                            return;
                        }
                        tile_data = static_cast<uint8_t*>(value->data);

                        cudaError_t cuda_status;
                        CUDA_ERROR(cudaMemcpy2D(dest_start_ptr + dest_pixel_index, dest_pixel_step_y,
                                                tile_data + nbytes_tile_index, nbytes_tw, nbytes_tile_pixel_size_x,
                                                tile_pixel_offset_ey - tile_pixel_offset_sy + 1,
                                                cudaMemcpyDeviceToDevice));
                    }
                    else
                    {
                        auto key = image_cache.create_key(ifd_hash_value, index);
                        image_cache.lock(index_hash);
                        auto value = image_cache.find(key);
                        if (value)
                        {
                            image_cache.unlock(index_hash);
                            tile_data = static_cast<uint8_t*>(value->data);
                        }
                        else
                        {
                            // Lifetime of tile_data is same with `value`
                            // : do not access this data when `value` is not accessible.
                            if (cache_type != cucim::cache::CacheType::kNoCache)
                            {
                                tile_data = static_cast<uint8_t*>(image_cache.allocate(tile_raster_nbytes));
                            }
                            else
                            {
                                // Allocate temporary buffer for tile data
                                tile_raster = std::unique_ptr<uint8_t, decltype(cucim_free)*>(
                                    reinterpret_cast<uint8_t*>(cucim_malloc(tile_raster_nbytes)), cucim_free);
                                tile_data = tile_raster.get();
                            }
                            {
                                // REMOVED: Legacy CPU decoder fallback code
                                // This code path should NOT be reached in a pure nvImageCodec build.
                                // All decoding should go through the nvImageCodec ROI path (lines 219-276).
                                // If you see this error, investigate why ROI decode failed.
                                throw std::runtime_error(fmt::format(
                                    "INTERNAL ERROR: Tile-based CPU decoder fallback reached. "
                                    "This should not happen in nvImageCodec build. "
                                    "Compression method: {}, tile offset: {}, size: {}",
                                    compression_method, tiledata_offset, tiledata_size));
                            }

                            value = image_cache.create_value(tile_data, tile_raster_nbytes);
                            image_cache.insert(key, value);
                            image_cache.unlock(index_hash);
                        }

                        for (uint32_t ty = tile_pixel_offset_sy; ty <= tile_pixel_offset_ey;
                             ++ty, dest_pixel_index += dest_pixel_step_y, nbytes_tile_index += nbytes_tw)
                        {
                            memcpy(dest_start_ptr + dest_pixel_index, tile_data + nbytes_tile_index,
                                   nbytes_tile_pixel_size_x);
                        }
                    }
                }
                else
                {
                    if (out_device.type() == cucim::io::DeviceType::kCPU)
                    {
                        for (uint32_t ty = tile_pixel_offset_sy; ty <= tile_pixel_offset_ey;
                             ++ty, dest_pixel_index += dest_pixel_step_y, nbytes_tile_index += nbytes_tw)
                        {
                            // Set background value such as (255,255,255)
                            memset(dest_start_ptr + dest_pixel_index, background_value, nbytes_tile_pixel_size_x);
                        }
                    }
                    else
                    {
                        cudaError_t cuda_status;
                        CUDA_ERROR(cudaMemset2D(dest_start_ptr + dest_pixel_index, dest_pixel_step_y, background_value,
                                                nbytes_tile_pixel_size_x,
                                                tile_pixel_offset_ey - tile_pixel_offset_sy + 1));
                    }
                }
                } catch (const std::exception& e) {
                    #ifdef DEBUG
                    fmt::print("❌ decode_func: Exception caught: {}\n", e.what());
                    #endif
                    throw;
                } catch (...) {
                    #ifdef DEBUG
                    fmt::print("❌ decode_func: Unknown exception caught\n");
                    #endif
                    throw;
                }
            };

            #ifdef DEBUG
            fmt::print("🔍 read_region_tiles: decode_func lambda created\n");
            #endif

            // TEMPORARY: Force single-threaded execution to test if decode works
            bool force_single_threaded = true;

            if (force_single_threaded || !loader || !(*loader))
            {
                #ifdef DEBUG
                fmt::print("🔍 read_region_tiles: Executing decode_func directly (FORCED SINGLE-THREADED TEST)\n");
                fflush(stdout);
                #endif
                decode_func();
                #ifdef DEBUG
                fmt::print("🔍 read_region_tiles: decode_func completed successfully!\n");
                fflush(stdout);
                #endif
            }
            else
            {
                #ifdef DEBUG
                fmt::print("🔍 read_region_tiles: Enqueueing task for tile index={}\n", index);
                #endif
                loader->enqueue(std::move(decode_func),
                                cucim::loader::TileInfo{ location_index, index, tiledata_offset, tiledata_size });
                #ifdef DEBUG
                fmt::print("🔍 read_region_tiles: Task enqueued\n");
                #endif
            }

            dest_pixel_index_x += nbytes_tile_pixel_size_x;
        }
        dest_start_ptr += dest_pixel_step_y * dest_pixel_offset_len_y;
    }

    return true;
}

bool IFD::read_region_tiles_boundary(const TIFF* tiff,
                                     const IFD* ifd,
                                     const int64_t* location,
                                     const int64_t location_index,
                                     const int64_t w,
                                     const int64_t h,
                                     void* raster,
                                     const cucim::io::Device& out_device,
                                     cucim::loader::ThreadBatchDataLoader* loader)
{
    PROF_SCOPED_RANGE(PROF_EVENT(ifd_read_region_tiles_boundary));
    (void)out_device;
    // Reference code: https://github.com/libjpeg-turbo/libjpeg-turbo/blob/master/tjexample.c
    int64_t sx = location[location_index * 2];
    int64_t sy = location[location_index * 2 + 1];

    uint8_t background_value = tiff->background_value_;
    uint16_t compression_method = ifd->compression_;
    int jpeg_color_space = ifd->jpeg_color_space_;
    int predictor = ifd->predictor_;

    int64_t ex = sx + w - 1;
    int64_t ey = sy + h - 1;

    uint32_t width = ifd->width_;
    uint32_t height = ifd->height_;

    // Memory for tile_raster would be manually allocated here, instead of using decode_libjpeg().
    // Need to free the manually. Usually it is set to nullptr and memory is created by decode_libjpeg() by using
    // tjAlloc() (Also need to free with tjFree() after use. See the documentation of tjAlloc() for the detail.)
    const int pixel_size_nbytes = ifd->pixel_size_nbytes();
    auto dest_start_ptr = static_cast<uint8_t*>(raster);

    bool is_out_of_image = (ex < 0 || width <= sx || ey < 0 || height <= sy);
    if (is_out_of_image)
    {
        // Fill background color(255,255,255) and return
        memset(dest_start_ptr, background_value, w * h * pixel_size_nbytes);
        return true;
    }
    cucim::cache::ImageCache& image_cache = cucim::CuImage::cache_manager().cache();
    cucim::cache::CacheType cache_type = image_cache.type();

    uint32_t tw = ifd->tile_width_;
    uint32_t th = ifd->tile_height_;

    const size_t tile_raster_nbytes = tw * th * pixel_size_nbytes;

    // TODO: revert this once we can get RGB data instead of RGBA
    uint32_t samples_per_pixel = 3; // ifd->samples_per_pixel();

    const void* jpegtable_data = ifd->jpegtable_.data();
    uint32_t jpegtable_count = ifd->jpegtable_.size();

    bool sx_in_range = (sx >= 0 && sx < width);
    bool ex_in_range = (ex >= 0 && ex < width);
    bool sy_in_range = (sy >= 0 && sy < height);
    bool ey_in_range = (ey >= 0 && ey < height);

    int64_t offset_boundary_x = (static_cast<int64_t>(width) - 1) / tw;
    int64_t offset_boundary_y = (static_cast<int64_t>(height) - 1) / th;

    int64_t offset_sx = sx / tw; // x-axis start offset for the requested region in the
                                 // ifd tile array as grid

    int64_t offset_ex = ex / tw; // x-axis end  offset for the requested region in the
                                 // ifd tile array as grid

    int64_t offset_sy = sy / th; // y-axis start offset for the requested region in the
                                 // ifd tile array as grid
    int64_t offset_ey = ey / th; // y-axis end offset for the requested region in the
                                 // ifd tile array as grid
    int64_t pixel_offset_sx = (sx % tw);
    int64_t pixel_offset_ex = (ex % tw);
    int64_t pixel_offset_sy = (sy % th);
    int64_t pixel_offset_ey = (ey % th);
    int64_t pixel_offset_boundary_x = ((width - 1) % tw);
    int64_t pixel_offset_boundary_y = ((height - 1) % th);

    // Make sure that division and modulo has same value with Python's one (e.g., making -1 / 3 == -1 instead of 0)
    if (pixel_offset_sx < 0)
    {
        pixel_offset_sx += tw;
        --offset_sx;
    }
    if (pixel_offset_ex < 0)
    {
        pixel_offset_ex += tw;
        --offset_ex;
    }
    if (pixel_offset_sy < 0)
    {
        pixel_offset_sy += th;
        --offset_sy;
    }
    if (pixel_offset_ey < 0)
    {
        pixel_offset_ey += th;
        --offset_ey;
    }
    int64_t offset_min_x = sx_in_range ? offset_sx : 0;
    int64_t offset_max_x = ex_in_range ? offset_ex : offset_boundary_x;
    int64_t offset_min_y = sy_in_range ? offset_sy : 0;
    int64_t offset_max_y = ey_in_range ? offset_ey : offset_boundary_y;

    uint32_t stride_y = width / tw + !!(width % tw); // # of tiles in a row(y) in the ifd tile array as grid

    int64_t start_index_y = offset_sy * stride_y;
    int64_t start_index_min_y = offset_min_y * stride_y;
    int64_t end_index_y = offset_ey * stride_y;
    int64_t end_index_max_y = offset_max_y * stride_y;
    int64_t boundary_index_y = offset_boundary_y * stride_y;


    int tiff_file = tiff->file_handle_shared_.get()->fd;
    uint64_t ifd_hash_value = ifd->hash_value_;

    uint32_t dest_pixel_step_y = w * samples_per_pixel;
    uint32_t nbytes_tw = tw * samples_per_pixel;


    // TODO: Current implementation doesn't consider endianness so need to consider later
    // TODO: Consider tile's depth tag.
    // TODO: update the type of variables (index, index_y) : other function uses uint32_t
    for (int64_t index_y = start_index_y; index_y <= end_index_y; index_y += stride_y)
    {
        uint32_t tile_pixel_offset_sy = (index_y == start_index_y) ? pixel_offset_sy : 0;
        uint32_t tile_pixel_offset_ey = (index_y == end_index_y) ? pixel_offset_ey : (th - 1);
        uint32_t dest_pixel_offset_len_y = tile_pixel_offset_ey - tile_pixel_offset_sy + 1;

        uint32_t dest_pixel_index_x = 0;

        int64_t index = index_y + offset_sx;
        for (int64_t offset_x = offset_sx; offset_x <= offset_ex; ++offset_x, ++index)
        {
            PROF_SCOPED_RANGE(PROF_EVENT_P(ifd_read_region_tiles_boundary_iter, index));
            uint64_t tiledata_offset = 0;
            uint64_t tiledata_size = 0;

            // Calculate a simple hash value for the tile index
            uint64_t index_hash = ifd_hash_value ^ (static_cast<uint64_t>(index) | (static_cast<uint64_t>(index) << 32));

            if (offset_x >= offset_min_x && offset_x <= offset_max_x && index_y >= start_index_min_y &&
                index_y <= end_index_max_y)
            {
                tiledata_offset = static_cast<uint64_t>(ifd->image_piece_offsets_[index]);
                tiledata_size = static_cast<uint64_t>(ifd->image_piece_bytecounts_[index]);
            }

            uint32_t tile_pixel_offset_x = (offset_x == offset_sx) ? pixel_offset_sx : 0;
            uint32_t nbytes_tile_pixel_size_x = (offset_x == offset_ex) ?
                                                    (pixel_offset_ex - tile_pixel_offset_x + 1) * samples_per_pixel :
                                                    (tw - tile_pixel_offset_x) * samples_per_pixel;

            uint32_t nbytes_tile_index_orig = (tile_pixel_offset_sy * tw + tile_pixel_offset_x) * samples_per_pixel;
            uint32_t dest_pixel_index_orig = dest_pixel_index_x;

            // Capture device type as integer to avoid copying Device object
            auto device_type_int = static_cast<int>(out_device.type());
            auto device_index = out_device.index();

            // Explicitly capture only what's needed to avoid issues with [=]
            auto decode_func = [
                // Tile identification
                index, index_hash,
                // Compression and decoding params
                compression_method, tiledata_offset, tiledata_size,
                // Tile geometry
                tile_pixel_offset_sy, tile_pixel_offset_ey, tile_pixel_offset_x,
                tw, th, samples_per_pixel, nbytes_tw, nbytes_tile_pixel_size_x, pixel_offset_ey,
                // Destination params - using _orig versions
                nbytes_tile_index_orig, dest_pixel_index_orig, dest_start_ptr, dest_pixel_step_y,
                // File and cache params
                tiff_file, ifd_hash_value, tile_raster_nbytes, cache_type,
                // JPEG params
                jpegtable_data, jpegtable_count, jpeg_color_space,
                // Other params
                background_value, predictor, device_type_int, device_index,
                // Boundary-specific params
                offset_x, offset_ex, offset_boundary_x, pixel_offset_boundary_x, pixel_offset_ex,
                offset_boundary_y, pixel_offset_boundary_y, dest_pixel_offset_len_y,
                // Loop/boundary indices
                index_y, boundary_index_y, end_index_y,
                // Loader pointer
                loader
            ]() {
                #ifdef DEBUG
                fmt::print("🔍🔍🔍 decode_func_boundary: LAMBDA INVOKED! index={}\n", index);
                fflush(stdout);
                #endif

                // Reconstruct Device object inside lambda
                cucim::io::Device out_device(static_cast<cucim::io::DeviceType>(device_type_int), device_index);

                PROF_SCOPED_RANGE(PROF_EVENT_P(ifd_read_region_tiles_boundary_task, index_hash));

                // Get image cache directly instead of capturing by reference
                cucim::cache::ImageCache& image_cache = cucim::CuImage::cache_manager().cache();

                uint32_t nbytes_tile_index = nbytes_tile_index_orig;
                uint32_t dest_pixel_index = dest_pixel_index_orig;

                if (tiledata_size > 0)
                {
                    bool copy_partial = false;
                    uint32_t fixed_nbytes_tile_pixel_size_x = nbytes_tile_pixel_size_x;
                    uint32_t fixed_tile_pixel_offset_ey = tile_pixel_offset_ey;

                    if (offset_x == offset_boundary_x)
                    {
                        copy_partial = true;
                        if (offset_x != offset_ex)
                        {
                            fixed_nbytes_tile_pixel_size_x =
                                (pixel_offset_boundary_x - tile_pixel_offset_x + 1) * samples_per_pixel;
                        }
                        else
                        {
                            fixed_nbytes_tile_pixel_size_x =
                                (std::min(pixel_offset_boundary_x, pixel_offset_ex) - tile_pixel_offset_x + 1) *
                                samples_per_pixel;
                        }
                    }
                    if (index_y == boundary_index_y)
                    {
                        copy_partial = true;
                        if (index_y != end_index_y)
                        {
                            fixed_tile_pixel_offset_ey = pixel_offset_boundary_y;
                        }
                        else
                        {
                            fixed_tile_pixel_offset_ey = std::min(pixel_offset_boundary_y, pixel_offset_ey);
                        }
                    }

                    uint8_t* tile_data = nullptr;
                    std::unique_ptr<uint8_t, decltype(cucim_free)*> tile_raster =
                        std::unique_ptr<uint8_t, decltype(cucim_free)*>(nullptr, cucim_free);

                    if (loader && loader->batch_data_processor())
                    {
                        switch (compression_method)
                        {
                        case COMPRESSION_JPEG:
                        case COMPRESSION_APERIO_JP2K_YCBCR: // 33003
                        case COMPRESSION_APERIO_JP2K_RGB:   // 33005
                            break;
                        default:
                            throw std::runtime_error("Unsupported compression method");
                        }
                        auto value = loader->wait_for_processing(index);
                        if (!value) // if shutdown
                        {
                            return;
                        }

                        tile_data = static_cast<uint8_t*>(value->data);

                        cudaError_t cuda_status;
                        if (copy_partial)
                        {
                            uint32_t fill_gap_x = nbytes_tile_pixel_size_x - fixed_nbytes_tile_pixel_size_x;
                            // Fill original, then fill white for remaining
                            if (fill_gap_x > 0)
                            {
                                CUDA_ERROR(cudaMemcpy2D(
                                    dest_start_ptr + dest_pixel_index, dest_pixel_step_y, tile_data + nbytes_tile_index,
                                    nbytes_tw, fixed_nbytes_tile_pixel_size_x,
                                    fixed_tile_pixel_offset_ey - tile_pixel_offset_sy + 1, cudaMemcpyDeviceToDevice));
                                CUDA_ERROR(cudaMemset2D(dest_start_ptr + dest_pixel_index + fixed_nbytes_tile_pixel_size_x,
                                                        dest_pixel_step_y, background_value, fill_gap_x,
                                                        fixed_tile_pixel_offset_ey - tile_pixel_offset_sy + 1));
                                dest_pixel_index +=
                                    dest_pixel_step_y * (fixed_tile_pixel_offset_ey - tile_pixel_offset_sy + 1);
                            }
                            else
                            {
                                CUDA_ERROR(cudaMemcpy2D(
                                    dest_start_ptr + dest_pixel_index, dest_pixel_step_y, tile_data + nbytes_tile_index,
                                    nbytes_tw, fixed_nbytes_tile_pixel_size_x,
                                    fixed_tile_pixel_offset_ey - tile_pixel_offset_sy + 1, cudaMemcpyDeviceToDevice));
                                dest_pixel_index +=
                                    dest_pixel_step_y * (fixed_tile_pixel_offset_ey - tile_pixel_offset_sy + 1);
                            }

                            CUDA_ERROR(cudaMemset2D(dest_start_ptr + dest_pixel_index, dest_pixel_step_y,
                                                    background_value, nbytes_tile_pixel_size_x,
                                                    tile_pixel_offset_ey - (fixed_tile_pixel_offset_ey + 1) + 1));
                        }
                        else
                        {
                            CUDA_ERROR(cudaMemcpy2D(dest_start_ptr + dest_pixel_index, dest_pixel_step_y,
                                                    tile_data + nbytes_tile_index, nbytes_tw, nbytes_tile_pixel_size_x,
                                                    tile_pixel_offset_ey - tile_pixel_offset_sy + 1,
                                                    cudaMemcpyDeviceToDevice));
                        }
                    }
                    else
                    {
                        auto key = image_cache.create_key(ifd_hash_value, index);
                        image_cache.lock(index_hash);
                        auto value = image_cache.find(key);
                        if (value)
                        {
                            image_cache.unlock(index_hash);
                            tile_data = static_cast<uint8_t*>(value->data);
                        }
                        else
                        {
                            // Lifetime of tile_data is same with `value`
                            // : do not access this data when `value` is not accessible.
                            if (cache_type != cucim::cache::CacheType::kNoCache)
                            {
                                tile_data = static_cast<uint8_t*>(image_cache.allocate(tile_raster_nbytes));
                            }
                            else
                            {
                                // Allocate temporary buffer for tile data
                                tile_raster = std::unique_ptr<uint8_t, decltype(cucim_free)*>(
                                    reinterpret_cast<uint8_t*>(cucim_malloc(tile_raster_nbytes)), cucim_free);
                                tile_data = tile_raster.get();
                            }
                            {
                                // REMOVED: Legacy CPU decoder fallback code (duplicate)
                                // This code path should NOT be reached in a pure nvImageCodec build.
                                // All decoding should go through the nvImageCodec ROI path (lines 219-276).
                                // If you see this error, investigate why ROI decode failed.
                                throw std::runtime_error(fmt::format(
                                    "INTERNAL ERROR: Tile-based CPU decoder fallback reached. "
                                    "This should not happen in nvImageCodec build. "
                                    "Compression method: {}, tile offset: {}, size: {}",
                                    compression_method, tiledata_offset, tiledata_size));
                            }
                            value = image_cache.create_value(tile_data, tile_raster_nbytes);
                            image_cache.insert(key, value);
                            image_cache.unlock(index_hash);
                        }
                        if (copy_partial)
                        {
                            uint32_t fill_gap_x = nbytes_tile_pixel_size_x - fixed_nbytes_tile_pixel_size_x;
                            // Fill original, then fill white for remaining
                            if (fill_gap_x > 0)
                            {
                                for (uint32_t ty = tile_pixel_offset_sy; ty <= fixed_tile_pixel_offset_ey;
                                     ++ty, dest_pixel_index += dest_pixel_step_y, nbytes_tile_index += nbytes_tw)
                                {
                                    memcpy(dest_start_ptr + dest_pixel_index, tile_data + nbytes_tile_index,
                                           fixed_nbytes_tile_pixel_size_x);
                                    memset(dest_start_ptr + dest_pixel_index + fixed_nbytes_tile_pixel_size_x,
                                           background_value, fill_gap_x);
                                }
                            }
                            else
                            {
                                for (uint32_t ty = tile_pixel_offset_sy; ty <= fixed_tile_pixel_offset_ey;
                                     ++ty, dest_pixel_index += dest_pixel_step_y, nbytes_tile_index += nbytes_tw)
                                {
                                    memcpy(dest_start_ptr + dest_pixel_index, tile_data + nbytes_tile_index,
                                           fixed_nbytes_tile_pixel_size_x);
                                }
                            }

                            for (uint32_t ty = fixed_tile_pixel_offset_ey + 1; ty <= tile_pixel_offset_ey;
                                 ++ty, dest_pixel_index += dest_pixel_step_y)
                            {
                                memset(dest_start_ptr + dest_pixel_index, background_value, nbytes_tile_pixel_size_x);
                            }
                        }
                        else
                        {
                            #ifdef DEBUG
                            fmt::print("🔍 MEMCPY_DETAILED: tile_pixel_offset_sy={}, tile_pixel_offset_ey={}\n",
                                      tile_pixel_offset_sy, tile_pixel_offset_ey);
                            fmt::print("🔍 MEMCPY_DETAILED: dest_start_ptr={}, dest_pixel_step_y={}\n",
                                      static_cast<void*>(dest_start_ptr), dest_pixel_step_y);
                            fmt::print("🔍 MEMCPY_DETAILED: initial dest_pixel_index={}, initial nbytes_tile_index={}\n",
                                      dest_pixel_index, nbytes_tile_index);
                            fmt::print("🔍 MEMCPY_DETAILED: nbytes_tile_pixel_size_x={}, nbytes_tw={}\n",
                                      nbytes_tile_pixel_size_x, nbytes_tw);
                            fmt::print("🔍 MEMCPY_DETAILED: tile_data={}\n", static_cast<void*>(tile_data));
                            #endif

                            // Calculate total buffer size needed
                            uint32_t num_rows = tile_pixel_offset_ey - tile_pixel_offset_sy + 1;
                            [[maybe_unused]] size_t total_dest_size_needed = dest_pixel_index + (num_rows - 1) * dest_pixel_step_y + nbytes_tile_pixel_size_x;
                            #ifdef DEBUG
                            fmt::print("🔍 MEMCPY_DETAILED: num_rows={}, total_dest_size_needed={}\n",
                                      num_rows, total_dest_size_needed);
                            #endif

                            for (uint32_t ty = tile_pixel_offset_sy; ty <= tile_pixel_offset_ey;
                                 ++ty, dest_pixel_index += dest_pixel_step_y, nbytes_tile_index += nbytes_tw)
                            {
                                #ifdef DEBUG
                                fmt::print("🔍 MEMCPY_ROW ty={}: dest_pixel_index={}, nbytes_tile_index={}, copy_size={}\n",
                                          ty, dest_pixel_index, nbytes_tile_index, nbytes_tile_pixel_size_x);
                                fmt::print("🔍 MEMCPY_ROW: dest_ptr={}, src_ptr={}\n",
                                          static_cast<void*>(dest_start_ptr + dest_pixel_index),
                                          static_cast<void*>(tile_data + nbytes_tile_index));
                                fflush(stdout);
                                #endif

                                memcpy(dest_start_ptr + dest_pixel_index, tile_data + nbytes_tile_index,
                                       nbytes_tile_pixel_size_x);

                                #ifdef DEBUG
                                fmt::print("🔍 MEMCPY_ROW ty={}: SUCCESS\n", ty);
                                fflush(stdout);
                                #endif
                            }
                        }
                    }
                }
                else
                {

                    if (out_device.type() == cucim::io::DeviceType::kCPU)
                    {
                        for (uint32_t ty = tile_pixel_offset_sy; ty <= tile_pixel_offset_ey;
                             ++ty, dest_pixel_index += dest_pixel_step_y, nbytes_tile_index += nbytes_tw)
                        {
                            // Set (255,255,255)
                            memset(dest_start_ptr + dest_pixel_index, background_value, nbytes_tile_pixel_size_x);
                        }
                    }
                    else
                    {
                        cudaError_t cuda_status;
                        CUDA_ERROR(cudaMemset2D(dest_start_ptr + dest_pixel_index, dest_pixel_step_y, background_value,
                                                nbytes_tile_pixel_size_x, tile_pixel_offset_ey - tile_pixel_offset_sy));
                    }
                }
            };

            if (loader && *loader)
            {
                loader->enqueue(std::move(decode_func),
                                cucim::loader::TileInfo{ location_index, index, tiledata_offset, tiledata_size });
            }
            else
            {
                decode_func();
            }

            dest_pixel_index_x += nbytes_tile_pixel_size_x;
        }
        dest_start_ptr += dest_pixel_step_y * dest_pixel_offset_len_y;
    }
    return true;
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
