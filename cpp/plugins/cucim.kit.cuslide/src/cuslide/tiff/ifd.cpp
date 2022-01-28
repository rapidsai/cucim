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

#include "ifd.h"

#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <thread>

#include <fmt/format.h>
#include <tiffio.h>
#include <tiffiop.h> // this is not included in the released library
#include <turbojpeg.h>

#include <cucim/codec/hash_function.h>
#include <cucim/cuimage.h>
#include <cucim/logger/timer.h>
#include <cucim/memory/memory_manager.h>
#include <cucim/profiler/nvtx3.h>
#include <cucim/util/cuda.h>

#include "cuslide/deflate/deflate.h"
#include "cuslide/jpeg/libjpeg_turbo.h"
#include "cuslide/jpeg2k/libopenjpeg.h"
#include "cuslide/loader/nvjpeg_processor.h"
#include "cuslide/lzw/lzw.h"
#include "cuslide/raw/raw.h"
#include "tiff.h"


namespace cuslide::tiff
{

IFD::IFD(TIFF* tiff, uint16_t index, ifd_offset_t offset) : tiff_(tiff), ifd_index_(index), ifd_offset_(offset)
{
    PROF_SCOPED_RANGE(PROF_EVENT(ifd_ifd));
    auto tif = tiff->client();

    char* software_char_ptr = nullptr;
    char* model_char_ptr = nullptr;

    // TODO: error handling
    TIFFGetField(tif, TIFFTAG_SOFTWARE, &software_char_ptr);
    software_ = std::string(software_char_ptr ? software_char_ptr : "");
    TIFFGetField(tif, TIFFTAG_MODEL, &model_char_ptr);
    model_ = std::string(model_char_ptr ? model_char_ptr : "");
    TIFFGetField(tif, TIFFTAG_IMAGEDESCRIPTION, &model_char_ptr);
    image_description_ = std::string(model_char_ptr ? model_char_ptr : "");

    TIFFDirectory& tif_dir = tif->tif_dir;
    flags_ = tif->tif_flags;

    width_ = tif_dir.td_imagewidth;
    height_ = tif_dir.td_imagelength;
    if ((flags_ & TIFF_ISTILED) != 0)
    {
        tile_width_ = tif_dir.td_tilewidth;
        tile_height_ = tif_dir.td_tilelength;
    }
    else
    {
        rows_per_strip_ = tif_dir.td_rowsperstrip;
    }
    bits_per_sample_ = tif_dir.td_bitspersample;
    samples_per_pixel_ = tif_dir.td_samplesperpixel;
    subfile_type_ = tif_dir.td_subfiletype;
    planar_config_ = tif_dir.td_planarconfig;
    photometric_ = tif_dir.td_photometric;
    compression_ = tif_dir.td_compression;
    TIFFGetField(tif, TIFFTAG_PREDICTOR, &predictor_);
    subifd_count_ = tif_dir.td_nsubifd;
    uint64_t* subifd_offsets = tif_dir.td_subifd;
    if (subifd_count_)
    {
        subifd_offsets_.resize(subifd_count_);
        subifd_offsets_.insert(subifd_offsets_.end(), &subifd_offsets[0], &subifd_offsets[subifd_count_]);
    }

    if (compression_ == COMPRESSION_JPEG)
    {
        uint8_t* jpegtable_data = nullptr;
        uint32_t jpegtable_count = 0;

        TIFFGetField(tif, TIFFTAG_JPEGTABLES, &jpegtable_count, &jpegtable_data);
        jpegtable_.reserve(jpegtable_count);
        jpegtable_.insert(jpegtable_.end(), jpegtable_data, jpegtable_data + jpegtable_count);

        if (photometric_ == PHOTOMETRIC_RGB)
        {
            jpeg_color_space_ = 2; // JCS_RGB
        }
        else if (photometric_ == PHOTOMETRIC_YCBCR)
        {
            jpeg_color_space_ = 3; // JCS_YCbCr
        }
    }

    image_piece_count_ = tif_dir.td_stripoffset_entry.tdir_count;

    image_piece_offsets_.reserve(image_piece_count_);
    uint64* td_stripoffset_p = tif_dir.td_stripoffset_p;
    uint64* td_stripbytecount_p = tif_dir.td_stripbytecount_p;

    // Copy data to vector
    image_piece_offsets_.insert(image_piece_offsets_.end(), &td_stripoffset_p[0], &td_stripoffset_p[image_piece_count_]);
    image_piece_bytecounts_.insert(
        image_piece_bytecounts_.end(), &td_stripbytecount_p[0], &td_stripbytecount_p[image_piece_count_]);

    // Calculate hash value with IFD index
    hash_value_ = tiff->file_handle_->hash_value ^ cucim::codec::splitmix64(index);

    //    TIFFPrintDirectory(tif, stdout, TIFFPRINT_STRIPS);
}

bool IFD::read(const TIFF* tiff,
               const cucim::io::format::ImageMetadataDesc* metadata,
               const cucim::io::format::ImageReaderRegionRequestDesc* request,
               cucim::io::format::ImageDataDesc* out_image_data)
{
    PROF_SCOPED_RANGE(PROF_EVENT(ifd_read));
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

    if (is_read_optimizable())
    {
        if (batch_size > 1)
        {
            ndim = 4;
        }
        int64_t* location = request->location;
        uint64_t location_len = request->location_len;
        const uint32_t num_workers = request->num_workers;
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

        if (location_len > 1 || batch_size > 1 || num_workers > 0)
        {
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
                uint8_t* raster_ptr = loader_ptr->raster_pointer(location_index);

                if (!read_region_tiles(tiff, ifd, location, location_index, w, h,
                                       raster_ptr, out_device, loader_ptr))
                {
                    fmt::print(stderr, "[Error] Failed to read region!\n");
                }
            };

            uint32_t maximum_tile_count = 0;

            std::unique_ptr<cucim::loader::BatchDataProcessor> batch_processor;

            // Set raster_type to CUDA because loader will handle this with nvjpeg
            if (out_device.type() == cucim::io::DeviceType::kCUDA)
            {
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
                    tiff->file_handle_, ifd, request_location->data(), request_size->data(), location_len, batch_size,
                    maximum_tile_count, static_cast<const uint8_t*>(jpegtable_data), jpegtable_size);

                // Update prefetch_factor
                prefetch_factor = nvjpeg_processor->preferred_loader_prefetch_factor();

                batch_processor = std::move(nvjpeg_processor);
            }

            auto loader = std::make_unique<cucim::loader::ThreadBatchDataLoader>(
                load_func, std::move(batch_processor), out_device, std::move(request_location), std::move(request_size),
                location_len, one_raster_size, batch_size, prefetch_factor, num_workers);

            const uint32_t load_size = std::min(static_cast<uint64_t>(batch_size) * (1 + prefetch_factor), location_len);

            loader->request(load_size);

            // If it reads entire image with multi threads (using loader), fetch the next item.
            if (location_len == 1 && batch_size == 1)
            {
                raster = loader->next_data();
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
    out_image_container.ctx = DLContext{ static_cast<DLDeviceType>(out_device.type()), out_device.index() };
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
    const int pixel_format = TJPF_RGB; // TODO: support other pixel format
    const int nbytes = tjPixelSize[pixel_format];
    return nbytes;
}

size_t IFD::tile_raster_size_nbytes() const
{
    const size_t nbytes = tile_width_ * tile_height_ * pixel_size_nbytes();
    return nbytes;
}

bool IFD::is_compression_supported() const
{
    switch (compression_)
    {
    case COMPRESSION_NONE:
    case COMPRESSION_JPEG:
    case COMPRESSION_ADOBE_DEFLATE:
    case COMPRESSION_DEFLATE:
    case cuslide::jpeg2k::kAperioJpeg2kYCbCr: // 33003: Jpeg 2000 with YCbCr format, possibly with a chroma subsampling
                                              // of 4:2:2
    case cuslide::jpeg2k::kAperioJpeg2kRGB: // 33005: Jpeg 2000 with RGB
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
    PROF_SCOPED_RANGE(PROF_EVENT(ifd_read_region_tiles));
    // Reference code: https://github.com/libjpeg-turbo/libjpeg-turbo/blob/master/tjexample.c

    int64_t sx = location[location_index * 2];
    int64_t sy = location[location_index * 2 + 1];
    int64_t ex = sx + w - 1;
    int64_t ey = sy + h - 1;

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

    int tiff_file = tiff->file_handle_->fd;
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
            PROF_SCOPED_RANGE(PROF_EVENT_P(ifd_read_region_tiles_iter, index));
            auto tiledata_offset = static_cast<uint64_t>(ifd->image_piece_offsets_[index]);
            auto tiledata_size = static_cast<uint64_t>(ifd->image_piece_bytecounts_[index]);

            // Calculate a simple hash value for the tile index
            uint64_t index_hash = ifd_hash_value ^ (static_cast<uint64_t>(index) | (static_cast<uint64_t>(index) << 32));

            uint32_t tile_pixel_offset_x = (offset_x == offset_sx) ? pixel_offset_sx : 0;
            uint32_t nbytes_tile_pixel_size_x = (offset_x == offset_ex) ?
                                                    (pixel_offset_ex - tile_pixel_offset_x + 1) * samples_per_pixel :
                                                    (tw - tile_pixel_offset_x) * samples_per_pixel;
            auto decode_func = [=, &image_cache]() {
                PROF_SCOPED_RANGE(PROF_EVENT_P(ifd_read_region_tiles_task, index_hash));
                uint32_t nbytes_tile_index = (tile_pixel_offset_sy * tw + tile_pixel_offset_x) * samples_per_pixel;
                uint32_t dest_pixel_index = dest_pixel_index_x;
                uint8_t* tile_data = nullptr;
                if (tiledata_size > 0)
                {
                    std::unique_ptr<uint8_t, decltype(cucim_free)*> tile_raster =
                        std::unique_ptr<uint8_t, decltype(cucim_free)*>(nullptr, cucim_free);

                    if (loader && loader->batch_data_processor())
                    {
                        switch (compression_method)
                        {
                        case COMPRESSION_JPEG:
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
                                PROF_SCOPED_RANGE(PROF_EVENT(ifd_decompression));
                                switch (compression_method)
                                {
                                case COMPRESSION_NONE:
                                    cuslide::raw::decode_raw(tiff_file, nullptr, tiledata_offset, tiledata_size,
                                                             &tile_data, tile_raster_nbytes, out_device);
                                    break;
                                case COMPRESSION_JPEG:
                                    cuslide::jpeg::decode_libjpeg(tiff_file, nullptr, tiledata_offset, tiledata_size,
                                                                  jpegtable_data, jpegtable_count, &tile_data,
                                                                  out_device, jpeg_color_space);
                                    break;
                                case COMPRESSION_ADOBE_DEFLATE:
                                case COMPRESSION_DEFLATE:
                                    cuslide::deflate::decode_deflate(tiff_file, nullptr, tiledata_offset, tiledata_size,
                                                                     &tile_data, tile_raster_nbytes, out_device);
                                    break;
                                case cuslide::jpeg2k::kAperioJpeg2kYCbCr: // 33003
                                    cuslide::jpeg2k::decode_libopenjpeg(tiff_file, nullptr, tiledata_offset,
                                                                        tiledata_size, &tile_data, tile_raster_nbytes,
                                                                        out_device, cuslide::jpeg2k::ColorSpace::kSYCC);
                                    break;
                                case cuslide::jpeg2k::kAperioJpeg2kRGB: // 33005
                                    cuslide::jpeg2k::decode_libopenjpeg(tiff_file, nullptr, tiledata_offset,
                                                                        tiledata_size, &tile_data, tile_raster_nbytes,
                                                                        out_device, cuslide::jpeg2k::ColorSpace::kRGB);
                                    break;
                                case COMPRESSION_LZW:
                                    cuslide::lzw::decode_lzw(tiff_file, nullptr, tiledata_offset, tiledata_size,
                                                             &tile_data, tile_raster_nbytes, out_device);
                                    // Apply unpredictor
                                    //   1: none, 2: horizontal differencing, 3: floating point predictor
                                    //   https://www.adobe.io/content/dam/udp/en/open/standards/tiff/TIFF6.pdf
                                    if (predictor == 2)
                                    {
                                        cuslide::lzw::horAcc8(tile_data, tile_raster_nbytes, nbytes_tw);
                                    }
                                    break;
                                default:
                                    throw std::runtime_error("Unsupported compression method");
                                }
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


    int tiff_file = tiff->file_handle_->fd;
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

            auto decode_func = [=, &image_cache]() {
                PROF_SCOPED_RANGE(PROF_EVENT_P(ifd_read_region_tiles_boundary_task, index_hash));
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
                                PROF_SCOPED_RANGE(PROF_EVENT(ifd_decompression));
                                switch (compression_method)
                                {
                                case COMPRESSION_NONE:
                                    cuslide::raw::decode_raw(tiff_file, nullptr, tiledata_offset, tiledata_size,
                                                             &tile_data, tile_raster_nbytes, out_device);
                                    break;
                                case COMPRESSION_JPEG:
                                    cuslide::jpeg::decode_libjpeg(tiff_file, nullptr, tiledata_offset, tiledata_size,
                                                                  jpegtable_data, jpegtable_count, &tile_data,
                                                                  out_device, jpeg_color_space);
                                    break;
                                case COMPRESSION_ADOBE_DEFLATE:
                                case COMPRESSION_DEFLATE:
                                    cuslide::deflate::decode_deflate(tiff_file, nullptr, tiledata_offset, tiledata_size,
                                                                     &tile_data, tile_raster_nbytes, out_device);
                                    break;
                                case cuslide::jpeg2k::kAperioJpeg2kYCbCr: // 33003
                                    cuslide::jpeg2k::decode_libopenjpeg(tiff_file, nullptr, tiledata_offset,
                                                                        tiledata_size, &tile_data, tile_raster_nbytes,
                                                                        out_device, cuslide::jpeg2k::ColorSpace::kSYCC);
                                    break;
                                case cuslide::jpeg2k::kAperioJpeg2kRGB: // 33005
                                    cuslide::jpeg2k::decode_libopenjpeg(tiff_file, nullptr, tiledata_offset,
                                                                        tiledata_size, &tile_data, tile_raster_nbytes,
                                                                        out_device, cuslide::jpeg2k::ColorSpace::kRGB);
                                    break;
                                case COMPRESSION_LZW:
                                    cuslide::lzw::decode_lzw(tiff_file, nullptr, tiledata_offset, tiledata_size,
                                                             &tile_data, tile_raster_nbytes, out_device);
                                    // Apply unpredictor
                                    //   1: none, 2: horizontal differencing, 3: floating point predictor
                                    //   https://www.adobe.io/content/dam/udp/en/open/standards/tiff/TIFF6.pdf
                                    if (predictor == 2)
                                    {
                                        cuslide::lzw::horAcc8(tile_data, tile_raster_nbytes, nbytes_tw);
                                    }
                                    break;
                                default:
                                    throw std::runtime_error("Unsupported compression method");
                                }
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
                            for (uint32_t ty = tile_pixel_offset_sy; ty <= tile_pixel_offset_ey;
                                 ++ty, dest_pixel_index += dest_pixel_step_y, nbytes_tile_index += nbytes_tw)
                            {
                                memcpy(dest_start_ptr + dest_pixel_index, tile_data + nbytes_tile_index,
                                       nbytes_tile_pixel_size_x);
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
void IFD::write_offsets_(const char* file_path)
{
    std::ofstream offsets(fmt::format("{}.offsets", file_path), std::ios::out | std::ios::binary | std::ios::trunc);
    std::ofstream bytecounts(fmt::format("{}.bytecounts", file_path), std::ios::out | std::ios::binary | std::ios::trunc);

    offsets.write(reinterpret_cast<char*>(&image_piece_count_), sizeof(image_piece_count_));
    bytecounts.write(reinterpret_cast<char*>(&image_piece_count_), sizeof(image_piece_count_));
    for (uint32_t i = 0; i < image_piece_count_; i++)
    {
        offsets.write(reinterpret_cast<char*>(&image_piece_offsets_[i]), sizeof(image_piece_offsets_[i]));
        bytecounts.write(reinterpret_cast<char*>(&image_piece_bytecounts_[i]), sizeof(image_piece_bytecounts_[i]));
    }
    bytecounts.close();
    offsets.close();
}

} // namespace cuslide::tiff
