/*
 * Apache License, Version 2.0
 * Copyright 2021 NVIDIA Corporation
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

#include "nvjpeg_processor.h"

#include <vector>

#include <cucim/cache/image_cache_manager.h>
#include <cucim/codec/hash_function.h>
#include <cucim/io/device.h>
#include <cucim/util/cuda.h>
#include <fmt/format.h>

#define ALIGN_UP(x, align_to) (((uint64_t)(x) + ((uint64_t)(align_to)-1)) & ~((uint64_t)(align_to)-1))
#define ALIGN_DOWN(x, align_to) ((uint64_t)(x) & ~((uint64_t)(align_to)-1))
namespace cuslide::loader
{

constexpr uint32_t MAX_CUDA_BATCH_SIZE = 1024;

NvJpegProcessor::NvJpegProcessor(CuCIMFileHandle* file_handle,
                                 const cuslide::tiff::IFD* ifd,
                                 int64_t* request_location,
                                 int64_t* request_size,
                                 uint64_t location_len,
                                 uint32_t batch_size,
                                 uint32_t maximum_tile_count,
                                 const uint8_t* jpegtable_data,
                                 const uint32_t jpegtable_size)
    : cucim::loader::BatchDataProcessor(batch_size), file_handle_(file_handle), ifd_(ifd)
{
    if (maximum_tile_count > 1)
    {
        // Calculate nearlest power of 2 that is equal or larger than the given number.
        // (Test with https://godbolt.org/z/n7qhPYzfP)
        int next_candidate = maximum_tile_count & (maximum_tile_count - 1);
        if (next_candidate > 0)
        {
            maximum_tile_count <<= 1;
            while (true)
            {
                next_candidate = maximum_tile_count & (maximum_tile_count - 1);
                if (next_candidate == 0)
                {
                    break;
                }
                maximum_tile_count = next_candidate;
            }
        }

        // Do not exceed MAX_CUDA_BATCH_SIZE for decoding JPEG with nvJPEG
        uint32_t cuda_batch_size = std::min(maximum_tile_count, MAX_CUDA_BATCH_SIZE);

        // Update prefetch_factor
        // (We can decode/cache tiles at least two times of the number of tiles for batch decoding)
        // E.g., (128 - 1) / 32 + 1 ~= 4 => 8 (for 256 tiles) for cuda_batch_size(=128) and batch_size(=32)
        preferred_loader_prefetch_factor_ = ((cuda_batch_size - 1) / batch_size_ + 1) * 2;

        // Create cuda image cache
        cucim::cache::ImageCacheConfig cache_config{};
        cache_config.type = cucim::cache::CacheType::kPerProcess;
        cache_config.memory_capacity = 1024 * 1024; // 1TB: set to fairly large memory so that memory_capacity is not a
                                                    // limiter.
        cache_config.capacity = cuda_batch_size * 2; // limit the number of cache item to cuda_batch_size * 2
        cuda_image_cache_ =
            std::move(cucim::cache::ImageCacheManager::create_cache(cache_config, cucim::io::DeviceType::kCUDA));

        cuda_batch_size_ = cuda_batch_size;

        // Initialize nvjpeg
        cudaError_t cuda_status;

        if (NVJPEG_STATUS_SUCCESS != nvjpegCreate(backend_, NULL, &handle_))
        {
            throw std::runtime_error(fmt::format("NVJPEG initialization error"));
        }
        if (NVJPEG_STATUS_SUCCESS != nvjpegJpegStateCreate(handle_, &state_))
        {
            throw std::runtime_error(fmt::format("JPEG state initialization error"));
        }

        nvjpegDecodeBatchedParseJpegTables(handle_, state_, jpegtable_data, jpegtable_size);
        nvjpegDecodeBatchedInitialize(handle_, state_, cuda_batch_size_, 1, output_format_);

        CUDA_ERROR(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));

        raw_cuda_inputs_.reserve(cuda_batch_size_);
        raw_cuda_inputs_len_.reserve(cuda_batch_size_);

        for (uint32_t i = 0; i < cuda_batch_size_; ++i)
        {
            raw_cuda_outputs_.emplace_back(); // add all-zero nvjpegImage_t object
        }

        // Read file block in advance
        tile_width_ = ifd->tile_width();
        tile_width_bytes_ = tile_width_ * ifd->pixel_size_nbytes();
        tile_height_ = ifd->tile_height();
        tile_raster_nbytes_ = tile_width_bytes_ * tile_height_;

        struct stat sb;
        fstat(file_handle_->fd, &sb);
        file_size_ = sb.st_size;
        file_start_offset_ = 0;
        file_block_size_ = file_size_;

        update_file_block_info(request_location, request_size, location_len);

        constexpr int BLOCK_SECTOR_SIZE = 4096;
        switch (backend_)
        {
        case NVJPEG_BACKEND_GPU_HYBRID:
            cufile_ = cucim::filesystem::open(file_handle->path, "rp");
            unaligned_host_ = static_cast<uint8_t*>(cucim_malloc(file_block_size_ + BLOCK_SECTOR_SIZE * 2));
            aligned_host_ = reinterpret_cast<uint8_t*>(ALIGN_UP(unaligned_host_, BLOCK_SECTOR_SIZE));
            cufile_->pread(aligned_host_, file_block_size_, file_start_offset_);
            break;
        case NVJPEG_BACKEND_GPU_HYBRID_DEVICE:
            cufile_ = cucim::filesystem::open(file_handle->path, "r");
            CUDA_ERROR(cudaMalloc(&unaligned_device_, file_block_size_ + BLOCK_SECTOR_SIZE));
            aligned_device_ = reinterpret_cast<uint8_t*>(ALIGN_UP(unaligned_device_, BLOCK_SECTOR_SIZE));
            cufile_->pread(aligned_device_, file_block_size_, file_start_offset_);
            break;
        default:
            throw std::runtime_error("Unsupported backend type");
        }
    }
}

NvJpegProcessor::~NvJpegProcessor()
{
    if (unaligned_host_)
    {
        cucim_free(unaligned_host_);
        unaligned_host_ = nullptr;
    }

    cudaError_t cuda_status;
    if (unaligned_device_)
    {
        CUDA_ERROR(cudaFree(unaligned_device_));
        unaligned_device_ = nullptr;
    }

    for (uint32_t i = 0; i < cuda_batch_size_; ++i)
    {
        if (raw_cuda_outputs_[i].channel[0])
        {
            CUDA_ERROR(cudaFree(raw_cuda_outputs_[i].channel[0]));
            raw_cuda_outputs_[i].channel[0] = nullptr;
        }
    }
}

uint32_t NvJpegProcessor::request(std::deque<uint32_t>& batch_item_counts, uint32_t num_remaining_patches)
{
    (void)batch_item_counts;
    std::vector<cucim::loader::TileInfo> tile_to_request;
    if (tiles_.empty())
    {
        return 0;
    }

    // Return if we need to wait until previous cuda batch is consumed.
    auto& first_tile = tiles_.front();
    if (first_tile.location_index <= fetch_after_.location_index)
    {
        if (first_tile.location_index < fetch_after_.location_index || first_tile.index < fetch_after_.index)
        {
            return 0;
        }
    }

    // Set fetch_after_ to the last tile info of previously processed cuda batch
    if (!cache_tile_queue_.empty())
    {
        fetch_after_ = cache_tile_map_[cache_tile_queue_.back()];
    }

    // Remove previous batch (keep last 'cuda_batch_size_' items) before adding/processing new cuda batch
    std::vector<cucim::loader::TileInfo> removed_tiles;
    while (cache_tile_queue_.size() > cuda_batch_size_)
    {
        uint32_t removed_tile_index = cache_tile_queue_.front();
        auto removed_tile = cache_tile_map_.find(removed_tile_index);
        removed_tiles.push_back(removed_tile->second);
        cache_tile_queue_.pop_front();
        cache_tile_map_.erase(removed_tile_index);
    }

    // Collect candidates
    for (auto tile : tiles_)
    {
        auto index = tile.index;
        if (tile_to_request.size() >= cuda_batch_size_)
        {
            break;
        }
        if (cache_tile_map_.find(index) == cache_tile_map_.end())
        {
            if (tile.size == 0)
            {
                continue;
            }
            cache_tile_queue_.emplace_back(index);
            cache_tile_map_.emplace(index, tile);
            tile_to_request.emplace_back(tile);
        }
    }

    // Return if we need to wait until more patches are requested
    if (tile_to_request.size() < cuda_batch_size_)
    {
        if (num_remaining_patches > 0)
        {
            // Restore cache_tile_queue_ and cache_tile_map_
            for (auto& added_tile : tile_to_request)
            {
                uint32_t added_index = added_tile.index;
                cache_tile_queue_.pop_back();
                cache_tile_map_.erase(added_index);
            }
            for (auto rit = removed_tiles.rbegin(); rit != removed_tiles.rend(); ++rit)
            {
                uint32_t removed_index = rit->index;
                cache_tile_queue_.emplace_front(removed_index);
                cache_tile_map_.emplace(removed_index, *rit);
            }
            return 0;
        }
        else
        {
            // Completed, set fetch_after_ to the last tile info.
            fetch_after_ = tiles_.back();
        }
    }

    uint8_t* file_block_ptr = nullptr;
    switch (backend_)
    {
    case NVJPEG_BACKEND_GPU_HYBRID:
        file_block_ptr = aligned_host_;
        break;
    case NVJPEG_BACKEND_GPU_HYBRID_DEVICE:
        file_block_ptr = aligned_device_;
        break;
    default:
        throw std::runtime_error("Unsupported backend type");
    }

    cudaError_t cuda_status;

    // Initialize batch data with the first data
    if (raw_cuda_inputs_.empty())
    {
        for (uint32_t i = 0; i < cuda_batch_size_; ++i)
        {
            uint8_t* mem_offset = nullptr;
            mem_offset = file_block_ptr + tile_to_request[0].offset - file_start_offset_;
            raw_cuda_inputs_.push_back((const unsigned char*)mem_offset);
            raw_cuda_inputs_len_.push_back(tile_to_request[0].size);
            CUDA_ERROR(cudaMallocPitch(
                &raw_cuda_outputs_[i].channel[0], &raw_cuda_outputs_[i].pitch[0], tile_width_bytes_, tile_height_));
        }
        CUDA_ERROR(cudaStreamSynchronize(stream_));
    }

    // Set inputs to nvJPEG
    size_t request_count = tile_to_request.size();
    for (uint32_t i = 0; i < request_count; ++i)
    {
        uint8_t* mem_offset = file_block_ptr + tile_to_request[i].offset - file_start_offset_;
        raw_cuda_inputs_[i] = mem_offset;
        raw_cuda_inputs_len_[i] = tile_to_request[i].size;
    }

    int error_code = nvjpegDecodeBatched(
        handle_, state_, raw_cuda_inputs_.data(), raw_cuda_inputs_len_.data(), raw_cuda_outputs_.data(), stream_);

    if (NVJPEG_STATUS_SUCCESS != error_code)
    {
        throw std::runtime_error(fmt::format("Error in batched decode: {}", error_code));
    }
    CUDA_ERROR(cudaStreamSynchronize(stream_));

    // Remove previous batch (keep last 'cuda_batch_size_' items) before adding to cuda_image_cache_
    // TODO: Utilize the removed tiles if next batch uses them.
    while (cuda_image_cache_->size() > cuda_batch_size_)
    {
        cuda_image_cache_->remove_front();
    }

    // Add to image cache
    for (uint32_t i = 0; i < request_count; ++i)
    {
        auto& added_tile = tile_to_request[i];

        uint32_t index = added_tile.index;
        uint64_t index_hash = cucim::codec::splitmix64(index);

        auto key = cuda_image_cache_->create_key(0, index);

        cuda_image_cache_->lock(index_hash);

        uint8_t* tile_data = static_cast<uint8_t*>(cuda_image_cache_->allocate(tile_raster_nbytes_));

        cudaError_t cuda_status;
        CUDA_TRY(cudaMemcpy2D(tile_data, tile_width_bytes_, raw_cuda_outputs_[i].channel[0],
                              raw_cuda_outputs_[i].pitch[0], tile_width_bytes_, tile_height_, cudaMemcpyDeviceToDevice));

        const size_t tile_raster_nbytes = raw_cuda_inputs_len_[i];
        auto value = cuda_image_cache_->create_value(tile_data, tile_raster_nbytes, cucim::io::DeviceType::kCUDA);
        cuda_image_cache_->insert(key, value);

        cuda_image_cache_->unlock(index_hash);
    }

    ++processed_cuda_batch_count_;

    cuda_batch_cond_.notify_all();
    return request_count;
}

uint32_t NvJpegProcessor::wait_batch(uint32_t index_in_task,
                                     std::deque<uint32_t>& batch_item_counts,
                                     uint32_t num_remaining_patches)
{
    // Check if the next (cuda) batch needs to be requested whenever an index in a task is divided by cuda batch size.
    // (each task which is for a patch consists of multiple tile processing)
    if (index_in_task % cuda_batch_size_ == 0)
    {
        request(batch_item_counts, num_remaining_patches);
    }
    return 0;
}

std::shared_ptr<cucim::cache::ImageCacheValue> NvJpegProcessor::wait_for_processing(uint32_t index)
{
    uint64_t index_hash = cucim::codec::splitmix64(index);
    std::mutex* m = reinterpret_cast<std::mutex*>(cuda_image_cache_->mutex(index_hash));
    std::shared_ptr<cucim::cache::ImageCacheValue> value;

    std::unique_lock<std::mutex> lock(*m);
    cuda_batch_cond_.wait(lock, [this, index, &value] {
        // Exit waiting if the thread needs to be stopped or cache value is available.
        if (stopped_)
        {
            value = std::shared_ptr<cucim::cache::ImageCacheValue>();
            return true;
        }
        auto key = cuda_image_cache_->create_key(0, index);
        value = cuda_image_cache_->find(key);
        return static_cast<bool>(value);
    });
    return value;
}

void NvJpegProcessor::shutdown()
{
    stopped_ = true;
    cuda_batch_cond_.notify_all();
}

uint32_t NvJpegProcessor::preferred_loader_prefetch_factor()
{
    return preferred_loader_prefetch_factor_;
}

void NvJpegProcessor::update_file_block_info(int64_t* request_location, int64_t* request_size, uint64_t location_len)
{

    uint32_t width = ifd_->width();
    uint32_t height = ifd_->height();
    uint32_t stride_y = width / tile_width_ + !!(width % tile_width_); // # of tiles in a row(y) in the ifd tile array
                                                                       // as grid
    uint32_t stride_x = height / tile_height_ + !!(height % tile_height_); // # of tiles in a col(x) in the ifd tile
                                                                           // array as grid
    int64_t min_tile_index = 1000000000;
    int64_t max_tile_index = 0;

    // Assume that offset for tiles are increasing as the index is increasing.
    for (size_t loc_index = 0; loc_index < location_len; ++loc_index)
    {
        int64_t sx = request_location[loc_index * 2];
        int64_t sy = request_location[loc_index * 2 + 1];
        int64_t offset_sx = static_cast<uint64_t>(sx) / tile_width_; // x-axis start offset for the requested region in
                                                                     // the ifd tile array as grid
        int64_t offset_sy = static_cast<uint64_t>(sy) / tile_height_; // y-axis start offset for the requested region in
                                                                      // the ifd tile array as grid
        int64_t tile_index = (offset_sy * stride_y) + offset_sx;
        min_tile_index = std::min(min_tile_index, tile_index);
        max_tile_index = std::max(max_tile_index, tile_index);
    }

    int64_t w = request_size[0];
    int64_t h = request_size[1];
    int64_t additional_index_x = (static_cast<uint64_t>(w) + (tile_width_ - 1)) / tile_width_;
    int64_t additional_index_y = (static_cast<uint64_t>(h) + (tile_height_ - 1)) / tile_height_;
    min_tile_index = std::max(min_tile_index, 0L);
    max_tile_index =
        std::min(stride_x * stride_y - 1,
                 static_cast<uint32_t>(max_tile_index + (additional_index_y * stride_y) + additional_index_x));

    auto& image_piece_offsets = const_cast<std::vector<uint64_t>&>(ifd_->image_piece_offsets());
    auto& image_piece_bytecounts = const_cast<std::vector<uint64_t>&>(ifd_->image_piece_bytecounts());

    uint64_t min_offset = image_piece_offsets[min_tile_index];
    uint64_t max_offset = image_piece_offsets[max_tile_index] + image_piece_bytecounts[max_tile_index];

    file_start_offset_ = min_offset;
    file_block_size_ = max_offset - min_offset + 1;
}

} // namespace cuslide::loader
