/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nvimgcodec_processor.h"

#ifdef CUCIM_HAS_NVIMGCODEC

#include <algorithm>
#include <cstring>

#include <cuda_runtime.h>
#include <fmt/format.h>

#include <cucim/util/cuda.h>
#include "cuslide/nvimgcodec/nvimgcodec_decoder.h"

namespace cuslide2::loader
{

// Maximum batch size for nvImageCodec decode
constexpr uint32_t MAX_NVIMGCODEC_BATCH_SIZE = 64;

NvImageCodecProcessor::NvImageCodecProcessor(
    cuslide2::nvimgcodec::TiffFileParser& tiff_parser,
    const int64_t* request_location,
    const int64_t* request_size,
    uint64_t location_len,
    uint32_t batch_size,
    uint32_t ifd_index,
    const cucim::io::Device& out_device)
    : cucim::loader::BatchDataProcessor(batch_size),
      tiff_parser_(tiff_parser),
      ifd_index_(ifd_index),
      out_device_(out_device),
      request_location_(request_location),
      location_len_(location_len)
{
    if (!tiff_parser_.is_valid())
    {
        throw std::runtime_error("Invalid TIFF parser");
    }

    // Get ROI dimensions
    roi_width_ = static_cast<uint32_t>(request_size[0]);
    roi_height_ = static_cast<uint32_t>(request_size[1]);

    // Get IFD info for channel count
    const auto& ifd_info = tiff_parser_.get_ifd(ifd_index_);
    uint32_t num_channels = ifd_info.num_channels > 0 ? ifd_info.num_channels : 3;
    roi_size_bytes_ = static_cast<size_t>(roi_width_) * roi_height_ * num_channels;

    // Determine output memory type
    std::string device_str = std::string(out_device_);
    use_device_memory_ = (device_str.find("cuda") != std::string::npos);

    // Check GPU availability
    if (use_device_memory_)
    {
        int device_count = 0;
        cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
        if (cuda_err != cudaSuccess || device_count == 0)
        {
            use_device_memory_ = false;
            #ifdef DEBUG
            fmt::print("⚠️  No GPU available, falling back to CPU memory\n");
            #endif
        }
    }

    // Calculate batch size for nvImageCodec
    // Capped by location_len and MAX_NVIMGCODEC_BATCH_SIZE.
    cuda_batch_size_ = std::min(
        static_cast<uint32_t>(location_len),
        std::min(batch_size, MAX_NVIMGCODEC_BATCH_SIZE));

    // Update prefetch_factor based on CUDA batch size (same logic as cuslide's NvJpegProcessor)
    // This ensures enough tiles are prefetched to keep the decoder busy.
    preferred_loader_prefetch_factor_ = ((cuda_batch_size_ - 1) / batch_size_ + 1) * 2;

    #ifdef DEBUG
    fmt::print("🔧 NvImageCodecProcessor initialized:\n");
    fmt::print("   ROI size: {}x{}, {} bytes\n", roi_width_, roi_height_, roi_size_bytes_);
    fmt::print("   Locations: {}, Batch size: {}\n", location_len_, batch_size_);
    fmt::print("   nvImageCodec batch size: {}\n", cuda_batch_size_);
    fmt::print("   Output: {}\n", use_device_memory_ ? "GPU" : "CPU");
    #endif
}

NvImageCodecProcessor::~NvImageCodecProcessor()
{
    shutdown();

    // Free CPU memory
    for (auto& [idx, ptr] : decoded_data_cpu_)
    {
        if (ptr)
        {
            free(ptr);
        }
    }
    decoded_data_cpu_.clear();

    // Free GPU memory
    for (auto& [idx, ptr] : decoded_data_gpu_)
    {
        if (ptr)
        {
            cudaFree(ptr);
        }
    }
    decoded_data_gpu_.clear();
}

void NvImageCodecProcessor::shutdown()
{
    {
        std::lock_guard<std::mutex> lock(request_mutex_);
        stopped_ = true;
    }
    cache_cond_.notify_all();
}

uint32_t NvImageCodecProcessor::preferred_loader_prefetch_factor() const
{
    return preferred_loader_prefetch_factor_;
}

uint32_t NvImageCodecProcessor::request(std::deque<uint32_t>& batch_item_counts, uint32_t num_remaining_patches)
{
    (void)batch_item_counts;
    (void)num_remaining_patches;

    // Build batch of ROI decode requests
    std::vector<RoiDecodeRequest> batch_requests;

    {
        std::lock_guard<std::mutex> lock(request_mutex_);

        while (next_decode_index_ < location_len_ && batch_requests.size() < cuda_batch_size_)
        {
            RoiDecodeRequest req;
            req.location_index = next_decode_index_;
            req.ifd_index = ifd_index_;
            req.x = static_cast<uint32_t>(request_location_[next_decode_index_ * 2]);
            req.y = static_cast<uint32_t>(request_location_[next_decode_index_ * 2 + 1]);
            req.width = roi_width_;
            req.height = roi_height_;

            batch_requests.push_back(req);
            ++next_decode_index_;
        }
    }

    if (batch_requests.empty())
    {
        return 0;
    }

    #ifdef DEBUG
    fmt::print("📦 Requesting batch decode of {} ROIs\n", batch_requests.size());
    #endif

    // Decode the batch
    if (!decode_roi_batch(batch_requests))
    {
        #ifdef DEBUG
        fmt::print("❌ Batch decode failed\n");
        #endif
        return 0;
    }

    return static_cast<uint32_t>(batch_requests.size());
}

uint32_t NvImageCodecProcessor::wait_batch(uint32_t index_in_task,
                                            std::deque<uint32_t>& batch_item_counts,
                                            uint32_t num_remaining_patches)
{
    (void)index_in_task;
    (void)batch_item_counts;
    (void)num_remaining_patches;

    // Acquire mutex to ensure any in-progress decode in request() has completed
    // (nvImageCodec decode is synchronous, so once we get the lock, decode is done)
    std::lock_guard<std::mutex> lock(nvimgcodec_mutex_);
    return batch_size_;
}

std::shared_ptr<cucim::cache::ImageCacheValue> NvImageCodecProcessor::wait_for_processing(uint32_t index)
{
    std::unique_lock<std::mutex> lock(cache_mutex_);

    // Wait until this index is decoded
    cache_cond_.wait(lock, [this, index]() {
        return stopped_ || decode_complete_.count(index) > 0;
    });

    if (stopped_)
    {
        return nullptr;
    }

    // Create cache value pointing to decoded data
    void* data_ptr = nullptr;
    cucim::io::DeviceType device_type = cucim::io::DeviceType::kCPU;

    if (use_device_memory_)
    {
        auto it = decoded_data_gpu_.find(index);
        if (it != decoded_data_gpu_.end())
        {
            data_ptr = it->second;
            device_type = cucim::io::DeviceType::kCUDA;
        }
    }
    else
    {
        auto it = decoded_data_cpu_.find(index);
        if (it != decoded_data_cpu_.end())
        {
            data_ptr = it->second;
            device_type = cucim::io::DeviceType::kCPU;
        }
    }

    auto value = std::make_shared<cucim::cache::ImageCacheValue>(
        data_ptr, roi_size_bytes_, nullptr, device_type);

    return value;
}

bool NvImageCodecProcessor::decode_roi_batch(const std::vector<RoiDecodeRequest>& requests)
{
    if (requests.empty())
    {
        return false;
    }

    #ifdef DEBUG
    fmt::print("🚀 Decoding batch of {} ROIs with nvImageCodec\n", requests.size());
    #endif

    // Build regions for batch decode
    std::vector<cuslide2::nvimgcodec::RoiRegion> regions;
    regions.reserve(requests.size());

    for (const auto& req : requests)
    {
        cuslide2::nvimgcodec::RoiRegion region;
        region.x = req.x;
        region.y = req.y;
        region.width = req.width;
        region.height = req.height;
        regions.push_back(region);
    }

    // Get IFD info for the batch (all regions decode from the same IFD)
    const auto& ifd_info = tiff_parser_.get_ifd(ifd_index_);

    // Perform batch decode using nvImageCodec API
    // Lock mutex since nvImageCodec is not thread-safe
    std::vector<cuslide2::nvimgcodec::BatchDecodeResult> results;
    {
        std::lock_guard<std::mutex> lock(nvimgcodec_mutex_);
        results = cuslide2::nvimgcodec::decode_batch_regions_nvimgcodec(
            ifd_info,
            tiff_parser_.get_main_code_stream(),
            regions,
            out_device_);
    }

    // Store results in cache
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);

        for (size_t i = 0; i < requests.size(); ++i)
        {
            uint64_t loc_idx = requests[i].location_index;

            if (results[i].success && results[i].buffer)
            {
                if (use_device_memory_)
                {
                    // Store GPU pointer directly
                    decoded_data_gpu_[loc_idx] = results[i].buffer;
                }
                else
                {
                    // Store CPU pointer directly (reuse allocated buffer)
                    decoded_data_cpu_[loc_idx] = results[i].buffer;
                }
                decode_complete_[loc_idx] = true;

                #ifdef DEBUG
                fmt::print("  ✅ ROI {} decoded successfully\n", loc_idx);
                #endif
            }
            else
            {
                #ifdef DEBUG
                fmt::print("  ❌ ROI {} decode failed\n", loc_idx);
                #endif
                decode_complete_[loc_idx] = false;
            }
        }
    }

    // Notify waiters
    cache_cond_.notify_all();

    return true;
}

} // namespace cuslide2::loader

#endif // CUCIM_HAS_NVIMGCODEC
