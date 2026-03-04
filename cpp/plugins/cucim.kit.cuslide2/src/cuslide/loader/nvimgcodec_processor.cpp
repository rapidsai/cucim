/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Standard library includes - MUST be before any headers that open namespaces
#include <algorithm>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

// Third-party includes - MUST be before any headers that open namespaces
// Note: fmt must be included in global namespace, before any namespace declarations
#include <fmt/format.h>

// CUDA includes
#include <cuda_runtime.h>

// cuCIM includes
#include <cucim/util/cuda.h>

#ifdef CUCIM_HAS_NVIMGCODEC

// Local includes - these may open namespaces, so include after std/fmt
#include "nvimgcodec_processor.h"

namespace cuslide2::loader
{

// Maximum batch size for nvImageCodec decode
constexpr uint32_t MAX_NVIMGCODEC_BATCH_SIZE = 64;

NvImageCodecProcessor::NvImageCodecProcessor(
    ::cuslide2::nvimgcodec::TiffFileParser& tiff_parser,
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
            ::fmt::print("⚠️  No GPU available, falling back to CPU memory\n");
            #endif
        }
    }

    // Create a dedicated CUDA stream for decode operations when using GPU.
    // This avoids blocking the default stream and allows overlapping decode
    // with other GPU work (e.g., inference, preprocessing).
    if (use_device_memory_)
    {
        cudaError_t stream_err = cudaStreamCreate(&decode_stream_);
        if (stream_err != cudaSuccess)
        {
            #ifdef DEBUG
            ::fmt::print("⚠️  Failed to create CUDA stream, falling back to default stream\n");
            #endif
            decode_stream_ = nullptr;
        }
    }

    // Calculate batch size for nvImageCodec
    // Capped by location_len and MAX_NVIMGCODEC_BATCH_SIZE.
    // MAX_NVIMGCODEC_BATCH_SIZE is defined in this namespace, so use it directly
    cuda_batch_size_ = std::min(
        static_cast<uint32_t>(location_len),
        std::min(batch_size, MAX_NVIMGCODEC_BATCH_SIZE));

    // Single batch prefetch: I/O is typically faster than decode work,
    // so one prefetch batch is sufficient to keep the decoder busy.
    // This uses 2 batch buffers (1 current + 1 prefetched) for reasonable memory usage.
    preferred_loader_prefetch_factor_ = 1;

    #ifdef DEBUG
    ::fmt::print("🔧 NvImageCodecProcessor initialized:\n");
    ::fmt::print("   ROI size: {}x{}, {} bytes\n", roi_width_, roi_height_, roi_size_bytes_);
    ::fmt::print("   Locations: {}, Batch size: {}\n", location_len_, batch_size_);
    ::fmt::print("   nvImageCodec batch size: {}\n", cuda_batch_size_);
    ::fmt::print("   Output: {} (stream: {})\n",
                 use_device_memory_ ? "GPU" : "CPU",
                 decode_stream_ ? "custom" : "default");
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

    // Destroy custom CUDA stream
    if (decode_stream_)
    {
        cudaStreamDestroy(decode_stream_);
        decode_stream_ = nullptr;
    }
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
    ::fmt::print("📦 Requesting batch decode of {} ROIs\n", batch_requests.size());
    #endif

    // Schedule batch decode asynchronously (don't wait yet)
    ::cuslide2::nvimgcodec::BatchDecodeState decode_state = schedule_roi_batch(batch_requests);

    // Check if the decode was actually scheduled (impl is always allocated,
    // but impl->future is null when scheduling fails).
    if (!decode_state.is_valid())
    {
        #ifdef DEBUG
        ::fmt::print("❌ Failed to schedule batch decode\n");
        #endif
        return 0;
    }

    // Store state and requests for later waiting
    {
        std::lock_guard<std::mutex> lock(nvimgcodec_mutex_);
        pending_batches_.push_back(std::move(decode_state));
        pending_requests_.push_back(batch_requests);
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

    // Pop the oldest pending batch under the lock, then release it before
    // the (potentially blocking) wait_batch_decode() call so that other
    // threads can continue to enqueue new batches concurrently.
    ::cuslide2::nvimgcodec::BatchDecodeState decode_state;
    std::vector<RoiDecodeRequest> requests;

    {
        std::lock_guard<std::mutex> lock(nvimgcodec_mutex_);

        if (pending_batches_.empty() || pending_requests_.empty())
        {
            return batch_size_;
        }

        decode_state = std::move(pending_batches_.front());
        requests = std::move(pending_requests_.front());
        pending_batches_.pop_front();
        pending_requests_.pop_front();
    }  // lock released — safe to block on GPU now

    // Wait for batch decode completion and process results
    std::vector<::cuslide2::nvimgcodec::BatchDecodeResult> results =
        ::cuslide2::nvimgcodec::wait_batch_decode(decode_state);

    // Store results in cache
    {
        std::lock_guard<std::mutex> cache_lock(cache_mutex_);

        for (size_t i = 0; i < requests.size(); ++i)
        {
            uint64_t loc_idx = requests[i].location_index;

            if (i < results.size() && results[i].success && results[i].buffer)
            {
                if (use_device_memory_)
                {
                    decoded_data_gpu_[loc_idx] = results[i].buffer;
                }
                else
                {
                    decoded_data_cpu_[loc_idx] = results[i].buffer;
                }
                decode_complete_[loc_idx] = true;

                #ifdef DEBUG
                ::fmt::print("  ✅ ROI {} decoded successfully\n", loc_idx);
                #endif
            }
            else
            {
                #ifdef DEBUG
                ::fmt::print("  ❌ ROI {} decode failed\n", loc_idx);
                #endif
                decode_complete_[loc_idx] = false;
            }
        }
    }

    // Notify waiters
    cache_cond_.notify_all();

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

::cuslide2::nvimgcodec::BatchDecodeState NvImageCodecProcessor::schedule_roi_batch(const std::vector<RoiDecodeRequest>& requests)
{
    ::cuslide2::nvimgcodec::BatchDecodeState state;

    if (requests.empty())
    {
        return state;
    }

    #ifdef DEBUG
    ::fmt::print("🚀 Scheduling batch decode of {} ROIs with nvImageCodec\n", requests.size());
    #endif

    // Build regions for batch decode
    std::vector<::cuslide2::nvimgcodec::RoiRegion> regions;
    regions.reserve(requests.size());

    for (const auto& req : requests)
    {
        ::cuslide2::nvimgcodec::RoiRegion region;
        region.x = req.x;
        region.y = req.y;
        region.width = req.width;
        region.height = req.height;
        regions.push_back(region);
    }

    // Get IFD info for the batch (all regions decode from the same IFD)
    const auto& ifd_info = tiff_parser_.get_ifd(ifd_index_);

    // Schedule batch decode asynchronously on the dedicated CUDA stream.
    // Using a custom stream avoids blocking the default stream and allows
    // overlapping decode with other GPU work (e.g., inference pipelines).
    // Note: nvImageCodec is not thread-safe, but we're calling this from request()
    // which is already protected by the caller's context.
    state = ::cuslide2::nvimgcodec::schedule_batch_decode(
        ifd_info,
        tiff_parser_.get_main_code_stream(),
        regions,
        out_device_,
        decode_stream_);

    return state;
}

} // namespace cuslide2::loader

#endif // CUCIM_HAS_NVIMGCODEC
