/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Standard library
#include <algorithm>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

// Third-party
#include <fmt/format.h>
#include <cuda_runtime.h>

// cuCIM
#include <cucim/util/cuda.h>

#ifdef CUCIM_HAS_NVIMGCODEC

// Local
#include "nvimgcodec_processor.h"

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

    // Create a dedicated CUDA stream for decode operations when using GPU.
    // This avoids blocking the default stream and allows overlapping decode
    // with other GPU work (e.g., inference, preprocessing).
    if (use_device_memory_)
    {
        cudaError_t stream_err = cudaStreamCreateWithFlags(&decode_stream_, cudaStreamNonBlocking);
        if (stream_err != cudaSuccess)
        {
            #ifdef DEBUG
            fmt::print("⚠️  Failed to create CUDA stream, falling back to default stream\n");
            #endif
            decode_stream_ = nullptr;
        }
    }

    cuda_batch_size_ = std::min(
        static_cast<uint32_t>(location_len),
        std::min(batch_size, MAX_NVIMGCODEC_BATCH_SIZE));

    #ifdef DEBUG
    fmt::print("🔧 NvImageCodecProcessor initialized:\n");
    fmt::print("   ROI size: {}x{}, {} bytes\n", roi_width_, roi_height_, roi_size_bytes_);
    fmt::print("   Locations: {}, Batch size: {}\n", location_len_, batch_size_);
    fmt::print("   nvImageCodec batch size: {}\n", cuda_batch_size_);
    fmt::print("   Output: {} (stream: {})\n",
                 use_device_memory_ ? "GPU" : "CPU",
                 decode_stream_ ? "custom" : "default");
    #endif
}

NvImageCodecProcessor::~NvImageCodecProcessor()
{
    shutdown();

    // Destroy custom CUDA stream
    if (decode_stream_)
    {
        cudaStreamDestroy(decode_stream_);
        decode_stream_ = nullptr;
    }
}

void NvImageCodecProcessor::shutdown()
{
    stopped_.store(true, std::memory_order_release);
}

void NvImageCodecProcessor::set_output_buffer_provider(OutputBufferProvider provider)
{
    output_buffer_provider_ = std::move(provider);
}

uint32_t NvImageCodecProcessor::preferred_loader_prefetch_factor() const
{
    return preferred_loader_prefetch_factor_;
}

uint32_t NvImageCodecProcessor::request(std::deque<uint32_t>& batch_item_counts, uint32_t num_remaining_patches)
{
    (void)batch_item_counts;
    (void)num_remaining_patches;

    if (stopped_)
    {
        return 0;
    }

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

    // nvImageCodec is not thread-safe, so schedule + enqueue must be
    // serialized under the same mutex.
    {
        std::lock_guard<std::mutex> lock(nvimgcodec_mutex_);

        cuslide2::nvimgcodec::BatchDecodeState decode_state = schedule_roi_batch(batch_requests);

        // Check if the decode was actually scheduled (impl is always allocated,
        // but impl->future is null when scheduling fails).
        if (!decode_state.is_valid())
        {
            #ifdef DEBUG
            fmt::print("❌ Failed to schedule batch decode\n");
            #endif
            return 0;
        }

        pending_batches_.push(std::move(decode_state));
    }

    return static_cast<uint32_t>(batch_requests.size());
}

uint32_t NvImageCodecProcessor::wait_batch(uint32_t index_in_task,
                                            std::deque<uint32_t>& batch_item_counts,
                                            uint32_t num_remaining_patches)
{
    (void)batch_item_counts;
    (void)num_remaining_patches;
    (void)index_in_task;

    #ifdef DEBUG
    fmt::print("🔍 wait_batch: index_in_task={}\n", index_in_task);
    #endif

    if (stopped_)
    {
        return 0;
    }

    // Pop the oldest pending batch under the lock, then release it before
    // the (potentially blocking) wait_batch_decode() call so that other
    // threads can continue to enqueue new batches concurrently.
    cuslide2::nvimgcodec::BatchDecodeState decode_state;

    {
        std::lock_guard<std::mutex> lock(nvimgcodec_mutex_);

        if (pending_batches_.empty())
        {
            return 0;  // No pending batches to process
        }

        decode_state = std::move(pending_batches_.front());
        pending_batches_.pop();

        // Also pop the corresponding request list (no longer needed for zero-copy)
        if (!pending_requests_.empty())
        {
            pending_requests_.pop();
        }
    }  // lock released — safe to block on GPU now

    // Wait for batch decode completion.  The decoded data was written directly
    // into the ThreadBatchDataLoader's raster ring buffer via the output_buffer
    // pointers passed to schedule_batch_decode(), so there is nothing to copy.
    std::vector<cuslide2::nvimgcodec::BatchDecodeResult> results =
        cuslide2::nvimgcodec::wait_batch_decode(decode_state);

    #ifdef DEBUG
    size_t success_count = 0;
    for (const auto& r : results) if (r.success) ++success_count;
    fmt::print("  ✅ wait_batch[{}]: {}/{} regions decoded successfully (zero-copy)\n",
               index_in_task, success_count, results.size());
    #endif

    return batch_size_;
}

std::shared_ptr<cucim::cache::ImageCacheValue> NvImageCodecProcessor::wait_for_processing(uint32_t index)
{
    (void)index;
    // With the zero-copy path, decoded data is written directly into the
    // raster ring buffer.  There is no intermediate cache to query.
    return nullptr;
}

cuslide2::nvimgcodec::BatchDecodeState NvImageCodecProcessor::schedule_roi_batch(const std::vector<RoiDecodeRequest>& requests)
{
    cuslide2::nvimgcodec::BatchDecodeState state;

    if (requests.empty())
    {
        return state;
    }

    #ifdef DEBUG
    fmt::print("🚀 Scheduling batch decode of {} ROIs with nvImageCodec\n", requests.size());
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

    // Build output buffer pointers (one per region) from the provider.
    // These point directly into the ThreadBatchDataLoader's raster ring buffer.
    std::vector<uint8_t*> output_buffers;
    if (output_buffer_provider_)
    {
        output_buffers.reserve(requests.size());
        for (const auto& req : requests)
        {
            output_buffers.push_back(output_buffer_provider_(req.location_index));
        }
    }

    // Get IFD info for the batch (all regions decode from the same IFD)
    const auto& ifd_info = tiff_parser_.get_ifd(ifd_index_);

    // Schedule batch decode asynchronously on the dedicated CUDA stream.
    // When output_buffers is non-empty, nvImageCodec decodes directly into
    // those caller-owned buffers (zero-copy).
    state = cuslide2::nvimgcodec::schedule_batch_decode(
        ifd_info,
        tiff_parser_.get_main_code_stream(),
        regions,
        out_device_,
        decode_stream_,
        output_buffers);

    return state;
}

} // namespace cuslide2::loader

#endif // CUCIM_HAS_NVIMGCODEC
