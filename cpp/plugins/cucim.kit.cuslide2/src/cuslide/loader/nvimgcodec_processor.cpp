/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Standard library
#include <algorithm>
#include <cassert>
#include <cstring>
#include <deque>
#include <format>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
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

    // Synchronize the decode stream to ensure all GPU work completes before
    // destroying it. This ensures nvImageCodec finishes all operations
    // enqueued on decode_stream_ before the stream is destroyed.
    if (decode_stream_)
    {
        cudaStreamSynchronize(decode_stream_);
        cudaStreamDestroy(decode_stream_);
        decode_stream_ = nullptr;
    }
}

void NvImageCodecProcessor::shutdown()
{
    stopped_.store(true, std::memory_order_release);
    PendingBatch discarded;
    {
        std::lock_guard<std::mutex> lock(nvimgcodec_mutex_);
        discarded = std::exchange(pending_batch_, {});
    }
    // Destroying each decode state waits for its writes to finish before
    // the loader releases the raster ring.
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

    std::lock_guard<std::mutex> lock(nvimgcodec_mutex_);
    if (stopped_ || next_decode_index_ == location_len_)
    {
        return 0;
    }

    assert(pending_batch_.chunks.empty() && !pending_batch_.submission_error);
    const auto count = static_cast<uint32_t>(std::min<uint64_t>(batch_size_, location_len_ - next_decode_index_));
    const uint64_t end = next_decode_index_ + count;

    // Queue failures with the batch they belong to. A prefetch failure must
    // not prevent next_data() from returning the previous, successful batch.
    try
    {
        while (next_decode_index_ < end)
        {
            const uint64_t chunk_end = std::min(next_decode_index_ + cuda_batch_size_, end);
            std::vector<RoiDecodeRequest> requests;
            requests.reserve(chunk_end - next_decode_index_);
            for (uint64_t i = next_decode_index_; i < chunk_end; ++i)
            {
                requests.push_back({ i,
                                     ifd_index_,
                                     static_cast<uint32_t>(request_location_[i * 2]),
                                     static_cast<uint32_t>(request_location_[i * 2 + 1]),
                                     roi_width_,
                                     roi_height_ });
            }

            auto state = schedule_roi_batch(requests);
            if (!state.is_valid())
            {
                throw std::runtime_error(std::format("Failed to schedule IFD[{}] batch decode with nvImageCodec",
                                                     ifd_index_));
            }
            pending_batch_.chunks.push_back(std::move(state));
            next_decode_index_ = chunk_end;
        }
    }
    catch (...)
    {
        pending_batch_.submission_error = std::current_exception();
        next_decode_index_ = end;
    }
    return count;
}

uint32_t NvImageCodecProcessor::wait_batch(uint32_t index_in_task,
                                          std::deque<uint32_t>& batch_item_counts,
                                          uint32_t num_remaining_patches)
{
    (void)batch_item_counts;
    (void)num_remaining_patches;

    // The loader calls this once per item. Each pending batch covers all
    // items in the output raster, so only its first item needs to wait.
    if (index_in_task != 0 || stopped_)
    {
        return 0;
    }

    PendingBatch batch;
    {
        std::lock_guard<std::mutex> lock(nvimgcodec_mutex_);
        batch = std::exchange(pending_batch_, {});
    }
    if (batch.submission_error)
    {
        std::rethrow_exception(batch.submission_error);
    }
    if (batch.chunks.empty())
    {
        throw std::runtime_error(std::format("No pending nvImageCodec batch for IFD[{}]", ifd_index_));
    }

    uint32_t decoded = 0;
    for (auto& chunk : batch.chunks)
    {
        const auto results = cuslide2::nvimgcodec::wait_batch_decode(chunk);
        if (results.empty() || std::any_of(results.begin(), results.end(),
                                           [](const auto& result) { return !result.success; }))
        {
            throw std::runtime_error(std::format("Failed to decode IFD[{}] batch with nvImageCodec. "
                                                 "Check that nvTIFF and the required codec libraries are installed.",
                                                 ifd_index_));
        }
        decoded += static_cast<uint32_t>(results.size());
    }
    return decoded;
}

std::shared_ptr<cucim::cache::ImageCacheValue> NvImageCodecProcessor::wait_for_processing(uint32_t index)
{
    (void)index;
    // Zero-copy path: decoded data is written directly into the raster ring
    // buffer via OutputBufferProvider.  This method should never be called on
    // NvImageCodecProcessor — use next_data() instead.
    assert(false && "wait_for_processing() should not be called in zero-copy mode; use next_data() instead");
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
