/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CUSLIDE2_NVIMGCODEC_PROCESSOR_H
#define CUSLIDE2_NVIMGCODEC_PROCESSOR_H

#ifdef CUCIM_HAS_NVIMGCODEC

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <nvimgcodec.h>
#include <cuda_runtime.h>

#include <cucim/io/device.h>
#include <cucim/loader/batch_data_processor.h>
#include <cucim/loader/tile_info.h>

#include "cuslide/nvimgcodec/nvimgcodec_tiff_parser.h"
#include "cuslide/nvimgcodec/nvimgcodec_decoder.h"

namespace cuslide2::loader
{

/**
 * @brief ROI decode request for batch processing
 */
struct RoiDecodeRequest
{
    uint64_t location_index;  // Index in the location array
    uint32_t ifd_index;       // IFD (resolution level) index
    uint32_t x;               // ROI start x
    uint32_t y;               // ROI start y
    uint32_t width;           // ROI width
    uint32_t height;          // ROI height
};

/**
 * @brief Batch data processor using nvImageCodec for ROI decoding
 *
 * This processor uses nvImageCodec's batch decoding API to decode multiple
 * ROI regions from a TIFF file in a single decoder.decode() call.
 *
 * Key features:
 * - Uses get_sub_code_stream() with ROI for each region
 * - Batches multiple ROI decodes into a single decode call
 * - Integrates with ThreadBatchDataLoader for multi-threaded loading
 *
 *  Read image into CodeStream, then call multiple get_sub_code_stream()
 *  on this main CodeStream with different ROI and decode them all in a
 *  single decoder.decode() call
 */
class NvImageCodecProcessor : public cucim::loader::BatchDataProcessor
{
public:
    /**
     * @brief Construct a new NvImageCodecProcessor
     *
     * @param tiff_parser Reference to the TIFF parser (provides main_code_stream and IFD info)
     * @param request_location Pointer to location array [x0,y0,x1,y1,...]
     * @param request_size Pointer to size array [w, h]
     * @param location_len Number of locations to decode
     * @param batch_size Batch size for output
     * @param ifd_index IFD index (resolution level) to decode from
     * @param out_device Output device (CPU or CUDA)
     */
    NvImageCodecProcessor(cuslide2::nvimgcodec::TiffFileParser& tiff_parser,
                          const int64_t* request_location,
                          const int64_t* request_size,
                          uint64_t location_len,
                          uint32_t batch_size,
                          uint32_t ifd_index,
                          const cucim::io::Device& out_device);

    ~NvImageCodecProcessor() override;

    /**
     * @brief Request decoding of the next batch of ROIs
     */
    uint32_t request(std::deque<uint32_t>& batch_item_counts, uint32_t num_remaining_patches) override;

    /**
     * @brief Wait for a batch to complete decoding
     */
    uint32_t wait_batch(uint32_t index_in_task,
                        std::deque<uint32_t>& batch_item_counts,
                        uint32_t num_remaining_patches) override;

    /**
     * @brief Wait for processing of a specific ROI and return cached result
     */
    std::shared_ptr<cucim::cache::ImageCacheValue> wait_for_processing(uint32_t index) override;

    /**
     * @brief Shutdown the processor
     */
    void shutdown() override;

    /**
     * @brief Get preferred prefetch factor for the loader
     */
    uint32_t preferred_loader_prefetch_factor() const;

private:
    /**
     * @brief Schedule a batch of ROIs for decoding using nvImageCodec batch API
     * @return BatchDecodeState that must be passed to wait_batch_decode()
     */
    cuslide2::nvimgcodec::BatchDecodeState schedule_roi_batch(const std::vector<RoiDecodeRequest>& requests);

    /**
     * @brief Store decoded batch results into the per-ROI cache and notify waiters.
     *
     * Acquires cache_mutex_ internally and calls cache_cond_.notify_all().
     */
    void store_batch_results(const std::vector<RoiDecodeRequest>& requests,
                             const std::vector<cuslide2::nvimgcodec::BatchDecodeResult>& results);

    bool stopped_ = false;
    uint32_t preferred_loader_prefetch_factor_ = 2;

    // TIFF parser and IFD info
    cuslide2::nvimgcodec::TiffFileParser& tiff_parser_;
    uint32_t ifd_index_ = 0;
    uint32_t roi_width_ = 0;
    uint32_t roi_height_ = 0;
    size_t roi_size_bytes_ = 0;

    // Device configuration
    cucim::io::Device out_device_;
    bool use_device_memory_ = false;

    // Custom CUDA stream for async decode (avoids blocking the default stream)
    cudaStream_t decode_stream_ = nullptr;

    // Batch configuration
    uint32_t cuda_batch_size_ = 1;

    // nvImageCodec thread safety (nvImageCodec is not thread-safe)
    std::mutex nvimgcodec_mutex_;

    // Request queue
    std::mutex request_mutex_;

    // Decoded data cache
    mutable std::mutex cache_mutex_;
    std::condition_variable cache_cond_;
    std::unordered_map<uint64_t, uint8_t*> decoded_data_cpu_;
    std::unordered_map<uint64_t, uint8_t*> decoded_data_gpu_;
    std::unordered_map<uint64_t, bool> decode_complete_;

    // Location info
    const int64_t* request_location_ = nullptr;
    uint64_t location_len_ = 0;

    // Decode batch tracking
    uint64_t next_decode_index_ = 0;

    // Asynchronous batch decode state (FIFO queues — push_back / pop_front)
    std::deque<cuslide2::nvimgcodec::BatchDecodeState> pending_batches_;
    std::deque<std::vector<RoiDecodeRequest>> pending_requests_;
};

} // namespace cuslide2::loader

#endif // CUCIM_HAS_NVIMGCODEC

#endif // CUSLIDE2_NVIMGCODEC_PROCESSOR_H
