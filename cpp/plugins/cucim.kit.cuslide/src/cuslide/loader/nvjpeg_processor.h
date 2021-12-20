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
#ifndef CUSLIDE_NVJPEG_PROCESSOR_H
#define CUSLIDE_NVJPEG_PROCESSOR_H


#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <vector>

#include <nvjpeg.h>

#include <cucim/filesystem/cufile_driver.h>
#include <cucim/filesystem/file_handle.h>
#include <cucim/io/device.h>
#include <cucim/loader/batch_data_processor.h>
#include <cucim/loader/tile_info.h>

#include "cuslide/tiff/ifd.h"

namespace cuslide::loader
{

class NvJpegProcessor : public cucim::loader::BatchDataProcessor
{
public:
    NvJpegProcessor(CuCIMFileHandle* file_handle,
                    const cuslide::tiff::IFD* ifd,
                    int64_t* request_location,
                    int64_t* request_size,
                    uint64_t location_len,
                    uint32_t batch_size,
                    uint32_t maximum_tile_count,
                    const uint8_t* jpegtable_data,
                    const uint32_t jpegtable_size);
    ~NvJpegProcessor();

    uint32_t request(std::deque<uint32_t>& batch_item_counts, uint32_t num_remaining_patches) override;
    uint32_t wait_batch(uint32_t index_in_task,
                        std::deque<uint32_t>& batch_item_counts,
                        uint32_t num_remaining_patches) override;

    std::shared_ptr<cucim::cache::ImageCacheValue> wait_for_processing(uint32_t index) override;

    void shutdown() override;

    uint32_t preferred_loader_prefetch_factor();

private:
    void update_file_block_info(int64_t* request_location, int64_t* request_size, uint64_t location_len);

    bool stopped_ = false;
    uint32_t preferred_loader_prefetch_factor_ = 2;

    CuCIMFileHandle* file_handle_ = nullptr;
    const cuslide::tiff::IFD* ifd_ = nullptr;
    std::shared_ptr<cucim::filesystem::CuFileDriver> cufile_;
    size_t tile_width_ = 0;
    size_t tile_width_bytes_ = 0;
    size_t tile_height_ = 0;
    size_t tile_raster_nbytes_ = 0;
    size_t file_size_ = 0;
    size_t file_start_offset_ = 0;
    size_t file_block_size_ = 0;

    uint32_t cuda_batch_size_ = 1;
    nvjpegHandle_t handle_ = nullptr;
    nvjpegOutputFormat_t output_format_ = NVJPEG_OUTPUT_RGBI;
    nvjpegJpegState_t state_;
    nvjpegBackend_t backend_ = NVJPEG_BACKEND_GPU_HYBRID_DEVICE;
    cudaStream_t stream_ = nullptr;

    std::condition_variable cuda_batch_cond_;
    std::unique_ptr<cucim::cache::ImageCache> cuda_image_cache_;
    uint64_t processed_cuda_batch_count_ = 0;
    cucim::loader::TileInfo fetch_after_{ -1, -1, 0, 0 };

    std::deque<uint32_t> cache_tile_queue_;
    std::unordered_map<uint32_t, cucim::loader::TileInfo> cache_tile_map_;

    uint8_t* unaligned_host_ = nullptr;
    uint8_t* aligned_host_ = nullptr;
    uint8_t* unaligned_device_ = nullptr;
    uint8_t* aligned_device_ = nullptr;

    std::vector<const unsigned char*> raw_cuda_inputs_;
    std::vector<size_t> raw_cuda_inputs_len_;
    std::vector<nvjpegImage_t> raw_cuda_outputs_;
};

} // namespace cuslide::loader

#endif // CUSLIDE_NVJPEG_PROCESSOR_H
