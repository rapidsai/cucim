/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#ifndef CUCIM_LOADER_THREAD_BATCH_DATA_LOADER_H
#define CUCIM_LOADER_THREAD_BATCH_DATA_LOADER_H

#include "cucim/macros/api_header.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>
#include <deque>

#include "cucim/concurrent/threadpool.h"

namespace cucim::loader
{

class EXPORT_VISIBLE ThreadBatchDataLoader
{
public:
    using LoadFunc = std::function<void(ThreadBatchDataLoader* loader_ptr, uint64_t location_index)>;

    ThreadBatchDataLoader(LoadFunc load_func,
                          std::unique_ptr<std::vector<int64_t>> location,
                          std::unique_ptr<std::vector<int64_t>> image_size,
                          uint64_t location_len,
                          size_t one_raster_size,
                          uint32_t batch_size,
                          uint32_t prefetch_factor,
                          uint32_t num_workers);

    operator bool() const;

    uint8_t* raster_pointer(const uint64_t location_index) const;
    uint32_t request(uint32_t load_size = 0);
    uint32_t wait_batch();
    /**
     * @brief Return the next batch of data.
     *
     * If the number of workers is zero, this function will return the ownership of the data.
     * @return uint8_t* The pointer to the data.
     */
    uint8_t* next_data();

    uint64_t size() const;
    uint32_t batch_size() const;

    uint64_t total_batch_count() const;
    uint64_t processed_batch_count() const;
    uint8_t* data() const;
    uint32_t data_batch_size() const;

    bool enqueue(std::function<void()> task);

private:
    LoadFunc load_func_;
    std::unique_ptr<std::vector<int64_t>> location_ = nullptr;
    std::unique_ptr<std::vector<int64_t>> image_size_ = nullptr;
    uint64_t location_len_ = 0;
    size_t one_rester_size_ = 0;
    uint32_t batch_size_ = 0;
    uint32_t prefetch_factor_ = 0;
    uint32_t num_workers_ = 0;

    size_t buffer_item_len_ = 0;
    size_t buffer_size_ = 0;
    std::vector<std::unique_ptr<uint8_t[]>> raster_data_;
    std::deque<std::future<void>> tasks_;
    // NOTE: the order is important ('thread_pool_' depends on 'raster_data_' and 'tasks_')
    cucim::concurrent::ThreadPool thread_pool_;

    uint64_t queued_item_count_ = 0;
    uint64_t buffer_item_head_index_ = 0;
    uint64_t buffer_item_tail_index_ = 0;

    std::deque<uint32_t> batch_item_counts_;
    uint64_t processed_batch_count_ = 0;
    uint8_t* current_data_ = nullptr;
    uint32_t current_data_batch_size_ = 0;
};

} // namespace cucim::loader

#endif // CUCIM_LOADER_THREAD_BATCH_DATA_LOADER_H
