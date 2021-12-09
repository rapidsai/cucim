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

#include "cucim/loader/thread_batch_data_loader.h"

#include <cassert>

#include <fmt/format.h>

#include "cucim/profiler/nvtx3.h"

namespace cucim::loader
{

ThreadBatchDataLoader::ThreadBatchDataLoader(LoadFunc load_func,
                                             std::unique_ptr<std::vector<int64_t>> location,
                                             std::unique_ptr<std::vector<int64_t>> image_size,
                                             uint64_t location_len,
                                             size_t one_raster_size,
                                             uint32_t batch_size,
                                             uint32_t prefetch_factor,
                                             uint32_t num_workers)
    : load_func_(load_func),
      location_(std::move(location)),
      image_size_(std::move(image_size)),
      location_len_(location_len),
      one_rester_size_(one_raster_size),
      batch_size_(batch_size),
      prefetch_factor_(prefetch_factor),
      num_workers_(num_workers),
      buffer_item_len_(std::min(static_cast<uint64_t>(location_len), static_cast<uint64_t>(1 + prefetch_factor))),
      buffer_size_(one_raster_size * batch_size),
      thread_pool_(num_workers),
      queued_item_count_(0),
      buffer_item_head_index_(0),
      buffer_item_tail_index_(0),
      processed_batch_count_(0),
      current_data_(nullptr),
      current_data_batch_size_(0)
{
    raster_data_.reserve(buffer_item_len_);
    for (size_t i = 0; i < buffer_item_len_; ++i)
    {
        raster_data_.emplace_back(std::make_unique<uint8_t[]>(buffer_size_));
    }
}

ThreadBatchDataLoader::operator bool() const
{
    return (num_workers_ > 0);
}

uint8_t* ThreadBatchDataLoader::raster_pointer(const uint64_t location_index) const
{
    uint64_t buffer_item_index = (location_index / batch_size_) % buffer_item_len_;
    uint32_t raster_data_index = location_index % batch_size_;

    assert(buffer_item_index < buffer_item_len_);

    uint8_t* batch_raster_ptr = raster_data_[buffer_item_index].get();

    return &batch_raster_ptr[raster_data_index * one_rester_size_];
}

uint32_t ThreadBatchDataLoader::request(uint32_t load_size)
{
    if (num_workers_ == 0)
    {
        return 0;
    }

    if (load_size == 0)
    {
        load_size = batch_size_;
    }

    uint32_t num_items_to_request = std::min(load_size, static_cast<uint32_t>(location_len_ - queued_item_count_));
    for (uint32_t i = 0; i < num_items_to_request; ++i)
    {
        uint32_t last_item_count = 0;
        if (!tasks_.empty())
        {
            last_item_count = tasks_.size();
        }
        load_func_(this, queued_item_count_);
        ++queued_item_count_;
        buffer_item_tail_index_ = queued_item_count_ % buffer_item_len_;
        // Append the number of added tasks to the batch count list.
        batch_item_counts_.emplace_back(tasks_.size() - last_item_count);
    }
    return num_items_to_request;
}

uint32_t ThreadBatchDataLoader::wait_batch()
{
    if (num_workers_ == 0)
    {
        return 0;
    }

    uint32_t num_items_waited = 0;
    for (uint32_t batch_item_index = 0; batch_item_index < batch_size_ && !batch_item_counts_.empty(); ++batch_item_index)
    {
        uint32_t batch_item_count = batch_item_counts_.front();
        for (uint32_t i = 0; i < batch_item_count; ++i)
        {
            auto& future = tasks_.front();
            future.wait();
            tasks_.pop_front();
        }
        batch_item_counts_.pop_front();
        num_items_waited += batch_item_count;
    }
    return num_items_waited;
}


uint8_t* ThreadBatchDataLoader::next_data()
{
    if (num_workers_ == 0)
    {
        uint8_t* batch_raster_ptr = raster_data_[0].release();
        return batch_raster_ptr;
    }

    if (processed_batch_count_ * batch_size_ >= location_len_)
    {
        // Remove buffer items that are no longer needed.
        for (size_t i = 0; i < buffer_item_len_; ++i)
        {
            raster_data_[i] = nullptr;
        }
        return nullptr;
    }

    // Wait until the batch is ready.
    wait_batch();

    uint8_t* batch_raster_ptr = raster_data_[buffer_item_head_index_].release();

    raster_data_[buffer_item_head_index_].reset(new uint8_t[buffer_size_]);
    buffer_item_head_index_ = (buffer_item_head_index_ + 1) % buffer_item_len_;

    current_data_ = batch_raster_ptr;
    current_data_batch_size_ =
        std::min(location_len_ - (processed_batch_count_ * batch_size_), static_cast<uint64_t>(batch_size_));

    ++processed_batch_count_;

    // Prepare the next batch
    request(batch_size_);
    return batch_raster_ptr;
}

uint64_t ThreadBatchDataLoader::size() const
{
    return location_len_;
}

uint32_t ThreadBatchDataLoader::batch_size() const
{
    return batch_size_;
}

uint64_t ThreadBatchDataLoader::total_batch_count() const
{
    return (location_len_ + batch_size_ - 1) / batch_size_;
}

uint64_t ThreadBatchDataLoader::processed_batch_count() const
{
    return processed_batch_count_;
}

uint8_t* ThreadBatchDataLoader::data() const
{
    return current_data_;
}

uint32_t ThreadBatchDataLoader::data_batch_size() const
{
    return current_data_batch_size_;
}

bool ThreadBatchDataLoader::enqueue(std::function<void()> task)
{
    if (num_workers_ > 0)
    {
        auto future = thread_pool_.enqueue(task);
        tasks_.emplace_back(std::move(future));
        return true;
    }
    return false;
}

} // namespace cucim::loader
