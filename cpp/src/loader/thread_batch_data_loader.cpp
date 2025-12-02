/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cucim/loader/thread_batch_data_loader.h"

#include <cassert>

#include <fmt/format.h>

#include "cucim/profiler/nvtx3.h"
#include "cucim/util/cuda.h"

namespace cucim::loader
{

ThreadBatchDataLoader::ThreadBatchDataLoader(LoadFunc load_func,
                                             std::unique_ptr<BatchDataProcessor> batch_data_processor,
                                             const cucim::io::Device out_device,
                                             std::unique_ptr<std::vector<int64_t>> location,
                                             std::unique_ptr<std::vector<int64_t>> image_size,
                                             const uint64_t location_len,
                                             const size_t one_raster_size,
                                             const uint32_t batch_size,
                                             const uint32_t prefetch_factor,
                                             const uint32_t num_workers)
    : load_func_(load_func),
      out_device_(out_device),
      location_(std::move(location)),
      image_size_(std::move(image_size)),
      location_len_(location_len),
      one_rester_size_(one_raster_size),
      batch_size_(batch_size),
      prefetch_factor_(prefetch_factor),
      num_workers_(num_workers),
      batch_data_processor_(std::move(batch_data_processor)),
      buffer_size_(one_raster_size * batch_size),
      thread_pool_(num_workers),
      queued_item_count_(0),
      buffer_item_head_index_(0),
      buffer_item_tail_index_(0),
      processed_batch_count_(0),
      current_data_(nullptr),
      current_data_batch_size_(0)
{
    buffer_item_len_ = std::min(static_cast<uint64_t>(location_len_), static_cast<uint64_t>(1 + prefetch_factor_)),

    raster_data_.reserve(buffer_item_len_);
    cucim::io::DeviceType device_type = out_device_.type();
    for (size_t i = 0; i < buffer_item_len_; ++i)
    {
        switch (device_type)
        {
        case io::DeviceType::kCPU:
            raster_data_.emplace_back(static_cast<uint8_t*>(cucim_malloc(buffer_size_)));
            break;
        case io::DeviceType::kCUDA: {
            cudaError_t cuda_status;
            void* image_data_ptr = nullptr;
            CUDA_ERROR(cudaMalloc(&image_data_ptr, buffer_size_));
            raster_data_.emplace_back(static_cast<uint8_t*>(image_data_ptr));
            break;
        }
        case io::DeviceType::kCUDAHost:
        case io::DeviceType::kCUDAManaged:
        case io::DeviceType::kCPUShared:
        case io::DeviceType::kCUDAShared:
            fmt::print(stderr, "Device type {} is not supported!\n", static_cast<int>(device_type));
            break;
        }
    }
}

ThreadBatchDataLoader::~ThreadBatchDataLoader()
{
    // Wait until all tasks are done.
    while (wait_batch() > 0);

    cucim::io::DeviceType device_type = out_device_.type();
    for (auto& raster_ptr : raster_data_)
    {
        switch (device_type)
        {
        case io::DeviceType::kCPU:
            if (raster_ptr)
            {
                cucim_free(raster_ptr);
            }
            break;
        case io::DeviceType::kCUDA:
            cudaError_t cuda_status;
            if (raster_ptr)
            {
                cuda_status = cudaSuccess;
                CUDA_TRY(cudaFree(raster_ptr));
            }
            break;
        case io::DeviceType::kCUDAHost:
        case io::DeviceType::kCUDAManaged:
        case io::DeviceType::kCPUShared:
        case io::DeviceType::kCUDAShared:
            fmt::print(stderr, "Device type {} is not supported!", static_cast<int>(device_type));
            break;
        }
        raster_ptr = nullptr;
    }
    if (batch_data_processor_)
    {
        stopped_ = true;
        batch_data_processor_->shutdown();
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

    uint8_t* batch_raster_ptr = raster_data_[buffer_item_index];

    return &batch_raster_ptr[raster_data_index * one_rester_size_];
}

uint32_t ThreadBatchDataLoader::request(uint32_t load_size)
{
#ifdef DEBUG
    fmt::print("🔍 request(): ENTRY - num_workers_={}, load_size={}, queued_item_count_={}\n",
              num_workers_, load_size, queued_item_count_);
#endif // DEBUG

    if (num_workers_ == 0)
    {
#ifdef DEBUG
        fmt::print("🔍 request(): num_workers==0, returning 0\n");
#endif // DEBUG
        return 0;
    }

    if (load_size == 0)
    {
        load_size = batch_size_;
    }

    uint32_t num_items_to_request = std::min(load_size, static_cast<uint32_t>(location_len_ - queued_item_count_));
#ifdef DEBUG
    fmt::print("🔍 request(): Will request {} items\n", num_items_to_request);
#endif // DEBUG

    for (uint32_t i = 0; i < num_items_to_request; ++i)
    {
        uint32_t last_item_count = 0;
        if (!tasks_.empty())
        {
            last_item_count = tasks_.size();
        }
#ifdef DEBUG
        fmt::print("🔍 request(): Calling load_func for item {} (location_index={})\n", i, queued_item_count_);
#endif // DEBUG
        load_func_(this, queued_item_count_);
#ifdef DEBUG
        fmt::print("🔍 request(): load_func returned, tasks added: {}\n", tasks_.size() - last_item_count);
#endif // DEBUG
        ++queued_item_count_;
        buffer_item_tail_index_ = queued_item_count_ % buffer_item_len_;
        // Append the number of added tasks to the batch count list.
        batch_item_counts_.emplace_back(tasks_.size() - last_item_count);
    }

    if (batch_data_processor_)
    {
        uint32_t num_remaining_patches = static_cast<uint32_t>(location_len_ - queued_item_count_);
        batch_data_processor_->request(batch_item_counts_, num_remaining_patches);
    }
    return num_items_to_request;
}

uint32_t ThreadBatchDataLoader::wait_batch()
{
#ifdef DEBUG
    fmt::print("🔍 wait_batch(): ENTRY - num_workers_={}, batch_item_counts_.size()={}, tasks_.size()={}\n",
              num_workers_, batch_item_counts_.size(), tasks_.size());
#endif // DEBUG

    if (num_workers_ == 0)
    {
        return 0;
    }

    uint32_t num_items_waited = 0;
    for (uint32_t batch_item_index = 0; batch_item_index < batch_size_ && !batch_item_counts_.empty(); ++batch_item_index)
    {
        uint32_t batch_item_count = batch_item_counts_.front();
#ifdef DEBUG
        fmt::print("🔍 wait_batch(): Processing batch_item_index={}, batch_item_count={}\n",
                  batch_item_index, batch_item_count);
#endif // DEBUG
        for (uint32_t i = 0; i < batch_item_count; ++i)
        {
#ifdef DEBUG
            fmt::print("🔍 wait_batch(): Waiting for task {} of {}\n", i, batch_item_count);
#endif // DEBUG
            auto& future = tasks_.front();
            try {
                future.wait();
#ifdef DEBUG
                fmt::print("🔍 wait_batch(): Task {} completed\n", i);
#endif // DEBUG
            } catch (const std::exception& e) {
#ifdef DEBUG
                fmt::print("❌ wait_batch(): Task {} threw exception: {}\n", i, e.what());
#endif // DEBUG
                throw;
            } catch (...) {
#ifdef DEBUG
                fmt::print("❌ wait_batch(): Task {} threw unknown exception\n", i);
#endif // DEBUG
                throw;
            }
            tasks_.pop_front();
            if (batch_data_processor_)
            {
                batch_data_processor_->remove_front_tile();
                uint32_t num_remaining_patches = static_cast<uint32_t>(location_len_ - queued_item_count_);
                batch_data_processor_->wait_batch(i, batch_item_counts_, num_remaining_patches);
            }
        }
        batch_item_counts_.pop_front();
        num_items_waited += batch_item_count;
    }
    return num_items_waited;
}


uint8_t* ThreadBatchDataLoader::next_data()
{
#ifdef DEBUG
    fmt::print("🔍 next_data(): ENTRY - num_workers_={}, processed_batch_count_={}, location_len_={}\n",
              num_workers_, processed_batch_count_, location_len_);
#endif // DEBUG

    if (num_workers_ == 0) // (location_len == 1 && batch_size == 1)
    {
#ifdef DEBUG
        fmt::print("🔍 next_data(): num_workers==0 path\n");
#endif // DEBUG
        // If it reads entire image with multi threads (using loader), release raster memory from batch data loader
        // by setting it to nullptr so that it will not be freed by ~ThreadBatchDataLoader (destructor).
        uint8_t* batch_raster_ptr = raster_data_[0];
        raster_data_[0] = nullptr;
        return batch_raster_ptr;
    }

    if (processed_batch_count_ * batch_size_ >= location_len_)
    {
#ifdef DEBUG
        fmt::print("🔍 next_data(): All batches processed, returning nullptr\n");
#endif // DEBUG
        // If all batches are processed, return nullptr.
        return nullptr;
    }

    // Wait until the batch is ready.
#ifdef DEBUG
    fmt::print("🔍 next_data(): About to call wait_batch()\n");
#endif // DEBUG
    wait_batch();
#ifdef DEBUG
    fmt::print("🔍 next_data(): wait_batch() completed\n");
#endif // DEBUG

    uint8_t* batch_raster_ptr = raster_data_[buffer_item_head_index_];

    cucim::io::DeviceType device_type = out_device_.type();
    switch (device_type)
    {
    case io::DeviceType::kCPU:
        raster_data_[buffer_item_head_index_] = static_cast<uint8_t*>(cucim_malloc(buffer_size_));
        break;
    case io::DeviceType::kCUDA: {
        cudaError_t cuda_status;
        CUDA_ERROR(cudaMalloc(&raster_data_[buffer_item_head_index_], buffer_size_));
        break;
    }
    case io::DeviceType::kCUDAHost:
    case io::DeviceType::kCUDAManaged:
    case io::DeviceType::kCPUShared:
    case io::DeviceType::kCUDAShared:
        fmt::print(stderr, "Device type {} is not supported!\n", static_cast<int>(device_type));
        break;
    }

    buffer_item_head_index_ = (buffer_item_head_index_ + 1) % buffer_item_len_;

    current_data_ = batch_raster_ptr;
    current_data_batch_size_ =
        std::min(location_len_ - (processed_batch_count_ * batch_size_), static_cast<uint64_t>(batch_size_));

    ++processed_batch_count_;

    // Prepare the next batch
    request(batch_size_);
    return batch_raster_ptr;
}

BatchDataProcessor* ThreadBatchDataLoader::batch_data_processor()
{
    return batch_data_processor_.get();
}

std::shared_ptr<cucim::cache::ImageCacheValue> ThreadBatchDataLoader::wait_for_processing(uint32_t index)
{
    if (batch_data_processor_ == nullptr || stopped_)
    {
        return std::shared_ptr<cucim::cache::ImageCacheValue>();
    }

    return batch_data_processor_->wait_for_processing(index);
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

bool ThreadBatchDataLoader::enqueue(std::function<void()> task, const TileInfo& tile)
{
#ifdef DEBUG
    fmt::print("🔍 enqueue(): ENTRY - num_workers_={}, tile.location_index={}, tile.index={}\n",
              num_workers_, tile.location_index, tile.index);
    fflush(stdout);
#endif // DEBUG

    if (num_workers_ > 0)
    {
#ifdef DEBUG
        fmt::print("🔍 enqueue(): About to enqueue task to thread pool\n");
        fflush(stdout);
#endif // DEBUG
        auto future = thread_pool_.enqueue(task);
#ifdef DEBUG
        fmt::print("🔍 enqueue(): Task enqueued, adding future to tasks_\n");
        fflush(stdout);
#endif // DEBUG
        tasks_.emplace_back(std::move(future));
#ifdef DEBUG
        fmt::print("🔍 enqueue(): tasks_.size()={}\n", tasks_.size());
        fflush(stdout);
#endif // DEBUG
        if (batch_data_processor_)
        {
            batch_data_processor_->add_tile(tile);
        }
#ifdef DEBUG
        fmt::print("🔍 enqueue(): Returning true\n");
        fflush(stdout);
#endif // DEBUG
        return true;
    }
    return false;
}

} // namespace cucim::loader
