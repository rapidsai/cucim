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

#include "cucim/loader/batch_data_processor.h"

#include <cuda_runtime.h>
#include <fmt/format.h>

#include "cucim/cache/image_cache_manager.h"

namespace cucim::loader
{

BatchDataProcessor::BatchDataProcessor(uint32_t batch_size) : batch_size_(batch_size), processed_index_count_(0)
{
}

BatchDataProcessor::~BatchDataProcessor()
{
}


void BatchDataProcessor::add_tile(const TileInfo& tile)
{
    tiles_.emplace_back(tile);
    ++total_index_count_;
}

TileInfo BatchDataProcessor::remove_front_tile()
{
    TileInfo tile = tiles_.front();
    tiles_.pop_front();
    ++processed_index_count_;
    return tile;
}

uint32_t BatchDataProcessor::request(std::deque<uint32_t>& batch_item_counts, uint32_t num_remaining_patches)
{
    (void)batch_item_counts;
    (void)num_remaining_patches;
    return 0;
}

uint32_t BatchDataProcessor::wait_batch(uint32_t index_in_task,
                                        std::deque<uint32_t>& batch_item_counts,
                                        uint32_t num_remaining_patches)
{
    (void)index_in_task;
    (void)batch_item_counts;
    (void)num_remaining_patches;
    return 0;
}

std::shared_ptr<cucim::cache::ImageCacheValue> BatchDataProcessor::wait_for_processing(uint32_t)
{
    return std::shared_ptr<cucim::cache::ImageCacheValue>();
}

void BatchDataProcessor::shutdown()
{
}

} // namespace cucim::loader
