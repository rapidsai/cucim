/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CUCIM_LOADER_BATCH_DATA_PROCESSOR_H
#define CUCIM_LOADER_BATCH_DATA_PROCESSOR_H

#include "cucim/macros/api_header.h"

#include <cstdint>
#include <deque>

#include <cucim/cache/image_cache.h>

#include "tile_info.h"

namespace cucim::loader
{

class EXPORT_VISIBLE BatchDataProcessor
{
public:
    BatchDataProcessor(uint32_t batch_size);
    virtual ~BatchDataProcessor();

    void add_tile(const TileInfo& tile);
    TileInfo remove_front_tile();

    virtual uint32_t request(std::deque<uint32_t>& batch_item_counts, const uint32_t num_remaining_patches);
    virtual uint32_t wait_batch(const uint32_t index_in_task,
                                std::deque<uint32_t>& batch_item_counts,
                                const uint32_t num_remaining_patches);

    virtual std::shared_ptr<cucim::cache::ImageCacheValue> wait_for_processing(uint32_t);

    virtual void shutdown();

protected:
    uint32_t batch_size_ = 1;
    uint64_t total_index_count_ = 0;
    uint64_t processed_index_count_ = 0;
    std::deque<TileInfo> tiles_;
};

} // namespace cucim::loader

#endif // CUCIM_LOADER_BATCH_DATA_PROCESSOR_H
