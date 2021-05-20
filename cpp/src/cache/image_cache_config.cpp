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

#include "cucim/cache/image_cache_config.h"

#include <fmt/format.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace cucim::cache
{

void ImageCacheConfig::load_config(void* json_obj)
{
    json& cache_config = *(static_cast<json*>(json_obj));

    if (cache_config["type"].is_string())
    {
        auto cache_type = cache_config.value("type", kDefaultCacheTypeStr);
        type = cucim::cache::lookup_cache_type(cache_type);
    }
    if (cache_config["memory_capacity"].is_number_unsigned())
    {
        memory_capacity = cache_config.value("memory_capacity", kDefaultCacheMemoryCapacity);
        capacity = calc_default_cache_capacity(kOneMiB * memory_capacity);
    }
    if (cache_config["capacity"].is_number_unsigned())
    {
        capacity = cache_config.value("capacity", calc_default_cache_capacity(kOneMiB * memory_capacity));
    }
    if (cache_config["mutex_pool_capacity"].is_number_unsigned())
    {
        mutex_pool_capacity = cache_config.value("mutex_pool_capacity", kDefaultCacheMutexPoolCapacity);
    }
    if (cache_config["list_padding"].is_number_unsigned())
    {
        list_padding = cache_config.value("list_padding", kDefaultCacheListPadding);
    }
    if (cache_config["extra_shared_memory_size"].is_number_unsigned())
    {
        extra_shared_memory_size = cache_config.value("extra_shared_memory_size", kDefaultCacheExtraSharedMemorySize);
    }
    if (cache_config["record_stat"].is_boolean())
    {
        record_stat = cache_config.value("record_stat", kDefaultCacheRecordStat);
    }
}

} // namespace cucim::cache