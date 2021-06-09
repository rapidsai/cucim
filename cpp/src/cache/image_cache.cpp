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

#include "cucim/cache/image_cache.h"

#include "cucim/cuimage.h"

namespace cucim::cache
{

ImageCacheKey::ImageCacheKey(uint64_t file_hash, uint64_t index) : file_hash(file_hash), location_hash(index)
{
}

ImageCacheValue::ImageCacheValue(void* data, uint64_t size, void* user_obj) : data(data), size(size), user_obj(user_obj)
{
}

ImageCacheValue::operator bool() const
{
    return data != nullptr;
}


ImageCache::ImageCache(const ImageCacheConfig& config, CacheType type) : type_(type), config_(config){};

CacheType ImageCache::type() const
{
    return type_;
}

const char* ImageCache::type_str() const
{
    return "nocache";
}

ImageCacheConfig& ImageCache::config()
{
    return config_;
}

ImageCacheConfig ImageCache::get_config() const
{
    return config_;
}


} // namespace cucim::cache
