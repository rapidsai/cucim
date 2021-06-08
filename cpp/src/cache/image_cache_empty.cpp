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

#include "image_cache_empty.h"

namespace cucim::cache
{

EmptyImageCache::EmptyImageCache(const ImageCacheConfig& config) : ImageCache(config){};


std::shared_ptr<ImageCacheKey> EmptyImageCache::create_key(uint64_t, uint64_t)
{
    return std::make_shared<ImageCacheKey>(0, 0);
}
std::shared_ptr<ImageCacheValue> EmptyImageCache::create_value(void*, uint64_t)
{
    return std::make_shared<ImageCacheValue>(nullptr, 0);
}

void* EmptyImageCache::allocate(std::size_t)
{
    return nullptr;
}

void EmptyImageCache::lock(uint64_t)
{
    return;
}

void EmptyImageCache::unlock(uint64_t)
{
    return;
}

bool EmptyImageCache::insert(std::shared_ptr<ImageCacheKey>&, std::shared_ptr<ImageCacheValue>&)
{
    return true;
}


uint32_t EmptyImageCache::size() const
{
    return 0;
}

uint64_t EmptyImageCache::memory_size() const
{
    return 0;
}
uint32_t EmptyImageCache::capacity() const
{
    return 0;
}
uint64_t EmptyImageCache::memory_capacity() const
{
    return 0;
}
uint64_t EmptyImageCache::free_memory() const
{
    return 0;
}

void EmptyImageCache::record(bool)
{
    return;
}

bool EmptyImageCache::record() const
{
    return false;
}

uint64_t EmptyImageCache::hit_count() const
{
    return 0;
}
uint64_t EmptyImageCache::miss_count() const
{
    return 0;
}

void EmptyImageCache::reserve(const ImageCacheConfig&)
{
}

std::shared_ptr<ImageCacheValue> EmptyImageCache::find(const std::shared_ptr<ImageCacheKey>&)
{
    return std::shared_ptr<ImageCacheValue>();
}

} // namespace cucim::cache
