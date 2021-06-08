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
#ifndef PYCUCIM_CACHE_IMAGE_CACHE_PYDOC_H
#define PYCUCIM_CACHE_IMAGE_CACHE_PYDOC_H

#include "../macros.h"

namespace cucim::cache::doc::ImageCache
{

// PYDOC(ImageCache, R"doc(
// Constructor of ImageCache.

// Args:

// )doc")

// virtual CacheType type() const;
PYDOC(type, R"doc(
A Cache type.
)doc")

// virtual uint32_t size() const = 0;
PYDOC(size, R"doc(
A size of list/hashmap.
)doc")


// virtual uint64_t memory_size() const = 0;
PYDOC(memory_size, R"doc(
A size of cache memory used.
)doc")

// virtual uint32_t capacity() const = 0;
PYDOC(capacity, R"doc(
A capacity of list/hashmap.
)doc")

// virtual uint64_t memory_capacity() const = 0;
PYDOC(memory_capacity, R"doc(
A capacity of cache memory.
)doc")

// virtual uint64_t free_memory() const = 0;
PYDOC(free_memory, R"doc(
A cache memory size available in the cache memory.
)doc")

// virtual void record(bool value) = 0;
// virtual bool record() const = 0;
PYDOC(record, R"doc(
A cache memory size available in the cache memory.
)doc")

// virtual uint64_t hit_count() const = 0;
PYDOC(hit_count, R"doc(
A cache hit count.
)doc")

// virtual uint64_t miss_count() const = 0;
PYDOC(miss_count, R"doc(
A cache miss count.
)doc")

// virtual void reserve(const ImageCacheConfig& config) = 0;
PYDOC(reserve, R"doc(
Reserves more memory if possible.
)doc")

} // namespace cucim::cache::doc::ImageCache

#endif // PYCUCIM_CACHE_IMAGE_CACHE_PYDOC_H
