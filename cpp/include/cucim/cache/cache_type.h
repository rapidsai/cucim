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

#ifndef CUCIM_CACHE_CACHE_TYPE_H
#define CUCIM_CACHE_CACHE_TYPE_H

#include "cucim/macros/api_header.h"

#include <array>
#include <cstdint>
#include <string_view>

namespace cucim::cache
{

constexpr std::size_t kCacheTypeCount = 3;
enum class CacheType : uint8_t
{
    kNoCache,
    kPerProcess,
    kSharedMemory
};

struct CacheTypeMap
{
    std::array<std::pair<std::string_view, CacheType>, kCacheTypeCount> data;

    [[nodiscard]] constexpr CacheType at(const std::string_view& key) const;
};

EXPORT_VISIBLE CacheType lookup_cache_type(const std::string_view sv);

struct CacheTypeStrMap
{
    std::array<std::pair<CacheType, std::string_view>, kCacheTypeCount> data;

    [[nodiscard]] constexpr std::string_view at(const CacheType& key) const;
};

EXPORT_VISIBLE std::string_view lookup_cache_type_str(const CacheType type);


} // namespace cucim::cache

#endif // CUCIM_CACHE_CACHE_TYPE_H
