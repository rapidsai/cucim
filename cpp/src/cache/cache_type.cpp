/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cucim/cache/cache_type.h"
#include "cucim/cpp20/find_if.h"


namespace cucim::cache
{

using namespace std::literals::string_view_literals;

constexpr CacheType CacheTypeMap::at(const std::string_view& key) const
{
    const auto itr = cucim::cpp20::find_if(begin(data), end(data), [&key](const auto& v) { return v.first == key; });

    if (itr != end(data))
    {
        return itr->second;
    }
    else
    {
        return CacheType::kNoCache;
    }
}

constexpr std::string_view CacheTypeStrMap::at(const CacheType& key) const
{
    const auto itr = cucim::cpp20::find_if(begin(data), end(data), [&key](const auto& v) { return v.first == key; });

    if (itr != end(data))
    {
        return itr->second;
    }
    else
    {
        return "nocache"sv;
    }
}

static constexpr std::array<std::pair<std::string_view, CacheType>, kCacheTypeCount> cache_type_values{
    { { "nocache"sv, CacheType::kNoCache },
      { "per_process"sv, CacheType::kPerProcess },
      { "shared_memory"sv, CacheType::kSharedMemory } }
};

CacheType lookup_cache_type(const std::string_view sv)
{
    static constexpr auto map = CacheTypeMap{ { cache_type_values } };
    return map.at(sv);
}

static constexpr std::array<std::pair<CacheType, std::string_view>, kCacheTypeCount> cache_type_str_values{
    { { CacheType::kNoCache, "nocache"sv },
      { CacheType::kPerProcess, "per_process"sv },
      { CacheType::kSharedMemory, "shared_memory"sv } }
};

std::string_view lookup_cache_type_str(const CacheType key)
{
    static constexpr auto map = CacheTypeStrMap{ { cache_type_str_values } };
    return map.at(key);
}

} // namespace cucim::cache
