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

#include "cucim/cache/cache_type.h"

#include <algorithm>


namespace cucim::cache
{

// https://en.cppreference.com/w/cpp/algorithm/find
#if __cplusplus < 202002L
template <class InputIt, class UnaryPredicate>
constexpr InputIt myfind_if(InputIt first, InputIt last, UnaryPredicate p)
{
    for (; first != last; ++first)
    {
        if (p(*first))
        {
            return first;
        }
    }
    return last;
}
#else
template <class InputIt, class UnaryPredicate>
constexpr InputIt myfind_if(InputIt first, InputIt last, UnaryPredicate p)
{
    return std::find_if(first, last, p);
}
#endif

using namespace std::literals::string_view_literals;

constexpr CacheType CacheTypeMap::at(const std::string_view& key) const
{
    const auto itr = myfind_if(begin(data), end(data), [&key](const auto& v) { return v.first == key; });

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
    const auto itr = myfind_if(begin(data), end(data), [&key](const auto& v) { return v.first == key; });

    if (itr != end(data))
    {
        return itr->second;
    }
    else
    {
        return "nocache"sv;
    }
}

static constexpr std::array<std::pair<std::string_view, CacheType>, 3> cache_type_values{
    { { "nocache"sv, CacheType::kNoCache },
      { "per_process"sv, CacheType::kPerProcess },
      { "shared_memory"sv, CacheType::kSharedMemory } }
};

CacheType lookup_cache_type(const std::string_view sv)
{
    static constexpr auto map = CacheTypeMap{ { cache_type_values } };
    return map.at(sv);
}

static constexpr std::array<std::pair<CacheType, std::string_view>, 3> cache_type_str_values{
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