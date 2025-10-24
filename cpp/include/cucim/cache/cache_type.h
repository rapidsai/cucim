/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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

// Using constexpr map (https://www.youtube.com/watch?v=INn3xa4pMfg)
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
