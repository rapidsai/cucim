/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef CUCIM_UTIL_CHECKED_MATH_H
#define CUCIM_UTIL_CHECKED_MATH_H

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace cucim::util
{

constexpr size_t kDefaultMaxTileBytes = 2ULL * 1024 * 1024 * 1024; // 2 GiB
constexpr size_t kDefaultMaxRasterBytes = 16ULL * 1024 * 1024 * 1024; // 16 GiB

namespace detail
{

inline size_t parse_env_size(const char* env_var, size_t default_value)
{
    const char* val = std::getenv(env_var);
    if (val == nullptr || val[0] == '\0')
    {
        return default_value;
    }
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(val, &end, 10);
    if (end == val || parsed == 0)
    {
        return default_value;
    }
    return static_cast<size_t>(parsed);
}

} // namespace detail

inline size_t max_tile_bytes()
{
    static const size_t value = detail::parse_env_size("CUCIM_MAX_TILE_BYTES", kDefaultMaxTileBytes);
    return value;
}

inline size_t max_raster_bytes()
{
    static const size_t value = detail::parse_env_size("CUCIM_MAX_RASTER_BYTES", kDefaultMaxRasterBytes);
    return value;
}

template <typename T>
inline T checked_mul(T a, T b)
{
    static_assert(std::is_unsigned_v<T>, "checked_mul requires unsigned types");
    T result;
    if (__builtin_mul_overflow(a, b, &result))
    {
        throw std::overflow_error("Integer overflow in buffer size calculation");
    }
    return result;
}

inline size_t checked_mul3(size_t a, size_t b, size_t c)
{
    return checked_mul(checked_mul(a, b), c);
}

inline size_t checked_tile_size(size_t width, size_t height, size_t pixel_bytes)
{
    size_t result = checked_mul3(width, height, pixel_bytes);
    size_t limit = max_tile_bytes();
    if (result > limit)
    {
        throw std::overflow_error("Tile size exceeds maximum allowed (" +
                                  std::to_string(result) + " > " +
                                  std::to_string(limit) + " bytes). "
                                  "Override with CUCIM_MAX_TILE_BYTES env var.");
    }
    return result;
}

inline size_t checked_raster_size(size_t width, size_t height, size_t pixel_bytes)
{
    size_t result = checked_mul3(width, height, pixel_bytes);
    size_t limit = max_raster_bytes();
    if (result > limit)
    {
        throw std::overflow_error("Raster size exceeds maximum allowed (" +
                                  std::to_string(result) + " > " +
                                  std::to_string(limit) + " bytes). "
                                  "Override with CUCIM_MAX_RASTER_BYTES env var.");
    }
    return result;
}

} // namespace cucim::util

#endif // CUCIM_UTIL_CHECKED_MATH_H
