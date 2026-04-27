/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef CUCIM_UTIL_CHECKED_MATH_H
#define CUCIM_UTIL_CHECKED_MATH_H

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace cucim::util
{

constexpr size_t kMaxTileBytes = 256ULL * 1024 * 1024; // 256 MiB
constexpr size_t kMaxRasterBytes = 1ULL * 1024 * 1024 * 1024; // 1 GiB

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
    if (result > kMaxTileBytes)
    {
        throw std::overflow_error("Tile size exceeds maximum allowed (" +
                                  std::to_string(result) + " > " +
                                  std::to_string(kMaxTileBytes) + " bytes)");
    }
    return result;
}

inline size_t checked_raster_size(size_t width, size_t height, size_t pixel_bytes)
{
    size_t result = checked_mul3(width, height, pixel_bytes);
    if (result > kMaxRasterBytes)
    {
        throw std::overflow_error("Raster size exceeds maximum allowed (" +
                                  std::to_string(result) + " > " +
                                  std::to_string(kMaxRasterBytes) + " bytes)");
    }
    return result;
}

} // namespace cucim::util

#endif // CUCIM_UTIL_CHECKED_MATH_H
