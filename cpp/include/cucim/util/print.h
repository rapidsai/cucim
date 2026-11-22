/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdio>
#include <format>
#include <string>
#include <string_view>

namespace cucim
{

inline void vprint(std::FILE* stream, std::string_view fmt, std::format_args args)
{
    const std::string formatted = std::vformat(fmt, args);
    [[maybe_unused]] const auto written = std::fwrite(formatted.data(), 1, formatted.size(), stream);
}

template <typename... Args>
inline void print(std::FILE* stream, std::format_string<Args...> fmt, Args&&... args)
{
    vprint(stream, fmt.get(), std::make_format_args(args...));
}

template <typename... Args>
inline void print(std::format_string<Args...> fmt, Args&&... args)
{
    vprint(stdout, fmt.get(), std::make_format_args(args...));
}

} // namespace cucim
