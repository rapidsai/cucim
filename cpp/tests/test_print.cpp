/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <catch2/catch_test_macros.hpp>
#include <cucim/util/print.h>

#include <cstdio>
#include <memory>
#include <string_view>
#include <system_error>

namespace
{
struct FileCloser
{
    void operator()(std::FILE* stream) const
    {
        std::fclose(stream);
    }
};
using FilePtr = std::unique_ptr<std::FILE, FileCloser>;
} // namespace

TEST_CASE("cucim::print writes the formatted text", "[test_print.cpp]")
{
    FilePtr stream(std::tmpfile());
    REQUIRE(stream);

    cucim::print(stream.get(), "{}-{}", 7, "x");

    std::rewind(stream.get());
    char text[8] = {};
    REQUIRE(std::fread(text, 1, sizeof(text) - 1, stream.get()) == 3);
    CHECK(std::string_view(text) == "7-x");
}

TEST_CASE("cucim::print throws std::system_error when the write fails", "[test_print.cpp]")
{
    FilePtr stream(std::fopen("/dev/full", "w"));
    if (!stream)
    {
        SKIP("/dev/full is not available");
    }
    // Unbuffered, so that fwrite itself fails instead of a later flush.
    REQUIRE(std::setvbuf(stream.get(), nullptr, _IONBF, 0) == 0);

    try
    {
        cucim::print(stream.get(), "{}", 42);
        FAIL("cucim::print returned without reporting the failed write");
    }
    catch (const std::system_error& e)
    {
        CHECK(e.code() == std::errc::no_space_on_device);
    }
}
