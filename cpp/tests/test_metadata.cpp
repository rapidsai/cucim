/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "config.h"
#include "cucim/cuimage.h"
#include "cucim/logger/timer.h"
#include "cucim/core/framework.h"
#include "cucim/io/format/image_format.h"

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <fcntl.h>
#include <unistd.h>
#include <string_view>
// Test
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <sys/mman.h>
#include <cuda_runtime.h>


#define ALIGN_UP(x, align_to) (((uint64_t)(x) + ((uint64_t)(align_to)-1)) & ~((uint64_t)(align_to)-1))

#define CUDA_ERROR(stmt)                                                                                               \
    {                                                                                                                  \
        cuda_status = stmt;                                                                                            \
        if (cudaSuccess != cuda_status)                                                                                \
        {                                                                                                              \
            INFO(fmt::format("Error message: {}", cudaGetErrorString(cuda_status)));                                   \
            REQUIRE(cudaSuccess == cuda_status);                                                                       \
        }                                                                                                              \
    }

#define POSIX_ERROR(stmt)                                                                                              \
    {                                                                                                                  \
        err = stmt;                                                                                                    \
        if (err < 0)                                                                                                   \
        {                                                                                                              \
            INFO(fmt::format("Error message: {}", std::strerror(errno)));                                              \
            REQUIRE(err >= 0);                                                                                         \
        }                                                                                                              \
    }

class Document
{
    bool is_cached_{};
    double rank_{};
    int id_{};
};
#include <string>
void test(float* haha)
{
    fmt::print("T {} {} {}\n", haha[0], haha[1], haha[2]);
}

TEST_CASE("Verify metadata", "[test_metadata.cpp]")
{
    cucim::Framework* framework = cucim::acquire_framework("sample.app");
    REQUIRE(framework != nullptr);
    std::string plugin_path = g_config.get_plugin_path();
    cucim::io::format::IImageFormat* image_format =
        framework->acquire_interface_from_library<cucim::io::format::IImageFormat>(plugin_path.c_str());
    // fmt::print("{}\n", image_format->formats[0].get_format_name());
    REQUIRE(image_format != nullptr);

    std::string input_path = g_config.get_input_path();
    std::shared_ptr<CuCIMFileHandle>* file_handle_shared = reinterpret_cast<std::shared_ptr<CuCIMFileHandle>*>(
        image_format->formats[0].image_parser.open(input_path.c_str()));

    std::shared_ptr<CuCIMFileHandle> file_handle = *file_handle_shared;
    delete file_handle_shared;

    // Set deleter to close the file handle
    file_handle->set_deleter(image_format->formats[0].image_parser.close);

    cucim::io::format::ImageMetadata metadata{};
    image_format->formats[0].image_parser.parse(file_handle.get(), &metadata.desc());

    // Using fmt::print() has a problem with TestMate VSCode plugin (output is not caught by the plugin)
    std::cout << fmt::format("metadata: {}\n", metadata.desc().raw_data);
    const uint8_t* buf = metadata.get_buffer();
    const uint8_t* buf2 = static_cast<uint8_t*>(metadata.allocate(1));
    std::cout << fmt::format("test: {}\n", buf2 - buf);

    // cucim::CuImage img{ g_config.get_input_path("private/philips_tiff_000.tif") };
    // const auto& img_metadata = img.metadata();
    // std::cout << fmt::format("metadata: {}\n", img_metadata);
    // auto v = img.spacing();
    // std::cout << fmt::format("spacing: {}\n", v.size());
    // delete ((cuslide::tiff::TIFF*)handle.client_data);
    // cucim_free(handle.client_data);
    // cucim_free(handle.client_data);

    // fmt::print("alignment: {}\n", alignof(int));
    // fmt::print("Document: {}\n", sizeof(Document));
    // fmt::print("max align: {}\n", alignof(size_t));
    // auto a = std::string{ "" };
    // fmt::print("size of ImageMetadataDesc :{}\n", sizeof(cucim::io::format::ImageMetadataDesc));
    // fmt::print("size of ImageMetadata :{}\n", sizeof(cucim::io::format::ImageMetadata));

    // cucim::io::format::ImageMetadata metadata;


    // test(std::array<float, 3>{ 1.0, 2.0, 3.0 }.data());


    // std::vector<float> d(3);


    // fmt::print("metadata: {} \n", (size_t)std::addressof(metadata));
    // fmt::print("handle: {} \n", (size_t)std::addressof(metadata.desc()));
    // fmt::print("ndim: {} \n", ((cucim::io::format::ImageMetadataDesc*)&metadata)->ndim);

    // // cucim::io::format::ImageMetadata a;

    // REQUIRE(1 == 1);
}

TEST_CASE("ImageMetadata::store_string copies into metadata memory", "[test_metadata.cpp]")
{
    cucim::io::format::ImageMetadata metadata{};

    SECTION("the copy is independent of the source and null-terminated")
    {
        std::string source{ "Cy5" };
        std::string_view stored = metadata.store_string(source);

        REQUIRE(stored == "Cy5");
        REQUIRE(stored.data() != source.data());
        // Consumers reach these through ImageMetadataDesc as plain `char*`.
        REQUIRE(stored.data()[stored.size()] == '\0');

        source[0] = 'X';
        REQUIRE(stored == "Cy5");
    }

    SECTION("a view into a longer buffer is terminated at its own end")
    {
        // The reason store_string() writes the terminator itself instead of
        // copying size() + 1 bytes from the source.
        std::string_view middle_of{ "DAPI/FITC", 4 };
        std::string_view stored = metadata.store_string(middle_of);

        REQUIRE(stored == "DAPI");
        REQUIRE(stored.size() == 4);
        REQUIRE(stored.data()[4] == '\0');
    }

    SECTION("an empty value is still a valid null-terminated string")
    {
        std::string_view stored = metadata.store_string(std::string_view{});

        REQUIRE(stored.empty());
        REQUIRE(stored.data() != nullptr);
        REQUIRE(stored.data()[0] == '\0');
    }

    SECTION("separate calls do not alias")
    {
        std::string_view first = metadata.store_string("R");
        std::string_view second = metadata.store_string("G");

        REQUIRE(first == "R");
        REQUIRE(second == "G");
        REQUIRE(first.data() != second.data());
    }
}

TEST_CASE("Load test", "[test_metadata.cpp]")
{
    cucim::CuImage img{ g_config.get_input_path("private/philips_tiff_000.tif") };
    REQUIRE(img.dtype() == DLDataType{ DLDataTypeCode::kDLUInt, 8, 1 });
    REQUIRE(img.typestr() == "|u1");

    auto test = img.read_region({ -10, -10 }, { 100, 100 });

    REQUIRE(test.dtype() == DLDataType{ DLDataTypeCode::kDLUInt, 8, 1 });
    REQUIRE(test.typestr() == "|u1");

    fmt::print("{}", img.metadata());
}
