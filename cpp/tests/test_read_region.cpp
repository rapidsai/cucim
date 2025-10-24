/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <iostream>

#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <openslide/openslide.h>

#include "config.h"
#include "cucim/core/framework.h"
#include "cucim/io/device.h"
#include "cucim/io/format/image_format.h"
#include "cucim/memory/memory_manager.h"


SCENARIO("Verify read_region()", "[test_read_region.cpp]")
{
    constexpr int test_sx = 200;
    constexpr int test_sy = 300;
    constexpr int test_width = 3;
    constexpr int test_height = 2;

    for (int iter=0; iter< 100; iter++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        std::string input_path = g_config.get_input_path();
        openslide_t* slide = openslide_open(input_path.c_str());
        REQUIRE(slide != nullptr);

        auto buf = static_cast<uint32_t*>(cucim_malloc(test_width * test_height * 4));
        int64_t w, h;
        openslide_get_level0_dimensions(slide, &w, &h);
        printf("w = %ld h=%ld\n", w, h);
        openslide_read_region(slide, buf, test_sx, test_sy, 0, test_width, test_height);
        auto out_image = reinterpret_cast<uint8_t*>(buf);
        int hash = 0;
        for(int i = 0 ;i < test_width * test_height * 4; i+= 4) {
            hash += out_image[i] + out_image[i+1] + out_image[i+2];
            printf("%d %d %d ", out_image[i + 2], out_image[i+1], out_image[i]);
        }
        printf("\nopenslide count: %d\n", hash);
        cucim_free(buf);
        openslide_close(slide);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        printf("time:%f\n", elapsed_seconds.count());
    }
    printf("\n\n");


    cucim::Framework* framework = cucim::acquire_framework("sample.app");

    //TODO: Parameterize input library/image
    cucim::io::format::IImageFormat* image_format =
        framework->acquire_interface_from_library<cucim::io::format::IImageFormat>(g_config.get_plugin_path().c_str());
    if (!image_format)
    {
        throw std::runtime_error("Cannot load plugin!");
    }
    std::cout << image_format->formats[0].get_format_name() << std::endl;

    for (int iter=0; iter< 100; iter++)
    {

        auto start = std::chrono::high_resolution_clock::now();

        std::string input_path = g_config.get_input_path();
        std::shared_ptr<CuCIMFileHandle>* file_handle_shared = reinterpret_cast<std::shared_ptr<CuCIMFileHandle>*>(
            image_format->formats[0].image_parser.open(input_path.c_str()));

        std::shared_ptr<CuCIMFileHandle> file_handle = *file_handle_shared;
        delete file_handle_shared;

        // Set deleter to close the file handle
        file_handle->set_deleter(image_format->formats[0].image_parser.close);

        cucim::io::format::ImageMetadata metadata{};
        image_format->formats[0].image_parser.parse(file_handle.get(), &metadata.desc());

        cucim::io::format::ImageReaderRegionRequestDesc request{};
        int64_t request_location[2] = { test_sx, test_sy };
        request.location = request_location;
        request.level = 0;
        int64_t request_size[2] = { test_width, test_height };
        request.size = request_size;
        request.device = const_cast<char*>("cpu");

        cucim::io::format::ImageDataDesc image_data{};

        image_format->formats[0].image_reader.read(
            file_handle.get(), &metadata.desc(), &request, &image_data, nullptr /*out_metadata*/);
        auto out_image = reinterpret_cast<uint8_t*>(image_data.container.data);

        int hash = 0;
        for (int i = 0; i < test_width * test_height * 3; i += 3)
        {
            hash += out_image[i] + out_image[i + 1] + out_image[i + 2];
            printf("%d %d %d ", out_image[i], out_image[i + 1], out_image[i + 2]);
        }
        printf("\ncucim count: %d\n", hash);
        //        for (int i = 0; i < test_width * test_height * 4; i += 4)
        //        {
        //            hash += out_image[i] + out_image[i + 1] + out_image[i + 2];
        //            printf("%d %d %d ", out_image[i], out_image[i + 1], out_image[i + 2]);
        //        }
        printf("\ncucim count: %d\n", hash);
        cucim_free(image_data.container.data);
        if (image_data.container.shape)
        {
            cucim_free(image_data.container.shape);
            image_data.container.shape = nullptr;
        }
        if (image_data.container.strides)
        {
            cucim_free(image_data.container.strides);
            image_data.container.strides = nullptr;
        }
        if (image_data.shm_name)
        {
            cucim_free(image_data.shm_name);
            image_data.shm_name = nullptr;
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        printf("time2:%f\n", elapsed_seconds.count());
    }



    REQUIRE(3 == 3);
}
