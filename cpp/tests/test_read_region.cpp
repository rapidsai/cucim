/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <chrono>
#include <iostream>

#include <catch2/catch.hpp>
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
        openslide_t* slide = openslide_open(g_config.get_input_path().c_str());
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
        auto handle = image_format->formats[0].image_parser.open(g_config.get_input_path().c_str());

        cucim::io::format::ImageMetadata metadata{};
        image_format->formats[0].image_parser.parse(&handle, &metadata.desc());

        cucim::io::format::ImageReaderRegionRequestDesc request{};
        int64_t request_location[2] = { test_sx, test_sy };
        request.location = request_location;
        request.level = 0;
        int64_t request_size[2] = { test_width, test_height };
        request.size = request_size;
        request.device = const_cast<char*>("cpu");

        cucim::io::format::ImageDataDesc image_data{};

        image_format->formats[0].image_reader.read(
            &handle, &metadata.desc(), &request, &image_data, nullptr /*out_metadata*/);
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
        image_format->formats[0].image_parser.close(&handle);

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        printf("time2:%f\n", elapsed_seconds.count());
    }



    REQUIRE(3 == 3);
}
