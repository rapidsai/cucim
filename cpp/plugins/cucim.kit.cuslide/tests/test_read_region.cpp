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

#include <catch2/catch.hpp>
#include <openslide/openslide.h>

#include <cucim/memory/memory_manager.h>

#include "config.h"
#include "cuslide/tiff/tiff.h"


TEST_CASE("Verify read_region()", "[test_read_region.cpp]")
{
    SECTION("Test with different parameters")
    {
        auto test_sx = GENERATE(as<int64_t>{}, 1, 255, 256, 511, 512);
        auto test_sy = GENERATE(as<int64_t>{}, 1, 255, 256, 511, 512);
        auto test_width = GENERATE(as<int64_t>{}, 1, 255, 256, 511, 512);
        auto test_height = GENERATE(as<int64_t>{}, 1, 255, 256, 511, 512);

        INFO("Execute with [sx:" << test_sx << ", sy:" << test_sy << ", width:" << test_width
                                 << ", height:" << test_height << "]");

        int openslide_count = 0;
        int cucim_count = 0;

        printf("[sx:%ld, sy:%ld, width:%ld, height:%ld]\n", test_sx, test_sy, test_width, test_height);
        {
            auto start = std::chrono::high_resolution_clock::now();

            openslide_t* slide = openslide_open(g_config.get_input_path().c_str());
            REQUIRE(slide != nullptr);

            auto buf = static_cast<uint32_t*>(cucim_malloc(test_width * test_height * 4));
            openslide_read_region(slide, buf, test_sx, test_sy, 0, test_width, test_height);

            openslide_close(slide);

            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
            printf("openslide: %f\n", elapsed_seconds.count());

            auto out_image = reinterpret_cast<uint8_t*>(buf);
            for (int i = 0; i < test_width * test_height * 4; i += 4)
            {
                openslide_count += out_image[i] + out_image[i + 1] + out_image[i + 2];
            }
            INFO("openslide value count: " << openslide_count);

            cucim_free(buf);
        }

        {
            auto start = std::chrono::high_resolution_clock::now();

            auto tif = std::make_shared<cuslide::tiff::TIFF>(g_config.get_input_path().c_str(),
                                                             O_RDONLY); // , cuslide::tiff::TIFF::kUseLibTiff
            tif->construct_ifds();

            cucim::io::format::ImageMetadata metadata{};
            cucim::io::format::ImageReaderRegionRequestDesc request{};
            cucim::io::format::ImageDataDesc image_data{};

            metadata.level_count(1).level_downsamples({ 1.0 }).level_ndim(3);

            int64_t request_location[2] = { test_sx, test_sy };
            request.location = request_location;
            request.level = 0;
            int64_t request_size[2] = { test_width, test_height };
            request.size = request_size;
            request.device = const_cast<char*>("cpu");

            tif->read(&metadata.desc(), &request, &image_data);

            tif->close();

            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

            printf("cucim: %f\n", elapsed_seconds.count());
            auto out_image = reinterpret_cast<uint8_t*>(image_data.container.data);
            for (int i = 0; i < test_width * test_height * 3; i += 3)
            {
                cucim_count += out_image[i] + out_image[i + 1] + out_image[i + 2];
            }
            INFO("cucim value count: " << cucim_count);

            cucim_free(image_data.container.data);
            printf("\n");
        }

        REQUIRE(openslide_count == cucim_count);

        /**
         * Note: Experiment with OpenSlide with various level values (2020-09-28)
         *
         * When other level (1~) is used (for example, sx=4, sy=4, level=2, assuming that down factor is 4 for
         * level 2), openslide's output is same with the values of cuCIM on the start position (sx/4, sy/4). If sx and
         * sy is not multiple of 4, openslide's output was not trivial and performance was low.
         */
    }
}