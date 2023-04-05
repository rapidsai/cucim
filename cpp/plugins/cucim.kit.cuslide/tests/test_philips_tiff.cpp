/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cucim/memory/memory_manager.h>
#include <openslide/openslide.h>
#include "cuslide/tiff/tiff.h"
#include "config.h"

#include <catch2/catch.hpp>
#include <chrono>

TEST_CASE("Verify philips tiff file", "[test_philips_tiff.cpp]")
{

    auto tif = std::make_shared<cuslide::tiff::TIFF>(g_config.get_input_path("private/philips_tiff_000.tif").c_str(),
                                                     O_RDONLY); // , cuslide::tiff::TIFF::kUseLibTiff
    tif->construct_ifds();

    int64_t test_sx = 0;
    int64_t test_sy = 0;

    int64_t test_width = 500;
    int64_t test_height = 500;

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

    request.associated_image_name = const_cast<char*>("label");
    tif->read(&metadata.desc(), &request, &image_data, nullptr /*out_metadata*/);

    tif->close();

    REQUIRE(1 == 1);
}
