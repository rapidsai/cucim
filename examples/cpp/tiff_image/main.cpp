/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cucim/cuimage.h>
#include <cucim/util/print.h>
#include <format>
#include <iterator>
#include <string>
#include <string_view>

namespace
{

template <typename Range>
std::string join_to_string(const Range& range, std::string_view separator)
{
    std::string out;
    bool first = true;
    for (const auto& elem : range)
    {
        if (!first)
        {
            out += separator;
        }
        first = false;
        std::format_to(std::back_inserter(out), "{}", elem);
    }
    return out;
}

} // namespace

int main(int argc, char* argv[])
{
    // Check the number of parameters
    if (argc < 3) {
        cucim::print(stderr, "Usage: {} INPUT_FILE_PATH OUTPUT_FOLDER\n", argv[0]);
        return 1;
    }
    const char* input_file_path = argv[1];
    const char* output_folder_path = argv[2];

    cucim::CuImage image = cucim::CuImage(input_file_path);

    cucim::print("is_loaded: {}\n", image.is_loaded());
    cucim::print("device: {}\n", std::string(image.device()));
    cucim::print("metadata: {}\n", image.metadata());
    cucim::print("dims: {}\n", image.dims());
    cucim::print("shape: ({})\n", join_to_string(image.shape(), ", "));
    cucim::print("size('XYC'): ({})\n", join_to_string(image.size("XYC"), ", "));
    cucim::print("channel_names: ({})\n", join_to_string(image.channel_names(), ", "));

    auto resolutions = image.resolutions();
    cucim::print("level_count: {}\n", resolutions.level_count());
    cucim::print("level_dimensions: ({})\n", join_to_string(resolutions.level_dimensions(), ", "));
    cucim::print("level_dimension (level 0): ({})\n", join_to_string(resolutions.level_dimension(0), ", "));
    cucim::print("level_downsamples: ({})\n", join_to_string(resolutions.level_downsamples(), ", "));
    cucim::print("level_tile_sizes: ({})\n", join_to_string(resolutions.level_tile_sizes(), ", "));

    auto associated_images = image.associated_images();
    cucim::print("associated_images: ({})\n", join_to_string(associated_images, ", "));

    cucim::print("#macro\n");
    auto associated_image = image.associated_image("macro");
    cucim::print("is_loaded: {}\n", associated_image.is_loaded());
    cucim::print("device: {}\n", std::string(associated_image.device()));
    cucim::print("metadata: {}\n", associated_image.metadata());
    cucim::print("dims: {}\n", associated_image.dims());
    cucim::print("shape: ({})\n", join_to_string(associated_image.shape(), ", "));
    cucim::print("size('XYC'): ({})\n", join_to_string(associated_image.size("XYC"), ", "));
    cucim::print("channel_names: ({})\n", join_to_string(associated_image.channel_names(), ", "));
    cucim::print("\n");

    cucim::CuImage region = image.read_region({ 10000, 10000 }, { 1024, 1024 }, 0);

    cucim::print("is_loaded: {}\n", region.is_loaded());
    cucim::print("device: {}\n", std::string(region.device()));
    cucim::print("metadata: {}\n", region.metadata());
    cucim::print("dims: {}\n", region.dims());
    cucim::print("shape: ({})\n", join_to_string(region.shape(), ", "));
    cucim::print("size('XY'): ({})\n", join_to_string(region.size("XY"), ", "));
    cucim::print("channel_names: ({})\n", join_to_string(region.channel_names(), ", "));

    resolutions = region.resolutions();
    cucim::print("level_count: {}\n", resolutions.level_count());
    cucim::print("level_dimensions: ({})\n", join_to_string(resolutions.level_dimensions(), ", "));
    cucim::print("level_dimension (level 0): ({})\n", join_to_string(resolutions.level_dimension(0), ", "));
    cucim::print("level_downsamples: ({})\n", join_to_string(resolutions.level_downsamples(), ", "));
    cucim::print("level_tile_sizes: ({})\n", join_to_string(resolutions.level_tile_sizes(), ", "));

    associated_images = region.associated_images();
    cucim::print("associated_images: ({})\n", join_to_string(associated_images, ", "));
    cucim::print("\n");

    region.save(std::format("{}/output.ppm", output_folder_path));

    cucim::CuImage region2 = image.read_region({ 5000, 5000 }, { 1024, 1024 }, 1);
    region2.save(std::format("{}/output2.ppm", output_folder_path));

    // Batch loading image
    // You need to create shared pointer for cucim::CuImage. Otherwise it will cause std::bad_weak_ptr exception.
    auto batch_image = std::make_shared<cucim::CuImage>(input_file_path);

    auto region3 = std::make_shared<cucim::CuImage>(image.read_region(
        { 0, 0, 100, 200, 300, 300, 400, 400, 500, 500, 600, 600, 700, 700, 800, 800, 900, 900, 1000, 1000 },
        { 200, 200 }, 0 /*level*/, 2 /*num_workers*/, 2 /*batch_size*/, false /*drop_last*/, 1 /*prefetch_factor*/,
        false /*shuffle*/, 0 /*seed*/));

    for (auto batch : *region3)
    {
        cucim::print("shape: {}, data size:{}\n", join_to_string(batch->shape(), ", "), batch->container().size());
    }
    return 0;
}
