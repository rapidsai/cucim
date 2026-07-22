/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cuslide::tiff::ome
{

struct Channel
{
    uint32_t index = 0;
    std::string id;
    std::string name;
    uint32_t samples_per_pixel = 1;
};

struct TiffDataEntry
{
    uint32_t ifd = 0;
    uint32_t first_c = 0;
    uint32_t first_z = 0;
    uint32_t first_t = 0;
    uint32_t plane_count = 1;
    std::string uuid;
    std::string file_name;
};

struct Pixels
{
    uint32_t size_x = 0;
    uint32_t size_y = 0;
    uint32_t size_c = 1;
    uint32_t size_z = 1;
    uint32_t size_t = 1;
    std::string type;
    std::string dimension_order = "XYCZT";
    double physical_size_x = 0.0;
    double physical_size_y = 0.0;
    std::string physical_size_x_unit;
    std::string physical_size_y_unit;
    bool has_physical_size_x = false;
    bool has_physical_size_y = false;
    std::vector<Channel> channels;
    std::vector<TiffDataEntry> tiff_data;
};

struct Model
{
    bool valid = false;
    Pixels pixels;
    std::string image_id;
    std::string image_name;
};

bool parse(const std::string& xml_text, Model* out_model, std::string* out_error = nullptr);

} // namespace cuslide::tiff::ome

