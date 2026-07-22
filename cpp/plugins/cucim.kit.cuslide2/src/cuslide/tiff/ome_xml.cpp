/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ome_xml.h"

#include <algorithm>
#include <string_view>

#include <pugixml.hpp>

namespace cuslide::tiff::ome
{

namespace
{
std::string_view local_name(const char* name)
{
    if (!name)
    {
        return {};
    }
    std::string_view value(name);
    auto pos = value.find(':');
    return (pos == std::string_view::npos) ? value : value.substr(pos + 1);
}

pugi::xml_node find_child_by_local_name(const pugi::xml_node& node, std::string_view name)
{
    for (auto child : node.children())
    {
        if (local_name(child.name()) == name)
        {
            return child;
        }
    }
    return {};
}
} // namespace

bool parse(const std::string& xml_text, Model* out_model, std::string* out_error)
{
    if (!out_model)
    {
        if (out_error)
        {
            *out_error = "Output model pointer is null";
        }
        return false;
    }

    *out_model = Model{};

    pugi::xml_document doc;
    pugi::xml_parse_result parsed = doc.load_string(xml_text.c_str());
    if (!parsed)
    {
        if (out_error)
        {
            *out_error = parsed.description();
        }
        return false;
    }

    pugi::xml_node ome_node = doc.document_element();
    if (local_name(ome_node.name()) != "OME")
    {
        if (out_error)
        {
            *out_error = "XML root is not OME";
        }
        return false;
    }

    pugi::xml_node image_node = find_child_by_local_name(ome_node, "Image");
    if (!image_node)
    {
        if (out_error)
        {
            *out_error = "OME Image node not found";
        }
        return false;
    }

    pugi::xml_node pixels_node = find_child_by_local_name(image_node, "Pixels");
    if (!pixels_node)
    {
        if (out_error)
        {
            *out_error = "OME Pixels node not found";
        }
        return false;
    }

    Model model;
    model.image_id = image_node.attribute("ID").as_string("");
    model.image_name = image_node.attribute("Name").as_string("");

    Pixels pixels;
    pixels.size_x = pixels_node.attribute("SizeX").as_uint(0);
    pixels.size_y = pixels_node.attribute("SizeY").as_uint(0);
    pixels.size_c = std::max<uint32_t>(1, pixels_node.attribute("SizeC").as_uint(1));
    pixels.size_z = std::max<uint32_t>(1, pixels_node.attribute("SizeZ").as_uint(1));
    pixels.size_t = std::max<uint32_t>(1, pixels_node.attribute("SizeT").as_uint(1));
    pixels.type = pixels_node.attribute("Type").as_string("");
    pixels.dimension_order = pixels_node.attribute("DimensionOrder").as_string("XYCZT");

    if (auto attr = pixels_node.attribute("PhysicalSizeX"))
    {
        pixels.physical_size_x = attr.as_double();
        pixels.has_physical_size_x = true;
    }
    if (auto attr = pixels_node.attribute("PhysicalSizeY"))
    {
        pixels.physical_size_y = attr.as_double();
        pixels.has_physical_size_y = true;
    }
    pixels.physical_size_x_unit = pixels_node.attribute("PhysicalSizeXUnit").as_string("");
    pixels.physical_size_y_unit = pixels_node.attribute("PhysicalSizeYUnit").as_string("");

    uint32_t channel_idx = 0;
    for (auto child : pixels_node.children())
    {
        const auto child_name = local_name(child.name());
        if (child_name == "Channel")
        {
            Channel channel;
            channel.index = channel_idx++;
            channel.id = child.attribute("ID").as_string("");
            channel.name = child.attribute("Name").as_string("");
            channel.samples_per_pixel = std::max<uint32_t>(1, child.attribute("SamplesPerPixel").as_uint(1));
            pixels.channels.emplace_back(std::move(channel));
        }
        else if (child_name == "TiffData")
        {
            TiffDataEntry entry;
            entry.ifd = child.attribute("IFD").as_uint(0);
            entry.first_c = child.attribute("FirstC").as_uint(0);
            entry.first_z = child.attribute("FirstZ").as_uint(0);
            entry.first_t = child.attribute("FirstT").as_uint(0);
            entry.plane_count = std::max<uint32_t>(1, child.attribute("PlaneCount").as_uint(1));

            pugi::xml_node uuid_node = find_child_by_local_name(child, "UUID");
            if (uuid_node)
            {
                entry.uuid = uuid_node.child_value();
                entry.file_name = uuid_node.attribute("FileName").as_string("");
            }

            pixels.tiff_data.emplace_back(std::move(entry));
        }
    }

    if (pixels.size_x == 0 || pixels.size_y == 0)
    {
        if (out_error)
        {
            *out_error = "OME Pixels is missing SizeX/SizeY";
        }
        return false;
    }

    model.pixels = std::move(pixels);
    model.valid = true;
    *out_model = std::move(model);
    return true;
}

} // namespace cuslide::tiff::ome

