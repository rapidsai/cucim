/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <catch2/catch_test_macros.hpp>

#include "cuslide/tiff/ome_xml.h"

TEST_CASE("OME parser reads pixels and channels", "[ome][parser]")
{
    constexpr const char* kOmeXml = R"OME(
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0" Name="multiplex">
    <Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint16"
            SizeX="1000" SizeY="500" SizeZ="2" SizeC="3" SizeT="4"
            PhysicalSizeX="0.325" PhysicalSizeY="0.325">
      <Channel ID="Channel:0:0" Name="DAPI" SamplesPerPixel="1"/>
      <Channel ID="Channel:0:1" Name="CD3" SamplesPerPixel="1"/>
      <Channel ID="Channel:0:2" Name="CD8" SamplesPerPixel="1"/>
      <TiffData IFD="0" FirstC="0" FirstZ="0" FirstT="0" PlaneCount="4"/>
    </Pixels>
  </Image>
</OME>
)OME";

    cuslide::tiff::ome::Model model;
    std::string error;
    REQUIRE(cuslide::tiff::ome::parse(kOmeXml, &model, &error));
    REQUIRE(model.valid);
    REQUIRE(model.pixels.size_x == 1000);
    REQUIRE(model.pixels.size_y == 500);
    REQUIRE(model.pixels.size_c == 3);
    REQUIRE(model.pixels.size_z == 2);
    REQUIRE(model.pixels.size_t == 4);
    REQUIRE(model.pixels.type == "uint16");
    REQUIRE(model.pixels.dimension_order == "XYZCT");
    REQUIRE(model.pixels.channels.size() == 3);
    REQUIRE(model.pixels.channels[0].name == "DAPI");
    REQUIRE(model.pixels.channels[2].name == "CD8");
    REQUIRE(model.pixels.tiff_data.size() == 1);
    REQUIRE(model.pixels.tiff_data[0].plane_count == 4);
}

TEST_CASE("OME parser reads UUID companion references", "[ome][parser]")
{
    constexpr const char* kOmeXml = R"OME(
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0">
    <Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="uint16"
            SizeX="256" SizeY="256" SizeZ="1" SizeC="2" SizeT="1">
      <Channel ID="Channel:0:0" Name="Marker0"/>
      <Channel ID="Channel:0:1" Name="Marker1"/>
      <TiffData IFD="0" FirstC="0" FirstZ="0" FirstT="0" PlaneCount="1">
        <UUID FileName="companion_a.ome.tif">urn:uuid:a</UUID>
      </TiffData>
      <TiffData IFD="3" FirstC="1" FirstZ="0" FirstT="0" PlaneCount="1">
        <UUID FileName="companion_b.ome.tif">urn:uuid:b</UUID>
      </TiffData>
    </Pixels>
  </Image>
</OME>
)OME";

    cuslide::tiff::ome::Model model;
    REQUIRE(cuslide::tiff::ome::parse(kOmeXml, &model));
    REQUIRE(model.pixels.tiff_data.size() == 2);
    REQUIRE(model.pixels.tiff_data[0].file_name == "companion_a.ome.tif");
    REQUIRE(model.pixels.tiff_data[0].uuid == "urn:uuid:a");
    REQUIRE(model.pixels.tiff_data[1].file_name == "companion_b.ome.tif");
    REQUIRE(model.pixels.tiff_data[1].ifd == 3);
    REQUIRE(model.pixels.tiff_data[1].first_c == 1);
}

