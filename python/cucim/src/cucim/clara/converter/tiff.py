#
# Copyright (c) 2020, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import concurrent.futures
import gc
import logging
import os
from itertools import repeat
from pathlib import Path

import cv2
import numpy as np
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator
from tifffile import TiffWriter

SUBFILETYPE_NONE = 0
SUBFILETYPE_REDUCEDIMAGE = 1

logger = logging.getLogger(__name__)


def filter_tile(
    tiles, dim_index, index, tile_size, output_array
):
    try:
        x, y = index
        tile = tiles.get_tile(dim_index, index)

        tile_width, tile_height = tile.size

        # Make image the same size for inference
        if tile.size != (tile_size, tile_size):
            tile = tile.crop((0, 0, tile_size, tile_size))
        ax = x * tile_size
        ay = y * tile_size

        tile_arr = np.array(tile)  # H x W x C

        output_array[ay: ay + tile_height, ax: ax + tile_width, :] = tile_arr[
            :tile_height, :tile_width]
    except Exception as e:
        logger.exception(e)


def svs2tif(input_file, output_folder, tile_size, overlap,
            num_workers=os.cpu_count(), output_filename="image.tif"):
    output_folder = str(output_folder)

    logger.info("Parameters")
    logger.info("       input file: %s", input_file)
    logger.info("    output folder: %s", output_folder)
    logger.info("        tile size: %d", tile_size)
    logger.info("          overlap: %d", overlap)
    logger.info("      num_workers: %d", num_workers)
    logger.info("  output filename: %s", output_filename)

    with OpenSlide(input_file) as slide:
        properties = slide.properties
        slide_dimensions = slide.dimensions

        tiles = DeepZoomGenerator(
            slide, tile_size=tile_size, overlap=overlap, limit_bounds=False
        )

        output_file = Path(output_folder) / output_filename

        np_memmap = []
        width, height = slide_dimensions
        img_w, img_h = width, height
        for level in range(tiles.level_count):
            memmap_filename = Path(output_folder, "level{}.mmap".format(level))
            memmap_shape = (img_h, img_w, 3)
            np_memmap_arr = np.memmap(
                memmap_filename, dtype=np.uint8, mode="w+", shape=memmap_shape
            )
            np_memmap.append(np_memmap_arr)
            logger.info("  Created %s %s", memmap_filename, repr(memmap_shape))

            img_w = round(img_w / 2)
            img_h = round(img_h / 2)
            if max(img_w, img_h) < tile_size:
                break
        try:

            # Multithread processing for each tile in the largest
            # image (index 0)
            logger.info("Processing tiles...")
            dim_index = tiles.level_count - 1
            tile_pos_x, tile_pos_y = tiles.level_tiles[dim_index]
            index_iter = np.ndindex(tile_pos_x, tile_pos_y)
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=num_workers) as executor:
                executor.map(
                    filter_tile,
                    repeat(tiles),
                    repeat(dim_index),
                    index_iter,
                    repeat(tile_size),
                    repeat(np_memmap[0]),
                )

            logger.info("Storing low resolution images...")
            for index in range(1, len(np_memmap)):
                src_arr = np_memmap[index - 1]
                target_arr = np_memmap[index]
                target_arr[:] = cv2.resize(
                    src_arr, (0, 0), fx=0.5, fy=0.5,
                    interpolation=cv2.INTER_AREA
                )
                # th, tw = target_arr.shape[:2]
                # target_arr[:] = src_arr[
                #     : th * 2 : 2, : tw * 2 : 2, :
                # ]  # Fast resizing. No anti-aliasing.
                logger.info("  Level %d: %s", index, repr(target_arr.shape))

            # Calculate resolution
            if (
                properties.get("tiff.ResolutionUnit")
                and properties.get("tiff.XResolution")
                and properties.get("tiff.YResolution")
            ):
                resolution_unit = properties.get("tiff.ResolutionUnit")
                x_resolution = float(properties.get("tiff.XResolution"))
                y_resolution = float(properties.get("tiff.YResolution"))
            else:
                resolution_unit = properties.get("tiff.ResolutionUnit", "inch")
                if properties.get("tiff.ResolutionUnit",
                                  "inch").lower() == "inch":
                    numerator = 25400  # Microns in Inch
                else:
                    numerator = 10000  # Microns in CM
                x_resolution = int(numerator
                                   // float(properties.get('openslide.mpp-x',
                                                           1)))
                y_resolution = int(numerator
                                   // float(properties.get('openslide.mpp-y',
                                                           1)))

            # Write TIFF file
            with TiffWriter(output_file, bigtiff=True) as tif:
                # Save from the largest image (openslide requires that)
                for level in range(len(np_memmap)):
                    src_arr = np_memmap[level]
                    height, width = src_arr.shape[:2]
                    logger.info("Saving Level %d image (%d x %d)...",
                                level, width, height)
                    if level:
                        subfiletype = SUBFILETYPE_REDUCEDIMAGE
                    else:
                        subfiletype = SUBFILETYPE_NONE

                    tif.save(
                        src_arr,
                        software="Glencoe/Faas pyramid",
                        metadata={"axes": "YXC"},
                        tile=(tile_size, tile_size),
                        photometric="RGB",
                        planarconfig="CONTIG",
                        resolution=(
                            x_resolution // 2 ** level,
                            y_resolution // 2 ** level,
                            resolution_unit,
                        ),
                        compress=("jpeg", 95),  # requires imagecodecs
                        subfiletype=subfiletype,
                    )
                logger.info("Done.")
        finally:
            # Remove memory-mapped file
            logger.info("Removing memmapped files...")
            src_arr = None
            target_arr = None
            np_memmap_arr = None
            del np_memmap
            gc.collect()
            mmap_file_iter = Path(output_folder).glob("*.mmap")
            for fp in mmap_file_iter:
                fp.unlink()
