#
# Copyright (c) 2021, NVIDIA CORPORATION.
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

from functools import reduce
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
from tifffile import TiffWriter

COMPRESSION_MAP = {'jpeg': ('jpeg', 95),
                   'deflate': 'deflate'}


class TiffGenerator:

    def get_image(self, pattern, image_size):
        try:
            func = getattr(self, pattern)
        except AttributeError:
            func = None

        if func:
            return func(image_size)
        return None

    def save_image(self, image_data, dest_folder, file_name, kind, subpath,
                   pattern, image_size, tile_size, compression):
        # You can add pyramid images (0: largest resolution)
        if isinstance(image_data, list):
            arr_stack = image_data
        elif isinstance(image_data, np.ndarray):
            arr_stack = [image_data]
        else:
            raise RuntimeError("'image_data' is neithor list or numpy.ndarray")

        compression = COMPRESSION_MAP.get(compression)
        if not compression:
            compression = ("jpeg", 95)

        # save as tif
        tiff_file_name = str(
            (Path(dest_folder) / f'{file_name}.tif').absolute())

        with TiffWriter(tiff_file_name, bigtiff=True) as tif:
            for level in range(len(arr_stack)):  # save from the largest image
                src_arr = arr_stack[level]

                tif.write(
                    src_arr,
                    software="tifffile",
                    metadata={"axes": "YXC"},
                    tile=(tile_size, tile_size),
                    photometric="RGB",
                    planarconfig="CONTIG",
                    compression=compression,  # requires imagecodecs
                    subfiletype=1 if level else 0,
                )
        return tiff_file_name

    def stripe(self, image_size):

        if 256 <= image_size[0] <= 4096:
            pyramid = True
        else:
            pyramid = False

        image_size_list = [image_size]
        if pyramid:
            width, height = image_size
            while width >= 256:
                width //= 2
                height //= 2
                image_size_list.append((width, height))

        array_list = []

        for level, size in enumerate(image_size_list):
            # create array
            shape = list(reversed(size)) + [3]
            area = reduce(lambda x, y: x * y, size)
            # Use mmap if image size is larger than 1GB
            if area * 3 > 2**20 * 1024:
                file_name = str(Path(mkdtemp()) / 'memmap.dat')
                array = np.memmap(file_name, dtype=np.uint8,
                                  mode='w+', shape=tuple(shape))
            else:
                array = np.zeros(shape, dtype=np.uint8)

            # Set different value depending on level
            array[::6, :, :] = (255 - level, 0, 0)
            array[1::6, :, :] = (255 - level, 0, 0)
            array[2::6, :, :] = (0, 255 - level, 0)
            array[3::6, :, :] = (0, 255 - level, 0)
            array[4::6, :, :] = (0, 0, 255 - level)
            array[5::6, :, :] = (0, 0, 255 - level)

            array_list.append(array)

        return array_list
