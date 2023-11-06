#
# Copyright (c) 2022, NVIDIA CORPORATION.
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

import os
from pathlib import Path

import pytest

# skip if imagecodecs and openslide packages are not available
pytest.importorskip("imagecodecs")
pytest.importorskip("openslide")


def test_image_converter_stripe_4096x4096_256_jpeg(
    tmp_path, testimg_tiff_stripe_4096x4096_256_jpeg
):
    import tifffile

    from cucim.clara.converter import tiff

    tile_size = 128
    overlap = 0
    num_workers = os.cpu_count()
    file_name = "test_image_converter_stripe_4096x4096_256_jpeg-output.tif"
    output_path = tmp_path / file_name
    tiff.svs2tif(
        testimg_tiff_stripe_4096x4096_256_jpeg,
        output_folder=Path(tmp_path),
        tile_size=tile_size,
        overlap=overlap,
        num_workers=num_workers,
        output_filename=file_name,
    )

    assert os.path.exists(output_path)

    with tifffile.TiffFile(output_path) as tif:
        assert len(tif.pages) == 6
        assert tif.pages[0].shape == (4096, 4096, 3)
        assert tif.pages[0].tile == (128, 128)
        assert tif.pages[0].compression == tifffile.COMPRESSION.JPEG
        assert tif.pages[5].shape == (128, 128, 3)
        assert tif.pages[5].tile == (128, 128)
        assert tif.pages[5].compression == tifffile.COMPRESSION.JPEG
