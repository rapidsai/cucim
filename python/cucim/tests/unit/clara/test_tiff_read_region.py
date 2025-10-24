#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest

from ...util.io import open_image_cucim

# skip if imagecodecs package not available (needed by ImageGenerator utility)
pytest.importorskip("imagecodecs")


def test_tiff_stripe_inner(testimg_tiff_stripe_32x24_16):
    cucim_img = open_image_cucim(testimg_tiff_stripe_32x24_16)
    width, height = cucim_img.size("XY")
    tile_width, tile_height = cucim_img.resolutions["level_tile_sizes"][0]

    # List of ((<start x>, <start y>), (<width>), (<height>))
    region_list = [
        ((0, 0), (width, height)),  # whole
        ((0, 0), (tile_width // 2, tile_height // 2)),  # left-top
        (
            (tile_width // 2, tile_height // 2),
            (tile_width, tile_height),
        ),  # middle
        (
            (width - (tile_width // 2), height - (tile_height // 2)),
            (tile_width // 2, tile_height // 2),
        ),  # right-bottom
    ]
    for start_pos, size in region_list:
        cucim_arr = np.asarray(cucim_img.read_region(start_pos, size))

        # Not all channel values are zero, so we need to check that.
        channel_value_count = np.count_nonzero(cucim_arr, axis=2)
        assert np.all(channel_value_count > 0)


def test_tiff_stripe_boundary(testimg_tiff_stripe_32x24_16):
    cucim_img = open_image_cucim(testimg_tiff_stripe_32x24_16)
    width, height = cucim_img.size("XY")
    tile_width, tile_height = cucim_img.resolutions["level_tile_sizes"][0]

    # List of ((<start x>, <start y>), (<width>), (<height>))
    region_list = [
        (
            (-(tile_width // 2), -(tile_height // 2)),
            (tile_width, tile_height),
        ),  # left top
        (
            (width - (tile_width // 2), -(tile_height // 2)),
            (tile_width, tile_height),
        ),  # right top
        (
            (-(tile_width // 2), height - (tile_height // 2)),
            (tile_width, tile_height),
        ),  # left bottom
        (
            (width - (tile_width // 2), height - (tile_height // 2)),
            (tile_width, tile_height),
        ),  # right bottom
    ]

    for start_pos, size in region_list:
        cucim_arr = np.asarray(cucim_img.read_region(start_pos, size))
        # Not all channel values are zero, so we need to check that.
        channel_value_count = np.count_nonzero(cucim_arr, axis=2)
        count_all_zero = np.count_nonzero(channel_value_count == 0)
        # 75% of the pixels would be all zero
        assert count_all_zero - (tile_width * tile_height * 0.75) < 5


def test_tiff_stripe_outside(testimg_tiff_stripe_32x24_16):
    cucim_img = open_image_cucim(testimg_tiff_stripe_32x24_16)
    width, height = cucim_img.size("XY")
    tile_width, tile_height = cucim_img.resolutions["level_tile_sizes"][0]

    # List of ((<start x>, <start y>), (<width>), (<height>))
    region_list = [
        (
            (-width - tile_width, -height - tile_height),
            (tile_width, tile_height),
        ),  # left top (outside)
        (
            (width + tile_height, height + tile_height),
            (tile_width, tile_height),
        ),  # right bottom (outside)
    ]

    for start_pos, size in region_list:
        cucim_arr = np.asarray(cucim_img.read_region(start_pos, size))
        # All channel values should be zero, so we need to check that.
        channel_value_count = np.count_nonzero(cucim_arr, axis=2)
        count_all_zero = np.count_nonzero(channel_value_count == 0)
        # All pixels would be zero.
        assert count_all_zero == (tile_width * tile_height)


def test_tiff_outside_of_resolution_level(testimg_tiff_stripe_4096x4096_256):
    cucim_img = open_image_cucim(testimg_tiff_stripe_4096x4096_256)
    with pytest.raises(ValueError, match=r"'level' should be less than "):
        _ = cucim_img.read_region(level=-1)

    with pytest.raises(ValueError, match=r"'level' should be less than "):
        _ = cucim_img.read_region(level=7)


def test_tiff_stripe_multiresolution(testimg_tiff_stripe_4096x4096_256):
    cucim_img = open_image_cucim(testimg_tiff_stripe_4096x4096_256)

    level_count = cucim_img.resolutions["level_count"]
    assert level_count == 6

    start_pos, size = ((0, 0), (256, 256))
    for level in range(level_count):
        cucim_arr = np.asarray(cucim_img.read_region(start_pos, size, level))
        # Not all channel values are zero, so we need to check that.
        channel_value_count = np.count_nonzero(cucim_arr, axis=2)
        count_all_zero = np.count_nonzero(channel_value_count == 0)
        img_size = cucim_img.resolutions["level_dimensions"][level]
        # Only outside of the box is zero.
        assert count_all_zero == 256 * 256 - (
            min(img_size[0], 256) * min(img_size[1], 256)
        )


def test_region_image_level_data(testimg_tiff_stripe_4096x4096_256):
    cucim_img = open_image_cucim(testimg_tiff_stripe_4096x4096_256)

    level_count = cucim_img.resolutions["level_count"]

    start_pos, size = ((-10, -20), (300, 400))
    for level in range(level_count):
        region_img = cucim_img.read_region(start_pos, size, level)
        resolutions = region_img.resolutions
        assert resolutions["level_count"] == 1
        assert resolutions["level_dimensions"][0] == (300, 400)
        assert resolutions["level_downsamples"] == (1.0,)
        assert resolutions["level_tile_sizes"][0] == (300, 400)


def test_region_image_dtype(testimg_tiff_stripe_4096x4096_256):
    from cucim.clara import DLDataType, DLDataTypeCode

    cucim_img = open_image_cucim(testimg_tiff_stripe_4096x4096_256)

    level_count = cucim_img.resolutions["level_count"]

    start_pos, size = ((0, 10), (20, 30))
    for level in range(level_count):
        region_img = cucim_img.read_region(start_pos, size, level)
        assert region_img.dtype == DLDataType(DLDataTypeCode.DLUInt, 8, 1)
        assert np.dtype(region_img.typestr) == np.uint8


def test_array_interface_support(testimg_tiff_stripe_32x24_16_jpeg):
    img = open_image_cucim(testimg_tiff_stripe_32x24_16_jpeg)
    whole_img = img.read_region()
    array_interface = whole_img.__array_interface__

    # {'data': (45867600, False), 'strides': None,
    #  'descr': [('', '|u1')], 'typestr': '|u1',
    #  'shape': (24, 32, 3), 'version': 3}
    assert array_interface["data"][0] is not None
    assert not array_interface["data"][1]
    assert array_interface["strides"] is None
    assert array_interface["descr"]
    assert array_interface["shape"] == tuple(whole_img.shape)
    assert array_interface["version"] == 3


def test_cuda_array_interface_support(testimg_tiff_stripe_32x24_16_jpeg):
    img = open_image_cucim(testimg_tiff_stripe_32x24_16_jpeg)
    whole_img = img.read_region(device="cuda")
    array_interface = whole_img.__cuda_array_interface__
    print(array_interface)

    # {'data': (81888083968, False), 'strides': None,
    #  'descr': [('', '|u1')], 'typestr': '|u1', 'shape': (24, 32, 3),
    #  'version': 3, 'mask': None, 'stream': 1}
    assert array_interface["data"][0] is not None
    assert not array_interface["data"][1]
    assert array_interface["strides"] is None
    assert array_interface["descr"]
    assert array_interface["typestr"]
    assert array_interface["shape"] == tuple(whole_img.shape)
    assert array_interface["version"] == 3
    assert array_interface["mask"] is None
    assert array_interface["stream"] == 1


def test_tiff_iterator(testimg_tiff_stripe_4096x4096_256):
    """Test that the iterator of read_region works as expected.
    See issue gh-592: https://github.com/rapidsai/cucim/issues/592
    """

    level = 0
    size = (256, 256)

    with open_image_cucim(testimg_tiff_stripe_4096x4096_256) as slide:
        x = slide.read_region([(0, 0), (0, 0)], size, level, num_workers=1)
        x = slide.read_region([(0, 0), (0, 0)], size, level, num_workers=1)

    with open_image_cucim(testimg_tiff_stripe_4096x4096_256) as slide:
        x = slide.read_region([(0, 0), (0, 0)], size, level, num_workers=1)
        _ = next(x)
        x = slide.read_region([(0, 0), (0, 0)], size, level, num_workers=1)
        x = slide.read_region([(0, 0), (0, 0)], size, level, num_workers=1)

    with open_image_cucim(testimg_tiff_stripe_4096x4096_256) as slide:
        x = slide.read_region([(0, 0), (0, 0)], size, level, num_workers=1)
        _ = next(x)
        x = slide.read_region([(0, 0), (0, 0)], size, level, num_workers=1)
        _ = next(x)
        x = slide.read_region([(0, 0), (0, 0)], size, level, num_workers=1)
        _ = next(x)
        x = slide.read_region(
            [(0, 0), (0, 0), (0, 0)], size, level, num_workers=1
        )
        _ = next(x)
        x = slide.read_region([(0, 0), (0, 0)], size, level, num_workers=1)
        _ = next(x)
        x = slide.read_region([(0, 0), (0, 0)], size, level, num_workers=1)
        _ = next(x)
        x = slide.read_region([(0, 0), (0, 0)], size, level, num_workers=1)
        _ = next(x)
