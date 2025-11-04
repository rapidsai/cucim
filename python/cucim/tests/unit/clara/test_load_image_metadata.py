#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import math

import pytest

from ...util.io import open_image_cucim

# skip if imagecodecs package not available (needed by ImageGenerator utility)
pytest.importorskip("imagecodecs")


def test_load_image_metadata(testimg_tiff_stripe_32x24_16):
    import numpy as np

    img = open_image_cucim(testimg_tiff_stripe_32x24_16)

    # True if image data is loaded & available.
    assert img.is_loaded
    # A device type.
    assert str(img.device) == "cpu"
    # The number of dimensions.
    assert img.ndim == 3
    # A string containing a list of dimensions being requested.
    assert img.dims == "YXC"
    # A tuple of dimension sizes (in the order of `dims`).
    assert img.shape == [24, 32, 3]
    # Returns size as a tuple for the given dimension order.
    assert img.size("XYC") == [32, 24, 3]
    # The data type of the image.
    dtype = img.dtype
    assert dtype.code == 1
    assert dtype.bits == 8
    assert dtype.lanes == 1
    # The typestr of the image.
    assert np.dtype(img.typestr) == np.uint8
    # A channel name list.
    assert img.channel_names == ["R", "G", "B"]
    # Returns physical size in tuple.
    assert img.spacing() == [1.0, 1.0, 1.0]
    # Units for each spacing element (size is same with `ndim`).
    assert img.spacing_units() == ["", "", "color"]
    # Physical location of (0, 0, 0) (size is always 3).
    assert img.origin == [0.0, 0.0, 0.0]
    # Direction cosines (size is always 3x3).
    assert img.direction == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    # Coordinate frame in which the direction cosines are measured.
    # Available Coordinate frame is not finalized yet.
    assert img.coord_sys == "LPS"
    # Returns a set of associated image names.
    assert img.associated_images == set()
    # Returns a dict that includes resolution information.
    assert img.resolutions == {
        "level_count": 1,
        "level_dimensions": ((32, 24),),
        "level_downsamples": (1.0,),
        "level_tile_sizes": ((16, 16),),
    }
    # A metadata object as `dict`
    metadata = img.metadata
    print(metadata)
    assert isinstance(metadata, dict)
    assert len(metadata) == 2  # 'cucim' and 'tiff'
    # A raw metadata string.
    assert img.raw_metadata == '{"axes": "YXC", "shape": [24, 32, 3]}'


def test_load_image_resolution_metadata(
    testimg_tiff_stripe_4096_4096_256_jpeg_resolution,
):  # noqa: E501
    image, resolution = testimg_tiff_stripe_4096_4096_256_jpeg_resolution
    img = open_image_cucim(image)

    x_resolution, y_resolution, resolution_unit = resolution

    if resolution_unit == "CENTIMETER":
        x_spacing = 10000.0 / x_resolution
        y_spacing = 10000.0 / y_resolution
        spacing_unit = "micrometer"
    elif resolution_unit == "INCH":
        x_spacing = 25400.0 / x_resolution
        y_spacing = 25400.0 / y_resolution
        spacing_unit = "micrometer"
    else:
        x_spacing = x_resolution
        y_spacing = y_resolution
        spacing_unit = ""

    # Returns physical size in tuple.
    assert all(
        map(
            lambda a, b: math.isclose(a, b, rel_tol=0.1),
            img.spacing(),
            (y_spacing, x_spacing, 1.0),
        )
    )
    # Units for each spacing element (size is same with `ndim`).
    assert img.spacing_units() == [spacing_unit, spacing_unit, "color"]

    # A metadata object as `dict`
    metadata = img.metadata
    print(metadata)
    assert isinstance(metadata, dict)
    assert len(metadata) == 2  # 'cucim' and 'tiff'
    assert math.isclose(
        metadata["tiff"]["x_resolution"], x_resolution, rel_tol=0.00001
    )
    assert math.isclose(
        metadata["tiff"]["y_resolution"], y_resolution, rel_tol=0.00001
    )
    unit_value = resolution_unit.lower() if resolution_unit != "NONE" else ""
    assert metadata["tiff"]["resolution_unit"] == unit_value

    # Check if lower resolution image's metadata has lower physical spacing.
    num_levels = img.resolutions["level_count"]
    for level in range(num_levels):
        lowres_img = img.read_region((0, 0), (100, 100), level=level)
        lowres_downsample = img.resolutions["level_downsamples"][level]
        assert all(
            map(
                lambda a, b: math.isclose(a, b, rel_tol=0.1),
                lowres_img.spacing(),
                (
                    y_spacing / lowres_downsample,
                    x_spacing / lowres_downsample,
                    1.0,
                ),
            )
        )


def test_load_rgba_image_metadata(tmpdir):
    """Test accessing RGBA image's metadata.

    - https://github.com/rapidsai/cucim/issues/262
    """
    import numpy as np
    from tifffile import imwrite

    from cucim import CuImage

    # Test with a 4-channel image
    img_array = np.ones((32, 32, 3)).astype(np.uint8)
    print(f"RGB image shape: {img_array.shape}")
    img_array = np.concatenate(
        [img_array, 255 * np.ones_like(img_array[..., 0])[..., np.newaxis]],
        axis=2,
    )
    print(f"RGBA image shape: {img_array.shape}")

    file_path_4ch = str(tmpdir.join("small_rgba_4ch.tiff"))
    imwrite(file_path_4ch, img_array, shape=img_array.shape, tile=(16, 16))

    obj = CuImage(file_path_4ch)
    assert obj.metadata["cucim"]["channel_names"] == ["R", "G", "B", "A"]
    obj2 = obj.read_region((0, 0), (16, 16))
    assert obj2.metadata["cucim"]["channel_names"] == ["R", "G", "B", "A"]

    # Test with a 1-channel image
    img_1ch_array = np.ones((32, 32, 1)).astype(np.uint8)
    file_path_1ch = str(tmpdir.join("small_rgba_1ch.tiff"))
    imwrite(
        file_path_1ch, img_1ch_array, shape=img_1ch_array.shape, tile=(16, 16)
    )

    obj = CuImage(file_path_1ch)
    assert obj.shape == [32, 32, 4]
    assert obj.metadata["cucim"]["channel_names"] == ["R", "G", "B", "A"]
    obj2 = obj.read_region((0, 0), (16, 16))
    assert obj2.metadata["cucim"]["channel_names"] == ["R", "G", "B", "A"]


def test_load_slow_path_warning(tmpdir, capfd):
    """Test showing a warning message when the image is loaded from a slow path.

    - https://github.com/rapidsai/cucim/issues/230
    """
    import re

    import numpy as np
    from tifffile import imwrite

    from cucim import CuImage

    # Test with a 1-channel image
    img_1ch_array = np.ones((32, 32, 1)).astype(np.uint8)
    file_path_1ch = str(tmpdir.join("small_rgba_1ch.tiff"))
    imwrite(file_path_1ch, img_1ch_array, shape=img_1ch_array.shape)

    img = CuImage(file_path_1ch)
    # Expect a warning message (one warning message per an image path)
    img.read_region()
    img.read_region((0, 0), (16, 16))
    img.read_region((0, 0), (16, 16))
    # Expect a warning message (one warning message per an image path)
    img2 = CuImage(file_path_1ch)
    img2.read_region()

    # Capture the warning message
    captured = capfd.readouterr()

    # Check the warning message
    warning_message = re.findall(
        r"\[Warning\] Loading image\('.*'\) with a slow-path", captured.err
    )
    assert len(captured.err) > 0
    assert len(warning_message) == 2
