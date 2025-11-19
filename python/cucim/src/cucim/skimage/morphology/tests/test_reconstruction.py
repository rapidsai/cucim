# SPDX-FileCopyrightText: Copyright (c) 2003-2009 Massachusetts Institute of Technology
# SPDX-FileCopyrightText: Copyright (c) 2009-2011 Broad Institute
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND (GPL-2.0-only OR BSD-3-Clause)

"""
These tests are originally part of CellProfiler, code licensed under both GPL
and BSD licenses.

Website: http://www.cellprofiler.org
Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.
Original author: Lee Kamentsky
"""

import cupy as cp
import numpy as np
import pytest
from cupy.testing import assert_array_almost_equal
from skimage import data
from skimage.morphology import reconstruction as reconstruction_cpu

from cucim.skimage.exposure import rescale_intensity
from cucim.skimage.morphology import reconstruction


def test_zeros():
    """Test reconstruction with image and mask of zeros"""
    assert_array_almost_equal(
        reconstruction(cp.zeros((5, 7)), cp.zeros((5, 7))), 0
    )


def test_image_equals_mask():
    """Test reconstruction where the image and mask are the same"""
    assert_array_almost_equal(
        reconstruction(cp.ones((7, 5)), cp.ones((7, 5))), 1
    )


def test_image_less_than_mask():
    """Test reconstruction where the image is uniform and less than mask"""
    image = cp.ones((5, 5))
    mask = cp.ones((5, 5)) * 2
    assert_array_almost_equal(reconstruction(image, mask), 1)


def test_one_image_peak():
    """Test reconstruction with one peak pixel"""
    image = cp.ones((5, 5))
    image[2, 2] = 2
    mask = cp.ones((5, 5)) * 3
    assert_array_almost_equal(reconstruction(image, mask), 2)


def test_two_image_peaks():
    """Test reconstruction with two peak pixels isolated by the mask"""
    # fmt: off
    image = cp.array([[1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 2, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 3, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1]])

    mask = cp.array([[4, 4, 4, 1, 1, 1, 1, 1],
                     [4, 4, 4, 1, 1, 1, 1, 1],
                     [4, 4, 4, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 4, 4, 4],
                     [1, 1, 1, 1, 1, 4, 4, 4],
                     [1, 1, 1, 1, 1, 4, 4, 4]])

    expected = cp.array([[2, 2, 2, 1, 1, 1, 1, 1],
                         [2, 2, 2, 1, 1, 1, 1, 1],
                         [2, 2, 2, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 3, 3, 3],
                         [1, 1, 1, 1, 1, 3, 3, 3],
                         [1, 1, 1, 1, 1, 3, 3, 3]])
    # fmt: on
    assert_array_almost_equal(reconstruction(image, mask), expected)


def test_zero_image_one_mask():
    """Test reconstruction with an image of all zeros and a mask that's not"""
    result = reconstruction(cp.zeros((10, 10)), cp.ones((10, 10)))
    assert_array_almost_equal(result, 0)


def test_fill_hole():
    """Test reconstruction by erosion, which should fill holes in mask."""
    seed = cp.array([0, 8, 8, 8, 8, 8, 8, 8, 8, 0])
    mask = cp.array([0, 3, 6, 2, 1, 1, 1, 4, 2, 0])
    result = reconstruction(seed, mask, method="erosion")
    assert_array_almost_equal(result, cp.array([0, 3, 6, 4, 4, 4, 4, 4, 2, 0]))


def test_invalid_seed():
    seed = cp.ones((5, 5))
    mask = cp.ones((5, 5))
    with pytest.raises(ValueError):
        reconstruction(seed * 2, mask, method="dilation")
    with pytest.raises(ValueError):
        reconstruction(seed * 0.5, mask, method="erosion")


@pytest.mark.parametrize(
    "unsigned_dtype", [cp.uint8, cp.uint16, cp.uint32, cp.uint64]
)
def test_erosion_uint8_input(unsigned_dtype):
    image = cp.asarray(data.moon())
    # Rescale image intensity so that we can see dim features.
    image = rescale_intensity(image, in_range=(50, 200))

    image2 = cp.copy(image)
    image2[1:-1, 1:-1] = image.max()
    mask = image

    image2 = image2.astype(unsigned_dtype)
    image2_erosion = reconstruction(image2, mask, method="erosion")

    expected_out_type = cp.promote_types(image2.dtype, cp.int8)
    assert image2_erosion.dtype == expected_out_type
    # promoted to signed dtype (or float in the case of cp.uint64)
    assert (
        image2_erosion.dtype.kind == "i" if unsigned_dtype != cp.uint64 else "f"
    )
    assert image2_erosion.dtype.itemsize >= image2.dtype.itemsize

    # compare to scikit-image CPU result
    image2_erosion_cpu = reconstruction_cpu(
        cp.asnumpy(image2), cp.asnumpy(mask), method="erosion"
    )
    # filled_cpu will be np.float64, so convert to the type returned by cuCIM
    cp.testing.assert_allclose(
        image2_erosion, image2_erosion_cpu.astype(image2_erosion.dtype)
    )


def test_invalid_footprint():
    seed = cp.ones((5, 5))
    mask = cp.ones((5, 5))
    with pytest.raises(ValueError):
        reconstruction(seed, mask, footprint=np.ones((4, 4)))
    with pytest.raises(ValueError):
        reconstruction(seed, mask, footprint=np.ones((3, 4)))
    reconstruction(seed, mask, footprint=np.ones((3, 3)))


def test_invalid_method():
    seed = cp.array([0, 8, 8, 8, 8, 8, 8, 8, 8, 0])
    mask = cp.array([0, 3, 6, 2, 1, 1, 1, 4, 2, 0])
    with pytest.raises(ValueError):
        reconstruction(seed, mask, method="foo")


def test_invalid_offset_not_none():
    """Test reconstruction with invalid not None offset parameter"""
    # fmt: off
    image = cp.array([[1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 2, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 3, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1]])

    mask = cp.array([[4, 4, 4, 1, 1, 1, 1, 1],
                     [4, 4, 4, 1, 1, 1, 1, 1],
                     [4, 4, 4, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 4, 4, 4],
                     [1, 1, 1, 1, 1, 4, 4, 4],
                     [1, 1, 1, 1, 1, 4, 4, 4]])
    # fmt: on
    with pytest.raises(ValueError):
        reconstruction(
            image,
            mask,
            method="dilation",
            footprint=cp.ones((3, 3)),
            offset=cp.array([3, 0]),
        )


def test_offset_not_none():
    """Test reconstruction with valid offset parameter"""
    seed = cp.array([0, 3, 6, 2, 1, 1, 1, 4, 2, 0])
    mask = cp.array([0, 8, 6, 8, 8, 8, 8, 4, 4, 0])
    expected = cp.array([0, 3, 6, 6, 6, 6, 6, 4, 4, 0])

    assert_array_almost_equal(
        reconstruction(
            seed,
            mask,
            method="dilation",
            footprint=cp.ones(3),
            offset=cp.array([0]),
        ),
        expected,
    )


def test_reconstruction_float_inputs():
    """Verifies fix for: https://github.com/rapidsai/cuci/issues/36

    Run the 2D example from the reconstruction docstring and compare the output
    to scikit-image.
    """

    y, x = np.mgrid[:20:0.5, :20:0.5]
    bumps = np.sin(x) + np.sin(y)
    h = 0.3
    seed = bumps - h
    background_cpu = reconstruction_cpu(seed, bumps)
    background = reconstruction(cp.asarray(seed), cp.asarray(bumps))
    cp.testing.assert_allclose(background, background_cpu)
