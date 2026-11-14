# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""test_watershed_skimage.py - tests the watershed function

Tests adapted from scikit-image's test_watershed.py. Original test names
are preserved in docstrings for traceability.
"""

import math
import warnings

import cupy as cp
import numpy as np
import pytest
from cupyx.scipy import ndimage as ndi

import cucim.skimage.measure
from cucim.skimage._shared.filters import gaussian
from cucim.skimage.feature import peak_local_max
from cucim.skimage.measure import label
from cucim.skimage.segmentation import watershed
from cucim.skimage.segmentation._watershed_ca import _get_neighbor_offsets

# fmt: off
blob = cp.array([[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],  # noqa: E501
                 [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],  # noqa: E501
                 [255, 255, 255, 255, 255, 204, 204, 204, 204, 204, 204, 255, 255, 255, 255, 255],  # noqa: E501
                 [255, 255, 255, 204, 204, 183, 153, 153, 153, 153, 183, 204, 204, 255, 255, 255],  # noqa: E501
                 [255, 255, 204, 183, 153, 141, 111, 103, 103, 111, 141, 153, 183, 204, 255, 255],  # noqa: E501
                 [255, 255, 204, 153, 111,  94,  72,  52,  52,  72,  94, 111, 153, 204, 255, 255],  # noqa: E501
                 [255, 255, 204, 153, 111,  72,  39,   1,   1,  39,  72, 111, 153, 204, 255, 255],  # noqa: E501
                 [255, 255, 204, 183, 141, 111,  72,  39,  39,  72, 111, 141, 183, 204, 255, 255],  # noqa: E501
                 [255, 255, 255, 204, 183, 141, 111,  72,  72, 111, 141, 183, 204, 255, 255, 255],  # noqa: E501
                 [255, 255, 255, 255, 204, 183, 141,  94,  94, 141, 183, 204, 255, 255, 255, 255],  # noqa: E501
                 [255, 255, 255, 255, 255, 204, 153, 103, 103, 153, 204, 255, 255, 255, 255, 255],  # noqa: E501
                 [255, 255, 255, 255, 204, 183, 141,  94,  94, 141, 183, 204, 255, 255, 255, 255],  # noqa: E501
                 [255, 255, 255, 204, 183, 141, 111,  72,  72, 111, 141, 183, 204, 255, 255, 255],  # noqa: E501
                 [255, 255, 204, 183, 141, 111,  72,  39,  39,  72, 111, 141, 183, 204, 255, 255],  # noqa: E501
                 [255, 255, 204, 153, 111,  72,  39,   1,   1,  39,  72, 111, 153, 204, 255, 255],  # noqa: E501
                 [255, 255, 204, 153, 111,  94,  72,  52,  52,  72,  94, 111, 153, 204, 255, 255],  # noqa: E501
                 [255, 255, 204, 183, 153, 141, 111, 103, 103, 111, 141, 153, 183, 204, 255, 255],  # noqa: E501
                 [255, 255, 255, 204, 204, 183, 153, 153, 153, 153, 183, 204, 204, 255, 255, 255],  # noqa: E501
                 [255, 255, 255, 255, 255, 204, 204, 204, 204, 204, 204, 255, 255, 255, 255, 255],  # noqa: E501
                 [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],  # noqa: E501
                 [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]])  # noqa: E501
# fmt: on


def _smooth_image_and_markers(shape, n_markers, seed=0):
    """Smooth random image with single-pixel markers at random locations."""
    rng = np.random.default_rng(seed)
    image = ndi.gaussian_filter(
        cp.asarray(rng.standard_normal(shape)), 2
    ).astype(cp.float32)
    flat = rng.choice(math.prod(shape), n_markers, replace=False)
    markers = cp.zeros(math.prod(shape), dtype=cp.int32)
    markers[cp.asarray(flat)] = cp.arange(1, n_markers + 1, dtype=cp.int32)
    return image, markers.reshape(shape)


# -----------------------------------------------------------------
# Tests adapted from scikit-image's TestWatershed class
# -----------------------------------------------------------------


def test_barrier_with_8conn():
    """skimage: test_watershed01 - barrier region with 8-connectivity."""
    # fmt: off
    data = cp.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 1, 1, 1, 1, 1, 0],
         [0, 1, 0, 0, 0, 1, 0],
         [0, 1, 0, 0, 0, 1, 0],
         [0, 1, 0, 0, 0, 1, 0],
         [0, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]],
        cp.uint8,
    )
    markers = cp.array(
        [[-1, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 1, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0]],
        cp.int8,
    )
    expected = cp.array(
        [[-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1,  1,  1,  1,  1,  1, -1],
         [-1,  1,  1,  1,  1,  1, -1],
         [-1,  1,  1,  1,  1,  1, -1],
         [-1,  1,  1,  1,  1,  1, -1],
         [-1,  1,  1,  1,  1,  1, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1]]
    )
    # fmt: on
    out = watershed(data, markers, connectivity=2)
    cp.testing.assert_array_equal(out, expected)


def test_barrier_with_4conn():
    """skimage: test_watershed02 - barrier with 4-connectivity."""
    # fmt: off
    data = cp.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 1, 1, 1, 1, 1, 0],
         [0, 1, 0, 0, 0, 1, 0],
         [0, 1, 0, 0, 0, 1, 0],
         [0, 1, 0, 0, 0, 1, 0],
         [0, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]],
        cp.uint8,
    )
    markers = cp.array(
        [[-1, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 1, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0]],
        cp.int8,
    )
    expected = cp.array(
        [[-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1,  1,  1,  1, -1, -1],
         [-1,  1,  1,  1,  1,  1, -1],
         [-1,  1,  1,  1,  1,  1, -1],
         [-1,  1,  1,  1,  1,  1, -1],
         [-1, -1,  1,  1,  1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1]]
    )
    # fmt: on
    out = watershed(data, markers)
    cp.testing.assert_array_equal(out, expected)


def test_two_basins_with_barrier_4conn():
    """skimage: test_watershed03 - two basins separated by barrier, 4-conn.

    The expected array has 0 at barrier pixels. scikit-image's priority
    queue leaves these as 0 (unlabeled), but the CA algorithm labels them.
    We only check non-barrier pixels match exactly.
    """
    # fmt: off
    data = cp.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 1, 1, 1, 1, 1, 0],
         [0, 1, 0, 1, 0, 1, 0],
         [0, 1, 0, 1, 0, 1, 0],
         [0, 1, 0, 1, 0, 1, 0],
         [0, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]],
        cp.uint8,
    )
    markers = cp.array(
        [[0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 2, 0, 3, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0, -1]],
        cp.int8,
    )
    expected = cp.array(
        [[-1, -1, -1, -1, -1, -1, -1],
         [-1,  0,  2,  0,  3,  0, -1],
         [-1,  2,  2,  0,  3,  3, -1],
         [-1,  2,  2,  0,  3,  3, -1],
         [-1,  2,  2,  0,  3,  3, -1],
         [-1,  0,  2,  0,  3,  0, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1]]
    )
    # fmt: on
    out = watershed(data, markers)
    # CA algorithm labels barrier pixels (expected==0) rather than leaving
    # them as 0. Only check non-barrier pixels.
    non_barrier = expected != 0
    cp.testing.assert_array_equal(out[non_barrier], expected[non_barrier])


def test_two_basins_with_barrier_8conn():
    """skimage: test_watershed04 - two basins separated by barrier, 8-conn.

    See test_two_basins_with_barrier_4conn for barrier pixel note.
    """
    # fmt: off
    data = cp.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 1, 1, 1, 1, 1, 0],
         [0, 1, 0, 1, 0, 1, 0],
         [0, 1, 0, 1, 0, 1, 0],
         [0, 1, 0, 1, 0, 1, 0],
         [0, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]],
        cp.uint8,
    )
    markers = cp.array(
        [[0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 2, 0, 3, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0, -1]],
        cp.int8,
    )
    expected = cp.array(
        [[-1, -1, -1, -1, -1, -1, -1],
         [-1,  2,  2,  0,  3,  3, -1],
         [-1,  2,  2,  0,  3,  3, -1],
         [-1,  2,  2,  0,  3,  3, -1],
         [-1,  2,  2,  0,  3,  3, -1],
         [-1,  2,  2,  0,  3,  3, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1]]
    )
    # fmt: on
    out = watershed(data, markers, connectivity=2)
    non_barrier = expected != 0
    cp.testing.assert_array_equal(out[non_barrier], expected[non_barrier])


def test_two_basins_swapped_labels():
    """skimage: test_watershed05 - same as 04 with swapped label values.

    See test_two_basins_with_barrier_4conn for barrier pixel note.
    """
    # fmt: off
    data = cp.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 1, 1, 1, 1, 1, 0],
         [0, 1, 0, 1, 0, 1, 0],
         [0, 1, 0, 1, 0, 1, 0],
         [0, 1, 0, 1, 0, 1, 0],
         [0, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]],
        cp.uint8,
    )
    markers = cp.array(
        [[0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 3, 0, 2, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0,  0],
         [0, 0, 0, 0, 0, 0, -1]],
        cp.int8,
    )
    expected = cp.array(
        [[-1, -1, -1, -1, -1, -1, -1],
         [-1,  3,  3,  0,  2,  2, -1],
         [-1,  3,  3,  0,  2,  2, -1],
         [-1,  3,  3,  0,  2,  2, -1],
         [-1,  3,  3,  0,  2,  2, -1],
         [-1,  3,  3,  0,  2,  2, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1]]
    )
    # fmt: on
    out = watershed(data, markers, connectivity=2)
    non_barrier = expected != 0
    cp.testing.assert_array_equal(out[non_barrier], expected[non_barrier])


def test_u_shaped_barrier():
    """skimage: test_watershed06 - U-shaped barrier region."""
    # fmt: off
    data = cp.array(
        [[0, 1, 0, 0, 0, 1, 0],
         [0, 1, 0, 0, 0, 1, 0],
         [0, 1, 0, 0, 0, 1, 0],
         [0, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]],
        cp.uint8,
    )
    markers = cp.array(
        [[ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 1, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0],
         [-1, 0, 0, 0, 0, 0, 0]],
        cp.int8,
    )
    expected = cp.array(
        [[-1,  1,  1,  1,  1,  1, -1],
         [-1,  1,  1,  1,  1,  1, -1],
         [-1,  1,  1,  1,  1,  1, -1],
         [-1,  1,  1,  1,  1,  1, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1]]
    )
    # fmt: on
    out = watershed(data, markers, connectivity=2)
    cp.testing.assert_array_equal(out, expected)


def test_competitive_blobs():
    """skimage: test_watershed07 - competitive case with two blobs."""
    data = blob
    mask = data != 255
    markers = cp.zeros(data.shape, int)
    markers[6, 7] = 1
    markers[14, 7] = 2
    out = watershed(data, markers, connectivity=2, mask=mask)
    size1 = int(cp.sum(out == 1))
    size2 = int(cp.sum(out == 2))
    assert abs(size1 - size2) <= 6


def test_competitive_blobs_equal_border():
    """skimage: test_watershed08 - border pixels + edge same value."""
    data = blob.copy()
    data[10, 7:9] = 141
    mask = data != 255
    markers = cp.zeros(data.shape, int)
    markers[6, 7] = 1
    markers[14, 7] = 2
    out = watershed(data, markers, connectivity=2, mask=mask)
    size1 = int(cp.sum(out == 1))
    size2 = int(cp.sum(out == 2))
    assert abs(size1 - size2) <= 6


def test_large_image():
    """skimage: test_watershed09 - reasonable size image for timing/memory."""
    image = cp.zeros((1000, 1000))
    coords = cp.random.uniform(0, 1000, (100, 2)).astype(int)
    markers = cp.zeros((1000, 1000), int)
    idx = 1
    for x, y in coords:
        image[x, y] = 1
        markers[x, y] = idx
        idx += 1
    image = gaussian(image, sigma=4, mode="reflect")
    watershed(image, markers, connectivity=2)


def test_plateau_four_markers():
    """skimage: test_watershed10 - four markers on uniform image."""
    # fmt: off
    data = cp.array(
        [[1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1]],
        cp.uint8,
    )
    markers = cp.array(
        [[1, 0, 0, 2],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [3, 0, 0, 4]],
        cp.int8,
    )
    expected = cp.array(
        [[1, 1, 2, 2],
         [1, 1, 2, 2],
         [3, 3, 4, 4],
         [3, 3, 4, 4]]
    )
    # fmt: on
    out = watershed(data, markers, connectivity=2)
    cp.testing.assert_array_equal(out, expected)


def test_plateau_closest_seed():
    """skimage: test_watershed11 - points assigned to closest seed on plateau.

    https://github.com/scikit-image/scikit-image/issues/803
    """
    image = cp.zeros((21, 21))
    markers = cp.zeros((21, 21), int)
    markers[5, 5] = 1
    markers[5, 10] = 2
    markers[10, 5] = 3
    markers[10, 10] = 4

    i, j = cp.mgrid[0:21, 0:21]
    d = cp.dstack(
        [
            cp.sqrt((i.astype(float) - i0) ** 2 + (j.astype(float) - j0) ** 2)
            for i0, j0 in ((5, 5), (5, 10), (10, 5), (10, 10))
        ]
    )
    dmin = cp.min(d, 2)

    # Steps within a flooding level are counted breadth-first, so on a
    # plateau every pixel is assigned to a closest seed.
    for use_block_async in (False, True):
        out = watershed(
            image, markers, connectivity=1, use_block_async=use_block_async
        )
        assert cp.all(d[i, j, out[i, j] - 1] == dmin)


def test_watershed_line_areas():
    """skimage: test_watershed12 - watershed line boundary areas."""
    # fmt: off
    data = cp.array(
        [[203, 255, 203, 153, 153, 153, 153, 153, 153, 153, 153, 153, 153, 153, 153, 153],  # noqa: E501
         [203, 255, 203, 153, 153, 153, 102, 102, 102, 102, 102, 102, 153, 153, 153, 153],  # noqa: E501
         [203, 255, 203, 203, 153, 153, 102, 102,  77,   0, 102, 102, 153, 153, 203, 203],  # noqa: E501
         [203, 255, 255, 203, 153, 153, 153, 102, 102, 102, 102, 153, 153, 203, 203, 255],  # noqa: E501
         [203, 203, 255, 203, 203, 203, 153, 153, 153, 153, 153, 153, 203, 203, 255, 255],  # noqa: E501
         [153, 203, 255, 255, 255, 203, 203, 203, 203, 203, 203, 203, 203, 255, 255, 203],  # noqa: E501
         [153, 203, 203, 203, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 203, 203],  # noqa: E501
         [153, 153, 153, 203, 203, 203, 203, 203, 255, 203, 203, 203, 203, 203, 203, 153],  # noqa: E501
         [102, 102, 153, 153, 153, 153, 203, 203, 255, 203, 203, 255, 203, 153, 153, 153],  # noqa: E501
         [102, 102, 102, 102, 102, 153, 203, 255, 255, 203, 203, 203, 203, 153, 102, 153],  # noqa: E501
         [102,  51,  51, 102, 102, 153, 203, 255, 203, 203, 153, 153, 153, 153, 102, 153],  # noqa: E501
         [ 77,  51,  51, 102, 153, 153, 203, 255, 203, 203, 203, 153, 102, 102, 102, 153],  # noqa: E501
         [ 77,   0,  51, 102, 153, 203, 203, 255, 203, 255, 203, 153, 102,  51, 102, 153],  # noqa: E501
         [ 77,   0,  51, 102, 153, 203, 255, 255, 203, 203, 203, 153, 102,   0, 102, 153],  # noqa: E501
         [102,   0,  51, 102, 153, 203, 255, 203, 203, 153, 153, 153, 102, 102, 102, 153],  # noqa: E501
         [102, 102, 102, 102, 153, 203, 255, 203, 153, 153, 153, 153, 153, 153, 153, 153]]  # noqa: E501
    )
    # fmt: on

    markerbin = data == 0
    marker = label(markerbin)
    ws = watershed(data, marker, connectivity=2, watershed_line=True)
    for lab, area in zip(range(4), [34, 74, 74, 74]):
        assert int(cp.sum(ws == lab)) == area


@pytest.mark.parametrize(
    "watershed_kwargs",
    [
        pytest.param({"use_block_async": False}, id="global"),
        pytest.param({"compactness": 0.1}, id="compact"),
        pytest.param({"use_block_async": True}, id="block-async"),
    ],
)
def test_watershed_line_preserves_adjacent_markers(watershed_kwargs):
    """Adjacent marker pixels must remain fixed instead of becoming lines."""
    image = cp.zeros((32, 32), dtype=cp.float32)
    markers = cp.zeros(image.shape, dtype=cp.int32)
    markers[16, 15] = 1
    markers[16, 16] = 2

    result = watershed(
        image,
        markers,
        watershed_line=True,
        **watershed_kwargs,
    )

    cp.testing.assert_array_equal(result[16, 15:17], cp.array([1, 2]))


def test_input_not_modified():
    """skimage: test_watershed_input_not_modified."""
    image = cp.random.default_rng().random(size=(21, 21))
    markers = cp.zeros((21, 21), dtype=cp.uint8)
    markers[[5, 5, 15, 15], [5, 15, 5, 15]] = [1, 2, 3, 4]
    original_markers = cp.copy(markers)
    result = watershed(image, markers)
    cp.testing.assert_array_equal(original_markers, markers)
    assert not cp.all(result == markers)


# -----------------------------------------------------------------
# Compact watershed tests
# -----------------------------------------------------------------


def test_compact_watershed():
    """skimage: test_compact_watershed."""
    image = cp.zeros((5, 6))
    image[:, 3] = 2  # watershed line
    image[:, 4:] = 1
    seeds = cp.zeros((5, 6), dtype=int)
    seeds[2, 0] = 1
    seeds[2, 5] = 2
    compact = watershed(image, seeds, compactness=0.01)
    expected = cp.array(
        [
            [1, 1, 1, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
        ],
        dtype=int,
    )
    cp.testing.assert_array_equal(compact, expected)

    normal = watershed(image, seeds)
    expected_skimage = cp.array(
        [
            [1, 1, 1, 1, 2, 2],
            [1, 1, 1, 1, 2, 2],
            [1, 1, 1, 1, 2, 2],
            [1, 1, 1, 1, 2, 2],
            [1, 1, 1, 1, 2, 2],
        ],
        dtype=int,
    )
    cp.testing.assert_array_equal(normal, expected_skimage)


def test_compact_watershed_uses_marker_image_value_as_initial_cost():
    """A high-cost marker must not flood as though its initial cost were 0."""
    image = cp.array([10.0, 0.0, 0.0, 0.0])
    markers = cp.array([1, 0, 0, 2], dtype=cp.int32)

    result = watershed(image, markers, compactness=0.1)

    expected = cp.array([1, 2, 2, 2], dtype=cp.int32)
    cp.testing.assert_array_equal(result, expected)


# -----------------------------------------------------------------
# Edge case / overspill tests
# -----------------------------------------------------------------


def test_watershed_with_markers_offset():
    """skimage: test_watershed_with_markers_offset (gh-6632 / gh-7661)."""
    x, y = cp.indices((80, 80))
    x1, y1, x2, y2 = 28, 28, 44, 52
    r1, r2 = 16, 20
    mask_circle1 = (x - x1) ** 2 + (y - y1) ** 2 < r1**2
    mask_circle2 = (x - x2) ** 2 + (y - y2) ** 2 < r2**2
    image = cp.logical_or(mask_circle1, mask_circle2)
    distance = ndi.distance_transform_edt(image)
    coords = peak_local_max(distance, footprint=cp.ones((3, 3)), labels=image)
    coords[:, 0] += 6
    mask = cp.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=image)
    props = cucim.skimage.measure.regionprops(labels, intensity_image=-distance)
    assert props[0].extent == 1
    expected_region = cp.arange(start=-10, stop=0, dtype=float).reshape(-1, 1)
    cp.testing.assert_array_equal(props[0].image_intensity, expected_region)
    assert props[0].num_pixels == 10
    assert props[1].num_pixels == 1928


def test_watershed_simple_basin_overspill():
    """skimage: test_watershed_simple_basin_overspill (gh-6632 / gh-7661)."""
    # Scenario 1
    # fmt: off
    image =    cp.array([[6, 5, 4, 3, 0, 3, 0, 1, 2],
                         [6, 5, 4, 3, 0, 3, 0, 1, 2]])
    markers =  cp.array([[0, 1, 0, 0, 0, 0, 0, 2, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    expected = cp.array([[1, 1, 2, 2, 2, 2, 2, 2, 2],
                         [2, 2, 2, 2, 2, 2, 2, 2, 2]])
    # fmt: on
    result = watershed(image, markers=markers)
    cp.testing.assert_array_equal(result, expected)

    # Scenario 2 (1D with mask)
    image = -cp.array([1, 2, 2, 2, 2, 2, 3])
    markers = cp.array([1, 0, 0, 0, 0, 0, 2])
    expected = cp.array([1, 2, 2, 2, 2, 2, 2])
    result = watershed(image, markers=markers, mask=image != 0)
    cp.testing.assert_array_equal(result, expected)


def test_watershed_evenly_distributed_overspill():
    """skimage: test_watershed_evenly_distributed_overspill."""
    # Scenario 1: markers start with the same value
    image =    cp.array([0, 2, 1, 1, 1, 1, 1, 1, 2, 0])  # fmt: skip
    markers =  cp.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 2])  # fmt: skip
    expected = cp.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])  # fmt: skip
    result = watershed(image, markers=markers)
    cp.testing.assert_array_equal(result, expected)

    # Scenario 2: markers start with different values
    image =    cp.array([2, 2, 1, 1, 1, 1, 1, 1, 2, 0])  # fmt: skip
    expected = cp.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])  # fmt: skip
    result = watershed(image, markers=markers)
    cp.testing.assert_array_equal(result, expected)


def test_markers_on_maxima():
    """skimage: test_markers_on_maxima (gh-7661)."""
    image =    cp.array([[0, 1, 2, 3, 4, 5, 4],
                         [0, 1, 2, 3, 4, 4, 4]])  # fmt: skip
    markers =  cp.array([[1, 0, 0, 0, 0, 2, 0],
                         [0, 0, 0, 0, 0, 0, 0]])  # fmt: skip
    expected = cp.array([[1, 1, 1, 1, 1, 2, 1],
                         [1, 1, 1, 1, 1, 1, 1]])  # fmt: skip
    result = watershed(image, markers=markers)
    cp.testing.assert_array_equal(result, expected)


def test_numeric_seed_watershed():
    """skimage: test_numeric_seed_watershed."""
    image = cp.zeros((5, 6))
    image[:, 3:] = 1
    compact = watershed(image, 2, compactness=0.01)
    expected = cp.array(
        [
            [1, 1, 1, 1, 2, 2],
            [1, 1, 1, 1, 2, 2],
            [1, 1, 1, 1, 2, 2],
            [1, 1, 1, 1, 2, 2],
            [1, 1, 1, 1, 2, 2],
        ],
        dtype=cp.int32,
    )
    cp.testing.assert_array_equal(compact, expected)


# -----------------------------------------------------------------
# Dtype, shape, and error handling tests
# -----------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype",
    [
        cp.uint8,
        cp.int8,
        cp.uint16,
        cp.int16,
        cp.uint32,
        cp.int32,
        cp.uint64,
        cp.int64,
    ],
)
def test_watershed_output_dtype(dtype):
    """skimage: test_watershed_output_dtype."""
    image = cp.zeros((100, 100))
    markers = cp.zeros((100, 100), dtype)
    out = watershed(image, markers)
    assert out.dtype == markers.dtype


@pytest.mark.parametrize(
    "dtype",
    [
        cp.float16,
        cp.float32,
        cp.float64,
        cp.complex64,
        cp.complex128,
        cp.bool_,
    ],
)
def test_watershed_rejects_unsupported_marker_dtype(dtype):
    image = cp.zeros((5, 6), dtype=cp.float32)
    markers = cp.zeros(image.shape, dtype=dtype)
    markers[1, 1] = 1

    with pytest.raises(TypeError, match="signed or unsigned integer dtype"):
        watershed(image, markers)


def test_incorrect_markers_shape():
    image = cp.ones((5, 6))
    markers = cp.ones((5, 7), dtype=cp.int32)
    with pytest.raises(ValueError):
        watershed(image, markers)


def test_broadcastable_markers_shape_with_mask_is_rejected():
    image = cp.ones((5, 6))
    markers = cp.ones((1, 6), dtype=cp.int32)
    mask = cp.ones(image.shape, dtype=bool)
    with pytest.raises(ValueError, match="markers shape"):
        watershed(image, markers, mask=mask)


def test_incorrect_mask_shape():
    image = cp.ones((5, 6))
    mask = cp.ones((5, 7))
    with pytest.raises(ValueError):
        watershed(image, markers=4, mask=mask)


def test_offset_not_implemented():
    """A non-default `offset` is accepted positionally (matching the
    scikit-image signature) but raises NotImplementedError."""
    image = cp.zeros((5, 6), dtype=cp.float32)
    markers = cp.zeros((5, 6), dtype=cp.int32)
    markers[1, 1] = 1
    markers[3, 4] = 2
    # offset is the 4th positional argument, as in scikit-image
    with pytest.raises(NotImplementedError):
        watershed(image, markers, 1, (1, 1))
    with pytest.raises(NotImplementedError):
        watershed(image, markers, offset=(1, 1))


@pytest.mark.parametrize(
    "inner_iterations, error",
    [
        pytest.param(0, ValueError, id="zero"),
        pytest.param(-1, ValueError, id="negative"),
        pytest.param(1.5, TypeError, id="fractional"),
        pytest.param("2", TypeError, id="string"),
    ],
)
def test_inner_iterations_invalid(inner_iterations, error):
    """inner_iterations must be a positive integer when provided."""
    image = cp.zeros((32, 32), dtype=cp.float32)
    markers = cp.zeros((32, 32), dtype=cp.int32)
    markers[1, 1] = 1
    markers[30, 30] = 2

    with pytest.raises(error, match="inner_iterations"):
        watershed(
            image,
            markers,
            use_block_async=True,
            inner_iterations=inner_iterations,
        )


@pytest.mark.parametrize("interval", [0, -1])
def test_convergence_check_interval_invalid(interval):
    """convergence_check_interval must be a positive integer."""
    image = cp.zeros((8, 8), dtype=cp.float32)
    markers = cp.zeros((8, 8), dtype=cp.int32)
    markers[1, 1] = 1
    markers[6, 6] = 2
    with pytest.raises(ValueError):
        watershed(image, markers, convergence_check_interval=interval)


@pytest.mark.parametrize("use_block_async", [False, True])
@pytest.mark.parametrize("interval", [1, 4, 64])
def test_convergence_check_interval_matches(use_block_async, interval):
    """Batching convergence checks must not change the result.

    Only the number of host syncs and no-op launches should differ.
    """
    image, markers = _smooth_image_and_markers((64, 64), 8)

    ref = watershed(
        image,
        markers,
        connectivity=1,
        use_block_async=use_block_async,
        convergence_check_interval=1,
    )
    out = watershed(
        image,
        markers,
        connectivity=1,
        use_block_async=use_block_async,
        convergence_check_interval=interval,
    )
    cp.testing.assert_array_equal(out, ref)


def test_convergence_check_interval_matches_compact():
    """Batching must not change the compact watershed result."""
    image = cp.zeros((5, 6))
    image[:, 3] = 2
    image[:, 4:] = 1
    seeds = cp.zeros((5, 6), dtype=int)
    seeds[2, 0] = 1
    seeds[2, 5] = 2
    ref = watershed(
        image, seeds, compactness=0.01, convergence_check_interval=1
    )
    out = watershed(
        image, seeds, compactness=0.01, convergence_check_interval=32
    )
    cp.testing.assert_array_equal(out, ref)


def test_markers_in_mask():
    data = blob
    mask = data != 255
    out = watershed(data, 25, connectivity=2, mask=mask)
    assert cp.all(out[~mask] == 0)


def test_integer_markers_with_empty_mask():
    image = cp.zeros((5, 6), dtype=cp.float32)
    mask = cp.zeros(image.shape, dtype=bool)
    out = watershed(image, 4, mask=mask)
    cp.testing.assert_array_equal(out, cp.zeros(image.shape, dtype=cp.int32))


def test_no_markers():
    data = blob
    mask = data != 255
    out = watershed(data, mask=mask)
    assert cp.max(out) == 2


def test_single_marker():
    """All pixels should be assigned to the only marker."""
    image = cp.random.default_rng(0).random((10, 10)).astype(cp.float32)
    markers = cp.zeros((10, 10), dtype=cp.int32)
    markers[5, 5] = 1
    labels = watershed(image, markers)
    cp.testing.assert_array_equal(labels, cp.ones_like(labels))


@pytest.mark.parametrize(
    "watershed_kwargs",
    [
        pytest.param({"use_block_async": False}, id="global"),
        pytest.param({"compactness": 0.1}, id="compact"),
        pytest.param({"use_block_async": True}, id="block-async"),
    ],
)
def test_reachable_float64_values_overflowing_float32_are_labeled(
    watershed_kwargs,
):
    """First arrival must label pixels even when their priority is infinite."""
    image = cp.full((32, 32), 1e40, dtype=cp.float64)
    markers = cp.zeros(image.shape, dtype=cp.int32)
    markers[0, 0] = 1

    labels = watershed(image, markers, **watershed_kwargs)

    cp.testing.assert_array_equal(labels, cp.ones_like(markers))


def test_empty_markers():
    """All-zero markers should produce all-zero output."""
    image = cp.random.default_rng(0).random((10, 10)).astype(cp.float32)
    markers = cp.zeros((10, 10), dtype=cp.int32)
    labels = watershed(image, markers)
    cp.testing.assert_array_equal(labels, cp.zeros_like(labels))


def test_all_markers():
    """When every pixel is a marker, output should equal input markers."""
    image = cp.random.default_rng(0).random((5, 5)).astype(cp.float32)
    markers = cp.arange(1, 26, dtype=cp.int32).reshape(5, 5)
    labels = watershed(image, markers)
    cp.testing.assert_array_equal(labels, markers)


def test_single_pixel():
    """Watershed on a 1x1 image."""
    labels = watershed(
        cp.array([[5.0]], dtype=cp.float32),
        cp.array([[1]], dtype=cp.int32),
    )
    assert labels.shape == (1, 1)
    assert int(labels[0, 0]) == 1


def test_border_markers():
    """Markers at all four corners should each claim a quadrant."""
    image = cp.random.default_rng(0).random((10, 10)).astype(cp.float32)
    markers = cp.zeros((10, 10), dtype=cp.int32)
    markers[0, 0] = 1
    markers[0, 9] = 2
    markers[9, 0] = 3
    markers[9, 9] = 4
    labels = watershed(image, markers)
    assert int(cp.max(labels)) == 4
    assert labels[0, 0] == 1
    assert labels[0, 9] == 2
    assert labels[9, 0] == 3
    assert labels[9, 9] == 4


def test_negative_markers():
    """Negative marker labels should be preserved."""
    image = cp.random.default_rng(0).random((10, 10)).astype(cp.float32)
    markers = cp.zeros((10, 10), dtype=cp.int32)
    markers[3, 3] = -1
    markers[7, 7] = 1
    labels = watershed(image, markers)
    assert labels[3, 3] == -1
    assert labels[7, 7] == 1
    assert int(cp.any(cp.unique(labels) < 0))
    assert int(cp.any(cp.unique(labels) > 0))


def test_numpy_input_rejected():
    """Watershed should reject NumPy arrays (CuPy arrays required)."""
    import numpy as np

    image_np = np.random.default_rng(0).random((10, 10)).astype(np.float32)
    markers_np = np.zeros((10, 10), dtype=np.int32)
    markers_np[3, 3] = 1
    markers_np[7, 7] = 2
    markers_cp = cp.asarray(markers_np)
    image_cp = cp.asarray(image_np)

    # NumPy image should be rejected
    with pytest.raises(TypeError):
        watershed(image_np, markers_cp)
    # NumPy markers should be rejected
    with pytest.raises(TypeError):
        watershed(image_cp, markers_np)


# -----------------------------------------------------------------
# nD and compact watershed tests
# -----------------------------------------------------------------


@pytest.mark.parametrize(
    "shape,marker_pos",
    [
        ((10,), [(0,), (9,)]),
        ((6, 8), [(0, 0), (5, 7)]),
        ((4, 5, 6), [(0, 0, 0), (3, 4, 5)]),
    ],
)
def test_compact_watershed_nd(shape, marker_pos):
    """Compact watershed should work for 1D, 2D, and 3D."""
    image = cp.zeros(shape, dtype=cp.float32)
    markers = cp.zeros(shape, dtype=cp.int32)
    markers[marker_pos[0]] = 1
    markers[marker_pos[1]] = 2
    result = watershed(image, markers, compactness=0.01)
    assert result.shape == shape
    assert set(cp.unique(result).tolist()) == {1, 2}


def test_watershed_4d():
    """Standard watershed should work for 4D images."""
    image = cp.zeros((4, 5, 6, 7), dtype=cp.float32)
    markers = cp.zeros_like(image, dtype=cp.int32)
    markers[0, 0, 0, 0] = 1
    markers[3, 4, 5, 6] = 2
    result = watershed(image, markers)
    assert result.shape == image.shape
    assert set(cp.unique(result).tolist()) == {1, 2}


# -----------------------------------------------------------------
# Connectivity / advanced feature tests
# -----------------------------------------------------------------


def test_connectivity():
    """skimage: test_connectivity - different connectivity gives different
    segmentation when markers are auto-generated.
    Issue = 5084
    """
    x, y = cp.indices((406, 270))
    x1, y1, x2, y2, x3, y3, x4, y4 = 200, 208, 300, 120, 100, 100, 340, 208
    r1, r2, r3, r4 = 100, 50, 40, 80
    mask_circle1 = (x - x1) ** 2 + (y - y1) ** 2 < r1**2
    mask_circle2 = (x - x2) ** 2 + (y - y2) ** 2 < r2**2
    mask_circle3 = (x - x3) ** 2 + (y - y3) ** 2 < r3**2
    mask_circle4 = (x - x4) ** 2 + (y - y4) ** 2 < r4**2
    image = cp.logical_or(mask_circle1, mask_circle2)
    image = cp.logical_or(image, mask_circle3)
    image = cp.logical_or(image, mask_circle4)

    DummyBT = ndi.distance_transform_edt(image)
    DummyBT_dis = cp.around(DummyBT / 12, decimals=0) * 12
    Img_mask = cp.where(DummyBT_dis == 0, 0, 1)

    labels_c1 = watershed(
        200 - DummyBT_dis, mask=Img_mask, connectivity=1, compactness=0.01
    )
    labels_c2 = watershed(
        200 - DummyBT_dis, mask=Img_mask, connectivity=2, compactness=0.01
    )

    assert cp.unique(labels_c1).shape[0] == 6
    assert cp.unique(labels_c2).shape[0] == 5

    # The CA relaxation differs from scikit-image on large plateau regions.
    # Use 20% tolerance for area checks.
    tol = 0.2

    for lab, area in zip(range(6), [61824, 3653, 20467, 11097, 1301, 11278]):
        assert (abs(int(cp.sum(labels_c1 == lab)) - area) / area) < tol

    for lab, area in zip(range(5), [61824, 3653, 20466, 12386, 11291]):
        assert (abs(int(cp.sum(labels_c2 == lab)) - area) / area) < tol


# -----------------------------------------------------------------
# Neighbor offset tests
# -----------------------------------------------------------------


class TestNeighborOffsets:
    """Tests for _get_neighbor_offsets."""

    def test_1d(self):
        offsets = _get_neighbor_offsets(1, 1)
        assert offsets == [(-1,), (1,)]

    def test_2d_connectivity1(self):
        offsets = _get_neighbor_offsets(2, 1)
        assert len(offsets) == 4
        assert set(offsets) == {(-1, 0), (1, 0), (0, -1), (0, 1)}

    def test_2d_connectivity2(self):
        offsets = _get_neighbor_offsets(2, 2)
        assert len(offsets) == 8
        for off in offsets[:4]:
            assert sum(c != 0 for c in off) == 1
        for off in offsets[4:]:
            assert sum(c != 0 for c in off) == 2

    def test_3d_connectivity1(self):
        assert len(_get_neighbor_offsets(3, 1)) == 6

    def test_3d_connectivity2(self):
        assert len(_get_neighbor_offsets(3, 2)) == 18

    def test_3d_connectivity3(self):
        assert len(_get_neighbor_offsets(3, 3)) == 26

    @pytest.mark.parametrize("ndim", [4, 5, 6])
    def test_nd_connectivity1(self, ndim):
        offsets = _get_neighbor_offsets(ndim, 1)
        assert len(offsets) == 2 * ndim
        for off in offsets:
            assert len(off) == ndim
            assert sum(c != 0 for c in off) == 1

    @pytest.mark.parametrize("ndim", [4, 5, 6])
    def test_nd_full_connectivity(self, ndim):
        assert len(_get_neighbor_offsets(ndim, ndim)) == 3**ndim - 1

    def test_offsets_sorted_by_connectivity_level(self):
        for ndim in range(1, 5):
            offsets = _get_neighbor_offsets(ndim, ndim)
            levels = [sum(c != 0 for c in off) for off in offsets]
            assert levels == sorted(levels)

    def test_no_origin(self):
        for ndim in range(1, 5):
            offsets = _get_neighbor_offsets(ndim, ndim)
            assert (0,) * ndim not in offsets


# -----------------------------------------------------------------
# Global vs. block-asynchronous kernels
# -----------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, connectivity",
    [
        ((96, 80), 1),
        ((96, 80), 2),
        ((5, 200), 1),
        ((3, 3), 2),
        ((24, 20, 28), 1),
        ((24, 20, 28), 2),
        ((24, 20, 28), 3),
        ((2, 40, 3), 1),
    ],
)
def test_kernels_identical(shape, connectivity):
    """All update orders converge to the same unique fixed point."""
    image, markers = _smooth_image_and_markers(
        shape, max(2, math.prod(shape) // 50)
    )
    ref = watershed(image, markers, connectivity, use_block_async=False)
    for kwargs in (
        {"use_block_async": True},
        {"use_block_async": True, "inner_iterations": 1},
        {"use_block_async": False, "convergence_check_interval": 1},
    ):
        out = watershed(image, markers, connectivity, **kwargs)
        cp.testing.assert_array_equal(out, ref)


@pytest.mark.parametrize(
    "shape, connectivity",
    [((128, 96), 1), ((128, 96), 2), ((32, 28, 24), 1), ((32, 28, 24), 3)],
)
@pytest.mark.parametrize("use_block_async", [False, True])
def test_regions_are_connected(shape, connectivity, use_block_async):
    """Each region is connected to (and contains) its single-pixel marker."""
    image, markers = _smooth_image_and_markers(shape, 40)
    out = watershed(
        image, markers, connectivity, use_block_async=use_block_async
    )
    footprint = ndi.generate_binary_structure(len(shape), connectivity)
    components, n_components = ndi.label(out > 0, structure=footprint)
    # Relabel each (label, component) pair; connected regions give one
    # component per label.
    pairs = cp.unique(out.astype(cp.int64) * (n_components + 1) + components)
    assert pairs.size == int(markers.max())


@pytest.mark.parametrize(
    "shape, connectivity",
    [((128, 96), 1), ((128, 96), 2), ((32, 28, 24), 1), ((32, 28, 24), 3)],
)
def test_matches_skimage_on_smooth_image(shape, connectivity):
    """Without plateaus, the result matches scikit-image exactly."""
    from skimage.segmentation import watershed as watershed_cpu

    image, markers = _smooth_image_and_markers(shape, 40)
    expected = watershed_cpu(
        cp.asnumpy(image), cp.asnumpy(markers), connectivity=connectivity
    )
    out = watershed(image, markers, connectivity)
    cp.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize(
    "shape, connectivity",
    [((128, 96), 1), ((128, 96), 2), ((32, 28, 24), 1), ((32, 28, 24), 3)],
)
@pytest.mark.parametrize("compactness", [0.1, 1.0])
def test_compact_close_to_skimage_on_smooth_image(
    shape, connectivity, compactness
):
    """Compact watershed differs from scikit-image only at a few pixels."""
    from skimage.segmentation import watershed as watershed_cpu

    image, markers = _smooth_image_and_markers(shape, 40)
    expected = watershed_cpu(
        cp.asnumpy(image),
        cp.asnumpy(markers),
        connectivity=connectivity,
        compactness=compactness,
    )
    out = watershed(image, markers, connectivity, compactness=compactness)
    mismatch = float(cp.mean(out != cp.asarray(expected)))
    assert mismatch < 0.005


@pytest.mark.parametrize("use_block_async", [None, False])
def test_long_parent_chains(use_block_async):
    """Parent chains much longer than one resolve launch's step budget are
    still resolved to the right marker (flat image, markers at the ends)."""
    from skimage.segmentation import watershed as watershed_cpu

    image = cp.zeros(5001, dtype=cp.float32)
    markers = cp.zeros(image.shape, dtype=cp.int32)
    markers[0] = 1
    markers[-1] = 2
    out = watershed(image, markers, use_block_async=use_block_async)
    expected = watershed_cpu(cp.asnumpy(image), cp.asnumpy(markers))
    cp.testing.assert_array_equal(out, expected)
    assert int(cp.sum(out == 1)) == 2501

    # A single marker in a corner gives chains as long as the image diagonal.
    image = cp.zeros((300, 200), dtype=cp.float32)
    markers = cp.zeros(image.shape, dtype=cp.int32)
    markers[0, 0] = 7
    out = watershed(image, markers, use_block_async=use_block_async)
    cp.testing.assert_array_equal(out, cp.full(image.shape, 7, dtype=cp.int32))


def test_large_image_all_labeled():
    """A larger image converges with every pixel labeled."""
    image = cp.zeros((256, 256))
    rng = cp.random.default_rng(42)
    coords = rng.integers(0, 256, (20, 2))
    markers = cp.zeros((256, 256), dtype=cp.int32)
    for i, (x, y) in enumerate(coords):
        image[x, y] = 1
        markers[x, y] = i + 1
    image = gaussian(image, sigma=4, mode="reflect")

    out = watershed(image, markers, connectivity=2)
    assert out.shape == (256, 256)
    assert int(cp.sum(out == 0)) == 0
    assert len(cp.unique(out)) == 20


# use size > 48 for markers, but also test odd sizes
@pytest.mark.parametrize("shape", ((64, 64), (49, 75)))
@pytest.mark.parametrize("use_block_async", [True, False])
def test_plateau_assigns_a_closest_marker(shape, use_block_async):
    """Both kernels assign each pixel to a closest marker on a flat image."""
    image = cp.zeros(shape)
    markers = cp.zeros(shape, dtype=cp.int32)
    markers[16, 16] = 1
    markers[16, 48] = 2
    markers[48, 16] = 3
    markers[48, 48] = 4

    out = watershed(
        image,
        markers,
        connectivity=1,
        use_block_async=use_block_async,
    )
    # d[:,:,k] = Euclidean distance from each pixel to marker k+1.
    # dmin = minimum distance to any marker at each pixel.
    # d[i, j, out[i,j]-1] = distance to the *assigned* marker.
    i, j = cp.mgrid[0 : shape[0], 0 : shape[1]]
    d = cp.dstack(
        [
            cp.sqrt((i.astype(float) - i0) ** 2 + (j.astype(float) - j0) ** 2)
            for i0, j0 in ((16, 16), (16, 48), (48, 16), (48, 48))
        ]
    )
    dmin = cp.min(d, 2)
    assert cp.all(d[i, j, out[i, j] - 1] == dmin)


@pytest.mark.parametrize(
    "kwargs, shape, match",
    [
        ({"compactness": 0.1}, (16, 16), "compactness"),
        ({}, (4, 4, 4, 4), "at most 3 dimensions"),
    ],
)
def test_block_async_unsupported_warns(kwargs, shape, match):
    """Forcing use_block_async=True where it does not apply warns and falls
    back to the global kernel."""
    image = cp.zeros(shape, dtype=cp.float32)
    markers = cp.zeros(shape, dtype=cp.int32)
    markers[(0,) * len(shape)] = 1
    markers[(-1,) * len(shape)] = 2
    with pytest.warns(UserWarning, match=match):
        out = watershed(image, markers, use_block_async=True, **kwargs)
    assert set(cp.unique(out).tolist()) == {1, 2}


@pytest.mark.parametrize(
    "shape, squeezed",
    [
        ((300,), (1, 300)),
        ((1, 40, 1, 30), (40, 30)),
        ((40, 1, 30), (40, 30)),
        ((1, 12, 10, 1, 8), (12, 10, 8)),
    ],
)
@pytest.mark.parametrize("full_connectivity", [False, True])
@pytest.mark.parametrize("watershed_line", [False, True])
def test_size_one_dimensions_are_dropped(
    shape, squeezed, full_connectivity, watershed_line
):
    """Dimensions of size one have no in-bounds neighbors, so an image with
    them (or a 1D image) gives the labels of the squeezed image, and can use
    the block-asynchronous kernel without a warning."""
    image, markers = _smooth_image_and_markers(squeezed, 6)
    connectivity = len(shape) if full_connectivity else 1
    expected = watershed(
        image,
        markers,
        min(connectivity, len(squeezed)),
        watershed_line=watershed_line,
    )
    for use_block_async in (None, True, False):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = watershed(
                image.reshape(shape),
                markers.reshape(shape),
                connectivity,
                watershed_line=watershed_line,
                use_block_async=use_block_async,
            )
        assert out.shape == shape
        cp.testing.assert_array_equal(out.reshape(squeezed), expected)


@pytest.mark.parametrize("use_block_async", [True, False])
def test_block_async_watershed_line(use_block_async):
    """Watershed line post-processing should work with both kernels."""
    image = cp.zeros((64, 64), dtype=cp.float32)
    image[:, 32] = 1.0  # vertical barrier
    markers = cp.zeros((64, 64), dtype=cp.int32)
    markers[32, 10] = 1
    markers[32, 50] = 2
    out = watershed(
        image,
        markers,
        connectivity=1,
        watershed_line=True,
        use_block_async=use_block_async,
    )
    # The barrier column is reached last from either side and becomes the
    # line.
    cp.testing.assert_array_equal(out[:, 32], 0)
    assert cp.all(out[:, :32] == 1)
    assert cp.all(out[:, 33:] == 2)


@pytest.mark.parametrize("use_block_async", [True, False])
def test_block_async_3d(use_block_async):
    """3D block-async should produce valid segmentation."""
    rng = cp.random.default_rng(42)
    image = rng.random((16, 16, 16)).astype(cp.float32)
    markers = cp.zeros_like(image, dtype=cp.int32)
    markers[4, 4, 4] = 1
    markers[4, 12, 4] = 2
    markers[12, 4, 12] = 3
    markers[12, 12, 12] = 4
    out = watershed(
        image,
        markers,
        use_block_async=use_block_async,
    )
    assert out.shape == (16, 16, 16)
    assert set(cp.unique(out).tolist()) == {1, 2, 3, 4}


@pytest.mark.parametrize("use_block_async", [True, False])
def test_block_async_3d_with_mask(use_block_async):
    """3D block-async with mask should only label masked pixels."""
    image = cp.zeros((16, 16, 16), dtype=cp.float32)
    markers = cp.zeros_like(image, dtype=cp.int32)
    markers[4, 4, 4] = 1
    markers[12, 12, 12] = 2
    mask = cp.zeros_like(image, dtype=bool)
    mask[2:14, 2:14, 2:14] = True
    out = watershed(
        image,
        markers,
        mask=mask,
        use_block_async=use_block_async,
    )
    assert cp.all(out[~mask] == 0)
    assert set(cp.unique(out[mask]).tolist()) == {1, 2}


def test_block_async_3d_vs_global_realistic():
    """Block-async and global 3D watershed give identical results on a
    realistic image generated from binary blobs with distance-transform
    markers."""
    from cucim.skimage.data import binary_blobs

    shape = (32, 32, 32)
    blobs = binary_blobs(max(shape), n_dim=3, rng=5)
    blobs = blobs[tuple(slice(s) for s in shape)]

    distance = ndi.distance_transform_edt(blobs)
    image = (cp.max(distance) - distance).astype(cp.float32)

    coords = peak_local_max(
        distance,
        min_distance=3,
        labels=blobs,
    )
    markers = cp.zeros(shape, dtype=cp.int32)
    markers[tuple(coords.T)] = cp.arange(1, len(coords) + 1)
    markers, _ = ndi.label(markers > 0)
    mask = blobs.astype(bool)

    n_markers = int(markers.max())
    assert n_markers >= 2, "Need at least 2 markers for a meaningful test"

    common = dict(markers=markers, mask=mask)
    out_global = watershed(image, use_block_async=False, **common)
    out_ba = watershed(image, use_block_async=True, **common)
    cp.testing.assert_array_equal(out_ba, out_global)
    assert cp.all(out_global[~mask] == 0)
