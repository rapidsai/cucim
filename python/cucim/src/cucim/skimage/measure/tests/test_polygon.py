# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import pytest
from cupy.testing import assert_array_equal
from numpy.testing import assert_equal

from cucim.skimage.measure import approximate_polygon, subdivide_polygon
from cucim.skimage.measure._polygon import _SUBDIVISION_MASKS

_square = cp.array(
    [
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 3],
        [2, 3],
        [3, 3],
        [3, 2],
        [3, 1],
        [3, 0],
        [2, 0],
        [1, 0],
        [0, 0],
    ]
)


@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
def test_approximate_polygon(dtype):
    square = _square.astype(dtype, copy=False)
    out = approximate_polygon(square, 0.1)
    assert out.dtype == dtype
    assert_array_equal(out, square[cp.asarray((0, 3, 6, 9, 12)), :])

    out = approximate_polygon(square, 2.2)
    assert_array_equal(out, square[cp.asarray((0, 6, 12)), :])

    out = approximate_polygon(
        square[cp.asarray((0, 1, 3, 4, 5, 6, 7, 9, 11, 12)), :], 0.1
    )
    assert_array_equal(out, square[cp.asarray((0, 3, 6, 9, 12)), :])

    out = approximate_polygon(square, -1)
    assert_array_equal(out, square)
    out = approximate_polygon(square, 0)
    assert_array_equal(out, square)


@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
def test_subdivide_polygon(dtype):
    square = _square.astype(dtype, copy=False)
    new_square1 = square
    new_square2 = square[:-1]
    new_square3 = square[:-1]
    # test iterative subdvision
    for _ in range(10):
        square1, square2, square3 = new_square1, new_square2, new_square3
        # test different B-Spline degrees
        for degree in range(1, 7):
            mask_len = len(_SUBDIVISION_MASKS[degree][0])
            # test circular
            new_square1 = subdivide_polygon(square1, degree)
            assert new_square1.dtype == dtype
            assert_array_equal(new_square1[-1], new_square1[0])
            assert_equal(new_square1.shape[0], 2 * square1.shape[0] - 1)
            # test non-circular
            new_square2 = subdivide_polygon(square2, degree)
            assert new_square3.dtype == dtype
            assert_equal(
                new_square2.shape[0], 2 * (square2.shape[0] - mask_len + 1)
            )
            # test non-circular, preserve_ends
            new_square3 = subdivide_polygon(square3, degree, True)
            assert new_square3.dtype == dtype
            assert_array_equal(new_square3[0], square3[0])
            assert_array_equal(new_square3[-1], square3[-1])

            assert_equal(
                new_square3.shape[0], 2 * (square3.shape[0] - mask_len + 2)
            )

    # not supported B-Spline degree
    with pytest.raises(ValueError):
        subdivide_polygon(square, 0)
    with pytest.raises(ValueError):
        subdivide_polygon(square, 8)
