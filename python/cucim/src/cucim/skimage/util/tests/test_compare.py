# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import pytest
from skimage._shared.testing import assert_stacklevel

from cucim.skimage.util.compare import compare_images


def test_compare_images_ValueError_shape():
    img1 = cp.zeros((10, 10), dtype=cp.uint8)
    img2 = cp.zeros((10, 1), dtype=cp.uint8)
    with pytest.raises(ValueError):
        compare_images(img1, img2)


def test_compare_images_ValueError_args():
    a = cp.ones((10, 10)) * 3
    b = cp.zeros((10, 10))
    with pytest.raises(ValueError):
        compare_images(a, b, method="unknown")


def test_compare_images_diff():
    img1 = cp.zeros((10, 10), dtype=cp.uint8)
    img1[3:8, 3:8] = 255
    img2 = cp.zeros_like(img1)
    img2[3:8, 0:8] = 255
    expected_result = cp.zeros_like(img1, dtype=cp.float64)
    expected_result[3:8, 0:3] = 1
    result = compare_images(img1, img2, method="diff")
    cp.testing.assert_array_equal(result, expected_result)


def test_compare_images_replaced_param():
    img1 = cp.zeros((10, 10), dtype=cp.uint8)
    img1[3:8, 3:8] = 255
    img2 = cp.zeros_like(img1)
    img2[3:8, 0:8] = 255
    expected_result = cp.zeros_like(img1, dtype=cp.float64)
    expected_result[3:8, 0:3] = 1

    regex = ".*Please use `image0, image1`.*"
    with pytest.warns(FutureWarning, match=regex) as record:
        result = compare_images(image1=img1, image2=img2)
    assert_stacklevel(record)
    cp.testing.assert_array_equal(result, expected_result)

    with pytest.warns(FutureWarning, match=regex) as record:
        result = compare_images(image0=img1, image2=img2)
    assert_stacklevel(record)
    cp.testing.assert_array_equal(result, expected_result)

    with pytest.warns(FutureWarning, match=regex) as record:
        result = compare_images(img1, image2=img2)
    assert_stacklevel(record)
    cp.testing.assert_array_equal(result, expected_result)

    # Test making "method" keyword-only here as well
    # so whole test can be removed in one go
    regex = ".*Please pass `method=`.*"
    with pytest.warns(FutureWarning, match=regex) as record:
        result = compare_images(img1, img2, "diff")
    assert_stacklevel(record)
    cp.testing.assert_array_equal(result, expected_result)


def test_compare_images_blend():
    img1 = cp.zeros((10, 10), dtype=cp.uint8)
    img1[3:8, 3:8] = 255
    img2 = cp.zeros_like(img1)
    img2[3:8, 0:8] = 255
    expected_result = cp.zeros_like(img1, dtype=cp.float64)
    expected_result[3:8, 3:8] = 1
    expected_result[3:8, 0:3] = 0.5
    result = compare_images(img1, img2, method="blend")
    cp.testing.assert_array_equal(result, expected_result)


def test_compare_images_checkerboard_default():
    img1 = cp.zeros((2**4, 2**4), dtype=cp.uint8)
    img2 = cp.full(img1.shape, fill_value=255, dtype=cp.uint8)
    res = compare_images(img1, img2, method="checkerboard")
    # fmt: off
    exp_row1 = cp.array(
        [0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1.]
    )
    exp_row2 = cp.array(
        [1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0.]
    )
    # fmt: on
    for i in (0, 1, 4, 5, 8, 9, 12, 13):
        cp.testing.assert_array_equal(res[i, :], exp_row1)
    for i in (2, 3, 6, 7, 10, 11, 14, 15):
        cp.testing.assert_array_equal(res[i, :], exp_row2)


def test_compare_images_checkerboard_tuple():
    img1 = cp.zeros((2**4, 2**4), dtype=cp.uint8)
    img2 = cp.full(img1.shape, fill_value=255, dtype=cp.uint8)
    res = compare_images(img1, img2, method="checkerboard", n_tiles=(4, 8))
    exp_row1 = cp.array(
        [
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            1.0,
            1.0,
        ]
    )
    exp_row2 = cp.array(
        [
            1.0,
            1.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
        ]
    )
    for i in (0, 1, 2, 3, 8, 9, 10, 11):
        cp.testing.assert_array_equal(res[i, :], exp_row1)
    for i in (4, 5, 6, 7, 12, 13, 14, 15):
        cp.testing.assert_array_equal(res[i, :], exp_row2)
