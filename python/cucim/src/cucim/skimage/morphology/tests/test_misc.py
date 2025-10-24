# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import pytest
from cupy.testing import assert_array_equal
from numpy.testing import assert_equal

from cucim.skimage._shared._warnings import expected_warnings
from cucim.skimage.morphology import remove_small_holes, remove_small_objects

# fmt: off
test_image = cp.array([[0, 0, 0, 1, 0],
                       [1, 1, 1, 0, 0],
                       [1, 1, 1, 0, 1]], bool)
# fmt: on


def test_one_connectivity():
    # fmt: off
    expected = cp.array([[0, 0, 0, 0, 0],
                         [1, 1, 1, 0, 0],
                         [1, 1, 1, 0, 0]], bool)
    # fmt: on
    observed = remove_small_objects(test_image, min_size=6)
    assert_array_equal(observed, expected)


def test_two_connectivity():
    # fmt: off
    expected = cp.array([[0, 0, 0, 1, 0],
                         [1, 1, 1, 0, 0],
                         [1, 1, 1, 0, 0]], bool)
    # fmt: on
    observed = remove_small_objects(test_image, min_size=7, connectivity=2)
    assert_array_equal(observed, expected)


def test_in_place():
    image = test_image.copy()
    observed = remove_small_objects(image, min_size=6, out=image)
    assert_equal(
        observed is image,
        True,
        "remove_small_objects in_place argument failed.",
    )


@pytest.mark.parametrize("in_dtype", [bool, int, cp.int32])
@pytest.mark.parametrize("out_dtype", [bool, int, cp.int32])
def test_out(in_dtype, out_dtype):
    image = test_image.astype(in_dtype, copy=True)
    expected_out = cp.empty_like(test_image, dtype=out_dtype)

    if out_dtype != bool:
        # object with only 1 label will warn on non-bool output dtype
        exp_warn = ["Only one label was provided"]
    else:
        exp_warn = []

    with expected_warnings(exp_warn):
        out = remove_small_objects(image, min_size=6, out=expected_out)

    assert out is expected_out


def test_labeled_image():
    # fmt: off
    labeled_image = cp.array([[2, 2, 2, 0, 1],
                              [2, 2, 2, 0, 1],
                              [2, 0, 0, 0, 0],
                              [0, 0, 3, 3, 3]], dtype=int)
    expected = cp.array([[2, 2, 2, 0, 0],
                         [2, 2, 2, 0, 0],
                         [2, 0, 0, 0, 0],
                         [0, 0, 3, 3, 3]], dtype=int)
    # fmt: on
    observed = remove_small_objects(labeled_image, min_size=3)
    assert_array_equal(observed, expected)


def test_uint_image():
    # fmt: off
    labeled_image = cp.array([[2, 2, 2, 0, 1],
                              [2, 2, 2, 0, 1],
                              [2, 0, 0, 0, 0],
                              [0, 0, 3, 3, 3]], dtype=cp.uint8)
    expected = cp.array([[2, 2, 2, 0, 0],
                         [2, 2, 2, 0, 0],
                         [2, 0, 0, 0, 0],
                         [0, 0, 3, 3, 3]], dtype=cp.uint8)
    # fmt: on
    observed = remove_small_objects(labeled_image, min_size=3)
    assert_array_equal(observed, expected)


def test_single_label_warning():
    # fmt: off
    image = cp.array([[0, 0, 0, 1, 0],
                      [1, 1, 1, 0, 0],
                      [1, 1, 1, 0, 0]], int)
    # fmt: on
    with expected_warnings(["use a boolean array?"]):
        remove_small_objects(image, min_size=6)


def test_float_input():
    float_test = cp.random.rand(5, 5)
    with pytest.raises(TypeError):
        remove_small_objects(float_test)


def test_negative_input():
    negative_int = cp.random.randint(-4, -1, size=(5, 5))
    with pytest.raises(ValueError):
        remove_small_objects(negative_int)


# fmt: off
test_holes_image = cp.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                             [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                             [0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
                             [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                             [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                             [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]], bool)
# fmt: on


def test_one_connectivity_holes():
    # fmt: off
    expected = cp.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]], bool)
    # fmt: on
    observed = remove_small_holes(test_holes_image, area_threshold=3)
    assert_array_equal(observed, expected)


def test_two_connectivity_holes():
    # fmt: off
    expected = cp.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                         [0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]], bool)
    # fmt: on
    observed = remove_small_holes(
        test_holes_image, area_threshold=3, connectivity=2
    )
    assert_array_equal(observed, expected)


def test_in_place_holes():
    image = test_holes_image.copy()
    observed = remove_small_holes(image, area_threshold=3, out=image)
    assert_equal(
        observed is image, True, "remove_small_holes in_place argument failed."
    )


def test_out_remove_small_holes():
    image = test_holes_image.copy()
    expected_out = cp.empty_like(image)
    out = remove_small_holes(image, area_threshold=3, out=expected_out)

    assert out is expected_out


def test_non_bool_out():
    image = test_holes_image.copy()
    expected_out = cp.empty_like(image, dtype=int)
    with pytest.raises(TypeError):
        remove_small_holes(image, area_threshold=3, out=expected_out)


def test_labeled_image_holes():
    # fmt: off
    labeled_holes_image = cp.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                                    [0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
                                    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 2, 2, 2],
                                    [0, 0, 0, 0, 0, 0, 0, 2, 0, 2],
                                    [0, 0, 0, 0, 0, 0, 0, 2, 2, 2]],
                                   dtype=int)
    expected = cp.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]], dtype=bool)
    # fmt: on
    with expected_warnings(["returned as a boolean array"]):
        observed = remove_small_holes(labeled_holes_image, area_threshold=3)
    assert_array_equal(observed, expected)


def test_uint_image_holes():
    # fmt: off
    labeled_holes_image = cp.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                                    [0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
                                    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 2, 2, 2],
                                    [0, 0, 0, 0, 0, 0, 0, 2, 0, 2],
                                    [0, 0, 0, 0, 0, 0, 0, 2, 2, 2]],
                                   dtype=cp.uint8)
    expected = cp.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]], dtype=bool)
    # fmt: on
    with expected_warnings(["returned as a boolean array"]):
        observed = remove_small_holes(labeled_holes_image, area_threshold=3)
    assert_array_equal(observed, expected)


def test_label_warning_holes():
    # fmt: off
    labeled_holes_image = cp.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                                    [0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
                                    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 2, 2, 2],
                                    [0, 0, 0, 0, 0, 0, 0, 2, 0, 2],
                                    [0, 0, 0, 0, 0, 0, 0, 2, 2, 2]],
                                   dtype=int)
    # fmt: on
    with expected_warnings(["use a boolean array?"]):
        remove_small_holes(labeled_holes_image, area_threshold=3)
    remove_small_holes(labeled_holes_image.astype(bool), area_threshold=3)


def test_float_input_holes():
    float_test = cp.random.rand(5, 5)
    with pytest.raises(TypeError):
        remove_small_holes(float_test)
