# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Morphological footprints
(skimage.morphology.footprint)

Author: Damian Eads
"""


import cupy as cp
import numpy as np
import pytest
from cupy.testing import assert_array_equal

from cucim.skimage._shared.testing import assert_stacklevel, fetch
from cucim.skimage.morphology import (
    footprint_from_sequence,
    footprint_rectangle,
    footprints,
)


class TestSElem:
    def strel_worker(self, fn, func):
        matlab_masks = np.load(fetch(fn))
        k = 0
        for arrname in sorted(matlab_masks):
            expected_mask = matlab_masks[arrname]
            actual_mask = func(k)
            if expected_mask.shape == (1,):
                expected_mask = expected_mask[:, np.newaxis]
            assert_array_equal(expected_mask, actual_mask)
            k = k + 1

    def strel_worker_3d(self, fn, func):
        matlab_masks = np.load(fetch(fn))
        k = 0
        for arrname in sorted(matlab_masks):
            expected_mask = matlab_masks[arrname]
            actual_mask = func(k)
            if expected_mask.shape == (1,):
                expected_mask = expected_mask[:, np.newaxis]
            # Test center slice for each dimension. This gives a good
            # indication of validity without the need for a 3D reference
            # mask.
            c = int(expected_mask.shape[0] / 2)
            assert_array_equal(expected_mask, actual_mask[c, :, :])
            assert_array_equal(expected_mask, actual_mask[:, c, :])
            assert_array_equal(expected_mask, actual_mask[:, :, c])
            k = k + 1

    def test_footprint_disk(self):
        """Test disk footprints"""
        self.strel_worker("data/disk-matlab-output.npz", footprints.disk)

    def test_footprint_diamond(self):
        """Test diamond footprints"""
        self.strel_worker("data/diamond-matlab-output.npz", footprints.diamond)

    def test_footprint_ball(self):
        """Test ball footprints"""
        self.strel_worker_3d("data/disk-matlab-output.npz", footprints.ball)

    def test_footprint_octahedron(self):
        """Test octahedron footprints"""
        self.strel_worker_3d(
            "data/diamond-matlab-output.npz", footprints.octahedron
        )

    def test_footprint_octagon(self):
        """Test octagon footprints"""
        # fmt: off
        expected_mask1 = cp.array([[0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                                   [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                   [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                   [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                   [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                   [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]],
                                  dtype=cp.uint8)
        actual_mask1 = footprints.octagon(5, 3)
        expected_mask2 = cp.array([[0, 1, 0],
                                   [1, 1, 1],
                                   [0, 1, 0]], dtype=cp.uint8)

        # fmt: on
        actual_mask2 = footprints.octagon(1, 1)
        assert_array_equal(expected_mask1, actual_mask1)
        assert_array_equal(expected_mask2, actual_mask2)

    def test_footprint_ellipse(self):
        """Test ellipse footprints"""
        # fmt: off
        expected_mask1 = cp.array([[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                   [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]],
                                  dtype=cp.uint8)
        actual_mask1 = footprints.ellipse(5, 3)
        expected_mask2 = cp.array([[1, 1, 1],
                                   [1, 1, 1],
                                   [1, 1, 1]], dtype=cp.uint8)
        # fmt: on
        actual_mask2 = footprints.ellipse(1, 1)
        assert_array_equal(expected_mask1, actual_mask1)
        assert_array_equal(expected_mask2, actual_mask2)
        assert_array_equal(expected_mask1, footprints.ellipse(3, 5).T)
        assert_array_equal(expected_mask2, footprints.ellipse(1, 1).T)

    def test_footprint_star(self):
        """Test star footprints"""
        # fmt: off
        expected_mask1 = cp.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                                   [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                   [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                   [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                   [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                   [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                   [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                   [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                   [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                   [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]],
                                  dtype=cp.uint8)
        actual_mask1 = footprints.star(4)
        expected_mask2 = cp.array([[1, 1, 1],
                                   [1, 1, 1],
                                   [1, 1, 1]], dtype=cp.uint8)
        # fmt: on
        actual_mask2 = footprints.star(1)
        assert_array_equal(expected_mask1, actual_mask1)
        assert_array_equal(expected_mask2, actual_mask2)


@pytest.mark.parametrize(
    "function, args, supports_sequence_decomposition",
    [
        (footprints.disk, (3,), True),
        (footprints.ball, (3,), True),
        (footprints.diamond, (3,), True),
        (footprints.octahedron, (3,), True),
        (footprint_rectangle, ((3, 5),), True),
        (footprints.ellipse, (3, 4), False),
        (footprints.octagon, (3, 4), True),
        (footprints.star, (3,), False),
    ],
)
@pytest.mark.parametrize("dtype", [np.uint8, np.float64])
def test_footprint_dtype(
    function, args, supports_sequence_decomposition, dtype
):
    # make sure footprint dtype matches what was requested
    footprint = function(*args, dtype=dtype)
    assert footprint.dtype == dtype

    if supports_sequence_decomposition:
        sequence = function(*args, dtype=dtype, decomposition="sequence")
        assert all([fp_tuple[0].dtype == dtype for fp_tuple in sequence])


@pytest.mark.parametrize("function", ["disk", "ball"])
@pytest.mark.parametrize(
    "radius", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 75, 100]
)
def test_nsphere_series_approximation(function, radius):
    fp_func = getattr(footprints, function)
    expected = fp_func(radius, strict_radius=False, decomposition=None)
    footprint_sequence = fp_func(
        radius, strict_radius=False, decomposition="sequence"
    )
    approximate = footprint_from_sequence(footprint_sequence)
    assert approximate.shape == expected.shape

    # verify that maximum error does not exceed some fraction of the size
    error = np.sum(np.abs(expected.astype(int) - approximate.astype(int)))
    if radius == 1:
        assert error == 0
    else:
        max_error = 0.1 if function == "disk" else 0.15
        assert error / expected.size <= max_error


@pytest.mark.parametrize("radius", [1, 2, 3, 4, 5, 10, 20, 50, 75])
@pytest.mark.parametrize("strict_radius", [False, True])
def test_disk_crosses_approximation(radius, strict_radius):
    fp_func = footprints.disk
    expected = fp_func(radius, strict_radius=strict_radius, decomposition=None)
    footprint_sequence = fp_func(
        radius, strict_radius=strict_radius, decomposition="crosses"
    )
    approximate = footprint_from_sequence(footprint_sequence)
    assert approximate.shape == expected.shape

    # verify that maximum error does not exceed some fraction of the size
    error = cp.sum(cp.abs(expected.astype(int) - approximate.astype(int)))
    max_error = 0.05
    assert error / expected.size <= max_error


@pytest.mark.parametrize("width", [3, 8, 20, 50])
@pytest.mark.parametrize("height", [3, 8, 20, 50])
def test_ellipse_crosses_approximation(width, height):
    fp_func = footprints.ellipse
    expected = fp_func(width, height, decomposition=None)
    footprint_sequence = fp_func(width, height, decomposition="crosses")
    approximate = footprint_from_sequence(footprint_sequence)
    assert approximate.shape == expected.shape

    # verify that maximum error does not exceed some fraction of the size
    error = cp.sum(cp.abs(expected.astype(int) - approximate.astype(int)))
    max_error = 0.05
    assert error / expected.size <= max_error


def test_disk_series_approximation_unavailable():
    # ValueError if radius is too large (only precomputed up to radius=250)
    with pytest.raises(ValueError):
        footprints.disk(radius=10000, decomposition="sequence")


def test_ball_series_approximation_unavailable():
    # ValueError if radius is too large (only precomputed up to radius=100)
    with pytest.raises(ValueError):
        footprints.ball(radius=10000, decomposition="sequence")


def assert_decomposition_equal(actual, desired):
    assert len(actual) == len(desired)
    for a, d in zip(actual, desired):
        assert_array_equal(a[0], d[0])
        assert a[1] == d[1]


class Test_footprint_rectangle:
    @pytest.mark.parametrize("i", [0, 1, 2, 3, 4])
    @pytest.mark.parametrize("j", [0, 1, 2, 3, 4])
    def test_rectangle(self, i, j):
        desired = cp.ones((i, j), dtype="uint8")
        actual = footprint_rectangle((i, j))
        assert_array_equal(actual, desired)

    @pytest.mark.parametrize("i", [0, 1, 2, 3, 4])
    @pytest.mark.parametrize("j", [0, 1, 2, 3, 4])
    @pytest.mark.parametrize("k", [0, 1, 2, 3, 4])
    def test_cuboid(self, i, j, k):
        desired = cp.ones((i, j, k), dtype="uint8")
        actual = footprint_rectangle((i, j, k))
        assert_array_equal(actual, desired)

    @pytest.mark.parametrize("shape", [(3,), (5, 5), (5, 5, 7)])
    @pytest.mark.parametrize("decomposition", ["separable", "sequence"])
    def test_decomposition(self, shape, decomposition):
        regular = footprint_rectangle(shape)
        decomposed = footprint_rectangle(shape, decomposition=decomposition)
        recomposed = footprint_from_sequence(decomposed)
        assert_array_equal(recomposed, regular)

    @pytest.mark.parametrize("shape", [(2,), (3, 4)])
    def test_uneven_sequence_decomposition_warning(self, shape):
        """Should fall back to decomposition="separable" for uneven footprint
        size.
        """
        desired = footprint_rectangle(shape, decomposition="separable")
        regex = (
            "decomposition='sequence' is only supported for uneven footprints"
        )
        with pytest.warns(UserWarning, match=regex) as record:
            actual = footprint_rectangle(shape, decomposition="sequence")
        assert_stacklevel(record)
        assert_decomposition_equal(actual, desired)
