"""
Tests for Morphological footprints
(skimage.morphology.footprint)

Author: Damian Eads
"""


import cupy as cp
import numpy as np
from cupy.testing import assert_array_equal

from cucim.skimage._shared.testing import fetch
from cucim.skimage.morphology import footprints


class TestSElem:
    def test_square_footprint(self):
        """Test square footprints"""
        for k in range(0, 5):
            actual_mask = footprints.square(k)
            expected_mask = np.ones((k, k), dtype='uint8')
            assert_array_equal(expected_mask, actual_mask)

    def test_rectangle_footprint(self):
        """Test rectangle footprints"""
        for i in range(0, 5):
            for j in range(0, 5):
                actual_mask = footprints.rectangle(i, j)
                expected_mask = np.ones((i, j), dtype='uint8')
                assert_array_equal(expected_mask, actual_mask)

    def test_cube_footprint(self):
        """Test cube footprints"""
        for k in range(0, 5):
            actual_mask = footprints.cube(k)
            expected_mask = np.ones((k, k, k), dtype='uint8')
            assert_array_equal(expected_mask, actual_mask)

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
        self.strel_worker_3d("data/diamond-matlab-output.npz",
                             footprints.octahedron)

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
