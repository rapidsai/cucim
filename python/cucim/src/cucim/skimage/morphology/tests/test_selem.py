"""
Tests for Morphological structuring elements
(skimage.morphology.selem)

Author: Damian Eads
"""


import cupy as cp
import numpy as np
from cupy.testing import assert_array_equal

from cucim.skimage._shared.testing import fetch
from cucim.skimage.morphology import selem


class TestSElem:
    def test_square_selem(self):
        """Test square structuring elements"""
        for k in range(0, 5):
            actual_mask = selem.square(k)
            expected_mask = cp.ones((k, k), dtype='uint8')
            assert_array_equal(expected_mask, actual_mask)

    def test_rectangle_selem(self):
        """Test rectangle structuring elements"""
        for i in range(0, 5):
            for j in range(0, 5):
                actual_mask = selem.rectangle(i, j)
                expected_mask = cp.ones((i, j), dtype='uint8')
                assert_array_equal(expected_mask, actual_mask)

    def test_cube_selem(self):
        """Test cube structuring elements"""
        for k in range(0, 5):
            actual_mask = selem.cube(k)
            expected_mask = cp.ones((k, k, k), dtype='uint8')
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

    def test_selem_disk(self):
        """Test disk structuring elements"""
        self.strel_worker("data/disk-matlab-output.npz", selem.disk)

    def test_selem_diamond(self):
        """Test diamond structuring elements"""
        self.strel_worker("data/diamond-matlab-output.npz", selem.diamond)

    def test_selem_ball(self):
        """Test ball structuring elements"""
        self.strel_worker_3d("data/disk-matlab-output.npz", selem.ball)

    def test_selem_octahedron(self):
        """Test octahedron structuring elements"""
        self.strel_worker_3d("data/diamond-matlab-output.npz",
                             selem.octahedron)

    def test_selem_octagon(self):
        """Test octagon structuring elements"""
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
        actual_mask1 = selem.octagon(5, 3)
        expected_mask2 = cp.array([[0, 1, 0],
                                   [1, 1, 1],
                                   [0, 1, 0]], dtype=cp.uint8)

        # fmt: on
        actual_mask2 = selem.octagon(1, 1)
        assert_array_equal(expected_mask1, actual_mask1)
        assert_array_equal(expected_mask2, actual_mask2)

    def test_selem_ellipse(self):
        """Test ellipse structuring elements"""
        # fmt: off
        expected_mask1 = cp.array([[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                   [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]],
                                  dtype=cp.uint8)
        actual_mask1 = selem.ellipse(5, 3)
        expected_mask2 = cp.array([[1, 1, 1],
                                   [1, 1, 1],
                                   [1, 1, 1]], dtype=cp.uint8)
        # fmt: on
        actual_mask2 = selem.ellipse(1, 1)
        assert_array_equal(expected_mask1, actual_mask1)
        assert_array_equal(expected_mask2, actual_mask2)
        assert_array_equal(expected_mask1, selem.ellipse(3, 5).T)
        assert_array_equal(expected_mask2, selem.ellipse(1, 1).T)

    def test_selem_star(self):
        """Test star structuring elements"""
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
        actual_mask1 = selem.star(4)
        expected_mask2 = cp.array([[1, 1, 1],
                                   [1, 1, 1],
                                   [1, 1, 1]], dtype=cp.uint8)
        # fmt: on
        actual_mask2 = selem.star(1)
        assert_array_equal(expected_mask1, actual_mask1)
        assert_array_equal(expected_mask2, actual_mask2)
