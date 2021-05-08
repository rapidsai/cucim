import cupy as cp
import numpy as np
import pytest
from cupy.testing import assert_array_equal

from cucim.skimage.morphology import thin
from cucim.skimage.morphology._skeletonize import (_G123_LUT, _G123P_LUT,
                                                   _generate_thin_luts)


class TestThin():
    @property
    def input_image(self):
        """image to test thinning with"""
        ii = cp.array([[0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 1, 0, 1, 1, 1, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0]], dtype=cp.uint8)
        return ii

    def test_zeros(self):
        assert cp.all(thin(cp.zeros((10, 10))) == False)

    def test_iter_1(self):
        result = thin(self.input_image, 1).astype(cp.uint8)
        expected = cp.array([[0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0],
                             [0, 1, 0, 1, 1, 0, 0],
                             [0, 0, 1, 1, 1, 0, 0],
                             [0, 0, 1, 1, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0]], dtype=cp.uint8)
        assert_array_equal(result, expected)

    def test_noiter(self):
        result = thin(self.input_image).astype(cp.uint8)
        expected = cp.array([[0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0],
                             [0, 1, 0, 1, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0]], dtype=cp.uint8)
        assert_array_equal(result, expected)

    def test_baddim(self):
        for ii in [cp.zeros((3)), cp.zeros((3, 3, 3))]:
            with pytest.raises(ValueError):
                thin(ii)

    def test_lut_generation(self):
        g123, g123p = _generate_thin_luts()

        assert_array_equal(cp.asarray(g123), _G123_LUT)
        assert_array_equal(cp.asarray(g123p), _G123P_LUT)
