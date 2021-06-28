import cupy as cp
import pytest
from cupy.testing import assert_array_equal
from skimage import data
from skimage.morphology import thin as thin_cpu

from cucim.skimage.morphology import thin


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
        assert cp.all(thin(cp.zeros((10, 10))) == 0)

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

    @pytest.mark.parametrize('invert', [False, True])
    def test_compare_skimage(self, invert):
        h = data.horse()
        if invert:
            h = ~h
        result = thin(cp.asarray(h))
        expected = thin_cpu(h)
        assert_array_equal(result, expected)
