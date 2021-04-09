import unittest

import cupy as cp
import pytest
from skimage import data

from cucim.skimage.filters import LPIFilter2D, inverse, wiener


class TestLPIFilter2D(unittest.TestCase):
    img = cp.array(data.camera()[:50, :50])

    def filt_func(self, r, c):
        return cp.exp(-cp.hypot(r, c) / 1)

    def setUp(self):
        self.f = LPIFilter2D(self.filt_func)

    def tst_shape(self, x):
        X = self.f(x)
        assert X.shape == x.shape

    def test_ip_shape(self):
        rows, columns = self.img.shape[:2]

        for c_slice in [slice(0, columns), slice(0, columns - 5),
                        slice(0, columns - 20)]:
            yield (self.tst_shape, self.img[:, c_slice])

    def test_inverse(self):
        F = self.f(self.img)
        g = inverse(F, predefined_filter=self.f)
        assert g.shape == self.img.shape

        g1 = inverse(F[::-1, ::-1], predefined_filter=self.f)
        assert (g - g1[::-1, ::-1]).sum() < 55

        # test cache
        g1 = inverse(F[::-1, ::-1], predefined_filter=self.f)
        assert (g - g1[::-1, ::-1]).sum() < 55

        g1 = inverse(F[::-1, ::-1], self.filt_func)
        assert (g - g1[::-1, ::-1]).sum() < 55

    def test_wiener(self):
        F = self.f(self.img)
        g = wiener(F, predefined_filter=self.f)
        assert g.shape == self.img.shape

        g1 = wiener(F[::-1, ::-1], predefined_filter=self.f)
        assert (g - g1[::-1, ::-1]).sum() < 1

        g1 = wiener(F[::-1, ::-1], self.filt_func)
        assert (g - g1[::-1, ::-1]).sum() < 1

    def test_non_callable(self):
        with pytest.raises(ValueError):
            LPIFilter2D(None)
