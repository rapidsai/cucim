import cupy as cp
import numpy as np
import pytest
from cupy.testing import assert_array_equal
from skimage import data
from skimage.morphology import thin as thin_cpu

from cucim.skimage.morphology import medial_axis, thin


class TestThin:
    @property
    def input_image(self):
        # Image to test thinning with
        ii = cp.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 2, 3, 4, 5, 0],
                [0, 1, 0, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 6, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=float,
        )
        return ii

    def test_all_zeros(self):
        image = cp.zeros((10, 10), dtype=bool)
        assert cp.all(thin(image) == 0)

    @pytest.mark.parametrize("dtype", [bool, float, int])
    def test_thin_copies_input(self, dtype):
        """Ensure thinning does not modify the input image."""
        image = self.input_image.astype(dtype)
        original = image.copy()
        thin(image)
        cp.testing.assert_array_equal(image, original)

    @pytest.mark.parametrize("dtype", [bool, float, int])
    def test_iter_1(self, dtype):
        image = self.input_image.astype(dtype)
        result = thin(image, 1).astype(bool)
        expected = cp.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 1, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [bool, float, int])
    def test_noiter(self, dtype):
        image = self.input_image.astype(dtype)
        result = thin(image).astype(bool)
        expected = cp.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        assert_array_equal(result, expected)

    def test_baddim(self):
        for ii in [cp.zeros(3, dtype=bool), cp.zeros((3, 3, 3), dtype=bool)]:
            with pytest.raises(ValueError):
                thin(ii)

    @pytest.mark.parametrize("invert", [False, True])
    def test_compare_skimage(self, invert):
        h = data.horse()
        if invert:
            h = ~h
        result = thin(cp.asarray(h))
        expected = thin_cpu(h)
        assert_array_equal(result, expected)


class TestMedialAxis:
    def test_all_zeros(self):
        result = medial_axis(cp.zeros((10, 10), dtype=bool))
        assert not cp.any(result)

    def test_all_zeros_masked(self):
        result = medial_axis(
            cp.zeros((10, 10), dtype=bool), cp.zeros((10, 10), dtype=bool)
        )
        assert not cp.any(result)

    def _test_vertical_line(self, dtype, **kwargs):
        """Test a thick vertical line, issue #3861"""
        image = cp.zeros((9, 9), dtype=dtype)
        image[:, 2] = 1
        image[:, 3] = 2
        image[:, 4] = 3

        expected = cp.full(image.shape, False)
        expected[:, 3] = True

        result = medial_axis(image, **kwargs)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [bool, float, int])
    def test_vertical_line(self, dtype):
        """Test a thick vertical line, issue #3861"""
        self._test_vertical_line(dtype=dtype)

    def test_rng_numpy(self):
        # NumPy Generator allowed
        self._test_vertical_line(dtype=bool, rng=np.random.default_rng())

    def test_rng_cupy(self):
        # CuPy Generator not currently supported
        with pytest.raises(ValueError):
            self._test_vertical_line(dtype=bool, rng=cp.random.default_rng())

    def test_rng_int(self):
        self._test_vertical_line(dtype=bool, rng=15)

    def test_rectangle(self):
        image = cp.zeros((9, 15), dtype=bool)
        image[1:-1, 1:-1] = True
        # Excepted are four diagonals from the corners, meeting in a horizontal
        # line
        expected = cp.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        result = medial_axis(image)
        assert cp.all(result == expected)
        result, distance = medial_axis(image, return_distance=True)
        assert distance.max() == 4

    def test_rectangle_with_hole(self):
        image = cp.zeros((9, 15), dtype=bool)
        image[1:-1, 1:-1] = True
        image[4, 4:-4] = False
        expected = cp.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        result = medial_axis(image)
        assert cp.all(result == expected)

    def test_narrow_image(self):
        # Image is a 1-pixel thin strip
        image = cp.zeros((1, 5), dtype=bool)
        image[:, 1:-1] = True
        result = medial_axis(image)
        assert cp.all(result == image)
