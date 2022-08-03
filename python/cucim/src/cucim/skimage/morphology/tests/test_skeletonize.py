import cupy as cp
import pytest
from cupy.testing import assert_array_equal
from skimage import data
from skimage.morphology import thin as thin_cpu

from cucim.skimage._shared._warnings import expected_warnings
from cucim.skimage.morphology import medial_axis, thin


class TestThin:
    @property
    def input_image(self):
        """image to test thinning with"""
        ii = cp.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 0, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=cp.uint8,
        )
        return ii

    def test_zeros(self):
        assert cp.all(thin(cp.zeros((10, 10))) == 0)

    def test_iter_1(self):
        result = thin(self.input_image, 1).astype(cp.uint8)
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
            dtype=cp.uint8,
        )
        assert_array_equal(result, expected)

    def test_max_iter_kwarg_deprecation(self):
        result1 = thin(self.input_image, max_num_iter=1).astype(cp.uint8)
        with expected_warnings(["`max_iter` is a deprecated argument name"]):
            result2 = thin(self.input_image, max_iter=1).astype(cp.uint8)
        assert_array_equal(result1, result2)

    def test_noiter(self):
        result = thin(self.input_image).astype(cp.uint8)
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
            dtype=cp.uint8,
        )
        assert_array_equal(result, expected)

    def test_baddim(self):
        for ii in [cp.zeros((3)), cp.zeros((3, 3, 3))]:
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
    def test_00_00_zeros(self):
        """Test skeletonize on an array of all zeros"""
        result = medial_axis(cp.zeros((10, 10), bool))
        assert not cp.any(result)

    def test_00_01_zeros_masked(self):
        """Test skeletonize on an array that is completely masked"""
        result = medial_axis(cp.zeros((10, 10), bool), cp.zeros((10, 10), bool))
        assert not cp.any(result)

    def test_vertical_line(self):
        """Test a thick vertical line, issue #3861"""
        img = cp.zeros((9, 9))
        img[:, 2] = 1
        img[:, 3] = 1
        img[:, 4] = 1

        expected = cp.full(img.shape, False)
        expected[:, 3] = True

        result = medial_axis(img)
        assert_array_equal(result, expected)

    def test_01_01_rectangle(self):
        """Test skeletonize on a rectangle"""
        image = cp.zeros((9, 15), bool)
        image[1:-1, 1:-1] = True
        #
        # The result should be four diagonals from the
        # corners, meeting in a horizontal line
        #
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

    def test_01_02_hole(self):
        """Test skeletonize on a rectangle with a hole in the middle"""
        image = cp.zeros((9, 15), bool)
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
        """Test skeletonize on a 1-pixel thin strip"""
        image = cp.zeros((1, 5), bool)
        image[:, 1:-1] = True
        result = medial_axis(image)
        assert cp.all(result == image)
