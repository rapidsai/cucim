import cupy as cp
import numpy as np
from cupyx.scipy import ndimage as ndi
from skimage import data

from cucim.skimage import color, transform
from cucim.skimage._shared._warnings import expected_warnings
from cucim.skimage._shared.testing import TestCase, fetch, parametrize
from cucim.skimage.morphology import grey, selem
from cucim.skimage.util import img_as_ubyte, img_as_uint


class TestMorphology(TestCase):

    # These expected outputs were generated with skimage v0.12.1
    # using:
    #
    #   from skimage.morphology.tests.test_grey import TestMorphology
    #   import numpy as np
    #   output = TestMorphology()._build_expected_output()
    #   np.savez_compressed('gray_morph_output.npz', **output)

    def _build_expected_output(self):
        funcs = (grey.erosion, grey.dilation, grey.opening, grey.closing,
                 grey.white_tophat, grey.black_tophat)
        selems_2D = (selem.square, selem.diamond,
                     selem.disk, selem.star)

        image = img_as_ubyte(transform.downscale_local_mean(
            color.rgb2gray(cp.array(data.coffee())), (20, 20)))

        output = {}
        for n in range(1, 4):
            for strel in selems_2D:
                for func in funcs:
                    key = '{0}_{1}_{2}'.format(
                        strel.__name__, n, func.__name__)
                    output[key] = func(image, strel(n))

        return output

    def test_gray_morphology(self):
        expected = dict(np.load(fetch('data/gray_morph_output.npz')))
        calculated = self._build_expected_output()
        for k, v in calculated.items():
            cp.testing.assert_array_equal(cp.asarray(expected[k]), v)


class TestEccentricStructuringElements(TestCase):
    def setUp(self):
        self.black_pixel = 255 * cp.ones((4, 4), dtype=cp.uint8)
        self.black_pixel[1, 1] = 0
        self.white_pixel = 255 - self.black_pixel
        self.selems = [
            selem.square(2),
            selem.rectangle(2, 2),
            selem.rectangle(2, 1),
            selem.rectangle(1, 2),
        ]

    def test_dilate_erode_symmetry(self):
        for s in self.selems:
            c = grey.erosion(self.black_pixel, s)
            d = grey.dilation(self.white_pixel, s)
            assert cp.all(c == (255 - d))

    def test_open_black_pixel(self):
        for s in self.selems:
            grey_open = grey.opening(self.black_pixel, s)
            assert cp.all(grey_open == self.black_pixel)

    def test_close_white_pixel(self):
        for s in self.selems:
            grey_close = grey.closing(self.white_pixel, s)
            assert cp.all(grey_close == self.white_pixel)

    def test_open_white_pixel(self):
        for s in self.selems:
            assert cp.all(grey.opening(self.white_pixel, s) == 0)

    def test_close_black_pixel(self):
        for s in self.selems:
            assert cp.all(grey.closing(self.black_pixel, s) == 255)

    def test_white_tophat_white_pixel(self):
        for s in self.selems:
            tophat = grey.white_tophat(self.white_pixel, s)
            cp.testing.assert_array_equal(tophat, self.white_pixel)

    def test_black_tophat_black_pixel(self):
        for s in self.selems:
            tophat = grey.black_tophat(self.black_pixel, s)
            cp.testing.assert_array_equal(tophat, 255 - self.black_pixel)

    def test_white_tophat_black_pixel(self):
        for s in self.selems:
            tophat = grey.white_tophat(self.black_pixel, s)
            assert cp.all(tophat == 0)

    def test_black_tophat_white_pixel(self):
        for s in self.selems:
            tophat = grey.black_tophat(self.white_pixel, s)
            assert cp.all(tophat == 0)


grey_functions = [grey.erosion, grey.dilation,
                  grey.opening, grey.closing,
                  grey.white_tophat, grey.black_tophat]


@parametrize("function", grey_functions)
def test_default_selem(function):
    strel = selem.diamond(radius=1)
    # fmt: off
    image = cp.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
                      [0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
                      [0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], cp.uint8)
    # fmt: on
    im_expected = function(image, strel)
    im_test = function(image)
    cp.testing.assert_array_equal(im_expected, im_test)


def test_3d_fallback_default_selem():
    # 3x3x3 cube inside a 7x7x7 image:
    image = cp.zeros((7, 7, 7), bool)
    image[2:-2, 2:-2, 2:-2] = 1

    opened = grey.opening(image)

    # expect a "hyper-cross" centered in the 5x5x5:
    image_expected = cp.zeros((7, 7, 7), dtype=bool)
    image_expected[2:5, 2:5, 2:5] = ndi.generate_binary_structure(3, 1)
    cp.testing.assert_array_equal(opened, image_expected)


grey_3d_fallback_functions = [grey.closing, grey.opening]


@parametrize("function", grey_3d_fallback_functions)
def test_3d_fallback_cube_selem(function):
    # 3x3x3 cube inside a 7x7x7 image:
    image = cp.zeros((7, 7, 7), bool)
    image[2:-2, 2:-2, 2:-2] = 1

    cube = cp.ones((3, 3, 3), dtype=cp.uint8)

    new_image = function(image, cube)
    cp.testing.assert_array_equal(new_image, image)


def test_3d_fallback_white_tophat():
    image = cp.zeros((7, 7, 7), dtype=bool)
    image[2, 2:4, 2:4] = 1
    image[3, 2:5, 2:5] = 1
    image[4, 3:5, 3:5] = 1

    with expected_warnings([r'operator.*deprecated|\A\Z']):
        new_image = grey.white_tophat(image)
    footprint = ndi.generate_binary_structure(3, 1)
    with expected_warnings([r'operator.*deprecated|\A\Z']):
        image_expected = ndi.white_tophat(
            image.view(dtype=cp.uint8), footprint=footprint
        )
    cp.testing.assert_array_equal(new_image, image_expected)


def test_3d_fallback_black_tophat():
    image = cp.ones((7, 7, 7), dtype=bool)
    image[2, 2:4, 2:4] = 0
    image[3, 2:5, 2:5] = 0
    image[4, 3:5, 3:5] = 0

    with expected_warnings([r'operator.*deprecated|\A\Z']):
        new_image = grey.black_tophat(image)
    footprint = ndi.generate_binary_structure(3, 1)
    with expected_warnings([r'operator.*deprecated|\A\Z']):
        image_expected = ndi.black_tophat(
            image.view(dtype=cp.uint8), footprint=footprint
        )
    cp.testing.assert_array_equal(new_image, image_expected)


def test_2d_ndimage_equivalence():
    image = cp.zeros((9, 9), cp.uint8)
    image[2:-2, 2:-2] = 128
    image[3:-3, 3:-3] = 196
    image[4, 4] = 255

    opened = grey.opening(image)
    closed = grey.closing(image)

    selem = ndi.generate_binary_structure(2, 1)
    ndimage_opened = ndi.grey_opening(image, footprint=selem)
    ndimage_closed = ndi.grey_closing(image, footprint=selem)

    cp.testing.assert_array_equal(opened, ndimage_opened)
    cp.testing.assert_array_equal(closed, ndimage_closed)


# float test images
# fmt: off
im = cp.array([[0.55, 0.72, 0.6 , 0.54, 0.42],   # noqa
               [0.65, 0.44, 0.89, 0.96, 0.38],
               [0.79, 0.53, 0.57, 0.93, 0.07],
               [0.09, 0.02, 0.83, 0.78, 0.87],
               [0.98, 0.8 , 0.46, 0.78, 0.12]])  # noqa

eroded = cp.array([[0.55, 0.44, 0.54, 0.42, 0.38],
                   [0.44, 0.44, 0.44, 0.38, 0.07],
                   [0.09, 0.02, 0.53, 0.07, 0.07],
                   [0.02, 0.02, 0.02, 0.78, 0.07],
                   [0.09, 0.02, 0.46, 0.12, 0.12]])

dilated = cp.array([[0.72, 0.72, 0.89, 0.96, 0.54],
                    [0.79, 0.89, 0.96, 0.96, 0.96],
                    [0.79, 0.79, 0.93, 0.96, 0.93],
                    [0.98, 0.83, 0.83, 0.93, 0.87],
                    [0.98, 0.98, 0.83, 0.78, 0.87]])

opened = cp.array([[0.55, 0.55, 0.54, 0.54, 0.42],
                   [0.55, 0.44, 0.54, 0.44, 0.38],
                   [0.44, 0.53, 0.53, 0.78, 0.07],
                   [0.09, 0.02, 0.78, 0.78, 0.78],
                   [0.09, 0.46, 0.46, 0.78, 0.12]])

closed = cp.array([[0.72, 0.72, 0.72, 0.54, 0.54],
                   [0.72, 0.72, 0.89, 0.96, 0.54],
                   [0.79, 0.79, 0.79, 0.93, 0.87],
                   [0.79, 0.79, 0.83, 0.78, 0.87],
                   [0.98, 0.83, 0.78, 0.78, 0.78]])
# fmt: on


def test_float():
    cp.testing.assert_allclose(grey.erosion(im), eroded)
    cp.testing.assert_allclose(grey.dilation(im), dilated)
    cp.testing.assert_allclose(grey.opening(im), opened)
    cp.testing.assert_allclose(grey.closing(im), closed)


def test_uint16():
    im16, eroded16, dilated16, opened16, closed16 = map(
        img_as_uint, [im, eroded, dilated, opened, closed]
    )
    cp.testing.assert_allclose(grey.erosion(im16), eroded16)
    cp.testing.assert_allclose(grey.dilation(im16), dilated16)
    cp.testing.assert_allclose(grey.opening(im16), opened16)
    cp.testing.assert_allclose(grey.closing(im16), closed16)


def test_discontiguous_out_array():
    # fmt: off
    image = cp.array([[5, 6, 2],
                      [7, 2, 2],
                      [3, 5, 1]], cp.uint8)
    # fmt: on
    out_array_big = cp.zeros((5, 5), cp.uint8)
    out_array = out_array_big[::2, ::2]
    # fmt: off
    expected_dilation = cp.array([[7, 0, 6, 0, 6],
                                  [0, 0, 0, 0, 0],
                                  [7, 0, 7, 0, 2],
                                  [0, 0, 0, 0, 0],
                                  [7, 0, 5, 0, 5]], cp.uint8)
    expected_erosion = cp.array([[5, 0, 2, 0, 2],
                                 [0, 0, 0, 0, 0],
                                 [2, 0, 2, 0, 1],
                                 [0, 0, 0, 0, 0],
                                 [3, 0, 1, 0, 1]], cp.uint8)
    # fmt: on
    grey.dilation(image, out=out_array)
    cp.testing.assert_array_equal(out_array_big, expected_dilation)
    grey.erosion(image, out=out_array)
    cp.testing.assert_array_equal(out_array_big, expected_erosion)


def test_1d_erosion():
    image = cp.array([1, 2, 3, 2, 1])
    expected = cp.array([1, 1, 2, 1, 1])
    eroded = grey.erosion(image)
    cp.testing.assert_array_equal(eroded, expected)
