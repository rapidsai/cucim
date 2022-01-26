import cupy as cp
import numpy as np
import pytest
from cupy import testing
from cupyx.scipy import ndimage as ndi
from skimage import data

from cucim.skimage import color, morphology
from cucim.skimage._shared._warnings import expected_warnings
from cucim.skimage.morphology import binary, gray
from cucim.skimage.util import img_as_bool

img = color.rgb2gray(cp.array(data.astronaut()))
bw_img = img > 100 / 255.0


def test_non_square_image():
    footprint = morphology.square(3)
    binary_res = binary.binary_erosion(bw_img[:100, :200], footprint)
    gray_res = img_as_bool(gray.erosion(bw_img[:100, :200], footprint))
    testing.assert_array_equal(binary_res, gray_res)


@pytest.mark.parametrize(
    'function',
    ['binary_erosion', 'binary_dilation', 'binary_closing', 'binary_opening']
)
def test_selem_kwarg_deprecation(function):
    with expected_warnings(["`selem` is a deprecated argument name"]):
        getattr(binary, function)(bw_img, selem=morphology.square(3))


def test_binary_erosion():
    footprint = morphology.square(3)
    binary_res = binary.binary_erosion(bw_img, footprint)
    gray_res = img_as_bool(gray.erosion(bw_img, footprint))
    testing.assert_array_equal(binary_res, gray_res)


def test_binary_dilation():
    footprint = morphology.square(3)
    binary_res = binary.binary_dilation(bw_img, footprint)
    gray_res = img_as_bool(gray.dilation(bw_img, footprint))
    testing.assert_array_equal(binary_res, gray_res)


def test_binary_closing():
    footprint = morphology.square(3)
    binary_res = binary.binary_closing(bw_img, footprint)
    gray_res = img_as_bool(gray.closing(bw_img, footprint))
    testing.assert_array_equal(binary_res, gray_res)


def test_binary_opening():
    footprint = morphology.square(3)
    binary_res = binary.binary_opening(bw_img, footprint)
    gray_res = img_as_bool(gray.opening(bw_img, footprint))
    testing.assert_array_equal(binary_res, gray_res)


def test_footprint_overflow():
    footprint = cp.ones((17, 17), dtype=cp.uint8)
    img = cp.zeros((20, 20), dtype=bool)
    img[2:19, 2:19] = True
    binary_res = binary.binary_erosion(img, footprint)
    gray_res = img_as_bool(gray.erosion(img, footprint))
    testing.assert_array_equal(binary_res, gray_res)


def test_out_argument():
    for func in (binary.binary_erosion, binary.binary_dilation):
        footprint = cp.ones((3, 3), dtype=cp.uint8)
        img = cp.ones((10, 10))
        out = cp.zeros_like(img)
        out_saved = out.copy()
        func(img, footprint, out=out)
        assert cp.any(out != out_saved)
        testing.assert_array_equal(out, func(img, footprint))


binary_functions = [binary.binary_erosion, binary.binary_dilation,
                    binary.binary_opening, binary.binary_closing]


@pytest.mark.parametrize("function", binary_functions)
def test_default_footprint(function):
    footprint = morphology.diamond(radius=1)
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
    im_expected = function(image, footprint)
    im_test = function(image)
    testing.assert_array_equal(im_expected, im_test)


def test_3d_fallback_default_footprint():
    # 3x3x3 cube inside a 7x7x7 image:
    image = cp.zeros((7, 7, 7), bool)
    image[2:-2, 2:-2, 2:-2] = 1

    opened = binary.binary_opening(image)

    # expect a "hyper-cross" centered in the 5x5x5:
    image_expected = cp.zeros((7, 7, 7), dtype=bool)
    image_expected[2:5, 2:5, 2:5] = ndi.generate_binary_structure(3, 1)
    testing.assert_array_equal(opened, image_expected)


binary_3d_fallback_functions = [binary.binary_opening, binary.binary_closing]


@pytest.mark.parametrize("function", binary_3d_fallback_functions)
def test_3d_fallback_cube_footprint(function):
    # 3x3x3 cube inside a 7x7x7 image:
    image = cp.zeros((7, 7, 7), bool)
    image[2:-2, 2:-2, 2:-2] = 1

    cube = cp.ones((3, 3, 3), dtype=cp.uint8)

    new_image = function(image, cube)
    testing.assert_array_equal(new_image, image)


def test_2d_ndimage_equivalence():
    image = cp.zeros((9, 9), cp.uint16)
    image[2:-2, 2:-2] = 2 ** 14
    image[3:-3, 3:-3] = 2 ** 15
    image[4, 4] = 2 ** 16 - 1

    bin_opened = binary.binary_opening(image)
    bin_closed = binary.binary_closing(image)

    footprint = ndi.generate_binary_structure(2, 1)
    ndimage_opened = ndi.binary_opening(image, structure=footprint)
    ndimage_closed = ndi.binary_closing(image, structure=footprint)

    testing.assert_array_equal(bin_opened, ndimage_opened)
    testing.assert_array_equal(bin_closed, ndimage_closed)


def test_binary_output_2d():
    image = cp.zeros((9, 9), cp.uint16)
    image[2:-2, 2:-2] = 2 ** 14
    image[3:-3, 3:-3] = 2 ** 15
    image[4, 4] = 2 ** 16 - 1

    bin_opened = binary.binary_opening(image)
    bin_closed = binary.binary_closing(image)

    int_opened = cp.empty_like(image, dtype=cp.uint8)
    int_closed = cp.empty_like(image, dtype=cp.uint8)
    binary.binary_opening(image, out=int_opened)
    binary.binary_closing(image, out=int_closed)

    np.testing.assert_equal(bin_opened.dtype, bool)
    np.testing.assert_equal(bin_closed.dtype, bool)

    np.testing.assert_equal(int_opened.dtype, np.uint8)
    np.testing.assert_equal(int_closed.dtype, np.uint8)


def test_binary_output_3d():
    image = cp.zeros((9, 9, 9), cp.uint16)
    image[2:-2, 2:-2, 2:-2] = 2 ** 14
    image[3:-3, 3:-3, 3:-3] = 2 ** 15
    image[4, 4, 4] = 2 ** 16 - 1

    bin_opened = binary.binary_opening(image)
    bin_closed = binary.binary_closing(image)

    int_opened = cp.empty_like(image, dtype=cp.uint8)
    int_closed = cp.empty_like(image, dtype=cp.uint8)
    binary.binary_opening(image, out=int_opened)
    binary.binary_closing(image, out=int_closed)

    np.testing.assert_equal(bin_opened.dtype, bool)
    np.testing.assert_equal(bin_closed.dtype, bool)

    np.testing.assert_equal(int_opened.dtype, np.uint8)
    np.testing.assert_equal(int_closed.dtype, np.uint8)
