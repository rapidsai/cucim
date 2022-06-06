import cupy as cp
import numpy as np
import pytest
from cupy import testing
from cupyx.scipy import ndimage as ndi
from skimage import data

from cucim.skimage import color, morphology
from cucim.skimage._shared._warnings import expected_warnings
from cucim.skimage.util import img_as_bool

img = color.rgb2gray(cp.array(data.astronaut()))
bw_img = img > 100 / 255.0


def test_non_square_image():
    footprint = morphology.square(3)
    binary_res = morphology.binary_erosion(bw_img[:100, :200], footprint)
    gray_res = img_as_bool(morphology.erosion(bw_img[:100, :200], footprint))
    testing.assert_array_equal(binary_res, gray_res)


@pytest.mark.parametrize(
    'function',
    ['binary_erosion', 'binary_dilation', 'binary_closing', 'binary_opening']
)
def test_selem_kwarg_deprecation(function):
    with expected_warnings(["`selem` is a deprecated argument name"]):
        getattr(morphology, function)(bw_img, selem=morphology.square(3))


def test_binary_erosion():
    footprint = morphology.square(3)
    binary_res = morphology.binary_erosion(bw_img, footprint)
    gray_res = img_as_bool(morphology.erosion(bw_img, footprint))
    testing.assert_array_equal(binary_res, gray_res)


def test_binary_dilation():
    footprint = morphology.square(3)
    binary_res = morphology.binary_dilation(bw_img, footprint)
    gray_res = img_as_bool(morphology.dilation(bw_img, footprint))
    testing.assert_array_equal(binary_res, gray_res)


def test_binary_closing():
    footprint = morphology.square(3)
    binary_res = morphology.binary_closing(bw_img, footprint)
    gray_res = img_as_bool(morphology.closing(bw_img, footprint))
    testing.assert_array_equal(binary_res, gray_res)


def test_binary_opening():
    footprint = morphology.square(3)
    binary_res = morphology.binary_opening(bw_img, footprint)
    gray_res = img_as_bool(morphology.opening(bw_img, footprint))
    testing.assert_array_equal(binary_res, gray_res)


def _get_decomp_test_data(function, ndim=2):
    if function == 'binary_erosion':
        img = cp.ones((17, ) * ndim, dtype=cp.uint8)
        img[8, 8] = 0
    elif function == 'binary_dilation':
        img = cp.zeros((17, ) * ndim, dtype=cp.uint8)
        img[8, 8] = 1
    else:
        img = cp.asarray(data.binary_blobs(32, n_dim=ndim, seed=1))
    return img


@pytest.mark.parametrize(
    "function",
    ["binary_erosion", "binary_dilation", "binary_closing", "binary_opening"],
)
@pytest.mark.parametrize("size", (3, 4, 11))
@pytest.mark.parametrize("decomposition", ['separable', 'sequence'])
def test_square_decomposition(function, size, decomposition):
    """Validate footprint decomposition for various shapes.

    comparison is made to the case without decomposition.
    """
    footprint_ndarray = morphology.square(size, decomposition=None)
    footprint = morphology.square(size, decomposition=decomposition)
    img = _get_decomp_test_data(function)
    func = getattr(morphology, function)
    expected = func(img, footprint=footprint_ndarray)
    out = func(img, footprint=footprint)
    testing.assert_array_equal(expected, out)


@pytest.mark.parametrize(
    "function",
    ["binary_erosion", "binary_dilation", "binary_closing", "binary_opening"],
)
@pytest.mark.parametrize("nrows", (3, 4, 11))
@pytest.mark.parametrize("ncols", (3, 4, 11))
@pytest.mark.parametrize("decomposition", ['separable', 'sequence'])
def test_rectangle_decomposition(function, nrows, ncols, decomposition):
    """Validate footprint decomposition for various shapes.

    comparison is made to the case without decomposition.
    """
    footprint_ndarray = morphology.rectangle(nrows, ncols, decomposition=None)
    footprint = morphology.rectangle(nrows, ncols, decomposition=decomposition)
    img = _get_decomp_test_data(function)
    func = getattr(morphology, function)
    expected = func(img, footprint=footprint_ndarray)
    out = func(img, footprint=footprint)
    testing.assert_array_equal(expected, out)


@pytest.mark.parametrize(
    "function",
    ["binary_erosion", "binary_dilation", "binary_closing", "binary_opening"],
)
@pytest.mark.parametrize("m", (0, 1, 2, 3, 4, 5))
@pytest.mark.parametrize("n", (0, 1, 2, 3, 4, 5))
@pytest.mark.parametrize("decomposition", ['sequence'])
def test_octagon_decomposition(function, m, n, decomposition):
    """Validate footprint decomposition for various shapes.

    comparison is made to the case without decomposition.
    """
    if m == 0 and n == 0:
        with pytest.raises(ValueError):
            morphology.octagon(m, n, decomposition=decomposition)
    else:
        footprint_ndarray = morphology.octagon(m, n, decomposition=None)
        footprint = morphology.octagon(m, n, decomposition=decomposition)
        img = _get_decomp_test_data(function)
        func = getattr(morphology, function)
        expected = func(img, footprint=footprint_ndarray)
        out = func(img, footprint=footprint)
        testing.assert_array_equal(expected, out)


@pytest.mark.parametrize(
    "function",
    ["binary_erosion", "binary_dilation", "binary_closing", "binary_opening"],
)
@pytest.mark.parametrize("radius", (1, 2, 5))
@pytest.mark.parametrize("decomposition", ['sequence'])
def test_diamond_decomposition(function, radius, decomposition):
    """Validate footprint decomposition for various shapes.

    comparison is made to the case without decomposition.
    """
    footprint_ndarray = morphology.diamond(radius, decomposition=None)
    footprint = morphology.diamond(radius, decomposition=decomposition)
    img = _get_decomp_test_data(function)
    func = getattr(morphology, function)
    expected = func(img, footprint=footprint_ndarray)
    out = func(img, footprint=footprint)
    testing.assert_array_equal(expected, out)


@pytest.mark.parametrize(
    "function",
    ["binary_erosion", "binary_dilation", "binary_closing", "binary_opening"],
)
@pytest.mark.parametrize("size", (3, 4, 5))
@pytest.mark.parametrize("decomposition", ['separable', 'sequence'])
def test_cube_decomposition(function, size, decomposition):
    """Validate footprint decomposition for various shapes.

    comparison is made to the case without decomposition.
    """
    footprint_ndarray = morphology.cube(size, decomposition=None)
    footprint = morphology.cube(size, decomposition=decomposition)
    img = _get_decomp_test_data(function, ndim=3)
    func = getattr(morphology, function)
    expected = func(img, footprint=footprint_ndarray)
    out = func(img, footprint=footprint)
    testing.assert_array_equal(expected, out)


@pytest.mark.parametrize(
    "function",
    ["binary_erosion", "binary_dilation", "binary_closing", "binary_opening"],
)
@pytest.mark.parametrize("radius", (1, 2, 3))
@pytest.mark.parametrize("decomposition", ['sequence'])
def test_octahedron_decomposition(function, radius, decomposition):
    """Validate footprint decomposition for various shapes.

    comparison is made to the case without decomposition.
    """
    footprint_ndarray = morphology.octahedron(radius, decomposition=None)
    footprint = morphology.octahedron(radius, decomposition=decomposition)
    img = _get_decomp_test_data(function, ndim=3)
    func = getattr(morphology, function)
    expected = func(img, footprint=footprint_ndarray)
    out = func(img, footprint=footprint)
    testing.assert_array_equal(expected, out)


def test_footprint_overflow():
    footprint = cp.ones((17, 17), dtype=cp.uint8)
    img = cp.zeros((20, 20), dtype=bool)
    img[2:19, 2:19] = True
    binary_res = morphology.binary_erosion(img, footprint)
    gray_res = img_as_bool(morphology.erosion(img, footprint))
    testing.assert_array_equal(binary_res, gray_res)


def test_out_argument():
    for func in (morphology.binary_erosion, morphology.binary_dilation):
        footprint = cp.ones((3, 3), dtype=cp.uint8)
        img = cp.ones((10, 10))
        out = cp.zeros_like(img)
        out_saved = out.copy()
        func(img, footprint, out=out)
        assert cp.any(out != out_saved)
        testing.assert_array_equal(out, func(img, footprint))


binary_functions = [morphology.binary_erosion, morphology.binary_dilation,
                    morphology.binary_opening, morphology.binary_closing]


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

    opened = morphology.binary_opening(image)

    # expect a "hyper-cross" centered in the 5x5x5:
    image_expected = cp.zeros((7, 7, 7), dtype=bool)
    image_expected[2:5, 2:5, 2:5] = ndi.generate_binary_structure(3, 1)
    testing.assert_array_equal(opened, image_expected)


binary_3d_fallback_functions = [
    morphology.binary_opening, morphology.binary_closing
]


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

    bin_opened = morphology.binary_opening(image)
    bin_closed = morphology.binary_closing(image)

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

    bin_opened = morphology.binary_opening(image)
    bin_closed = morphology.binary_closing(image)

    int_opened = cp.empty_like(image, dtype=cp.uint8)
    int_closed = cp.empty_like(image, dtype=cp.uint8)
    morphology.binary_opening(image, out=int_opened)
    morphology.binary_closing(image, out=int_closed)

    np.testing.assert_equal(bin_opened.dtype, bool)
    np.testing.assert_equal(bin_closed.dtype, bool)

    np.testing.assert_equal(int_opened.dtype, np.uint8)
    np.testing.assert_equal(int_closed.dtype, np.uint8)


def test_binary_output_3d():
    image = cp.zeros((9, 9, 9), cp.uint16)
    image[2:-2, 2:-2, 2:-2] = 2 ** 14
    image[3:-3, 3:-3, 3:-3] = 2 ** 15
    image[4, 4, 4] = 2 ** 16 - 1

    bin_opened = morphology.binary_opening(image)
    bin_closed = morphology.binary_closing(image)

    int_opened = cp.empty_like(image, dtype=cp.uint8)
    int_closed = cp.empty_like(image, dtype=cp.uint8)
    morphology.binary_opening(image, out=int_opened)
    morphology.binary_closing(image, out=int_closed)

    np.testing.assert_equal(bin_opened.dtype, bool)
    np.testing.assert_equal(bin_closed.dtype, bool)

    np.testing.assert_equal(int_opened.dtype, np.uint8)
    np.testing.assert_equal(int_closed.dtype, np.uint8)
