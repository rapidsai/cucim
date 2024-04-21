import cupy as cp
import numpy as np
import pytest
import skimage
from cupy import testing
from cupyx.scipy import ndimage as ndi
from packaging.version import Version
from skimage import data, morphology as morphology_cpu

from cucim.skimage import color, morphology, transform
from cucim.skimage._shared._warnings import expected_warnings
from cucim.skimage._shared.testing import assert_stacklevel
from cucim.skimage.morphology import gray
from cucim.skimage.util import img_as_ubyte, img_as_uint


@pytest.fixture
def cam_image():
    from skimage import data

    return cp.ascontiguousarray(cp.array(data.camera()[64:112, 64:96]))


@pytest.fixture
def cell3d_image():
    from skimage import data

    return cp.ascontiguousarray(
        cp.array(data.cells3d()[30:48, 0, 20:36, 20:32])
    )


gray_morphology_funcs = (
    "erosion",
    "dilation",
    "opening",
    "closing",
    "white_tophat",
    "black_tophat",
)


class TestMorphology:
    @pytest.mark.parametrize(
        "footprint, footprint_kwargs",
        (
            (morphology.square, dict(decomposition=None)),
            (morphology.diamond, dict(decomposition=None)),
            (morphology.disk, dict(decomposition=None)),
            (morphology.star, {}),
        ),
    )
    @pytest.mark.parametrize("func", gray_morphology_funcs)
    @pytest.mark.parametrize("n", range(1, 5))
    def test_gray_morphology(self, footprint, footprint_kwargs, func, n):
        if (n % 2 == 0) and (Version(skimage.__version__) < Version("0.23.0")):
            # even sized footprint bug fixes require recent skimage
            pytest.skip("direct CPU comparison requires skimage >= 0.23.0")

        image = img_as_ubyte(
            transform.downscale_local_mean(
                color.rgb2gray(cp.array(data.coffee())), (20, 20)
            )
        )
        gpu_fp = footprint(n, **footprint_kwargs)
        if isinstance(gpu_fp, tuple):
            # skimage doesn't support a tuple footprint
            cpu_fp = np.ones(gpu_fp, dtype=np.uint8)
        else:
            cpu_fp = cp.asnumpy(gpu_fp)
        cpu_func = getattr(morphology_cpu, func)
        cpu_result = cpu_func(cp.asnumpy(image), cpu_fp)

        gpu_func = getattr(morphology, func)
        gpu_result = gpu_func(image, gpu_fp)

        cp.testing.assert_array_equal(cpu_result, gpu_result)

    def test_gray_closing_extensive(self):
        img = cp.array(data.coins())
        footprint = cp.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]])

        # Default mode="reflect" is not extensive for backwards-compatibility
        result_default = morphology.closing(img, footprint=footprint)
        assert not cp.all(result_default >= img)

        result = morphology.closing(img, footprint=footprint, mode="ignore")
        assert cp.all(result >= img)

    def test_gray_opening_anti_extensive(self):
        img = cp.array(data.coins())
        footprint = cp.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]])

        # Default mode="reflect" is not extensive for backwards-compatibility
        result_default = morphology.opening(img, footprint=footprint)
        assert not cp.all(result_default <= img)

        result_ignore = morphology.opening(
            img, footprint=footprint, mode="ignore"
        )
        assert cp.all(result_ignore <= img)

    @pytest.mark.parametrize("func", gray_morphology_funcs)
    @pytest.mark.parametrize("mode", gray._SUPPORTED_MODES)
    def test_supported_mode(self, func, mode):
        img = cp.ones((10, 10))
        func = getattr(morphology, func)
        func(img, mode=mode)

    @pytest.mark.parametrize("func", gray_morphology_funcs)
    @pytest.mark.parametrize("mode", ["", "symmetric", 3, None])
    def test_unsupported_mode(self, func, mode):
        img = cp.ones((10, 10))
        with pytest.raises(ValueError, match="unsupported mode"):
            func = getattr(morphology, func)
            func(img, mode=mode)


eccentric_footprint_args = [
    (morphology.square, (2,)),
    (morphology.rectangle, (2, 2)),
    (morphology.rectangle, (2, 1)),
    (morphology.rectangle, (1, 2)),
]
eccentric_params = ("footprint_func, args", eccentric_footprint_args)


class TestEccentricStructuringElements:
    def setup_method(self):
        self.black_pixel = 255 * cp.ones((6, 6), dtype=cp.uint8)
        self.black_pixel[2, 2] = 0
        self.white_pixel = 255 - self.black_pixel

    @pytest.mark.parametrize(*eccentric_params)
    def test_dilate_erode_symmetry(self, footprint_func, args):
        s = footprint_func(*args)
        c = morphology.erosion(self.black_pixel, s)
        d = morphology.dilation(self.white_pixel, s)
        assert cp.all(c == (255 - d))

    @pytest.mark.parametrize(*eccentric_params)
    def test_open_black_pixel(self, footprint_func, args):
        s = footprint_func(*args)
        gray_open = morphology.opening(self.black_pixel, s)
        assert cp.all(gray_open == self.black_pixel)

    @pytest.mark.parametrize(*eccentric_params)
    def test_close_white_pixel(self, footprint_func, args):
        s = footprint_func(*args)
        gray_close = morphology.closing(self.white_pixel, s)
        assert cp.all(gray_close == self.white_pixel)

    @pytest.mark.parametrize(*eccentric_params)
    def test_open_white_pixel(self, footprint_func, args):
        s = footprint_func(*args)
        assert cp.all(morphology.opening(self.white_pixel, s) == 0)

    @pytest.mark.parametrize(*eccentric_params)
    def test_close_black_pixel(self, footprint_func, args):
        s = footprint_func(*args)
        assert cp.all(morphology.closing(self.black_pixel, s) == 255)

    @pytest.mark.parametrize(*eccentric_params)
    def test_white_tophat_white_pixel(self, footprint_func, args):
        s = footprint_func(*args)
        tophat = morphology.white_tophat(self.white_pixel, s)
        cp.testing.assert_array_equal(tophat, self.white_pixel)

    @pytest.mark.parametrize(*eccentric_params)
    def test_black_tophat_black_pixel(self, footprint_func, args):
        s = footprint_func(*args)
        tophat = morphology.black_tophat(self.black_pixel, s)
        cp.testing.assert_array_equal(tophat, self.white_pixel)

    @pytest.mark.parametrize(*eccentric_params)
    def test_white_tophat_black_pixel(self, footprint_func, args):
        s = footprint_func(*args)
        tophat = morphology.white_tophat(self.black_pixel, s)
        assert cp.all(tophat == 0)

    @pytest.mark.parametrize(*eccentric_params)
    def test_black_tophat_white_pixel(self, footprint_func, args):
        s = footprint_func(*args)
        tophat = morphology.black_tophat(self.white_pixel, s)
        assert cp.all(tophat == 0)


@pytest.mark.parametrize("func_name", gray_morphology_funcs)
def test_default_footprint(func_name):
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
    function = getattr(morphology, func_name)
    im_expected = function(image, footprint)
    im_test = function(image)
    cp.testing.assert_array_equal(im_expected, im_test)


def test_3d_fallback_default_footprint():
    # 3x3x3 cube inside a 7x7x7 image:
    image = cp.zeros((7, 7, 7), bool)
    image[2:-2, 2:-2, 2:-2] = 1

    opened = morphology.opening(image)

    # expect a "hyper-cross" centered in the 5x5x5:
    image_expected = cp.zeros((7, 7, 7), dtype=bool)
    image_expected[2:5, 2:5, 2:5] = ndi.generate_binary_structure(3, 1)
    cp.testing.assert_array_equal(opened, image_expected)


gray_3d_fallback_functions = [morphology.closing, morphology.opening]


@pytest.mark.parametrize("function", gray_3d_fallback_functions)
def test_3d_fallback_cube_footprint(function):
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

    with expected_warnings([r"operator.*deprecated|\A\Z"]):
        new_image = morphology.white_tophat(image)
    footprint = ndi.generate_binary_structure(3, 1)
    with expected_warnings([r"operator.*deprecated|\A\Z"]):
        image_expected = ndi.white_tophat(
            image.view(dtype=cp.uint8), footprint=footprint
        )
    cp.testing.assert_array_equal(new_image, image_expected)


def test_3d_fallback_black_tophat():
    image = cp.ones((7, 7, 7), dtype=bool)
    image[2, 2:4, 2:4] = 0
    image[3, 2:5, 2:5] = 0
    image[4, 3:5, 3:5] = 0

    with expected_warnings([r"operator.*deprecated|\A\Z"]):
        new_image = morphology.black_tophat(image)
    footprint = ndi.generate_binary_structure(3, 1)
    with expected_warnings([r"operator.*deprecated|\A\Z"]):
        image_expected = ndi.black_tophat(
            image.view(dtype=cp.uint8), footprint=footprint
        )
    cp.testing.assert_array_equal(new_image, image_expected)


def test_2d_ndimage_equivalence():
    image = cp.zeros((9, 9), cp.uint8)
    image[2:-2, 2:-2] = 128
    image[3:-3, 3:-3] = 196
    image[4, 4] = 255

    opened = morphology.opening(image)
    closed = morphology.closing(image)

    footprint = ndi.generate_binary_structure(2, 1)
    ndimage_opened = ndi.grey_opening(image, footprint=footprint)
    ndimage_closed = ndi.grey_closing(image, footprint=footprint)

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
    cp.testing.assert_allclose(morphology.erosion(im), eroded)
    cp.testing.assert_allclose(morphology.dilation(im), dilated)
    cp.testing.assert_allclose(morphology.opening(im), opened)
    cp.testing.assert_allclose(morphology.closing(im), closed)


def test_uint16():
    im16, eroded16, dilated16, opened16, closed16 = map(
        img_as_uint, [im, eroded, dilated, opened, closed]
    )
    cp.testing.assert_allclose(morphology.erosion(im16), eroded16)
    cp.testing.assert_allclose(morphology.dilation(im16), dilated16)
    cp.testing.assert_allclose(morphology.opening(im16), opened16)
    cp.testing.assert_allclose(morphology.closing(im16), closed16)


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
    morphology.dilation(image, out=out_array)
    cp.testing.assert_array_equal(out_array_big, expected_dilation)
    morphology.erosion(image, out=out_array)
    cp.testing.assert_array_equal(out_array_big, expected_erosion)


def test_1d_erosion():
    image = cp.array([1, 2, 3, 2, 1])
    expected = cp.array([1, 1, 2, 1, 1])
    eroded = morphology.erosion(image)
    cp.testing.assert_array_equal(eroded, expected)


@pytest.mark.parametrize(
    "function",
    [
        "erosion",
        "dilation",
        "closing",
        "opening",
        "white_tophat",
        "black_tophat",
    ],
)
@pytest.mark.parametrize("size", (7,))
@pytest.mark.parametrize("decomposition", ["separable", "sequence"])
def test_square_decomposition(cam_image, function, size, decomposition):
    """Validate footprint decomposition for various shapes.

    comparison is made to the case without decomposition.
    """
    footprint_ndarray = morphology.square(size, decomposition=None)
    footprint = morphology.square(size, decomposition=decomposition)
    func = getattr(morphology, function)
    expected = func(cam_image, footprint=footprint_ndarray)
    out = func(cam_image, footprint=footprint)
    cp.testing.assert_array_equal(expected, out)


@pytest.mark.parametrize(
    "function",
    [
        "erosion",
        "dilation",
        "closing",
        "opening",
        "white_tophat",
        "black_tophat",
    ],
)
@pytest.mark.parametrize("nrows", (3, 11))
@pytest.mark.parametrize("ncols", (3, 11))
@pytest.mark.parametrize("decomposition", ["separable", "sequence"])
def test_rectangle_decomposition(
    cam_image, function, nrows, ncols, decomposition
):
    """Validate footprint decomposition for various shapes.

    comparison is made to the case without decomposition.
    """
    footprint_ndarray = morphology.rectangle(nrows, ncols, decomposition=None)
    footprint = morphology.rectangle(nrows, ncols, decomposition=decomposition)
    func = getattr(morphology, function)
    expected = func(cam_image, footprint=footprint_ndarray)
    out = func(cam_image, footprint=footprint)
    cp.testing.assert_array_equal(expected, out)


@pytest.mark.parametrize(
    "function",
    [
        "erosion",
        "dilation",
        "closing",
        "opening",
        "white_tophat",
        "black_tophat",
    ],
)
@pytest.mark.parametrize("radius", (2, 3))
@pytest.mark.parametrize("decomposition", ["sequence"])
def test_diamond_decomposition(cam_image, function, radius, decomposition):
    """Validate footprint decomposition for various shapes.

    comparison is made to the case without decomposition.
    """
    footprint_ndarray = morphology.diamond(radius, decomposition=None)
    footprint = morphology.diamond(radius, decomposition=decomposition)
    func = getattr(morphology, function)
    expected = func(cam_image, footprint=footprint_ndarray)
    out = func(cam_image, footprint=footprint)
    cp.testing.assert_array_equal(expected, out)


@pytest.mark.parametrize(
    "function",
    [
        "erosion",
        "dilation",
        "closing",
        "opening",
        "white_tophat",
        "black_tophat",
    ],
)
@pytest.mark.parametrize("m", (0, 1, 3, 5))
@pytest.mark.parametrize("n", (0, 1, 2, 3))
@pytest.mark.parametrize("decomposition", ["sequence"])
def test_octagon_decomposition(cam_image, function, m, n, decomposition):
    """Validate footprint decomposition for various shapes.

    comparison is made to the case without decomposition.
    """
    if m == 0 and n == 0:
        with pytest.raises(ValueError):
            morphology.octagon(m, n, decomposition=decomposition)
    else:
        footprint_ndarray = morphology.octagon(m, n, decomposition=None)
        footprint = morphology.octagon(m, n, decomposition=decomposition)
        func = getattr(morphology, function)
        expected = func(cam_image, footprint=footprint_ndarray)
        out = func(cam_image, footprint=footprint)
        cp.testing.assert_array_equal(expected, out)


@pytest.mark.parametrize(
    "function",
    [
        "erosion",
        "dilation",
        "closing",
        "opening",
        "white_tophat",
        "black_tophat",
    ],
)
@pytest.mark.parametrize("size", (5,))
@pytest.mark.parametrize("decomposition", ["separable", "sequence"])
def test_cube_decomposition(cell3d_image, function, size, decomposition):
    """Validate footprint decomposition for various shapes.

    comparison is made to the case without decomposition.
    """
    footprint_ndarray = morphology.cube(size, decomposition=None)
    footprint = morphology.cube(size, decomposition=decomposition)
    func = getattr(morphology, function)
    expected = func(cell3d_image, footprint=footprint_ndarray)
    out = func(cell3d_image, footprint=footprint)
    cp.testing.assert_array_equal(expected, out)


@pytest.mark.parametrize(
    "function",
    [
        "erosion",
        "dilation",
        "closing",
        "opening",
        "white_tophat",
        "black_tophat",
    ],
)
@pytest.mark.parametrize("radius", (3,))
@pytest.mark.parametrize("decomposition", ["sequence"])
def test_octahedron_decomposition(
    cell3d_image, function, radius, decomposition
):
    """Validate footprint decomposition for various shapes.

    comparison is made to the case without decomposition.
    """
    footprint_ndarray = morphology.octahedron(radius, decomposition=None)
    footprint = morphology.octahedron(radius, decomposition=decomposition)
    func = getattr(morphology, function)
    expected = func(cell3d_image, footprint=footprint_ndarray)
    out = func(cell3d_image, footprint=footprint)
    cp.testing.assert_array_equal(expected, out)


@pytest.mark.parametrize(
    "function",
    ["erosion", "dilation", "closing", "opening"],
)
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("odd_only", [False, True])
def test_tuple_as_footprint(function, ndim, odd_only):
    """Validate footprint decomposition for various shapes.

    comparison is made to the case without decomposition.
    """
    if odd_only:
        footprint_shape = (3,) * ndim
    else:
        footprint_shape = tuple(range(2, 2 + ndim))
    footprint_ndarray = cp.ones(footprint_shape, dtype=bool)

    rng = cp.random.default_rng(5)
    img = rng.standard_normal((16,) * ndim, dtype=cp.float32)
    func = getattr(morphology, function)
    expected = func(img, footprint=footprint_ndarray)
    out = func(img, footprint=footprint_shape)
    testing.assert_array_equal(expected, out)


@pytest.mark.parametrize("func", [morphology.erosion, morphology.dilation])
@pytest.mark.parametrize("name", ["shift_x", "shift_y"])
@pytest.mark.parametrize("value", [True, False, None])
def test_deprecated_shift(func, name, value):
    img = cp.ones(10)
    func(img)  # Shouldn't warn

    regex = "`shift_x` and `shift_y` are deprecated"
    with pytest.warns(FutureWarning, match=regex) as record:
        func(img, **{name: value})
    assert_stacklevel(record)
