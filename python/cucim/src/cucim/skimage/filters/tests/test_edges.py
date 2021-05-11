import cupy as cp
import numpy as np
import pytest
from cupy.testing import assert_allclose, assert_array_almost_equal
from numpy.testing import assert_

from cucim.skimage import filters
from cucim.skimage.data import binary_blobs
from cucim.skimage.filters.edges import _mask_filter_result


def test_roberts_zeros():
    """Roberts' filter on an array of all zeros."""
    result = filters.roberts(cp.zeros((10, 10)), cp.ones((10, 10), bool))
    assert cp.all(result == 0)


def test_roberts_diagonal1():
    """Roberts' filter on a diagonal edge should be a diagonal line."""
    image = cp.tri(10, 10, 0)
    expected = ~(cp.tri(10, 10, -1).astype(bool) |
                 cp.tri(10, 10, -2).astype(bool).transpose())
    expected[-1, -1] = 0  # due to 'reflect' & image shape, last pixel not edge
    result = filters.roberts(image).astype(bool)
    assert_array_almost_equal(result, expected)


def test_roberts_diagonal2():
    """Roberts' filter on a diagonal edge should be a diagonal line."""
    image = cp.rot90(cp.tri(10, 10, 0), 3)
    expected = ~cp.rot90(cp.tri(10, 10, -1).astype(bool) |
                         cp.tri(10, 10, -2).astype(bool).transpose())
    expected = _mask_filter_result(expected, None)
    result = filters.roberts(image).astype(bool)
    assert_array_almost_equal(result, expected)


def test_sobel_zeros():
    """Sobel on an array of all zeros."""
    result = filters.sobel(cp.zeros((10, 10)), cp.ones((10, 10), bool))
    assert cp.all(result == 0)


@pytest.mark.parametrize('function', ['sobel', 'prewitt', 'scharr'])
@pytest.mark.parametrize('dtype', [cp.float16, cp.float32, cp.float64])
def test_sobel_float_output_dtype(function, dtype):
    """Sobel on an array of all zeros."""
    x = cp.ones((4, 4), dtype=dtype)
    filter_func = getattr(filters, function)
    result = filter_func(x)
    assert result.dtype == dtype


def test_sobel_mask():
    """Sobel on a masked array should be zero."""
    result = filters.sobel(cp.random.uniform(size=(10, 10)),
                           cp.zeros((10, 10), dtype=bool))
    assert cp.all(result == 0)


def test_sobel_horizontal():
    """Sobel on a horizontal edge should be a horizontal line."""
    i, j = cp.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(float)
    result = filters.sobel(image) * np.sqrt(2)
    # Check if result match transform direction

    assert_allclose(result[i == 0], 1)
    assert_allclose(result[cp.abs(i) > 1], 0)


def test_sobel_vertical():
    """Sobel on a vertical edge should be a vertical line."""
    i, j = cp.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(float)
    result = filters.sobel(image) * np.sqrt(2)
    assert_allclose(result[j == 0], 1)
    assert cp.all(result[cp.abs(j) > 1] == 0)


def test_sobel_h_zeros():
    """Horizontal sobel on an array of all zeros."""
    result = filters.sobel_h(cp.zeros((10, 10)), cp.ones((10, 10), dtype=bool))
    assert cp.all(result == 0)


def test_sobel_h_mask():
    """Horizontal Sobel on a masked array should be zero."""
    result = filters.sobel_h(cp.random.uniform(size=(10, 10)),
                             cp.zeros((10, 10), dtype=bool))
    assert cp.all(result == 0)


def test_sobel_h_horizontal():
    """Horizontal Sobel on an edge should be a horizontal line."""
    i, j = cp.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(float)
    result = filters.sobel_h(image)
    # Check if result match transform direction
    assert cp.all(result[i == 0] == 1)
    assert cp.all(result[cp.abs(i) > 1] == 0)


def test_sobel_h_vertical():
    """Horizontal Sobel on a vertical edge should be zero."""
    i, j = cp.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(float) * np.sqrt(2)
    result = filters.sobel_h(image)
    assert_allclose(result, 0, atol=1e-10)


def test_sobel_v_zeros():
    """Vertical sobel on an array of all zeros."""
    result = filters.sobel_v(cp.zeros((10, 10)), cp.ones((10, 10), dtype=bool))
    assert_allclose(result, 0)


def test_sobel_v_mask():
    """Vertical Sobel on a masked array should be zero."""
    result = filters.sobel_v(cp.random.uniform(size=(10, 10)),
                             cp.zeros((10, 10), dtype=bool))
    assert_allclose(result, 0)


def test_sobel_v_vertical():
    """Vertical Sobel on an edge should be a vertical line."""
    i, j = cp.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(float)
    result = filters.sobel_v(image)
    # Check if result match transform direction
    assert cp.all(result[j == 0] == 1)
    assert cp.all(result[cp.abs(j) > 1] == 0)


def test_sobel_v_horizontal():
    """vertical Sobel on a horizontal edge should be zero."""
    i, j = cp.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(float)
    result = filters.sobel_v(image)
    assert_allclose(result, 0)


def test_scharr_zeros():
    """Scharr on an array of all zeros."""
    result = filters.scharr(cp.zeros((10, 10)), cp.ones((10, 10), dtype=bool))
    assert cp.all(result < 1e-16)


def test_scharr_mask():
    """Scharr on a masked array should be zero."""
    result = filters.scharr(cp.random.uniform(size=(10, 10)),
                            cp.zeros((10, 10), dtype=bool))
    assert_allclose(result, 0)


def test_scharr_horizontal():
    """Scharr on an edge should be a horizontal line."""
    i, j = cp.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(float)
    result = filters.scharr(image) * np.sqrt(2)
    # Check if result match transform direction
    assert_allclose(result[i == 0], 1)
    assert cp.all(result[cp.abs(i) > 1] == 0)


def test_scharr_vertical():
    """Scharr on a vertical edge should be a vertical line."""
    i, j = cp.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(float)
    result = filters.scharr(image) * np.sqrt(2)
    assert_allclose(result[j == 0], 1)
    assert cp.all(result[cp.abs(j) > 1] == 0)


def test_scharr_h_zeros():
    """Horizontal Scharr on an array of all zeros."""
    result = filters.scharr_h(cp.zeros((10, 10)),
                              cp.ones((10, 10), dtype=bool))
    assert_allclose(result, 0)


def test_scharr_h_mask():
    """Horizontal Scharr on a masked array should be zero."""
    result = filters.scharr_h(cp.random.uniform(size=(10, 10)),
                              cp.zeros((10, 10), dtype=bool))
    assert_allclose(result, 0)


def test_scharr_h_horizontal():
    """Horizontal Scharr on an edge should be a horizontal line."""
    i, j = cp.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(float)
    result = filters.scharr_h(image)
    # Check if result match transform direction
    assert cp.all(result[i == 0] == 1)
    assert cp.all(result[cp.abs(i) > 1] == 0)


def test_scharr_h_vertical():
    """Horizontal Scharr on a vertical edge should be zero."""
    i, j = cp.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(float)
    result = filters.scharr_h(image)
    assert_allclose(result, 0)


def test_scharr_v_zeros():
    """Vertical Scharr on an array of all zeros."""
    result = filters.scharr_v(cp.zeros((10, 10)),
                              cp.ones((10, 10), dtype=bool))
    assert_allclose(result, 0)


def test_scharr_v_mask():
    """Vertical Scharr on a masked array should be zero."""
    result = filters.scharr_v(cp.random.uniform(size=(10, 10)),
                              cp.zeros((10, 10), dtype=bool))
    assert_allclose(result, 0)


def test_scharr_v_vertical():
    """Vertical Scharr on an edge should be a vertical line."""
    i, j = cp.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(float)
    result = filters.scharr_v(image)
    # Check if result match transform direction
    assert cp.all(result[j == 0] == 1)
    assert cp.all(result[cp.abs(j) > 1] == 0)


def test_scharr_v_horizontal():
    """vertical Scharr on a horizontal edge should be zero."""
    i, j = cp.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(float)
    result = filters.scharr_v(image)
    assert_allclose(result, 0)


def test_prewitt_zeros():
    """Prewitt on an array of all zeros."""
    result = filters.prewitt(cp.zeros((10, 10)),
                             cp.ones((10, 10), dtype=bool))
    assert_allclose(result, 0)


def test_prewitt_mask():
    """Prewitt on a masked array should be zero."""
    result = filters.prewitt(cp.random.uniform(size=(10, 10)),
                             cp.zeros((10, 10), dtype=bool))
    assert_allclose(cp.abs(result), 0)


def test_prewitt_horizontal():
    """Prewitt on an edge should be a horizontal line."""
    i, j = cp.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(float)
    result = filters.prewitt(image) * np.sqrt(2)
    # Check if result match transform direction
    assert_allclose(result[i == 0], 1)
    assert_allclose(result[cp.abs(i) > 1], 0, atol=1e-10)


def test_prewitt_vertical():
    """Prewitt on a vertical edge should be a vertical line."""
    i, j = cp.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(float)
    result = filters.prewitt(image) * np.sqrt(2)
    assert_allclose(result[j == 0], 1)
    assert_allclose(result[cp.abs(j) > 1], 0, atol=1e-10)


def test_prewitt_h_zeros():
    """Horizontal prewitt on an array of all zeros."""
    result = filters.prewitt_h(cp.zeros((10, 10)),
                               cp.ones((10, 10), dtype=bool))
    assert_allclose(result, 0)


def test_prewitt_h_mask():
    """Horizontal prewitt on a masked array should be zero."""
    result = filters.prewitt_h(cp.random.uniform(size=(10, 10)),
                               cp.zeros((10, 10), dtype=bool))
    assert_allclose(result, 0)


def test_prewitt_h_horizontal():
    """Horizontal prewitt on an edge should be a horizontal line."""
    i, j = cp.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(float)
    result = filters.prewitt_h(image)
    # Check if result match transform direction
    assert cp.all(result[i == 0] == 1)
    assert_allclose(result[cp.abs(i) > 1], 0, atol=1e-10)


def test_prewitt_h_vertical():
    """Horizontal prewitt on a vertical edge should be zero."""
    i, j = cp.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(float)
    result = filters.prewitt_h(image)
    assert_allclose(result, 0, atol=1e-10)


def test_prewitt_v_zeros():
    """Vertical prewitt on an array of all zeros."""
    result = filters.prewitt_v(cp.zeros((10, 10)),
                               cp.ones((10, 10), dtype=bool))
    assert_allclose(result, 0)


def test_prewitt_v_mask():
    """Vertical prewitt on a masked array should be zero."""
    result = filters.prewitt_v(cp.random.uniform(size=(10, 10)),
                               cp.zeros((10, 10), dtype=bool))
    assert_allclose(result, 0)


def test_prewitt_v_vertical():
    """Vertical prewitt on an edge should be a vertical line."""
    i, j = cp.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(float)
    result = filters.prewitt_v(image)
    # Check if result match transform direction
    assert cp.all(result[j == 0] == 1)
    assert_allclose(result[cp.abs(j) > 1], 0, atol=1e-10)


def test_prewitt_v_horizontal():
    """Vertical prewitt on a horizontal edge should be zero."""
    i, j = cp.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(float)
    result = filters.prewitt_v(image)
    assert_allclose(result, 0)


def test_laplace_zeros():
    """Laplace on a square image."""
    # Create a synthetic 2D image
    image = cp.zeros((9, 9))
    image[3:-3, 3:-3] = 1
    result = filters.laplace(image)
    # fmt: off
    check_result = cp.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., -1., -1., -1., 0., 0., 0.],
                             [0., 0., -1., 2., 1., 2., -1., 0., 0.],
                             [0., 0., -1., 1., 0., 1., -1., 0., 0.],
                             [0., 0., -1., 2., 1., 2., -1., 0., 0.],
                             [0., 0., 0., -1., -1., -1., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    # fmt: on
    assert_allclose(result, check_result)


def test_laplace_mask():
    """Laplace on a masked array should be zero."""
    # Create a synthetic 2D image
    image = cp.zeros((9, 9))
    image[3:-3, 3:-3] = 1
    # Define the mask
    result = filters.laplace(image, ksize=3, mask=cp.zeros((9, 9), dtype=bool))
    assert cp.all(result == 0)


def test_farid_zeros():
    """Farid on an array of all zeros."""
    result = filters.farid(cp.zeros((10, 10)),
                           mask=cp.ones((10, 10), dtype=bool))
    assert cp.all(result == 0)


def test_farid_mask():
    """Farid on a masked array should be zero."""
    result = filters.farid(cp.random.uniform(size=(10, 10)),
                           mask=cp.zeros((10, 10), dtype=bool))
    assert (cp.all(result == 0))


def test_farid_horizontal():
    """Farid on a horizontal edge should be a horizontal line."""
    i, j = cp.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(float)
    result = filters.farid(image) * np.sqrt(2)
    # Check if result match transform direction
    assert cp.all(result[i == 0] == result[i == 0][0])
    assert_allclose(result[cp.abs(i) > 2], 0, atol=1e-10)


def test_farid_vertical():
    """Farid on a vertical edge should be a vertical line."""
    i, j = cp.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(float)
    result = filters.farid(image) * np.sqrt(2)
    assert cp.all(result[j == 0] == result[j == 0][0])
    assert_allclose(result[cp.abs(j) > 2], 0, atol=1e-10)


def test_farid_h_zeros():
    """Horizontal Farid on an array of all zeros."""
    result = filters.farid_h(cp.zeros((10, 10)),
                             mask=cp.ones((10, 10), dtype=bool))
    assert (cp.all(result == 0))


def test_farid_h_mask():
    """Horizontal Farid on a masked array should be zero."""
    result = filters.farid_h(cp.random.uniform(size=(10, 10)),
                             mask=cp.zeros((10, 10), dtype=bool))
    assert cp.all(result == 0)


def test_farid_h_horizontal():
    """Horizontal Farid on an edge should be a horizontal line."""
    i, j = cp.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(float)
    result = filters.farid_h(image)
    # Check if result match transform direction
    assert cp.all(result[i == 0] == result[i == 0][0])
    assert_allclose(result[cp.abs(i) > 2], 0, atol=1e-10)


def test_farid_h_vertical():
    """Horizontal Farid on a vertical edge should be zero."""
    i, j = cp.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(float) * np.sqrt(2)
    result = filters.farid_h(image)
    assert_allclose(result, 0, atol=1e-10)


def test_farid_v_zeros():
    """Vertical Farid on an array of all zeros."""
    result = filters.farid_v(cp.zeros((10, 10)),
                             mask=cp.ones((10, 10), dtype=bool))
    assert_allclose(result, 0, atol=1e-10)


def test_farid_v_mask():
    """Vertical Farid on a masked array should be zero."""
    result = filters.farid_v(cp.random.uniform(size=(10, 10)),
                             mask=cp.zeros((10, 10), dtype=bool))
    assert_allclose(result, 0)


def test_farid_v_vertical():
    """Vertical Farid on an edge should be a vertical line."""
    i, j = cp.mgrid[-5:6, -5:6]
    image = (j >= 0).astype(float)
    result = filters.farid_v(image)
    # Check if result match transform direction
    assert cp.all(result[j == 0] == result[j == 0][0])
    assert_allclose(result[cp.abs(j) > 2], 0, atol=1e-10)


def test_farid_v_horizontal():
    """vertical Farid on a horizontal edge should be zero."""
    i, j = cp.mgrid[-5:6, -5:6]
    image = (i >= 0).astype(float)
    result = filters.farid_v(image)
    assert_allclose(result, 0, atol=1e-10)


@pytest.mark.parametrize(
    "grad_func", (filters.prewitt_h, filters.sobel_h, filters.scharr_h)
)
def test_horizontal_mask_line(grad_func):
    """Horizontal edge filters mask pixels surrounding input mask."""
    vgrad, _ = cp.mgrid[:1:11j, :1:11j]  # vertical gradient with spacing 0.1
    vgrad[5, :] = 1  # bad horizontal line

    mask = cp.ones_like(vgrad)
    mask[5, :] = 0  # mask bad line

    expected = cp.zeros_like(vgrad)
    expected[1:-1, 1:-1] = 0.2  # constant gradient for most of image,
    expected[4:7, 1:-1] = 0  # but line and neighbors masked

    result = grad_func(vgrad, mask)
    assert_allclose(result, expected)


@pytest.mark.parametrize(
    "grad_func", (filters.prewitt_v, filters.sobel_v, filters.scharr_v)
)
def test_vertical_mask_line(grad_func):
    """Vertical edge filters mask pixels surrounding input mask."""
    _, hgrad = cp.mgrid[:1:11j, :1:11j]  # horizontal gradient with spacing 0.1
    hgrad[:, 5] = 1  # bad vertical line

    mask = cp.ones_like(hgrad)
    mask[:, 5] = 0  # mask bad line

    expected = cp.zeros_like(hgrad)
    expected[1:-1, 1:-1] = 0.2  # constant gradient for most of image,
    expected[1:-1, 4:7] = 0  # but line and neighbors masked

    result = grad_func(hgrad, mask)
    assert_allclose(result, expected)


# The below three constant 3x3x3 cubes were empirically found to maximise the
# output of each of their respective filters. We use them to test that the
# output of the filter on the blobs image matches expectation in terms of
# scale.

# maximum Sobel 3D edge on axis 0
# fmt: off
MAX_SOBEL_0 = cp.array([
    [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]],
    [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]],
    [[1, 1, 1],
     [1, 1, 1],
     [1, 1, 1]],
]).astype(float)

# maximum Sobel 3D edge in magnitude
MAX_SOBEL_ND = cp.array([
    [[1, 0, 0],
     [1, 0, 0],
     [1, 0, 0]],

    [[1, 0, 0],
     [1, 1, 0],
     [1, 1, 0]],

    [[1, 1, 0],
     [1, 1, 0],
     [1, 1, 0]]
]).astype(float)

# maximum Scharr 3D edge in magnitude. This illustrates the better rotation
# invariance of the Scharr filter!
MAX_SCHARR_ND = cp.array([
    [[0, 0, 0],
     [0, 0, 1],
     [0, 1, 1]],
    [[0, 0, 1],
     [0, 1, 1],
     [0, 1, 1]],
    [[0, 0, 1],
     [0, 1, 1],
     [1, 1, 1]]
]).astype(float)
# fmt: on


@pytest.mark.parametrize(
    ('func', 'max_edge'),
    [(filters.prewitt, MAX_SOBEL_ND),
     (filters.sobel, MAX_SOBEL_ND),
     (filters.scharr, MAX_SCHARR_ND)]
)
def test_3d_edge_filters(func, max_edge):
    blobs = binary_blobs(length=128, n_dim=3)
    edges = func(blobs)
    assert_allclose(cp.max(edges), func(max_edge)[1, 1, 1])


@pytest.mark.parametrize(
    'func', (filters.prewitt, filters.sobel, filters.scharr)
)
def test_3d_edge_filters_single_axis(func):
    blobs = binary_blobs(length=128, n_dim=3)
    edges0 = func(blobs, axis=0)
    assert_allclose(cp.max(edges0), func(MAX_SOBEL_0, axis=0)[1, 1, 1])


@pytest.mark.parametrize(
    'detector',
    [filters.sobel, filters.scharr, filters.prewitt,
     filters.roberts, filters.farid]
)
def test_range(detector):
    """Output of edge detection should be in [0, 1]"""
    image = cp.random.random((100, 100))
    out = detector(image)
    assert_(
        out.min() >= 0, f'Minimum of `{detector.__name__}` is smaller than 0.'
    )
    assert_(
        out.max() <= 1, f'Maximum of `{detector.__name__}` is larger than 1.'
    )
