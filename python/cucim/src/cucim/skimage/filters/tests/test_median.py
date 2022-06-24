import cupy as cp
import pytest
from cupy.testing import assert_allclose
from cupyx.scipy import ndimage
from skimage import data

from cucim.skimage._shared.testing import expected_warnings
from cucim.skimage.filters import median


@pytest.fixture
def image():
    return cp.array([[1, 2, 3, 2, 1],
                     [1, 1, 2, 2, 3],
                     [3, 2, 1, 2, 1],
                     [3, 2, 1, 1, 1],
                     [1, 2, 1, 2, 3]],
                    dtype=cp.uint8)


@pytest.fixture
def camera():
    return cp.array(data.camera())


# TODO: mode='rank' disabled until it has been implmented
@pytest.mark.parametrize(
    "mode, cval, behavior, warning_type",
    [('nearest', 0.0, 'ndimage', []),
     # ('constant', 0.0, 'rank', (UserWarning,)),
     # ('nearest', 0.0, 'rank', []),
     ('nearest', 0.0, 'ndimage', [])]
)
def test_median_warning(image, mode, cval, behavior, warning_type):

    if warning_type:
        with pytest.warns(warning_type):
            median(image, mode=mode, behavior=behavior)
    else:
        median(image, mode=mode, behavior=behavior)


def test_selem_kwarg_deprecation(image):
    with expected_warnings(["`selem` is a deprecated argument name"]):
        median(image, selem=None)


# TODO: update if rank.median implemented
@pytest.mark.parametrize(
    'behavior, func', [('ndimage', ndimage.median_filter)],
    # ('rank', rank.median, {'footprint': cp.ones((3, 3), dtype=cp.uint8)})]
)
@pytest.mark.parametrize(
    'mode', ['reflect', 'mirror', 'nearest', 'constant', 'wrap']
)
# include even shapes and singleton shape that force non-histogram code path.
# include some large shapes that always take the histogram-based code path.
@pytest.mark.parametrize(
    'footprint_shape', [
        (3, 3), (5, 5), (9, 15), (2, 2), (1, 1), (2, 7), (23, 23), (15, 35),
    ]
)
@pytest.mark.parametrize('out', [None, cp.uint8, cp.float32, 'array'])
def test_median_behavior(camera, behavior, func, mode, footprint_shape, out):
    footprint = cp.ones(footprint_shape, dtype=bool)
    cam2 = camera[:, :177]  # use anisotropic size
    assert cam2.dtype == cp.uint8
    if out == 'array':
        out = cp.zeros_like(cam2)
    assert_allclose(
        median(cam2, footprint, mode=mode, behavior=behavior, out=out),
        func(cam2, size=footprint.shape, mode=mode, output=out),
    )


@pytest.mark.parametrize(
    'behavior, func', [('ndimage', ndimage.median_filter)],
    # ('rank', rank.median, {'footprint': cp.ones((3, 3), dtype=cp.uint8)})]
)
@pytest.mark.parametrize(
    'mode', ['reflect', 'mirror', 'nearest', 'constant', 'wrap']
)
# use an anisotropic footprint large enough to trigger the histogram-based path
@pytest.mark.parametrize('footprint_shape', [(15, 23)])
@pytest.mark.parametrize(
    'int_dtype', [cp.uint8, cp.uint16, cp.int8, cp.int16]
)
def test_median_hist_dtypes(
    camera, behavior, func, mode, footprint_shape, int_dtype
):
    footprint = cp.ones(footprint_shape, dtype=bool)
    rng = cp.random.default_rng(123)
    shape = (350, 407)  # use anisotropic size
    if int_dtype == cp.uint8:
        img = rng.integers(0, 256, shape, dtype=cp.uint8)
    elif int_dtype == cp.int8:
        img = rng.integers(-128, 128, shape, dtype=int).astype(cp.int8)
    elif int_dtype == cp.uint16:
        # test with 12-bit range stored in 16-bit integers (e.g. DICOM)
        img = rng.integers(0, 4096, shape, dtype=cp.uint16)
    elif int_dtype == cp.int16:
        # chose a limited range of values to test 512 hist_size case
        img = rng.integers(-128, 384, shape, dtype=int).astype(cp.int16)
    out = median(img, footprint, mode=mode, behavior=behavior)
    expected = func(img, size=footprint.shape, mode=mode)
    assert_allclose(expected, out)


@pytest.mark.parametrize('int_dtype', [cp.uint16, cp.int16])
def test_median_hist_kernel_resource_limit_try_except(int_dtype):
    # use an anisotropic footprint large enough to trigger
    # the histogram-based path
    footprint = cp.ones((15, 23), dtype=bool)
    mode = 'nearest'
    rng = cp.random.default_rng(123)
    shape = (350, 407)  # use anisotropic size
    if int_dtype == cp.uint16:
        # test with range likely to exceed the shared memory limit
        img = rng.integers(0, 65536, shape, dtype=cp.uint16)
    elif int_dtype == cp.int16:
        # test with range likely to exceed the shared memory limit
        img = rng.integers(-32768, 32767, shape, dtype=int).astype(cp.int16)
    out = median(img, footprint, mode=mode)
    expected = ndimage.median_filter(img, size=footprint.shape, mode=mode)
    assert_allclose(expected, out)


@pytest.mark.parametrize(
    "dtype", [cp.uint8, cp.uint16, cp.float32, cp.float64]
)
def test_median_preserve_dtype(image, dtype):
    median_image = median(image.astype(dtype), behavior='ndimage')
    assert median_image.dtype == dtype


# TODO: update if rank.median implemented
# def test_median_error_ndim():
#     img = cp.random.randint(0, 10, size=(5, 5, 5), dtype=cp.uint8)
#     with pytest.raises(ValueError):
#         median(img, behavior='rank')


# TODO: update if rank.median implemented
@pytest.mark.parametrize(
    "img, behavior",
    # (cp.random.randint(0, 10, size=(3, 3), dtype=cp.uint8), 'rank'),
    [(cp.random.randint(0, 10, size=(3, 3), dtype=cp.uint8), 'ndimage'),
     (cp.random.randint(0, 10, size=(3, 3, 3), dtype=cp.uint8), 'ndimage')]
)
def test_median(img, behavior):
    median(img, behavior=behavior)
