import cupy as cp
import pytest
from cupy.testing import assert_allclose
from cupyx.scipy import ndimage
from skimage import data

from cucim.skimage._shared.testing import expected_warnings
from cucim.skimage.filters import median

try:
    from math import prod
except ImportError:
    from functools import reduce
    from operator import mul

    def prod(x):
        return reduce(mul, x)


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
    'mode', ['reflect', 'mirror', 'nearest', 'constant', 'wrap']
)
# use an anisotropic footprint large enough to trigger the histogram-based path
@pytest.mark.parametrize('footprint_shape', [(3, 3), (3, 5), (15, 23)])
@pytest.mark.parametrize(
    'int_dtype', [cp.uint8, cp.int8, cp.uint16, cp.int16]
)
@pytest.mark.parametrize(
    'algorithm', ['auto', 'histogram', 'sorting']
)
@pytest.mark.parametrize(
    'algorithm_kwargs', [{}, {'partitions': 32}]
)
def test_median_hist_dtypes(
    mode, footprint_shape, int_dtype, algorithm, algorithm_kwargs,
):
    footprint = cp.ones(footprint_shape, dtype=bool)
    rng = cp.random.default_rng(123)
    shape = (350, 407)
    if int_dtype == cp.uint8:
        img = rng.integers(0, 256, shape, dtype=cp.uint8)
    elif int_dtype == cp.int8:
        img = rng.integers(-128, 128, shape, dtype=int).astype(cp.int8)
    elif int_dtype == cp.uint16:
        if False:
            # test with 12-bit range stored in 16-bit integers (e.g. DICOM)
            img = rng.integers(0, 4096, shape, dtype=cp.uint16)
        else:
            # smaller dynamic range
            #    (range 4096 fails only on CI, but couldn't reproduce locally)
            img = rng.integers(0, 1024, shape, dtype=cp.uint16)
    elif int_dtype == cp.int16:
        # chose a limited range of values to test 512 hist_size case
        img = rng.integers(-128, 384, shape, dtype=int).astype(cp.int16)

    # 150 is the value used to auto-select between sorting vs. histogram
    small_kernel = prod(footprint_shape) < 150
    if algorithm_kwargs and (
        algorithm == 'sorting'
        or (algorithm == 'auto' and small_kernel)
    ):
        msg = ["algorithm_kwargs={'partitions': 32} ignored"]
    else:
        msg = []
    with expected_warnings(msg):
        out = median(img, footprint, mode=mode, behavior='ndimage',
                     algorithm=algorithm, algorithm_kwargs=algorithm_kwargs)
    expected = ndimage.median_filter(img, size=footprint.shape, mode=mode)
    assert_allclose(expected, out)


# TODO: Determine source of isolated remote test failures when 16-bit range
#       is > 1024. Could not reproduce locally.
@pytest.mark.parametrize('mode', ['reflect', ])
# use an anisotropic footprint large enough to trigger the histogram-based path
@pytest.mark.parametrize('footprint_shape', [(7, 11)])
@pytest.mark.parametrize(
    'int_dtype, irange',
    [
        (cp.uint16, (0, 256)),
        (cp.uint16, (0, 15)),
        (cp.uint16, (128, 384)),
        (cp.uint16, (0, 200)),
        (cp.uint16, (0, 510)),
        (cp.uint16, (500, 550)),
        (cp.uint16, (0, 1024)),
        pytest.param(cp.uint16, (0, 2048), marks=pytest.mark.skip(reason="isolated failure on CI only")),  # noqa
        pytest.param(cp.uint16, (1024, 3185), marks=pytest.mark.skip(reason="isolated failure on CI only")),  # noqa
        (cp.int16, (0, 256)),
        (cp.int16, (-15, 15)),
        (cp.int16, (128, 384)),
        (cp.int16, (-128, 384)),
        (cp.int16, (-400, 400)),
        pytest.param(cp.int16, (-1024, 2048), marks=pytest.mark.skip(reason="isolated failure on CI only")),  # noqa
        pytest.param(cp.int16, (150, 2048), marks=pytest.mark.skip(reason="isolated failure on CI only")),  # noqa
    ]
)
def test_median_hist_16bit_offsets(mode, footprint_shape, int_dtype, irange):
    """Make sure 16-bit cases are robust to various value ranges"""
    footprint = cp.ones(footprint_shape, dtype=bool)
    rng = cp.random.default_rng(123)
    shape = (350, 407)
    if int_dtype == cp.uint16:
        # test with 12-bit range stored in 16-bit integers (e.g. DICOM)
        img = rng.integers(irange[0], irange[1], shape, dtype=cp.uint16)
    elif int_dtype == cp.int16:
        # chose a limited range of values to test 512 hist_size case
        img = rng.integers(irange[0], irange[1], shape, dtype=int)
        img = img.astype(cp.int16)
    out = median(img, footprint, mode=mode, behavior='ndimage',
                 algorithm='histogram')
    expected = ndimage.median_filter(img, size=footprint.shape, mode=mode)
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
    'algorithm', ['auto', 'histogram', 'sorting', 'invalid']
)
def test_median_algorithm_parameter(algorithm):
    """Call all algorithms for float32 input.
    """
    footprint = cp.ones((15, 23), dtype=bool)
    mode = 'nearest'
    rng = cp.random.default_rng(123)
    shape = (350, 407)  # use anisotropic size
    img = rng.standard_normal(shape, dtype=cp.float32)
    if algorithm in ['invalid', 'histogram']:
        # histogram supports only integer-valued dtypes
        # 'invalid' is an uncrecognized algorithm
        with pytest.raises(ValueError):
            median(img, footprint, mode=mode, algorithm=algorithm)
    else:
        out = median(img, footprint, mode=mode, algorithm=algorithm)
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
