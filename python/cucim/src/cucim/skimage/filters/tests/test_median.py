import cupy as cp
import pytest
from cupy.testing import assert_allclose
from cupyx.scipy import ndimage

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
    "behavior, func, params",
    [('ndimage', ndimage.median_filter, {'size': (3, 3)})]
    # ('rank', rank.median, {'footprint': cp.ones((3, 3), dtype=cp.uint8)})]
)
def test_median_behavior(image, behavior, func, params):
    assert_allclose(median(image, behavior=behavior), func(image, **params))


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
