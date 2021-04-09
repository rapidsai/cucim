import cupy as cp
import numpy as np
import pytest
from cupy.testing import assert_allclose
from cupyx.scipy import ndimage

from cucim.skimage.filters import median

# from cucim.skimage.filters import rank


@pytest.fixture
def image():
    return cp.array([[1, 2, 3, 2, 1],
                     [1, 1, 2, 2, 3],
                     [3, 2, 1, 2, 1],
                     [3, 2, 1, 1, 1],
                     [1, 2, 1, 2, 3]],
                    dtype=np.uint8)


# TODO: mode='rank' disabled until it has been implmented
@pytest.mark.parametrize(
    "mode, cval, behavior, n_warning, warning_type",
    [('nearest', 0.0, 'ndimage', 0, []),
     # ('constant', 0.0, 'rank', 1, (UserWarning,)),
     # ('nearest', 0.0, 'rank', 0, []),
     ('nearest', 0.0, 'ndimage', 0, [])]
)
def test_median_warning(image, mode, cval, behavior,
                        n_warning, warning_type):

    with pytest.warns(None) as records:
        median(image, mode=mode, behavior=behavior)

    assert len(records) == n_warning
    for rec in records:
        assert isinstance(rec.message, warning_type)


# TODO: update if rank.median implemented
@pytest.mark.parametrize(
    "behavior, func, params",
    [('ndimage', ndimage.median_filter, {'size': (3, 3)})]
    # ('rank', rank.median, {'selem': np.ones((3, 3), dtype=np.uint8)})]
)
def test_median_behavior(image, behavior, func, params):
    assert_allclose(median(image, behavior=behavior), func(image, **params))


@pytest.mark.parametrize(
    "dtype", [np.uint8, np.uint16, np.float32, np.float64]
)
def test_median_preserve_dtype(image, dtype):
    median_image = median(image.astype(dtype), behavior='ndimage')
    assert median_image.dtype == dtype


# TODO: update if rank.median implemented
# def test_median_error_ndim():
#     img = cp.random.randint(0, 10, size=(5, 5, 5), dtype=np.uint8)
#     with pytest.raises(ValueError):
#         median(img, behavior='rank')


# TODO: update if rank.median implemented
@pytest.mark.parametrize(
    "img, behavior",
    # (np.random.randint(0, 10, size=(3, 3), dtype=np.uint8), 'rank'),
    [(cp.random.randint(0, 10, size=(3, 3), dtype=np.uint8), 'ndimage'),
     (cp.random.randint(0, 10, size=(3, 3, 3), dtype=np.uint8), 'ndimage')]
)
def test_median(img, behavior):
    median(img, behavior=behavior)
