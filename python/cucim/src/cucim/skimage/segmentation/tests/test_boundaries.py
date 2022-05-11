import cupy as cp
import numpy as np
import pytest
from cupy.testing import assert_allclose, assert_array_equal

from cucim.skimage._shared.utils import _supported_float_type
from cucim.skimage.segmentation import find_boundaries, mark_boundaries

white = (1, 1, 1)


def test_find_boundaries():
    image = cp.zeros((10, 10), dtype=cp.uint8)
    image[2:7, 2:7] = 1

    # fmt: off
    ref = cp.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    # fmt: on
    result = find_boundaries(image)
    assert_array_equal(result, ref)


def test_find_boundaries_bool():
    image = cp.zeros((5, 5), dtype=bool)
    image[2:5, 2:5] = True

    # fmt: off
    ref = cp.array([[False, False, False, False, False],  # noqa
                    [False, False,  True,  True,  True],  # noqa
                    [False,  True,  True,  True,  True],  # noqa
                    [False,  True,  True, False, False],  # noqa
                    [False,  True,  True, False, False]], dtype=bool)  # noqa
    # fmt: on
    result = find_boundaries(image)
    assert_array_equal(result, ref)


@pytest.mark.parametrize(
    'dtype', [cp.uint8, cp.float16, cp.float32, cp.float64]
)
def test_mark_boundaries(dtype):
    image = cp.zeros((10, 10), dtype=dtype)
    label_image = cp.zeros((10, 10), dtype=cp.uint8)
    label_image[2:7, 2:7] = 1

    # fmt: off
    ref = cp.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    # fmt: on
    marked = mark_boundaries(image, label_image, color=white, mode='thick')
    assert marked.dtype == _supported_float_type(dtype)
    result = cp.mean(marked, axis=-1)
    assert_array_equal(result, ref)
    # fmt: off
    ref = cp.array([[0, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                    [2, 2, 1, 1, 1, 1, 1, 2, 2, 0],
                    [2, 1, 1, 1, 1, 1, 1, 1, 2, 0],
                    [2, 1, 1, 2, 2, 2, 1, 1, 2, 0],
                    [2, 1, 1, 2, 0, 2, 1, 1, 2, 0],
                    [2, 1, 1, 2, 2, 2, 1, 1, 2, 0],
                    [2, 1, 1, 1, 1, 1, 1, 1, 2, 0],
                    [2, 2, 1, 1, 1, 1, 1, 2, 2, 0],
                    [0, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    # fmt: on
    marked = mark_boundaries(image, label_image, color=white,
                             outline_color=(2, 2, 2), mode='thick')
    result = cp.mean(marked, axis=-1)
    assert_array_equal(result, ref)


def test_mark_boundaries_bool():
    image = cp.zeros((10, 10), dtype=bool)
    label_image = cp.zeros((10, 10), dtype=cp.uint8)
    label_image[2:7, 2:7] = 1
    # fmt: off
    ref = cp.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    # fmt: on
    marked = mark_boundaries(image, label_image, color=white, mode='thick')
    result = cp.mean(marked, axis=-1)
    assert_array_equal(result, ref)


@pytest.mark.parametrize('dtype', [cp.float16, cp.float32, cp.float64])
def test_mark_boundaries_subpixel(dtype):
    # fmt: off
    labels = cp.array([[0, 0, 0, 0],
                       [0, 0, 5, 0],
                       [0, 1, 5, 0],
                       [0, 0, 5, 0],
                       [0, 0, 0, 0]], dtype=cp.uint8)
    np.random.seed(0)
    # fmt: on
    # Note: use np.random to have same seed as NumPy
    # Note: use np.round until cp.around is fixed upstream
    image = cp.asarray(np.round(np.random.rand(*labels.shape), 2))
    image = image.astype(dtype, copy=False)
    marked = mark_boundaries(image, labels, color=white, mode='subpixel')
    assert marked.dtype == _supported_float_type(dtype)
    marked_proj = cp.asarray(cp.around(cp.mean(marked, axis=-1), 2))

    # fmt: off
    ref_result = cp.array(
        [[0.55, 0.63, 0.72, 0.69, 0.6 , 0.55, 0.54],   # noqa
         [0.45, 0.58, 0.72, 1.  , 1.  , 1.  , 0.69],   # noqa
         [0.42, 0.54, 0.65, 1.  , 0.44, 1.  , 0.89],   # noqa
         [0.69, 1.  , 1.  , 1.  , 0.69, 1.  , 0.83],   # noqa
         [0.96, 1.  , 0.38, 1.  , 0.79, 1.  , 0.53],   # noqa
         [0.89, 1.  , 1.  , 1.  , 0.38, 1.  , 0.16],   # noqa
         [0.57, 0.78, 0.93, 1.  , 0.07, 1.  , 0.09],   # noqa
         [0.2 , 0.52, 0.92, 1.  , 1.  , 1.  , 0.54],   # noqa
         [0.02, 0.35, 0.83, 0.9 , 0.78, 0.81, 0.87]])  # noqa
    # fmt: on

    # TODO: get fully equivalent interpolation/boundary as skimage
    #       I think this requires fixing mode='reflect' upstream in SciPy
    if False:
        assert_allclose(marked_proj, ref_result, atol=0.01)
    else:
        # Note: grlee77: only test locations of ones, due to different default
        #                interpolation settings in CuPy version of mark
        #                 boundaries
        assert_allclose(marked_proj == 1, ref_result == 1, atol=0.01)
