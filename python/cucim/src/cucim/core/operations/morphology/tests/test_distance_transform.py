from copy import copy

import cupy as cp
import numpy as np
import pytest
from cupy import testing
import scipy.ndimage as ndi_cpu

from cucim.core.operations.morphology import distance_transform_edt



def binary_image(shape, pct_true=50):
    rng = cp.random.default_rng(123)
    x = rng.integers(0, 100, size=shape,  dtype=cp.uint8)
    return x >=pct_true


def assert_percentile_equal(arr1, arr2, pct=95):
    """Assert that at least pct% of the entries in arr1 and arr2 are equal."""
    pct_mismatch = (100 - pct) / 100
    arr1 = cp.asnumpy(arr1)
    arr2 = cp.asnumpy(arr2)
    mismatch = np.sum(arr1 != arr2) / arr1.size
    assert mismatch < pct_mismatch


@pytest.mark.parametrize('block_params', [(1, 1, 1), None])
@pytest.mark.parametrize('sampling', [None, (1.5, 1.5)])
@pytest.mark.parametrize('shape', [(256, 256), (537, 236)])
@pytest.mark.parametrize('density', [20, 50, 80])
def test_distance_transform_edt_2d(shape, sampling, block_params, density):

    kwargs_scipy = dict(sampling=sampling)
    kwargs_cucim = dict(sampling=sampling, block_params=block_params)
    img = binary_image(shape, pct_true=density)
    out = distance_transform_edt(img, **kwargs_cucim)
    expected = ndi_cpu.distance_transform_edt(cp.asnumpy(img), **kwargs_scipy)
    cp.testing.assert_allclose(out, expected)


@pytest.mark.parametrize('sx', list(range(32)))
@pytest.mark.parametrize('sy', list(range(16)))
def test_distance_transform_edt_2d_aniso_block_params(sx, sy):
    """ensure default block_params is robust to anisotropic shape."""
    shape = (128 + sx, 128 + sy)
    img = binary_image(shape, pct_true=80)
    out = distance_transform_edt(img)
    expected = ndi_cpu.distance_transform_edt(cp.asnumpy(img))
    cp.testing.assert_allclose(out, expected)


@pytest.mark.parametrize('value', [0, 1, 3])
def test_distance_transform_edt_2d_uniform_valued(value):
    """ensure default block_params is robust to anisotropic shape."""
    img = cp.full((64, 64), value, dtype=cp.uint8)
    # ensure there is at least 1 pixel at background intensity
    img[13, 13] = 0
    out = distance_transform_edt(img)
    expected = ndi_cpu.distance_transform_edt(cp.asnumpy(img))
    cp.testing.assert_allclose(out, expected)


@pytest.mark.parametrize('return_indices', [False, True])
@pytest.mark.parametrize('return_distances', [False, True])
@pytest.mark.parametrize('sampling', [None, (1.5, 1.5)])
@pytest.mark.parametrize('shape', [(256, 256), (65, 128)])
def test_distance_transform_edt_2d_returns(shape, sampling, return_distances,
                                           return_indices):

    if not (return_indices or return_distances):
        return

    kwargs = dict(
        sampling=sampling,
        return_distances=return_distances,
        return_indices=return_indices,
    )
    img = binary_image(shape, pct_true=50)
    out = distance_transform_edt(img, **kwargs)
    expected = ndi_cpu.distance_transform_edt(cp.asnumpy(img), **kwargs)
    if return_indices and return_distances:
        assert len(out) == 2
        cp.testing.assert_allclose(out[0], expected[0])
        # May differ at a small % of coordinates where multiple points were
        # equidistant.
        assert_percentile_equal(out[1], expected[1], pct=95)
    elif return_distances:
        cp.testing.assert_allclose(out, expected)
    elif return_indices:
        assert_percentile_equal(out, expected, pct=95)
