from copy import copy

import cupy as cp
import numpy as np
import pytest
import scipy.ndimage as ndi_cpu

from cucim.core.operations.morphology import distance_transform_edt


def binary_image(shape, pct_true=50):
    rng = cp.random.default_rng(123)
    x = rng.integers(0, 100, size=shape, dtype=cp.uint8)
    return x >= pct_true


def assert_percentile_equal(arr1, arr2, pct=95):
    """Assert that at least pct% of the entries in arr1 and arr2 are equal."""
    pct_mismatch = (100 - pct) / 100
    arr1 = cp.asnumpy(arr1)
    arr2 = cp.asnumpy(arr2)
    mismatch = np.sum(arr1 != arr2) / arr1.size
    assert mismatch < pct_mismatch


@pytest.mark.parametrize('return_indices', [False, True])
@pytest.mark.parametrize('return_distances', [False, True])
@pytest.mark.parametrize(
    'shape, sampling',
    [
        ((256, 128), None),
        ((384, 256), (1.5, 1.5)),
        ((14, 32, 50), None),
        ((50, 32, 24), (2, 2, 2)),
    ],
)
@pytest.mark.parametrize('density', [5, 50, 95])
@pytest.mark.parametrize('block_params', [None, (1, 1, 1)])
def test_distance_transform_edt(
    shape, sampling, return_distances, return_indices, density, block_params
):

    if not (return_indices or return_distances):
        return

    kwargs_scipy = dict(
        sampling=sampling,
        return_distances=return_distances,
        return_indices=return_indices,
    )
    kwargs_cucim = copy(kwargs_scipy)
    kwargs_cucim['block_params'] = block_params
    img = binary_image(shape, pct_true=density)
    out = distance_transform_edt(img, **kwargs_cucim)
    expected = ndi_cpu.distance_transform_edt(cp.asnumpy(img), **kwargs_scipy)
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


@pytest.mark.parametrize(
    'shape',
    (
        [(s,) * 2 for s in range(512, 512 + 32)]
        + [(s,) * 2 for s in range(1024, 1024 + 16)]
        + [(s,) * 2 for s in range(2050, 2050)]
        + [(s,) * 2 for s in range(4100, 4100)]
    ),
)
@pytest.mark.parametrize('density', [2, 98])
def test_distance_transform_edt_additional_shapes(shape, density):

    kwargs_scipy = dict(return_distances=True, return_indices=False)
    kwargs_cucim = copy(kwargs_scipy)
    img = binary_image(shape, pct_true=density)
    distances = distance_transform_edt(img, **kwargs_cucim)
    expected = ndi_cpu.distance_transform_edt(cp.asnumpy(img), **kwargs_scipy)
    cp.testing.assert_allclose(distances, expected)


@pytest.mark.parametrize(
    'shape',
    [(s,) * 2 for s in range(1024, 1024 + 4)],
)
@pytest.mark.parametrize(
    'block_params',
    [(1, 1, 1), (5, 4, 2), (3, 8, 4), (7, 16, 1), (11, 32, 3), (1, 1, 16)]
)
def test_distance_transform_edt_block_params(shape, block_params):

    kwargs_scipy = dict(return_distances=True, return_indices=False)
    kwargs_cucim = copy(kwargs_scipy)
    kwargs_cucim['block_params'] = block_params
    img = binary_image(shape, pct_true=4)
    distances = distance_transform_edt(img, **kwargs_cucim)
    expected = ndi_cpu.distance_transform_edt(cp.asnumpy(img), **kwargs_scipy)
    cp.testing.assert_allclose(distances, expected)


@pytest.mark.parametrize(
    'block_params', [
        (0, 1, 1), (1, 0, 1), (1, 1, 0),  # no elements can be < 1
        (1, 3, 1), (1, 5, 1), (1, 7, 1),  # 2nd element must be a power of 2
        (128, 1, 1),  # m1 too large for the array size
        (1, 128, 1),  # m2 too large for the array size
        (1, 1, 128),  # m3 too large for the array size
    ]
)
def test_distance_transform_edt_block_params_invalid(block_params):
    img = binary_image((512, 512), pct_true=4)
    with pytest.raises(ValueError):
        distance_transform_edt(img, block_params=block_params)


@pytest.mark.parametrize('return_indices', [False, True])
@pytest.mark.parametrize('return_distances', [False, True])
@pytest.mark.parametrize(
    'shape, sampling',
    [
        ((384, 256), (1, 3)),
        ((50, 32, 24), (1, 2, 4)),
    ]
)
@pytest.mark.parametrize('density', [5, 50, 95])
def test_distance_transform_edt_nonuniform_sampling(
    shape, sampling, return_distances, return_indices, density
):

    if not (return_indices or return_distances):
        return

    kwargs_scipy = dict(
        sampling=sampling,
        return_distances=return_distances,
        return_indices=return_indices,
    )
    kwargs_cucim = copy(kwargs_scipy)
    img = binary_image(shape, pct_true=density)
    if sampling is not None and len(np.unique(sampling)) != 1:
        with pytest.raises(NotImplementedError):
            distance_transform_edt(img, **kwargs_cucim)
        return


@pytest.mark.parametrize('value', [0, 1, 3])
@pytest.mark.parametrize('ndim', [2, 3])
def test_distance_transform_edt_uniform_valued(value, ndim):
    """ensure default block_params is robust to anisotropic shape."""
    img = cp.full((48, ) * ndim, value, dtype=cp.uint8)
    # ensure there is at least 1 pixel at background intensity
    img[(slice(24, 25),) * ndim] = 0
    out = distance_transform_edt(img)
    expected = ndi_cpu.distance_transform_edt(cp.asnumpy(img))
    cp.testing.assert_allclose(out, expected)


@pytest.mark.parametrize('sx', list(range(16)))
@pytest.mark.parametrize('sy', list(range(16)))
def test_distance_transform_edt_2d_aniso(sx, sy):
    """ensure default block_params is robust to anisotropic shape."""
    shape = (128 + sy, 128 + sx)
    img = binary_image(shape, pct_true=80)
    out = distance_transform_edt(img)
    expected = ndi_cpu.distance_transform_edt(cp.asnumpy(img))
    cp.testing.assert_allclose(out, expected)


@pytest.mark.parametrize('sx', list(range(4)))
@pytest.mark.parametrize('sy', list(range(4)))
@pytest.mark.parametrize('sz', list(range(4)))
def test_distance_transform_edt_3d_aniso(sx, sy, sz):
    """ensure default block_params is robust to anisotropic shape."""
    shape = (16 + sz, 32 + sy, 48 + sx)
    img = binary_image(shape, pct_true=80)
    out = distance_transform_edt(img)
    expected = ndi_cpu.distance_transform_edt(cp.asnumpy(img))
    cp.testing.assert_allclose(out, expected)


@pytest.mark.parametrize('ndim', [1, 4, 5])
def test_distance_transform_edt_unsupported_ndim(ndim):
    with pytest.raises(NotImplementedError):
        distance_transform_edt(cp.zeros((8,) * ndim))


@pytest.mark.skip(reason="excessive memory requirement")
def test_distance_transform_edt_3d_int64():
    shape = (1280, 1280, 1280)
    img = binary_image(shape, pct_true=80)
    distance_transform_edt(img)
    # Note: no validation vs. scipy.ndimage due to excessive run time
    return
