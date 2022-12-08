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
        ((384, 256), (3, 2)),  # integer-valued anisotropic
        ((384, 256), (2.25, .85)),
        ((14, 32, 50), None),
        ((50, 32, 24), (2., 2., 2.)),
        ((50, 32, 24), (3, 1, 2)),  # integer-valued anisotropic
    ],
)
@pytest.mark.parametrize('density', ['single_point', 5, 50, 95])
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
    if density == 'single_point':
        img = cp.ones(shape, dtype=bool)
        img[tuple(s // 2 for s in shape)] = 0
    else:
        img = binary_image(shape, pct_true=density)

    out = distance_transform_edt(img, **kwargs_cucim)
    expected = ndi_cpu.distance_transform_edt(cp.asnumpy(img), **kwargs_scipy)
    if sampling is None:
        target_pct = 95
    else:
        target_pct = 90
    if return_indices and return_distances:
        assert len(out) == 2
        cp.testing.assert_allclose(out[0], expected[0], rtol=1e-6)
        # May differ at a small % of coordinates where multiple points were
        # equidistant.
        assert_percentile_equal(out[1], expected[1], pct=target_pct)
    elif return_distances:
        cp.testing.assert_allclose(out, expected, rtol=1e-6)
    elif return_indices:
        assert_percentile_equal(out, expected, pct=target_pct)


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


@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('sampling', [None, 'iso', 'aniso'])
def test_distance_transform_inplace_distance(ndim, sampling):
    img = binary_image((32, ) * ndim, pct_true=80)
    distances = cp.empty(img.shape, dtype=cp.float32)
    if sampling == 'iso':
        sampling = (1.5,) * ndim
    elif sampling == 'aniso':
        sampling = tuple(range(1, ndim + 1))
    distance_transform_edt(img, sampling=sampling, distances=distances)
    expected = ndi_cpu.distance_transform_edt(
        cp.asnumpy(img), sampling=sampling
    )
    cp.testing.assert_allclose(distances, expected)


@pytest.mark.parametrize('ndim', [2, 3])
def test_distance_transform_inplace_distance_errors(ndim):
    img = binary_image((32, ) * ndim, pct_true=80)

    # for binary input, distances output is float32. Other dtypes raise
    with pytest.raises(RuntimeError):
        distances = cp.empty(img.shape, dtype=cp.float64)
        distance_transform_edt(img, distances=distances)
    with pytest.raises(RuntimeError):
        distances = cp.empty(img.shape, dtype=cp.int32)
        distance_transform_edt(img, distances=distances)

    # wrong shape
    with pytest.raises(RuntimeError):
        distances = cp.empty(img.shape + (2,), dtype=cp.float32)
        distance_transform_edt(img, distances=distances)

    # can't provide indices array when return_indices is False
    with pytest.raises(RuntimeError):
        distances = cp.empty(img.shape, dtype=cp.float32)
        distance_transform_edt(img, distances=distances,
                               return_distances=False, return_indices=True)


@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('sampling', [None, 'iso', 'aniso'])
@pytest.mark.parametrize('dtype', [cp.int16, cp.uint16, cp.uint32, cp.int32,
                                   cp.uint64, cp.int64])
@pytest.mark.parametrize('return_distances', [False, True])
def test_distance_transform_inplace_indices(
    ndim, sampling, dtype, return_distances
):
    img = binary_image((32, ) * ndim, pct_true=80)
    if ndim == 3 and dtype in [cp.int16, cp.uint16]:
        pytest.skip(reason="3d case requires at least 32-bit integer output")
    if sampling == 'iso':
        sampling = (1.5,) * ndim
    elif sampling == 'aniso':
        sampling = tuple(range(1, ndim + 1))
    common_kwargs = dict(
        sampling=sampling, return_distances=return_distances,
        return_indices=True
    )
    # verify that in-place and out-of-place results agree
    indices = cp.empty((ndim,) + img.shape, dtype=dtype)
    distance_transform_edt(img, indices=indices, **common_kwargs)
    expected = distance_transform_edt(img, **common_kwargs)
    if return_distances:
        cp.testing.assert_array_equal(indices, expected[1])
    else:
        cp.testing.assert_array_equal(indices, expected)


@pytest.mark.parametrize('ndim', [2, 3])
def test_distance_transform_inplace_indices_errors(ndim):
    img = binary_image((32, ) * ndim, pct_true=80)
    common_kwargs = dict(return_distances=False, return_indices=True)

    # int8 has itemsize too small
    with pytest.raises(RuntimeError):
        indices = cp.empty((ndim,) + img.shape, dtype=cp.int8)
        distance_transform_edt(img, indices=indices, **common_kwargs)

    # float not allowed
    with pytest.raises(RuntimeError):
        indices = cp.empty((ndim,) + img.shape, dtype=cp.float64)
        distance_transform_edt(img, indices=indices, **common_kwargs)

    # wrong shape
    with pytest.raises(RuntimeError):
        indices = cp.empty((ndim,), dtype=cp.float32)
        distance_transform_edt(img, indices=indices, **common_kwargs)

    # can't provide indices array when return_indices is False
    with pytest.raises(RuntimeError):
        indices = cp.empty((ndim,) + img.shape, dtype=cp.int32)
        distance_transform_edt(img, indices=indices, return_indices=False)


@pytest.mark.parametrize('sx', list(range(4)))
@pytest.mark.parametrize('sy', list(range(4)))
@pytest.mark.parametrize('sz', list(range(4)))
def test_distance_transform_edt_3d_aniso(sx, sy, sz):
    """ensure default block_params is robust to anisotropic shape."""
    shape = (16 + sz, 32 + sy, 48 + sx)
    img = binary_image(shape, pct_true=80)
    out = distance_transform_edt(img)
    print(f"{out.dtype=}")
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
