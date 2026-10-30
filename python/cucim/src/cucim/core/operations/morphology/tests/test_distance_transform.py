# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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


@pytest.mark.parametrize("return_indices", [False, True])
@pytest.mark.parametrize("return_distances", [False, True])
@pytest.mark.parametrize(
    "shape, sampling",
    [
        ((256, 128), None),
        ((384, 256), (1.5, 1.5)),
        ((384, 256), (3, 2)),  # integer-valued anisotropic
        ((384, 256), (2.25, 0.85)),
        ((14, 32, 50), None),
        ((50, 32, 24), (2.0, 2.0, 2.0)),
        ((50, 32, 24), (3, 1, 2)),  # integer-valued anisotropic
    ],
)
@pytest.mark.parametrize("density", ["single_point", 5, 50, 95])
@pytest.mark.parametrize("block_params", [None, (1, 1, 1)])
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
    kwargs_cucim["block_params"] = block_params
    if density == "single_point":
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
    "shape",
    (
        [(s,) * 2 for s in range(512, 512 + 32)]
        + [(s,) * 2 for s in range(1024, 1024 + 16)]
        + [(s,) * 2 for s in range(2050, 2050)]
        + [(s,) * 2 for s in range(4100, 4100)]
    ),
)
@pytest.mark.parametrize("density", [2, 98])
def test_distance_transform_edt_additional_shapes(shape, density):
    kwargs_scipy = dict(return_distances=True, return_indices=False)
    kwargs_cucim = copy(kwargs_scipy)
    img = binary_image(shape, pct_true=density)
    distances = distance_transform_edt(img, **kwargs_cucim)
    expected = ndi_cpu.distance_transform_edt(cp.asnumpy(img), **kwargs_scipy)
    cp.testing.assert_allclose(distances, expected)


@pytest.mark.parametrize(
    "shape",
    [(s,) * 2 for s in range(1024, 1024 + 4)],
)
@pytest.mark.parametrize(
    "block_params",
    [(1, 1, 1), (5, 4, 2), (3, 8, 4), (7, 16, 1), (11, 32, 3), (1, 1, 16)],
)
def test_distance_transform_edt_block_params(shape, block_params):
    kwargs_scipy = dict(return_distances=True, return_indices=False)
    kwargs_cucim = copy(kwargs_scipy)
    kwargs_cucim["block_params"] = block_params
    img = binary_image(shape, pct_true=4)
    distances = distance_transform_edt(img, **kwargs_cucim)
    expected = ndi_cpu.distance_transform_edt(cp.asnumpy(img), **kwargs_scipy)
    cp.testing.assert_allclose(distances, expected)


@pytest.mark.parametrize("shape", [(288, 288), (300, 200), (320, 1024)])
@pytest.mark.parametrize(
    "block_params", [(6, 1, 1), (3, 4, 2), (7, 4, 3), (5, 2, 1)]
)
def test_distance_transform_edt_block_params_band_size(shape, block_params):
    """The phase 1 band size, shape // m1, must be a multiple of 32."""
    img = binary_image(shape, pct_true=4)
    distances = distance_transform_edt(img, block_params=block_params)
    expected = ndi_cpu.distance_transform_edt(cp.asnumpy(img))
    cp.testing.assert_allclose(distances, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "block_params",
    [
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),  # no elements can be < 1
        (1, 3, 1),
        (1, 5, 1),
        (1, 7, 1),  # 2nd element must be a power of 2
        (128, 1, 1),  # m1 too large for the array size
        (1, 128, 1),  # m2 too large for the array size
        (1, 1, 128),  # m3 too large for the array size
    ],
)
def test_distance_transform_edt_block_params_invalid(block_params):
    img = binary_image((512, 512), pct_true=4)
    with pytest.raises(ValueError):
        distance_transform_edt(img, block_params=block_params)


@pytest.mark.parametrize("value", [0, 1, 3])
@pytest.mark.parametrize("ndim", [2, 3])
def test_distance_transform_edt_uniform_valued(value, ndim):
    """ensure default block_params is robust to anisotropic shape."""
    img = cp.full((48,) * ndim, value, dtype=cp.uint8)
    # ensure there is at least 1 pixel at background intensity
    img[(slice(24, 25),) * ndim] = 0
    out = distance_transform_edt(img)
    expected = ndi_cpu.distance_transform_edt(cp.asnumpy(img))
    cp.testing.assert_allclose(out, expected)


@pytest.mark.parametrize("sx", list(range(16)))
@pytest.mark.parametrize("sy", list(range(16)))
def test_distance_transform_edt_2d_aniso(sx, sy):
    """ensure default block_params is robust to anisotropic shape."""
    shape = (128 + sy, 128 + sx)
    img = binary_image(shape, pct_true=80)
    out = distance_transform_edt(img)
    expected = ndi_cpu.distance_transform_edt(cp.asnumpy(img))
    cp.testing.assert_allclose(out, expected)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("sampling", [None, "iso", "aniso"])
def test_distance_transform_inplace_distance(ndim, sampling):
    img = binary_image((32,) * ndim, pct_true=80)
    distances = cp.empty(img.shape, dtype=cp.float32)
    if sampling == "iso":
        sampling = (1.5,) * ndim
    elif sampling == "aniso":
        sampling = tuple(range(1, ndim + 1))
    distance_transform_edt(img, sampling=sampling, distances=distances)
    expected = ndi_cpu.distance_transform_edt(
        cp.asnumpy(img), sampling=sampling
    )
    cp.testing.assert_allclose(distances, expected)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("sampling", [None, "iso", "aniso"])
@pytest.mark.parametrize("inplace", [False, True])
def test_distance_transform_edt_float64_distances(ndim, sampling, inplace):
    img = binary_image((40,) * ndim, pct_true=80)
    if sampling == "iso":
        sampling = (1.5,) * ndim
    elif sampling == "aniso":
        sampling = tuple(range(1, ndim + 1))
    kwargs = dict(sampling=sampling, float64_distances=True)
    if inplace:
        distances = cp.empty(img.shape, dtype=cp.float64)
        distance_transform_edt(img, distances=distances, **kwargs)
    else:
        distances = distance_transform_edt(img, **kwargs)
    assert distances.dtype == cp.float64
    expected = ndi_cpu.distance_transform_edt(
        cp.asnumpy(img), sampling=sampling
    )
    # the distances are exact to double precision unless there is a spacing
    # (the comparisons in the kernels with a spacing are in float)
    rtol = 1e-12 if sampling is None else 1e-6
    cp.testing.assert_allclose(distances, expected, rtol=rtol)


@pytest.mark.parametrize("ndim", [2, 3])
def test_distance_transform_float32_distances_default(ndim):
    img = binary_image((40,) * ndim, pct_true=80)
    assert distance_transform_edt(img).dtype == cp.float32


@pytest.mark.parametrize("ndim", [2, 3])
def test_distance_transform_inplace_distance_errors(ndim):
    img = binary_image((32,) * ndim, pct_true=80)

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
        distance_transform_edt(
            img,
            distances=distances,
            return_distances=False,
            return_indices=True,
        )


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("sampling", [None, "iso", "aniso"])
@pytest.mark.parametrize(
    "dtype", [cp.int16, cp.uint16, cp.uint32, cp.int32, cp.uint64, cp.int64]
)
@pytest.mark.parametrize("return_distances", [False, True])
def test_distance_transform_inplace_indices(
    ndim, sampling, dtype, return_distances
):
    img = binary_image((32,) * ndim, pct_true=80)
    if ndim == 3 and dtype in [cp.int16, cp.uint16]:
        pytest.skip(reason="3d case requires at least 32-bit integer output")
    if sampling == "iso":
        sampling = (1.5,) * ndim
    elif sampling == "aniso":
        sampling = tuple(range(1, ndim + 1))
    common_kwargs = dict(
        sampling=sampling,
        return_distances=return_distances,
        return_indices=True,
    )
    # verify that in-place and out-of-place results agree
    indices = cp.empty((ndim,) + img.shape, dtype=dtype)
    distance_transform_edt(img, indices=indices, **common_kwargs)
    expected = distance_transform_edt(img, **common_kwargs)
    if return_distances:
        cp.testing.assert_array_equal(indices, expected[1])
    else:
        cp.testing.assert_array_equal(indices, expected)


@pytest.mark.parametrize(
    "shape",
    [(40, 70), (33, 100), (1, 50), (33, 40, 50), (5, 30, 70), (64, 5, 33)],
)
@pytest.mark.parametrize("return_distances", [False, True])
def test_distance_transform_inplace_indices_unpadded_shape(
    shape, return_distances
):
    """The shape of `indices` is that of the input, not of the padded input."""
    ndim = len(shape)
    img = binary_image(shape, pct_true=80)
    common_kwargs = dict(return_distances=return_distances, return_indices=True)
    indices = cp.empty((ndim,) + shape, dtype=cp.int32)
    distance_transform_edt(img, indices=indices, **common_kwargs)
    expected = distance_transform_edt(img, **common_kwargs)
    if return_distances:
        expected = expected[1]
    cp.testing.assert_array_equal(indices, expected)


@pytest.mark.parametrize("ndim", [2, 3])
def test_distance_transform_inplace_indices_errors(ndim):
    img = binary_image((32,) * ndim, pct_true=80)
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


@pytest.mark.parametrize("sx", list(range(4)))
@pytest.mark.parametrize("sy", list(range(4)))
@pytest.mark.parametrize("sz", list(range(4)))
def test_distance_transform_edt_3d_aniso(sx, sy, sz):
    """ensure default block_params is robust to anisotropic shape."""
    shape = (16 + sz, 32 + sy, 48 + sx)
    img = binary_image(shape, pct_true=80)
    out = distance_transform_edt(img)
    print(f"{out.dtype=}")
    expected = ndi_cpu.distance_transform_edt(cp.asnumpy(img))
    cp.testing.assert_allclose(out, expected)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 40, 70),  # single slice
        (3, 33, 31),
        (5, 64, 64),
        (100, 32, 32),  # z longest
        (37, 200, 333),
        (8, 5, 300),  # x longest
        (300, 5, 8),  # z longest, tiny y and x
        (2, 257, 129),
        (64, 96, 160),
    ],
)
@pytest.mark.parametrize("sampling", [None, (2.0, 0.7, 1.3)])
@pytest.mark.parametrize("density", [2, 50, 98])
def test_distance_transform_edt_3d_strongly_aniso(shape, sampling, density):
    """Axes of very different size are not padded to a common size."""
    img = binary_image(shape, pct_true=density)
    # ensure there is at least one background voxel
    img[(0,) * 3] = False
    dist, ind = distance_transform_edt(
        img, sampling=sampling, return_indices=True
    )
    img_cpu = cp.asnumpy(img)
    expected = ndi_cpu.distance_transform_edt(img_cpu, sampling=sampling)
    cp.testing.assert_allclose(dist, expected, rtol=1e-6)

    # The indices may differ from SciPy's at equidistant voxels, so check that
    # they are valid background locations that reproduce the distances.
    ind = cp.asnumpy(ind)
    assert not img_cpu[tuple(ind)].any()
    scale = np.ones(3) if sampling is None else np.asarray(sampling)
    offsets = ind - np.indices(shape)
    from_indices = np.sqrt(
        sum((o * s) ** 2 for o, s in zip(offsets, scale)), dtype=np.float64
    )
    np.testing.assert_allclose(from_indices, expected, rtol=1e-5)


@pytest.mark.parametrize(
    "shape, expected",
    [
        ((64, 512, 512), None),  # nothing to pad
        ((64, 100, 70), ((0, 0), (0, 28), (0, 26))),
        ((3, 33, 31), ((0, 1), (0, 31), (0, 1))),
        ((1, 1, 1), ((0, 3), (0, 31), (0, 31))),
    ],
)
def test_pba3d_determine_padding(shape, expected):
    """Each axis is padded separately, not up to a common size."""
    from cucim.core.operations.morphology._pba_3d import _determine_padding

    # block_size = 32, blockx = 32 and blocky = 4 are the default values
    assert _determine_padding(shape, 32, 32, 4) == expected


def test_distance_transform_edt_3d_too_many_elements():
    # broadcast_to avoids allocating the (2 GiB) array
    img = cp.broadcast_to(cp.zeros((1, 1, 1), dtype=bool), (2048, 1024, 1024))
    with pytest.raises(ValueError, match="too many elements"):
        distance_transform_edt(img)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 40),
        (40, 1),
        (3, 500),
        (500, 3),
        (33, 1000),
        (1000, 33),
        (64, 4096),  # one axis much longer than the other
        (4096, 64),
        (37, 129),
        (100, 300),
        (2000, 70),
    ],
)
@pytest.mark.parametrize("sampling", [None, (2.25, 0.85)])
@pytest.mark.parametrize("density", [2, 50, 98])
def test_distance_transform_edt_2d_strongly_aniso(shape, sampling, density):
    """Axes of very different size are not padded to a common size."""
    img = binary_image(shape, pct_true=density)
    # ensure there is at least one background pixel
    img[(0,) * 2] = False
    dist, ind = distance_transform_edt(
        img, sampling=sampling, return_indices=True
    )
    img_cpu = cp.asnumpy(img)
    expected = ndi_cpu.distance_transform_edt(img_cpu, sampling=sampling)
    cp.testing.assert_allclose(dist, expected, rtol=1e-6)

    # The indices may differ from SciPy's at equidistant pixels, so check that
    # they are valid background locations that reproduce the distances.
    ind = cp.asnumpy(ind)
    assert not img_cpu[tuple(ind)].any()
    scale = np.ones(2) if sampling is None else np.asarray(sampling)
    offsets = ind - np.indices(shape)
    from_indices = np.sqrt(
        sum((o * s) ** 2 for o, s in zip(offsets, scale)), dtype=np.float64
    )
    np.testing.assert_allclose(from_indices, expected, rtol=1e-5)


@pytest.mark.parametrize(
    "shape, block_params",
    [
        # the size of the bands has to be a multiple of 32
        ((288, 288), (6, 1, 1)),
        ((300, 520), (6, 4, 2)),
        ((100, 2000), (2, 16, 3)),
        ((513, 70), (7, 2, 1)),
        ((96, 1024), (3, 8, 4)),
    ],
)
def test_distance_transform_edt_block_params_non_isotropic(shape, block_params):
    img = binary_image(shape, pct_true=4)
    distances = distance_transform_edt(img, block_params=block_params)
    expected = ndi_cpu.distance_transform_edt(cp.asnumpy(img))
    cp.testing.assert_allclose(distances, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "shape, block_params, expected",
    [
        ((256, 4096), None, None),  # nothing to pad
        ((100, 300), None, ((0, 28), (0, 20))),
        ((300, 100), None, ((0, 20), (0, 28))),
        ((1, 1), None, ((0, 31), (0, 31))),
        # m1 bands of a multiple of 32 rows each
        ((288, 288), (6, 1, 1), ((0, 96), (0, 0))),
        ((288, 288), (1, 64, 1), ((0, 0), (0, 64 - 32))),
    ],
)
def test_pba2d_determine_padding(shape, block_params, expected):
    """Each axis is padded separately, not up to a common size."""
    from cucim.core.operations.morphology._pba_2d import (
        _default_block_params,
        _determine_padding,
    )

    # block_size = 32 is the default value
    if block_params is None:
        block_params = _default_block_params(shape, 32)
    m1, m2, _ = block_params
    assert _determine_padding(shape, 32, m1, m2) == expected


@pytest.mark.parametrize(
    "shape",
    [
        (64, 4096),  # processed transposed
        (300, 1000),  # processed transposed
        (1024, 2048),  # processed transposed
        (1024, 1500),
        (2048, 4096),  # too large to be processed transposed
    ],
)
@pytest.mark.parametrize("sampling", [None, (2.25, 0.85)])
def test_distance_transform_edt_2d_orientation(shape, sampling):
    """The distances do not depend on the order of the axes."""
    img = binary_image(shape, pct_true=5)
    # (the distances are computed in the transposed orientation if the first
    # axis is short, but not if the indices are requested)
    dist = distance_transform_edt(img, sampling=sampling)
    sampling_t = None if sampling is None else sampling[::-1]
    dist_t = distance_transform_edt(
        cp.ascontiguousarray(img.T), sampling=sampling_t
    )
    cp.testing.assert_allclose(dist, dist_t.T, rtol=1e-6)

    img_cpu = cp.asnumpy(img)
    expected = ndi_cpu.distance_transform_edt(img_cpu, sampling=sampling)
    cp.testing.assert_allclose(dist, expected, rtol=1e-6)

    # the same distances are returned if the indices are also requested, and
    # these are background locations that reproduce the distances
    dist2, ind = distance_transform_edt(
        img, sampling=sampling, return_indices=True
    )
    cp.testing.assert_array_equal(dist, dist2)
    ind = cp.asnumpy(ind)
    assert not img_cpu[tuple(ind)].any()
    scale = np.ones(2) if sampling is None else np.asarray(sampling)
    offsets = ind - np.indices(shape)
    from_indices = np.sqrt(
        sum((o * s) ** 2 for o, s in zip(offsets, scale)), dtype=np.float64
    )
    np.testing.assert_allclose(from_indices, expected, rtol=1e-5)


def test_pba2d_default_block_params():
    from cucim.core.operations.morphology._pba_2d import _default_block_params

    # m1 is set by the first axis and m2 by the second axis
    assert _default_block_params((100, 300), 32) == (4, 8, 8)
    assert _default_block_params((4096, 256), 32) == (128, 8, 8)
    assert _default_block_params((256, 4096), 32) == (8, 32, 16)
    assert _default_block_params((32, 32), 32) == (1, 1, 1)
    assert _default_block_params((64, 64), 32) == (2, 2, 2)
    assert _default_block_params((8192, 8192), 32) == (256, 32, 16)


def test_distance_transform_edt_2d_too_many_elements():
    # broadcast_to avoids allocating the (2 GiB) array
    img = cp.broadcast_to(cp.zeros((1, 1), dtype=bool), (46400, 46400))
    with pytest.raises(ValueError, match="too many elements"):
        distance_transform_edt(img)


@pytest.mark.parametrize("shape", [(40, 70), (7, 5), (33, 40, 50), (6, 3, 4)])
@pytest.mark.parametrize("sampling", [None, 2.0, "aniso"])
@pytest.mark.parametrize("float64_distances", [False, True])
@pytest.mark.parametrize("return_distances", [False, True])
def test_distance_transform_edt_no_background(
    shape, sampling, float64_distances, return_distances
):
    """With no background element there is no nearest background element."""
    ndim = len(shape)
    if sampling == "aniso":
        sampling = tuple(range(1, ndim + 1))
    img = cp.ones(shape, dtype=bool)
    out = distance_transform_edt(
        img,
        sampling=sampling,
        return_distances=return_distances,
        return_indices=True,
        float64_distances=float64_distances,
    )
    if return_distances:
        distances, indices = out
        assert cp.isinf(distances).all()
    else:
        indices = out
    assert (indices < 0).all()

    if return_distances:
        # distances only (the 3D case uses a separate, fused kernel)
        distances = distance_transform_edt(
            img, sampling=sampling, float64_distances=float64_distances
        )
        assert cp.isinf(distances).all()


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize(
    "container", [list, tuple, np.asarray, cp.asarray, "scalar", "np_scalar"]
)
@pytest.mark.parametrize("aniso", [False, True])
def test_distance_transform_edt_sampling_types(ndim, container, aniso):
    """`sampling` can be a scalar, or a sequence of any type."""
    img = binary_image((24,) * ndim, pct_true=80)
    if container in ("scalar", "np_scalar"):
        if aniso:
            pytest.skip("a scalar sampling is isotropic")
        sampling = 1.5 if container == "scalar" else np.float32(1.5)
    else:
        values = (
            tuple(1.0 + 0.5 * i for i in range(ndim)) if aniso else (2,) * ndim
        )
        sampling = container(values)
    out = distance_transform_edt(img, sampling=sampling)
    expected = ndi_cpu.distance_transform_edt(
        cp.asnumpy(img),
        sampling=cp.asnumpy(sampling)
        if isinstance(sampling, cp.ndarray)
        else sampling,
    )
    cp.testing.assert_allclose(out, expected, rtol=1e-6)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize(
    "sampling", [(1.0, 2.0, 3.0, 4.0), [1.0] * 5, [[1.0, 2.0]]]
)
def test_distance_transform_edt_sampling_invalid(ndim, sampling):
    img = binary_image((8,) * ndim, pct_true=80)
    with pytest.raises(ValueError, match="sampling"):
        distance_transform_edt(img, sampling=sampling)


def test_distance_transform_edt_sampling_not_on_device(monkeypatch):
    """Checking `sampling` must not use the device (it would synchronize)."""

    def fail(*args, **kwargs):
        raise AssertionError("cp.unique should not be used")

    monkeypatch.setattr(cp, "unique", fail)
    img = binary_image((24, 24), pct_true=80)
    distance_transform_edt(img, sampling=(1.5, 1.5))
    distance_transform_edt(img, sampling=(1.5, 2.0))


def test_pba_kernels_are_cached():
    """The kernels are looked up once, not for every call."""
    from cucim.core.operations.morphology._pba_2d import _get_pba2d_kernels
    from cucim.core.operations.morphology._pba_3d import _get_pba3d_kernels

    for get_kernels, args in [
        (_get_pba2d_kernels, (32, -32768, "short2")),
        (_get_pba3d_kernels, (32, (10, 10, False))),
    ]:
        plain = get_kernels(*args, False)
        assert get_kernels(*args, False) is plain
        spacing = get_kernels(*args, True)
        assert get_kernels(*args, True) is spacing
        # the variants with a spacing are different kernels with the same keys
        assert plain.keys() == spacing.keys()
        # the kernels with a spacing in single precision are also different
        assert get_kernels(*args, True, "float") is not spacing
        assert get_kernels(*args, True, "float") is get_kernels(
            *args, True, "float"
        )
        assert (
            plain["kernelColor" if len(args) == 3 else "kernelColorAxis"]
            is not spacing[
                "kernelColor" if len(args) == 3 else "kernelColorAxis"
            ]
        )


@pytest.mark.parametrize("ndim", [2, 3])
def test_distance_transform_edt_repeated_calls(ndim):
    """Calls with and without spacing in any order use the right kernels."""
    img = binary_image((40,) * ndim, pct_true=80)
    sampling = tuple(1.0 + 0.5 * i for i in range(ndim))
    expected = ndi_cpu.distance_transform_edt(cp.asnumpy(img))
    expected_s = ndi_cpu.distance_transform_edt(
        cp.asnumpy(img), sampling=sampling
    )
    for _ in range(2):
        cp.testing.assert_allclose(
            distance_transform_edt(img), expected, rtol=1e-6
        )
        cp.testing.assert_allclose(
            distance_transform_edt(img, sampling=sampling),
            expected_s,
            rtol=1e-6,
        )


@pytest.mark.parametrize(
    "shape",
    [
        (40, 50),
        (64, 2000),
        (300, 70),
        (24, 24, 24),
        (5, 64, 100),
        (100, 20, 33),
    ],
)
@pytest.mark.parametrize("return_indices", [False, True])
@pytest.mark.parametrize("density", [2, 50])
def test_distance_transform_edt_float32_spacing(shape, return_indices, density):
    """The spacing can be used in single precision (a little less exact)."""
    ndim = len(shape)
    sampling = (2.25, 0.85, 1.3)[:ndim]
    img = binary_image(shape, pct_true=density)
    img[(0,) * ndim] = False
    out = distance_transform_edt(
        img,
        sampling=sampling,
        return_indices=return_indices,
        float32_spacing=True,
    )
    expected = ndi_cpu.distance_transform_edt(
        cp.asnumpy(img), sampling=sampling
    )
    distances = out[0] if return_indices else out
    cp.testing.assert_allclose(distances, expected, rtol=1e-5)
    if return_indices:
        # the indices are background locations that reproduce the distances
        ind = cp.asnumpy(out[1])
        assert not cp.asnumpy(img)[tuple(ind)].any()
        offsets = ind - np.indices(shape)
        from_indices = np.sqrt(
            sum((o * s) ** 2 for o, s in zip(offsets, sampling))
        )
        np.testing.assert_allclose(from_indices, expected, rtol=1e-5)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("sampling", [None, 2.0, "iso"])
def test_distance_transform_edt_float32_spacing_unused(ndim, sampling):
    """float32_spacing has no effect if there is no (anisotropic) spacing."""
    img = binary_image((32,) * ndim, pct_true=80)
    if sampling == "iso":
        sampling = (1.5,) * ndim
    default = distance_transform_edt(img, sampling=sampling)
    out = distance_transform_edt(img, sampling=sampling, float32_spacing=True)
    cp.testing.assert_array_equal(out, default)


@pytest.mark.parametrize(
    "shape, expected",
    [
        ((32, 32, 32), (5, 5, False)),
        ((1024, 1024, 1024), (10, 10, False)),  # 30 bits
        ((1, 1, 1), (1, 1, False)),
        ((64, 2048, 2048), (11, 6, False)),  # a long axis uses fewer bits
        ((4, 1100, 40), (11, 2, False)),
        ((2, 8192, 8192), (13, 1, False)),
        ((1025, 1025, 1025), (11, 11, True)),  # 33 bits
        ((300, 1100, 20), (11, 9, True)),
    ],
)
def test_pba3d_encoding(shape, expected):
    """The number of bits of the coordinates follows from the shape."""
    from cucim.core.operations.morphology._pba_3d import _get_encoding

    assert _get_encoding(shape) == expected


def test_pba3d_encoding_too_large():
    from cucim.core.operations.morphology._pba_3d import _get_encoding

    with pytest.raises(ValueError):
        _get_encoding((2**21, 2**21, 2**21))


@pytest.mark.parametrize("shape", [(5, 17, 9), (3, 1100, 6), (300, 1100, 20)])
def test_pba3d_encode_decode(shape):
    """Encoding and decoding coordinates is the identity for 32 and 64 bits."""
    from cucim.core.operations.morphology._pba_3d import (
        _get_encoding,
        decode3d,
        encode3d,
    )

    encoding = _get_encoding(shape)
    # shapes that use a 32-bit and a 64-bit integer are included
    assert encoding[2] == (shape == (300, 1100, 20))
    arr = cp.zeros(shape, dtype=bool)
    encoded = encode3d(arr, encoding=encoding)
    assert encoded.dtype == (cp.int64 if encoding[2] else cp.int32)
    x, y, z = decode3d(encoded, encoding)
    zz, yy, xx = cp.meshgrid(*(cp.arange(n) for n in shape), indexing="ij")
    cp.testing.assert_array_equal(x, xx)
    cp.testing.assert_array_equal(y, yy)
    cp.testing.assert_array_equal(z, zz)


@pytest.mark.parametrize(
    "shape",
    [
        (3, 1500, 70),  # 32-bit integers, although an axis is larger than 1024
        (4, 1100, 40),
        (1, 40, 2000),
        (300, 1100, 20),  # needs 64-bit integers
        (260, 1025, 10),
        (20, 1030, 1030),
    ],
)
@pytest.mark.parametrize("sampling", [None, (2.0, 0.7, 1.3)])
def test_distance_transform_edt_3d_long_axis(shape, sampling):
    img = binary_image(shape, pct_true=30)
    img[(0,) * 3] = False
    dist, ind = distance_transform_edt(
        img, sampling=sampling, return_indices=True
    )
    img_cpu = cp.asnumpy(img)
    expected = ndi_cpu.distance_transform_edt(img_cpu, sampling=sampling)
    cp.testing.assert_allclose(dist, expected, rtol=1e-6)
    cp.testing.assert_allclose(
        distance_transform_edt(img, sampling=sampling), expected, rtol=1e-6
    )
    assert not img_cpu[tuple(cp.asnumpy(ind))].any()


@pytest.mark.parametrize("ndim", [1, 4, 5])
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
