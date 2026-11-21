# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact constant-extension boundaries ('nearest', 'grid-constant').

The interpolation functions prefilter these modes with exact boundary
initializations and evaluate coefficients outside the array in closed form,
instead of padding by 12 samples. The reference is the infinite extension,
approximated by padding far enough that the truncation is below float64
rounding.
"""

import cupy as cp
import numpy as np
import pytest
from scipy import ndimage as scipy_ndi

from cucim.skimage import transform
from cucim.skimage._vendored import (
    _ndimage_interp_kernels as kernels,
    _ndimage_interpolation as interpolation,
    _ndimage_spline_prefilter_core as core,
    ndimage as cndi,
)

_PAD = 64
_CVAL = 0.4


def _exact_coefficients(image, order, mode, cval=_CVAL, batch_axes=()):
    pad_width = [
        (0, 0) if a in batch_axes else (_PAD, _PAD) for a in range(image.ndim)
    ]
    if mode == "nearest":
        padded = np.pad(image, pad_width, mode="edge")
    else:
        padded = np.pad(image, pad_width, mode="constant", constant_values=cval)
    coef = padded.astype(np.float64)
    for axis in range(image.ndim):
        if axis not in batch_axes:
            coef = scipy_ndi.spline_filter1d(coef, order, axis, mode="mirror")
    return coef


def _exact_map_coordinates(image, coords, order, mode, cval=_CVAL):
    coef = _exact_coefficients(image, order, mode, cval)
    return scipy_ndi.map_coordinates(
        coef, coords + _PAD, order=order, prefilter=False, mode="nearest"
    )


def _random_coords(shape, n, rng, margin):
    lo = [-margin] * len(shape)
    hi = [s - 1 + margin for s in shape]
    return rng.uniform(lo, hi, (n, len(shape))).T


@pytest.mark.parametrize("order", [2, 3, 4, 5])
@pytest.mark.parametrize("nearest", [True, False])
def test_exterior_coefficient_terms(order, nearest):
    roots, U = core.exterior_coefficient_terms(order, nearest)
    U = np.asarray(U)
    for m in range(0, 6):
        E = sum(U[r] * roots[r] ** m for r in range(len(roots)))
        if m == 0:
            # the tail at m = 0 is the first in-domain coefficient
            np.testing.assert_allclose(E, np.eye(len(E))[0], atol=1e-14)
        if nearest:
            # a constant line is reproduced
            np.testing.assert_allclose(E.sum(), 1.0, atol=1e-13)


@pytest.mark.parametrize("order", [2, 3, 4, 5])
@pytest.mark.parametrize("mode", ["nearest", "grid-constant"])
@pytest.mark.parametrize(
    "shape",
    [
        (23, 31),  # strided axis (segmented kernel), short last axis
        (40, 70),  # contiguous last axis >= 32 (warp kernel)
        (6, 9, 40),  # 3D with an axis of exactly order + 1 samples
    ],
)
def test_map_coordinates_exact(order, mode, shape):
    rng = np.random.default_rng(5)
    image = rng.standard_normal(shape)
    coords = np.concatenate(
        [
            _random_coords(shape, 200, rng, 0.5),
            _random_coords(shape, 200, rng, 20.0),  # far outside as well
        ],
        axis=1,
    )
    expected = _exact_map_coordinates(image, coords, order, mode)
    image_gpu = cp.asarray(image)
    assert interpolation._use_exact_constant_boundary(
        image_gpu, mode, order, True, None
    )
    actual = cndi.map_coordinates(
        image_gpu, cp.asarray(coords), order=order, mode=mode, cval=_CVAL
    )
    cp.testing.assert_allclose(actual, expected, rtol=0, atol=1e-11)


@pytest.mark.parametrize("order", [3, 5])
@pytest.mark.parametrize("mode", ["nearest", "grid-constant"])
def test_matches_scipy_inside(order, mode):
    """SciPy's 12-sample padding is accurate inside the array."""
    rng = np.random.default_rng(6)
    image = rng.standard_normal((30, 45))
    coords = _random_coords(image.shape, 500, rng, 0.0)
    expected = scipy_ndi.map_coordinates(
        image, coords, order=order, mode=mode, cval=_CVAL
    )
    actual = cndi.map_coordinates(
        cp.asarray(image),
        cp.asarray(coords),
        order=order,
        mode=mode,
        cval=_CVAL,
    )
    cp.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)


@pytest.mark.parametrize("order", [2, 3, 5])
@pytest.mark.parametrize("mode", ["nearest", "grid-constant"])
def test_shift_zoom_affine_exact(order, mode):
    rng = np.random.default_rng(7)
    image = rng.standard_normal((37, 52))
    image_gpu = cp.asarray(image)
    kw = dict(order=order, mode=mode, cval=_CVAL)
    # shift: output[i] = input[i - s]
    s = (2.3, -4.6)
    out = cndi.shift(image_gpu, s, **kw)
    grid = np.indices(image.shape, dtype=float)
    coords = grid - np.asarray(s)[:, None, None]
    expected = _exact_map_coordinates(image, coords, order, mode)
    cp.testing.assert_allclose(out, expected, rtol=0, atol=1e-11)
    # affine with a rotation, reaching outside the array
    theta = np.deg2rad(17)
    mat = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    offset = np.array([-5.0, 8.0])
    out = cndi.affine_transform(image_gpu, mat, offset, **kw)
    coords = np.tensordot(mat, grid, axes=1) + offset[:, None, None]
    expected = _exact_map_coordinates(image, coords, order, mode)
    cp.testing.assert_allclose(out, expected, rtol=0, atol=1e-11)
    # zoom (grid_mode reaches half a pixel outside the array)
    out = cndi.zoom(image_gpu, 1.7, grid_mode=True, **kw)
    zooms = [s_in / s_out for s_in, s_out in zip(image.shape, out.shape)]
    grid_out = np.indices(out.shape, dtype=float)
    coords = np.stack([(g + 0.5) * z - 0.5 for g, z in zip(grid_out, zooms)])
    expected = _exact_map_coordinates(image, coords, order, mode)
    cp.testing.assert_allclose(out, expected, rtol=0, atol=1e-11)


@pytest.mark.parametrize("mode", ["nearest", "grid-constant"])
def test_batch_axes_and_channels(mode):
    """Looped last batch axis (RGB) and the interleaved warp prefilter."""
    order = 3
    rng = np.random.default_rng(8)
    image = rng.standard_normal((33, 47, 3))
    out = cndi.shift(
        cp.asarray(image), (1.5, -30.25, 0), order=order, mode=mode, cval=_CVAL
    )
    coef = _exact_coefficients(image, order, mode, batch_axes=(2,))
    grid = np.indices(image.shape[:2], dtype=float)
    coords = grid - np.asarray([1.5, -30.25])[:, None, None]
    expected = np.stack(
        [
            scipy_ndi.map_coordinates(
                coef[..., ch],
                coords + _PAD,
                order=order,
                prefilter=False,
                mode="nearest",
            )
            for ch in range(3)
        ],
        axis=-1,
    )
    cp.testing.assert_allclose(out, expected, rtol=0, atol=1e-11)


@pytest.mark.parametrize("mode", ["nearest", "grid-constant"])
def test_integer_output(mode):
    rng = np.random.default_rng(9)
    image = rng.integers(0, 255, (40, 40)).astype(np.uint8)
    out = cndi.shift(
        cp.asarray(image), (3.5, -2.25), order=3, mode=mode, cval=10
    )
    grid = np.indices(image.shape, dtype=float)
    coords = grid - np.asarray([3.5, -2.25])[:, None, None]
    expected = _exact_map_coordinates(
        image.astype(float), coords, 3, mode, cval=10
    )
    # Spline overshoot outside [0, 255] is cast without clipping (as for the
    # other modes), so compare only representable values.
    valid = (expected > 0.5) & (expected < 254.5)
    expected = np.rint(expected)
    # float32 working precision may flip a few exact .5 roundings
    diff = np.abs(out.get().astype(int) - expected.astype(int))[valid]
    assert int(diff.max()) <= 1


@pytest.mark.parametrize("order", [3, 5])
def test_short_axis_falls_back_to_padding(order):
    image = cp.asarray(np.random.default_rng(10).standard_normal((order, 30)))
    _, npad, exact = interpolation._filter_input(
        image, True, "nearest", 0.0, order
    )
    assert not exact
    assert npad == 12


def test_public_spline_filter_unchanged():
    """spline_filter1d keeps SciPy's behavior for 'nearest'."""
    image = np.random.default_rng(11).standard_normal((20, 30))
    for mode in ("nearest", "grid-constant"):
        expected = scipy_ndi.spline_filter(image, 3, mode=mode)
        actual = cndi.spline_filter(cp.asarray(image), 3, mode=mode)
        cp.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    "cval", [np.nan, float("nan"), np.float32("nan"), np.inf, -np.inf]
)
def test_rotate_with_nonfinite_constant_fill(cval):
    image = cp.arange(64, dtype=cp.float64).reshape(8, 8)
    kwargs = dict(order=3, mode="constant", cval=cval, clip=False)
    # Identity rotation skips all prefilter axes, so it uses no fill values.
    cp.testing.assert_array_equal(transform.rotate(image, 0, **kwargs), image)
    # The padded cubic prefilter propagates the nonfinite fill.
    rotated = transform.rotate(image, 15, **kwargs)
    assert cp.isnan(rotated).all()


@pytest.mark.parametrize("order", [2, 3, 4, 5])
@pytest.mark.parametrize("mode", ["nearest", "grid-constant"])
@pytest.mark.parametrize(
    "dtype, coords_dtype",
    [
        (np.float32, np.float32),
        (np.float32, np.float64),
        (np.float64, np.float64),
    ],
)
def test_exact_extension_beyond_index_range(order, mode, dtype, coords_dtype):
    image = cp.arange(8, dtype=dtype)
    coords = cp.array(
        [[-1e30, -(2**31), -(2**31) + 1, 2**31 - 1, 2**31, 1e30]],
        dtype=coords_dtype,
    )
    out = cndi.map_coordinates(image, coords, order=order, mode=mode, cval=-3)
    # All nonconstant tails have decayed to zero. Only the constant
    # extension of the original samples remains.
    expected = [0, 0, 0, 7, 7, 7] if mode == "nearest" else [-3] * 6
    cp.testing.assert_allclose(out, expected, rtol=0, atol=2e-6)


@pytest.mark.parametrize("channels", [None, 3])
@pytest.mark.parametrize("force_serial", [False, True])
def test_zoom_reuses_interpolation_and_prefilter_kernels(
    channels, force_serial, monkeypatch
):
    if force_serial:
        monkeypatch.setattr(core, "supports_warp_spline1d", lambda *args: False)
        monkeypatch.setattr(
            core, "supports_segmented_spline1d", lambda *args: False
        )

    def record_getter(module, name):
        records = []
        original = getattr(module, name)

        def record(*args, **kwargs):
            kernel = original(*args, **kwargs)
            records.append(kernel)
            return kernel

        monkeypatch.setattr(module, name, record)
        return records

    interpolation_kernels = record_getter(kernels, "_get_interp_kernel")
    prefilter_kernels = [
        record_getter(core, name)
        for name in (
            "get_raw_spline1d_kernel",
            "get_segmented_spline1d_kernel",
            "get_warp_spline1d_kernel",
        )
    ]

    rng = np.random.default_rng(531)
    spatial_shape = (40, 70)
    batch_shape = () if channels is None else (channels,)
    image = rng.normal(size=spatial_shape + batch_shape).astype(np.float32)
    image_gpu = cp.asarray(image)
    for shape, cval in [((53, 91), -0.375), ((61, 111), 0.625)]:
        factors = tuple(out / src for src, out in zip(spatial_shape, shape))
        actual = cndi.zoom(
            image_gpu,
            factors + (1,) * len(batch_shape),
            order=3,
            grid_mode=True,
            mode="grid-constant",
            cval=cval,
        )
        coords = np.stack(
            [
                (grid + 0.5) * (src / out) - 0.5
                for grid, src, out in zip(
                    np.indices(shape), spatial_shape, shape
                )
            ]
        )
        if channels is None:
            expected = _exact_map_coordinates(
                image, coords, 3, "grid-constant", cval
            )
        else:
            expected = np.stack(
                [
                    _exact_map_coordinates(
                        image[..., j], coords, 3, "grid-constant", cval
                    )
                    for j in range(channels)
                ],
                axis=-1,
            )
        cp.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-5)

    assert len(interpolation_kernels) == 2
    assert interpolation_kernels[0] is interpolation_kernels[1]
    assert sum(map(len, prefilter_kernels)) == 4
    for records in prefilter_kernels:
        midpoint = len(records) // 2
        assert records[:midpoint] == records[midpoint:]


@pytest.mark.parametrize(
    "dtype, cval, expected",
    [
        (cp.float32, 1.0 + 2.0**-30, 1.0),
        (cp.float64, 1.0 + 2.0**-30, 1.0 + 2.0**-30),
        (cp.float32, 1e40, np.inf),
        (cp.float64, 1e40, 1e40),
    ],
)
def test_sampler_fill_output_precision(dtype, cval, expected):
    with np.errstate(over="raise"):
        actual = cndi.map_coordinates(
            cp.zeros(2, dtype=cp.float32),
            cp.asarray([[-0.5]]),
            order=1,
            mode="constant",
            cval=cval,
            output=dtype,
            prefilter=False,
        )
    cp.testing.assert_array_equal(actual, cp.asarray([expected], dtype=dtype))


@pytest.mark.parametrize("dtype", [cp.uint8, cp.int16])
@pytest.mark.parametrize(
    "cval, literal",
    [
        (np.nan, "CUDART_NAN"),
        (np.inf, "CUDART_INF"),
        (-np.inf, "(-CUDART_INF)"),
    ],
)
def test_grid_constant_nonfinite_integer_fill(dtype, cval, literal):
    # Reproduce the old output-type conversion on the running CUDA compiler.
    legacy_cast = cp.ElementwiseKernel(
        "",
        "Y out",
        f"out = (Y){literal};",
        "test_interp_legacy_integer_fill",
        preamble="#include <cupy/math_constants.h>",
    )
    fill = cp.empty(1, dtype=dtype)
    legacy_cast(fill)
    expected = cp.rint(fill.astype(cp.float64) * 0.5).astype(dtype)
    actual = cndi.map_coordinates(
        cp.zeros(2, dtype=cp.float64),
        cp.asarray([[-0.5]]),
        order=1,
        mode="grid-constant",
        cval=cval,
        output=dtype,
        prefilter=False,
    )
    cp.testing.assert_array_equal(actual, expected)
