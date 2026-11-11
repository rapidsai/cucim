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

from cucim.skimage._vendored import (
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
