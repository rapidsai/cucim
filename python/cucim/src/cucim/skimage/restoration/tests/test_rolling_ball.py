# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause


"""
Tests for Rolling Ball Filter
(skimage.restoration.rolling_ball)
"""

import math

import cupy as cp
import numpy as np
import pytest
from skimage import data
from skimage.restoration import (
    ellipsoid_kernel as skimage_ellipsoid_kernel,
    rolling_ball as skimage_rolling_ball,
)

from cucim.skimage import util
from cucim.skimage.metrics import normalized_root_mse
from cucim.skimage.restoration import (
    ball_kernel,
    ellipsoid_kernel,
    rolling_ball,
)


def _ball_kernel_reference(
    radius, ndim, dtype=np.float64, structure_and_footprint=False
):
    """Simple numpy-based reference implementation of the ball kernel.

    used to validate ball_kernel elementwise kernel implementation
    """
    dtype = np.dtype(dtype)
    if dtype.kind != "f":
        raise ValueError("dtype must be a floating-point type")

    half_size = math.ceil(radius)
    coords = np.meshgrid(
        *[
            np.arange(-half_size, half_size + 1, dtype=dtype)
            for _ in range(ndim)
        ],
        indexing="ij",
    )
    sum_of_squares = sum(c**2 for c in coords)
    distance_from_center = np.sqrt(sum_of_squares)
    kernel = np.sqrt(np.clip(radius**2 - sum_of_squares, 0, None))
    if structure_and_footprint:
        center_height = radius
        kernel -= center_height
        footprint = np.zeros(kernel.shape, dtype=bool)
        footprint[distance_from_center <= radius] = True
        kernel[~footprint] = np.inf
        return cp.asarray(kernel), cp.asarray(footprint)
    else:
        kernel[distance_from_center > radius] = np.inf
        return cp.asarray(kernel), None


def _ellipsoid_kernel_reference(
    shape, intensity, dtype=np.float64, structure_and_footprint=False
):
    """Simple numpy-based reference implementation of the ellipsoid kernel.

    Used to validate ellipsoid_kernel elementwise kernel implementation.
    """
    shape = np.asarray(shape)
    semi_axis = np.clip(shape // 2, 1, None)
    dtype = np.dtype(dtype)
    if dtype.kind != "f":
        raise ValueError("dtype must be a floating-point type")

    grids = [np.arange(-x, x + 1, dtype=dtype) for x in semi_axis]
    kernel_coords = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1)

    norm_sq = np.sum((kernel_coords / semi_axis) ** 2, axis=-1)
    intensity_scaling = 1 - norm_sq
    kernel = intensity * np.sqrt(np.clip(intensity_scaling, 0, None))
    if structure_and_footprint:
        kernel -= intensity
        footprint = np.zeros(kernel.shape, dtype=bool)
        footprint[intensity_scaling >= 0] = True
        kernel[~footprint] = np.inf
        return cp.asarray(kernel), cp.asarray(footprint)
    else:
        kernel[intensity_scaling < 0] = np.inf
        return cp.asarray(kernel), None


# ---------------------------------------------------------------------------
# Tests comparing ball_kernel / ellipsoid_kernel to reference implementations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("radius", [1, 2, 2.5, 5, 10])
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ball_kernel_vs_reference(radius, ndim, dtype):
    """ball_kernel output matches the numpy-based reference implementation."""
    ref, fp_ref = _ball_kernel_reference(
        radius, ndim, dtype=dtype, structure_and_footprint=True
    )
    structure, fp = ball_kernel(
        radius, ndim, dtype=cp.dtype(dtype), structure_and_footprint=True
    )
    assert fp.shape == ref.shape
    mask = cp.isfinite(structure)
    cp.testing.assert_allclose(structure[mask], ref[mask], rtol=1e-5, atol=1e-5)
    cp.testing.assert_array_equal(cp.isposinf(structure), cp.isposinf(ref))
    cp.testing.assert_array_equal(fp, fp_ref)


def test_float_radius_matches_skimage_without_downscaling():
    img_cpu = np.random.default_rng(0).integers(
        0, 256, size=(16, 16), dtype=np.uint8
    )
    img = cp.asarray(img_cpu)

    out = rolling_ball(img, radius=2.5, downscale=None)
    expected = skimage_rolling_ball(img_cpu, radius=2.5)

    cp.testing.assert_allclose(out, expected)


@pytest.mark.parametrize("shape", [(5, 5), (3, 7), (4, 4, 4)])
@pytest.mark.parametrize("intensity", [1.0, 50.0, 100.0])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ellipsoid_kernel_vs_reference(shape, intensity, dtype):
    """ellipsoid_kernel output matches the numpy-based reference implementation."""
    ref, fp_ref = _ellipsoid_kernel_reference(
        shape, intensity, dtype=dtype, structure_and_footprint=True
    )
    structure, fp = ellipsoid_kernel(
        shape, intensity, dtype=cp.dtype(dtype), structure_and_footprint=True
    )
    assert fp.shape == ref.shape
    mask = cp.isfinite(structure)
    cp.testing.assert_allclose(structure[mask], ref[mask], rtol=1e-5, atol=1e-5)
    cp.testing.assert_array_equal(cp.isposinf(structure), cp.isposinf(ref))
    cp.testing.assert_array_equal(fp, fp_ref)


# -------------------------------
# Tests adapted from scikit-image
# -------------------------------


@pytest.mark.parametrize(
    "dtype", [np.uint8, np.int32, np.float16, np.float32, np.float64]
)
@pytest.mark.parametrize("structure_and_footprint", [False, True])
def test_ellipsoid_const(dtype, structure_and_footprint):
    img = cp.full((100, 100), 155, dtype=dtype)
    if structure_and_footprint:
        structure, footprint = ellipsoid_kernel(
            (25, 53), 50, structure_and_footprint=structure_and_footprint
        )
        assert footprint.dtype == bool
        background = rolling_ball(img, kernel=(structure, footprint))
    else:
        kernel = ellipsoid_kernel((25, 53), 50)
        background = rolling_ball(img, kernel=kernel)
    assert cp.allclose(img - background, np.zeros_like(img))
    assert background.dtype == img.dtype


@pytest.mark.skip(
    reason="nansafe=True requires mask support in vendored ndimage"
)
def test_nan_const():
    img_cpu = np.full((100, 100), 123, dtype=float)
    img_cpu[20, 20] = np.nan
    img_cpu[50, 53] = np.nan
    img = cp.asarray(img_cpu)

    kernel_shape = (10, 10)
    nan_spread_like_in_skimage = False
    if nan_spread_like_in_skimage:
        # For scikit-image, any NaN pixels contaminate a radius around them.
        x = np.arange(-kernel_shape[1] // 2, kernel_shape[1] // 2 + 1)
        x = x[np.newaxis, :]
        y = np.arange(-kernel_shape[0] // 2, kernel_shape[0] // 2 + 1)
        y = y[:, np.newaxis]
        expected_img = np.zeros_like(img_cpu)
        expected_img[y + 20, x + 20] = np.nan
        expected_img[y + 50, x + 53] = np.nan
    else:
        # No spreading of NaN values.
        expected_img = np.zeros_like(img)
        expected_img[np.isnan(img_cpu)] = np.nan
    expected_img = cp.asarray(expected_img)

    kernel = ellipsoid_kernel(kernel_shape, 100)
    background = rolling_ball(img, kernel=kernel, nansafe=True)
    assert cp.allclose(img - background, expected_img, equal_nan=True)


@pytest.mark.parametrize("radius", [1, 2.5, 10.346, 50])
def test_const_image(radius):
    # infinite plane light source at top left corner
    img = cp.full((100, 100), 23, dtype=np.uint8)
    background = rolling_ball(img, radius=radius)
    assert cp.allclose(img - background, np.zeros_like(img))


def test_radial_gradient():
    # spot light source at top left corner
    spot_radius = 50
    x, y = np.meshgrid(range(5), range(5))
    img = np.sqrt(np.clip(spot_radius**2 - y**2 - x**2, 0, None))
    img = cp.asarray(img)

    background = rolling_ball(img, radius=5)
    assert cp.allclose(img - background, cp.zeros_like(img))


def test_linear_gradient():
    # linear light source centered at top left corner
    x, y = np.meshgrid(range(100), range(100))
    img = cp.asarray(y * 20 + x * 20)

    expected_img = cp.full_like(img, 19)
    expected_img[0, 0] = 0

    background = rolling_ball(img, radius=1)
    assert cp.allclose(img - background, expected_img)


@pytest.mark.parametrize("radius", [2, 10, 12.5, 50])
def test_preserve_peaks(radius):
    x, y = np.meshgrid(range(100), range(100))
    img = 0 * x + 0 * y + 10
    img[10, 10] = 20
    img[20, 20] = 35
    img[45, 26] = 156
    img = cp.asarray(img)

    expected_img = img - 10
    background = rolling_ball(img, radius=radius)
    assert cp.allclose(img - background, expected_img)


@pytest.mark.parametrize("workers", [None, 2])
def test_workers(workers):
    # workers is unused, just verifying that the API allows specifying it
    img = cp.full((100, 100), 23, dtype=np.uint8)
    rolling_ball(img, radius=10, workers=workers)
    rolling_ball(img, radius=10, nansafe=False, workers=workers)


def test_num_threads_deprecated():
    img = cp.full((100, 100), 23, dtype=np.uint8)

    with pytest.warns(FutureWarning, match="`num_threads` is deprecated"):
        rolling_ball(img, radius=10, num_threads=2)


def test_ndim():
    image = data.cells3d()[:5, 1, ...]
    kernel_args = ((3, 50, 50), 50)
    kernel = ellipsoid_kernel(*kernel_args)
    out = rolling_ball(image, kernel=kernel)

    # validate kernel against scikit-image
    skimage_kernel = skimage_ellipsoid_kernel(*kernel_args)
    mask = np.isfinite(skimage_kernel)
    cp.testing.assert_allclose(
        kernel[mask], skimage_kernel[mask], atol=1e-5, rtol=1e-5
    )

    # validate filtering result against scikit-image
    out_cpu = skimage_rolling_ball(cp.asnumpy(image), kernel=skimage_kernel)
    cp.testing.assert_allclose(out, out_cpu)


# --------------------------------
# Extra tests covering downscaling
# --------------------------------


@pytest.mark.parametrize("downscale_factor", [2, 3, 4])
def test_downscale_nrmse_shape_and_dtype(downscale_factor):
    """With downscaling, computation is faster and output shape/dtype match no-downscale case."""

    tmp = util.invert(cp.asarray(data.page()))
    tmp = cp.concatenate([tmp, tmp[::-1, :], tmp, tmp[::-1, :]], axis=0)
    img = cp.concatenate([tmp, tmp[:, ::-1]], axis=1)
    shape = img.shape

    radius = 60

    # Baseline: no downscaling
    background_full = rolling_ball(img, radius=radius, downscale=None)

    # With downscaling (e.g. 4x smaller each dimension)
    background_down = rolling_ball(
        img, radius=radius, downscale=downscale_factor
    )

    nrmse = normalized_root_mse(background_full, background_down)

    # result with and without downscaling should be similar
    assert nrmse < 0.5

    assert background_down.shape == shape, (
        f"output shape with downscale must match input shape, got {background_down.shape}"
    )
    assert background_down.dtype == background_full.dtype == img.dtype, (
        f"output dtype must match input and no-downscale result, got {background_down.dtype}"
    )


def test_downscale_invalid():
    """downscale < 1 must raise ValueError."""
    img = cp.full((64, 64), 23, dtype=np.uint8)
    with pytest.raises(ValueError, match="downscale must be >= 1"):
        rolling_ball(img, radius=10, downscale=0)
    with pytest.raises(ValueError, match="downscale must be >= 1"):
        rolling_ball(img, radius=10, downscale=0.5)
