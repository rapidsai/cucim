# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pytest

from cucim.skimage.transform import pyramids
from cucim.skimage.transform._pyramid_kernels import _gaussian_decimate
from cucim.skimage.transform._warps import resize


@pytest.mark.parametrize(
    "shape,channel_axis",
    [((8, 10), None), ((8, 10, 3), -1), ((3, 8, 10), 0), ((4, 6, 8), None)],
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("mode", ["reflect", "constant", "wrap"])
@pytest.mark.parametrize("sigma", [0, 2 / 3, 1.25, 4])
def test_gaussian_decimate_matches_smooth_resize(
    shape, channel_axis, dtype, mode, sigma
):
    image = cp.asarray(
        np.random.default_rng(0).standard_normal(shape).astype(dtype)
    )
    axes = tuple(
        ax
        for ax in range(image.ndim)
        if channel_axis is None or ax != channel_axis % image.ndim
    )
    smoothed = pyramids._smooth(image, sigma, mode, -2.0, channel_axis)
    out_shape = tuple(d // 2 if ax in axes else d for ax, d in enumerate(shape))
    expected = resize(
        smoothed,
        out_shape,
        order=1,
        mode=mode,
        cval=-2,
        preserve_range=True,
        anti_aliasing=False,
    )
    actual = _gaussian_decimate(image, sigma, axes, mode, -2.0)
    cp.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    public_result = pyramids.pyramid_reduce(
        image,
        sigma=sigma,
        mode=mode,
        cval=-2.0,
        channel_axis=channel_axis,
        preserve_range=True,
    )
    cp.testing.assert_array_equal(public_result, actual)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("mode", ["reflect", "constant", "wrap"])
@pytest.mark.parametrize("sigma", [0, 2 / 3])
def test_gaussian_decimate_nonfinite(value, mode, sigma):
    image = cp.ones((12, 14, 3), dtype=cp.float32)
    image[0, 1, 1] = value
    smoothed = pyramids._smooth(image, sigma, mode, 0, -1)
    expected = resize(
        smoothed, (6, 7, 3), order=1, mode=mode, anti_aliasing=False
    )
    actual = _gaussian_decimate(image, sigma, (0, 1), mode, 0)
    cp.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(
    "case",
    [
        "odd",
        "strided",
        "sigma_sequence",
        "large_sigma",
        "scale",
        "order",
        "nonfinite_cval",
    ],
)
def test_gaussian_decimate_fallback(case, monkeypatch):
    image = cp.arange(16 * 20, dtype=cp.float32).reshape(16, 20) / 100
    kwargs = dict(downscale=2, sigma=2 / 3, order=1, mode="reflect", cval=0)
    if case == "odd":
        image = image[:-1]
    elif case == "strided":
        image = image[:, ::2]
    elif case == "sigma_sequence":
        kwargs["sigma"] = (0.5, 1.0)
    elif case == "large_sigma":
        kwargs["sigma"] = 5
    elif case == "scale":
        kwargs["downscale"] = 3
    elif case == "order":
        kwargs["order"] = 3
    else:
        kwargs["cval"] = np.nan

    def fail_fused(*args, **kwargs):
        pytest.fail("unsupported input dispatched to Gaussian decimation")

    monkeypatch.setattr(pyramids, "_gaussian_decimate", fail_fused)
    expected_shape = tuple(
        int(np.ceil(d / kwargs["downscale"])) for d in image.shape
    )
    smoothed = pyramids._smooth(
        image, kwargs["sigma"], kwargs["mode"], kwargs["cval"], None
    )
    expected = resize(
        smoothed,
        expected_shape,
        order=kwargs["order"],
        mode=kwargs["mode"],
        cval=kwargs["cval"],
        preserve_range=True,
        anti_aliasing=False,
    )
    actual = pyramids.pyramid_reduce(image, preserve_range=True, **kwargs)
    cp.testing.assert_array_equal(actual, expected)
