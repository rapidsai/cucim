# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pytest
from cupyx import allow_synchronize

from cucim.skimage.transform._warps import _clip_warp_output

pytestmark = pytest.mark.filterwarnings(
    "ignore:cupyx.allow_synchronize will be removed:UserWarning"
)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("mode", ["reflect", "constant"])
@pytest.mark.parametrize("input_nan", [False, True])
@pytest.mark.parametrize("output_nan", [False, True])
@pytest.mark.parametrize("cval", [-1, 1.5, 3, np.nan])
def test_clip_warp_output_without_synchronizing(
    dtype, mode, input_nan, output_nan, cval
):
    image = np.array([1, 2, np.nan] if input_nan else [1, 2], dtype=dtype)
    output = np.array([-2, 0, 1, 1.5, 2, 3, 4], dtype=dtype)
    if output_nan:
        output[-1] = np.nan

    min_val, max_val = np.nanmin(image).item(), np.nanmax(image).item()
    min_func, max_func = (
        (np.nanmin, np.nanmax) if input_nan else (np.min, np.max)
    )
    if (
        mode == "constant"
        and not min_val <= cval <= max_val
        and min_func(output) <= cval <= max_func(output)
    ):
        min_val = min(min_val, image.dtype.type(cval))
        max_val = max(max_val, image.dtype.type(cval))
    expected = np.clip(output, min_val, max_val)

    image_gpu, output_gpu = cp.asarray(image), cp.asarray(output)
    with allow_synchronize(False):
        _clip_warp_output(image_gpu, output_gpu, mode, cval, 3, True)
    cp.testing.assert_array_equal(output_gpu, expected)


@pytest.mark.parametrize("mode", ["reflect", "constant"])
@pytest.mark.parametrize("cval", [-1, 0, 3])
@pytest.mark.parametrize("device_bounds", [False, True])
def test_clip_warp_output_with_precomputed_bounds(mode, cval, device_bounds):
    bounds = (1.0, 2.0)
    if device_bounds:
        bounds = tuple(cp.asarray(value) for value in bounds)
    output = cp.asarray([0.5, 1.5, 3.0])
    max_val = 3 if mode == "constant" and cval == 3 else 2
    with allow_synchronize(False):
        _clip_warp_output(bounds, output, mode, cval, 3, True)
    cp.testing.assert_array_equal(output, [1, 1.5, max_val])


@pytest.mark.parametrize("mode", ["reflect", "constant"])
def test_clip_warp_output_unused_cval(mode):
    image = cp.asarray([1.0, 2.0])
    output = cp.asarray([0.5, 1.5, 2.5])
    with allow_synchronize(False):
        _clip_warp_output(image, output, mode, 0, 3, True)
    cp.testing.assert_array_equal(output, [1, 1.5, 2])


@pytest.mark.parametrize("mode", ["reflect", "constant"])
def test_clip_warp_output_all_nan(mode):
    image = cp.full(4, cp.nan)
    output = cp.full(4, cp.nan)
    with allow_synchronize(False):
        _clip_warp_output(image, output, mode, 0, 3, True)
    cp.testing.assert_array_equal(output, cp.full(4, cp.nan))
