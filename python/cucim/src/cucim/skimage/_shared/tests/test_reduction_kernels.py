# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pytest
from cupyx import allow_synchronize

from cucim.skimage._shared import _reduction_kernels
from cucim.skimage._shared._reduction_kernels import _get_minmax

pytestmark = pytest.mark.filterwarnings(
    "ignore:cupyx.allow_synchronize will be removed:UserWarning"
)


@pytest.fixture(params=["combined", "packed"])
def extrema_dispatch(request, monkeypatch):
    """Check NaN propagation and precision with both extrema strategies."""
    monkeypatch.setattr(_reduction_kernels, "_RANGE_DIRECT_MAX_BYTES", 0)
    minimum = 0 if request.param == "packed" else float("inf")
    maximum = float("inf") if request.param == "packed" else 0
    monkeypatch.setattr(
        _reduction_kernels, "_MINMAX_SEPARATE_MIN_BYTES", minimum
    )
    monkeypatch.setattr(
        _reduction_kernels, "_MINMAX_SEPARATE_MAX_BYTES", maximum
    )


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
@pytest.mark.parametrize("shape", [(7, 9), (257, 513)])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("nan_pattern", ["none", "mixed", "all"])
def test_minmax_propagates_nan(
    dtype, shape, transpose, nan_pattern, extrema_dispatch
):
    image = cp.linspace(-1, 2, np.prod(shape), dtype=dtype).reshape(shape)
    if nan_pattern == "mixed":
        image[0, 0] = cp.nan
    elif nan_pattern == "all":
        image[:] = cp.nan
    if transpose:
        image = image.T
    expected = cp.stack((image.min(), image.max()))
    with allow_synchronize(False):
        bounds = _get_minmax(image)
    assert bounds.shape == (2,)
    assert bounds.dtype == image.dtype
    cp.testing.assert_array_equal(bounds, expected)


@pytest.mark.parametrize(
    "dtype",
    [
        "bool",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
    ],
)
def test_minmax_integer_precision(dtype, extrema_dispatch):
    extrema = (
        (False, True)
        if dtype == "bool"
        else (np.iinfo(dtype).min, np.iinfo(dtype).max)
    )
    image = cp.asarray(extrema, dtype=dtype)
    with allow_synchronize(False):
        bounds = _get_minmax(image)
    assert bounds.dtype == image.dtype
    assert bounds.get().tolist() == list(extrema)


@pytest.mark.parametrize("shape", [(7, 9), (257, 513)])
@pytest.mark.parametrize("transpose", [False, True])
def test_minmax_preserves_float32_subnormals(
    shape, transpose, extrema_dispatch
):
    tiny = np.nextafter(np.float32(0), np.float32(1))
    image = np.zeros(shape, dtype=np.float32)
    image.flat[1], image.flat[-2] = -tiny, tiny
    image_gpu = cp.asarray(image)
    if transpose:
        image_gpu = image_gpu.T
    with allow_synchronize(False):
        bounds = _get_minmax(image_gpu)
    cp.testing.assert_array_equal(bounds, [image.min(), image.max()])


def test_minmax_without_accelerated_reductions(monkeypatch):
    image = cp.linspace(-1, 2, 131073, dtype=cp.float32)
    monkeypatch.setattr(
        _reduction_kernels._accelerator,
        "get_reduction_accelerators",
        lambda: [],
    )

    def fail_pack(*args, **kwargs):
        pytest.fail("selected accelerated reduction path without CUB")

    monkeypatch.setattr(_reduction_kernels, "_pack_minmax_kernel", fail_pack)
    with allow_synchronize(False):
        bounds = _get_minmax(image)
    cp.testing.assert_array_equal(bounds, [-1, 2])
