# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pytest
from cupyx import allow_synchronize

from cucim.skimage._shared import _reduction_kernels
from cucim.skimage._shared._reduction_kernels import (
    _get_minmax_with_nan as _get_warp_range,
)
from cucim.skimage.transform import _warps
from cucim.skimage.transform._warps import _clip_warp_output

pytestmark = pytest.mark.filterwarnings(
    "ignore:cupyx.allow_synchronize will be removed:UserWarning"
)


@pytest.fixture(params=["direct", "staged"])
def range_dispatch(request, monkeypatch):
    """Exercise both reduction paths on the same dtype and NaN cases."""
    if request.param == "staged":
        monkeypatch.setattr(_reduction_kernels, "_RANGE_DIRECT_MAX_BYTES", 0)


@pytest.fixture(params=["device", "host"])
def clipping_dispatch(request, monkeypatch):
    """Check both constant-mode strategies against the same reference values."""
    threshold = 0 if request.param == "host" else float("inf")
    monkeypatch.setattr(_warps, "_CLIP_CONSTANT_HOST_MIN_BYTES", threshold)
    monkeypatch.setattr(_warps, "_CLIP_CONSTANT_HOST_MAX_BYTES", float("inf"))
    return request.param


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("mode", ["reflect", "constant"])
@pytest.mark.parametrize("input_nan", [False, True])
@pytest.mark.parametrize("output_nan", [False, True])
@pytest.mark.parametrize("cval", [-1, 1.5, 3, np.nan])
def test_clip_warp_output(
    dtype, mode, input_nan, output_nan, cval, clipping_dispatch
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
    with allow_synchronize(mode == "constant" and clipping_dispatch == "host"):
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
def test_clip_warp_output_unused_cval(mode, clipping_dispatch):
    image = cp.asarray([1.0, 2.0])
    output = cp.asarray([0.5, 1.5, 2.5])
    with allow_synchronize(mode == "constant" and clipping_dispatch == "host"):
        _clip_warp_output(image, output, mode, 0, 3, True)
    cp.testing.assert_array_equal(output, [1, 1.5, 2])


@pytest.mark.parametrize("mode", ["reflect", "constant"])
def test_clip_warp_output_all_nan(mode, clipping_dispatch):
    image = cp.full(4, cp.nan)
    output = cp.full(4, cp.nan)
    with allow_synchronize(mode == "constant" and clipping_dispatch == "host"):
        _clip_warp_output(image, output, mode, 0, 3, True)
    cp.testing.assert_array_equal(output, cp.full(4, cp.nan))


@pytest.mark.parametrize("cval", [0.1, 2.1])
def test_clip_warp_output_cval_uses_input_dtype(cval, clipping_dispatch):
    image = cp.asarray([1, 2], dtype=cp.float32)
    output = cp.asarray([-1, 0, 1.5, 3], dtype=cp.float64)
    min_val = min(1, np.float32(cval))
    max_val = max(2, np.float32(cval))
    expected = np.clip(np.array([-1, 0, 1.5, 3]), min_val, max_val)
    with allow_synchronize(clipping_dispatch == "host"):
        _clip_warp_output(image, output, "constant", cval, 3, True)
    cp.testing.assert_array_equal(output, expected)


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
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ],
)
@pytest.mark.parametrize("size", [1, 31, 4096, 4097, 12288, 12289])
def test_warp_range(dtype, size, range_dispatch):
    dtype = np.dtype(dtype)
    image = np.random.default_rng(0).integers(0, 100, size=size).astype(dtype)
    if size > 1:
        if dtype.kind in "iu":
            image[0], image[-1] = np.iinfo(dtype).min, np.iinfo(dtype).max
        elif dtype.kind in "fc":
            image[0], image[-1] = -np.inf, np.inf
    image_gpu = cp.asarray(image)
    with allow_synchronize(False):
        minimum, maximum, has_nan = _get_warp_range(image_gpu)
    assert minimum.dtype == maximum.dtype == dtype
    cp.testing.assert_array_equal(minimum, image.min())
    cp.testing.assert_array_equal(maximum, image.max())
    cp.testing.assert_array_equal(has_nan, False)


@pytest.mark.parametrize(
    "dtype", ["float16", "float32", "float64", "complex64", "complex128"]
)
@pytest.mark.parametrize("nan_pattern", ["all", "mixed", "chunk"])
def test_warp_range_with_nans(dtype, nan_pattern, range_dispatch):
    image = cp.asarray(np.arange(12289).astype(dtype))
    if nan_pattern == "all":
        image[:] = cp.nan
    elif nan_pattern == "mixed":
        image[[0, 4095, 4096, -1]] = cp.nan
    else:
        image[:8192] = cp.nan
    expected_min = cp._core.nanmin(image)
    expected_max = cp._core.nanmax(image)
    with allow_synchronize(False):
        minimum, maximum, has_nan = _get_warp_range(image)
    cp.testing.assert_array_equal(minimum, expected_min)
    cp.testing.assert_array_equal(maximum, expected_max)
    cp.testing.assert_array_equal(has_nan, True)


@pytest.mark.parametrize(
    "layout", ["transpose", "slice", "reverse", "fortran", "strided_1d"]
)
def test_warp_range_strided(layout):
    image = cp.asarray(
        np.random.default_rng(0).standard_normal((129, 257)).astype(np.float32)
    )
    if layout == "transpose":
        image = image.T
    elif layout == "slice":
        image = image[::2, ::2]
    elif layout == "reverse":
        image = image[::-2, ::3]
    elif layout == "fortran":
        image = cp.asfortranarray(image)
    else:
        image = image.reshape(-1)[::2]
    expected_min, expected_max = image.min(), image.max()
    with allow_synchronize(False):
        minimum, maximum, has_nan = _get_warp_range(image)
    cp.testing.assert_array_equal(minimum, expected_min)
    cp.testing.assert_array_equal(maximum, expected_max)
    cp.testing.assert_array_equal(has_nan, False)


@pytest.mark.parametrize("shape", [(1, 32771), (2, 32771), (32771, 2)])
def test_warp_range_long_strided_axis(shape):
    image = cp.arange(np.prod(shape) * 2, dtype=cp.float32)[::2].reshape(shape)
    image[-1, -1] = cp.nan
    expected_min = cp._core.nanmin(image)
    expected_max = cp._core.nanmax(image)
    with allow_synchronize(False):
        minimum, maximum, has_nan = _get_warp_range(image)
    cp.testing.assert_array_equal(minimum, expected_min)
    cp.testing.assert_array_equal(maximum, expected_max)
    cp.testing.assert_array_equal(has_nan, True)


@pytest.mark.parametrize("value", [1.0, np.nan])
def test_warp_range_scalar(value):
    image = cp.asarray(value)
    with allow_synchronize(False):
        minimum, maximum, has_nan = _get_warp_range(image)
    cp.testing.assert_array_equal(minimum, value)
    cp.testing.assert_array_equal(maximum, value)
    cp.testing.assert_array_equal(has_nan, np.isnan(value))


def test_warp_range_empty():
    with pytest.raises(ValueError, match="empty image"):
        _get_warp_range(cp.empty(0))


@pytest.mark.parametrize("dtype", ["float32", "float64", "uint64"])
def test_warp_range_parallel_streams(dtype):
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
    pending = []
    for repeat in range(12):
        for index, stream in enumerate(streams):
            with stream:
                size = 131073 + repeat * 13 + index
                image = cp.arange(size, dtype=dtype)
                if dtype != "uint64":
                    image[-1] = cp.nan
                result = _get_warp_range(image)
                pending.append((image, result, size))
    for stream in streams:
        stream.synchronize()
    for image, (minimum, maximum, has_nan), size in pending:
        cp.testing.assert_array_equal(minimum, 0)
        cp.testing.assert_array_equal(
            maximum, size - (1 if dtype == "uint64" else 2)
        )
        cp.testing.assert_array_equal(has_nan, dtype != "uint64")


@pytest.mark.parametrize("size", [31, 131073])
def test_warp_range_subnormal_extrema(size):
    tiny = np.nextafter(np.float32(0), np.float32(1))
    image = cp.full(size, tiny, dtype=cp.float32)
    image[0] = -tiny
    minimum, maximum, has_nan = _get_warp_range(image)
    cp.testing.assert_array_equal(minimum, -tiny)
    cp.testing.assert_array_equal(maximum, tiny)
    cp.testing.assert_array_equal(has_nan, False)


@pytest.mark.parametrize("mode", ["reflect", "constant"])
@pytest.mark.parametrize("input_nan", [False, True])
def test_clip_warp_output_multiple_chunks(
    mode, input_nan, range_dispatch, clipping_dispatch
):
    image = cp.ones(16385, dtype=cp.float32)
    image[-1] = 2
    if input_nan:
        image[0] = cp.nan
    output = cp.full(16385, 1.5, dtype=cp.float32)
    output[0], output[-1], output[4096] = 0, 3, cp.nan
    expected = output.copy()
    expected[0] = 1
    expected[-1] = 3 if mode == "constant" and input_nan else 2
    with allow_synchronize(mode == "constant" and clipping_dispatch == "host"):
        _clip_warp_output(image, output, mode, 3, 3, True)
    cp.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_warp_range_direct_cutoff(dtype, offset):
    dtype = cp.dtype(dtype)
    size = _reduction_kernels._RANGE_DIRECT_MAX_BYTES // dtype.itemsize + offset
    image = cp.arange(size, dtype=dtype)
    image[-1] = cp.nan
    with allow_synchronize(False):
        minimum, maximum, has_nan = _get_warp_range(image)
    cp.testing.assert_array_equal(minimum, 0)
    cp.testing.assert_array_equal(maximum, size - 2)
    cp.testing.assert_array_equal(has_nan, True)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("size", [31, 131073])
@pytest.mark.parametrize("mode", ["reflect", "constant"])
@pytest.mark.parametrize("cval", [-1, 1.5, 3])
def test_clip_warp_output_assume_finite(monkeypatch, dtype, size, mode, cval):
    image = cp.linspace(1, 2, size, dtype=dtype)
    output = cp.linspace(-2, 4, size, dtype=dtype)
    expected = output.copy()
    _clip_warp_output(image, expected, mode, cval, 3, True)

    def fail_nan_path(*args, **kwargs):
        pytest.fail("finite clipping used NaN detection or host range checks")

    monkeypatch.setattr(_warps, "_get_warp_range", fail_nan_path)
    monkeypatch.setattr(_warps, "_clip_constant_warp_output", fail_nan_path)
    with allow_synchronize(False):
        _clip_warp_output(
            image, output, mode, cval, 3, True, assume_finite=True
        )
    cp.testing.assert_array_equal(output, expected)
