# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pytest

from cucim.skimage._vendored import _ndimage_filters as filters

pytestmark = pytest.mark.filterwarnings("error::UserWarning")


@pytest.mark.parametrize(
    "shape,axis",
    [
        ((7, 19, 3), 1),
        ((5, 17, 4), 1),
        ((2, 3, 17, 2, 3), 2),
        ((3, 17, 8), 1),
        ((2, 17, 2, 16), 1),
        ((2, 3, 17), 2),
        ((1, 1, 3), 1),
        ((3, 2, 7), 1),
    ],
)
@pytest.mark.parametrize(
    "mode", ["reflect", "mirror", "nearest", "wrap", "constant"]
)
@pytest.mark.parametrize("convolution", [False, True])
def test_filter_axis_views(shape, axis, mode, convolution):
    image = cp.asarray(
        np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    )
    weights = cp.asarray([0.1, -0.2, 0.3, 0.4], dtype=cp.float32)
    function = filters.convolve1d if convolution else filters.correlate1d
    kwargs = dict(axis=axis - image.ndim, mode=mode, cval=2.0, origin=-1)
    expected = function(image, weights, algorithm="elementwise", **kwargs)
    output = cp.empty_like(image)
    result = function(
        image, weights, output=output, algorithm="shared_memory", **kwargs
    )
    assert result is output
    cp.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(
    "dtype", ["float16", "float64", "int16", "uint8", "complex64", "complex128"]
)
def test_filter_axis_view_dtypes(dtype):
    image = cp.asarray(
        np.random.default_rng(1).integers(0, 20, (5, 13, 3))
    ).astype(dtype)
    weights = cp.asarray([0.125, 0.25, -0.5, 1], dtype=cp.float64)
    if image.dtype.kind == "c":
        image += 0.5j
        weights = weights + 0.25j
    for output in (
        None,
        cp.float64 if image.dtype.kind != "c" else cp.complex128,
    ):
        expected = filters.correlate1d(
            image, weights, axis=1, output=output, algorithm="elementwise"
        )
        result = filters.correlate1d(
            image, weights, axis=1, output=output, algorithm="shared_memory"
        )
        cp.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(
    "width,origin", [(1, 0), (7, -3), (7, 3), (65, -32), (65, 32)]
)
def test_filter_axis_view_large_halo(width, origin):
    # Multiple reflections and an asymmetric halo can extend beyond a tile.
    image = cp.arange(2 * 5 * 3, dtype=cp.float32).reshape(2, 5, 3)
    weights = cp.full(width, 1 / width, dtype=cp.float32)
    expected = filters.correlate1d(
        image,
        weights,
        axis=1,
        origin=origin,
        mode="mirror",
        algorithm="elementwise",
    )
    result = filters.correlate1d(
        image,
        weights,
        axis=1,
        origin=origin,
        mode="mirror",
        algorithm="shared_memory",
    )
    cp.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-6)


def test_filter_axis_view_resource_fallback():
    image = cp.arange(2 * 9 * 16, dtype=cp.float32).reshape(2, 9, 16)
    weights = cp.full(255, 1 / 255, dtype=cp.float32)
    expected = filters.correlate1d(
        image, weights, axis=1, algorithm="elementwise"
    )
    result = filters.correlate1d(image, weights, axis=1)
    cp.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("trailing_size", [None, 3, 32])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_filter_axis_views_zero_weights(trailing_size, value):
    shape = (3, 9) if trailing_size is None else (3, 9, trailing_size)
    image = cp.ones(shape, dtype=cp.float32)
    image[(1, 4) if trailing_size is None else (1, 4, 0)] = value
    weights = cp.asarray([-1, 0, 1], dtype=cp.float32)
    expected = filters.correlate1d(
        image, weights, axis=1, algorithm="elementwise"
    )
    result = filters.correlate1d(
        image, weights, axis=1, algorithm="shared_memory"
    )
    cp.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("axis", [1, 2, -1])
def test_filter_axis_view_default_dispatch(axis, monkeypatch):
    image = cp.arange(2 * 17 * 13 * 3, dtype=cp.float32).reshape(2, 17, 13, 3)
    if axis == -1:
        image = image.reshape(2, 17, 39)
    weights = cp.asarray([0.25, 0.5, 0.25], dtype=cp.float32)
    expected = filters.correlate1d(
        image, weights, axis=axis, algorithm="elementwise"
    )
    shared = filters._shmem_convolve1d
    calls = []

    def record_shared(*args, **kwargs):
        calls.append(args[0].shape)
        return shared(*args, **kwargs)

    monkeypatch.setattr(filters, "_shmem_convolve1d", record_shared)
    result = filters.correlate1d(image, weights, axis=axis)
    assert len(calls) == 1
    cp.testing.assert_array_equal(result, expected)


def test_filter_axis_view_many_rows():
    # More than 65535 row tiles must spill into grid.z, not exceed grid.y.
    image = cp.arange(524281, dtype=cp.float32).reshape(-1, 1, 1)
    weights = cp.asarray([0.25, 0.5, 0.25], dtype=cp.float32)
    result = filters.correlate1d(
        image, weights, axis=1, algorithm="shared_memory"
    )
    cp.testing.assert_array_equal(result, image)


@pytest.mark.parametrize(
    "shape", [(4 * 65535 + 1, 8, 17), (1, 16 * 65535 + 1, 17)]
)
def test_filter_axis_view_grid_limit_fallback(shape, monkeypatch):
    image = cp.zeros(shape, dtype=cp.float32)
    weights = cp.asarray([0.25, 0.5, 0.25], dtype=cp.float32)

    def fail_shared(*args, **kwargs):
        pytest.fail("wide-stride tile grid exceeds a CUDA launch limit")

    monkeypatch.setattr(filters, "_shmem_convolve1d", fail_shared)
    result = filters.correlate1d(image, weights, axis=1)
    cp.testing.assert_array_equal(result, image)


@pytest.mark.parametrize(
    "layout",
    ["strided_input", "strided_output", "overlap", "short_axis", "short_row"],
)
def test_filter_axis_view_fallback(layout, monkeypatch):
    image = cp.arange(8 * 19 * 3, dtype=cp.float32).reshape(8, 19, 3)
    if layout == "short_axis":
        image = cp.arange(1024 * 3 * 32, dtype=cp.float32).reshape(1024, 3, 32)
    elif layout == "short_row":
        image = cp.arange(4096 * 8 * 4, dtype=cp.float32).reshape(4096, 8, 4)
    if layout == "strided_input":
        image = image[:, ::2]
    output = cp.empty_like(image)
    if layout == "strided_output":
        output = cp.empty(image.shape + (2,), dtype=image.dtype)[..., 0]
    elif layout == "overlap":
        output = image
    weights = cp.asarray([0.25, 0.5, 0.25], dtype=cp.float32)
    expected = filters.correlate1d(
        image, weights, axis=1, algorithm="elementwise"
    )

    def fail_shared(*args, **kwargs):
        pytest.fail("unsupported layout entered the view path")

    monkeypatch.setattr(filters, "_shmem_convolve1d", fail_shared)
    result = filters.correlate1d(image, weights, axis=1, output=output)
    assert result is output
    cp.testing.assert_array_equal(result, expected)
