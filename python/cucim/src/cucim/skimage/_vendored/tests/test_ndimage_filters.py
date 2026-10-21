# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pytest
from scipy import ndimage as scipy_ndi

from cucim.skimage._vendored import _ndimage_filters as filters


@pytest.mark.parametrize("shape", [(17, 19, 3), (13, 7, 5, 4)])
@pytest.mark.parametrize("dtype", ["float32", "float64", "int16", "complex64"])
@pytest.mark.parametrize(
    "mode", ["reflect", "mirror", "nearest", "wrap", "constant"]
)
@pytest.mark.parametrize("convolution", [False, True])
def test_axis0_shared_memory_view(shape, dtype, mode, convolution):
    image = cp.asarray(
        np.random.default_rng(0).standard_normal(shape) * 20
    ).astype(dtype)
    weights = cp.asarray([0.1, 0.2, 0.3, 0.4])
    function = filters.convolve1d if convolution else filters.correlate1d
    expected = function(
        image,
        weights,
        axis=0,
        origin=-1,
        mode=mode,
        cval=2,
        algorithm="elementwise",
    )
    output = cp.empty_like(image)
    actual = function(
        image,
        weights,
        axis=-image.ndim,
        origin=-1,
        mode=mode,
        cval=2,
        output=output,
    )
    assert actual is output
    cp.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    "layout", ["contiguous", "strided_input", "strided_output", "overlap"]
)
def test_axis0_shared_memory_view_dispatch(monkeypatch, layout):
    image = cp.arange(9 * 11 * 3, dtype=cp.float32).reshape(9, 11, 3)
    if layout == "strided_input":
        image = image[:, ::2]
    output = cp.empty_like(image)
    if layout == "strided_output":
        output = cp.empty(image.shape + (2,), image.dtype)[..., 0]
    elif layout == "overlap":
        output = image
    weights = cp.asarray([0.25, 0.5, 0.25], dtype=cp.float32)
    expected = filters.correlate1d(
        image, weights, axis=0, algorithm="elementwise"
    )
    calls = []
    original = filters._shmem_convolve1d

    def spy(image, *args, **kwargs):
        calls.append(image.shape)
        return original(image, *args, **kwargs)

    monkeypatch.setattr(filters, "_shmem_convolve1d", spy)
    actual = filters.correlate1d(image, weights, axis=0, output=output)
    assert calls == ([(9, 33)] if layout == "contiguous" else [])
    cp.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("order", [0, 1])
@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64"])
def test_cached_gaussian_kernel_is_initialized_on_private_stream(
    monkeypatch, order, dtype
):
    asarray_calls = []
    original_asarray = cp.asarray

    def asarray_spy(*args, **kwargs):
        result = original_asarray(*args, **kwargs)
        stream = cp.cuda.get_current_stream()
        ready = cp.cuda.Event()
        ready.record(stream)
        asarray_calls.append((stream, ready))
        return result

    monkeypatch.setattr(filters.cupy, "asarray", asarray_spy)
    filters._cached_gaussian_kernel1d_for_device.cache_clear()

    args = (1.0, order, 4, cp.dtype(dtype).str)
    producer = cp.cuda.Stream(non_blocking=True)
    consumer = cp.cuda.Stream(non_blocking=True)
    with producer:
        first = filters._cached_gaussian_kernel1d(*args)
        assert cp.cuda.get_current_stream() == producer
    with consumer:
        second = filters._cached_gaussian_kernel1d(*args)
        copied = second.copy()
    consumer.synchronize()

    assert first is second
    assert first.dtype == cp.dtype(dtype)
    assert len(asarray_calls) == 1
    stream, ready = asarray_calls[0]
    assert stream.ptr not in (
        producer.ptr,
        consumer.ptr,
        cp.cuda.Stream.null.ptr,
    )
    assert stream.is_non_blocking
    assert ready.done
    cp.testing.assert_array_equal(copied, first)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("algorithm", ["shared_memory", "elementwise"])
def test_cached_gaussian_filter_matches_scipy(dtype, order, algorithm):
    image = np.random.default_rng(0).standard_normal((16, 20)).astype(dtype)
    image_gpu = cp.asarray(image)
    expected = scipy_ndi.gaussian_filter(image, sigma=1, order=order)
    for _ in range(2):
        actual = filters.gaussian_filter(
            image_gpu, sigma=1, order=order, algorithm=algorithm
        )
        cp.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)


def test_gaussian_cache_byte_limit_uses_requested_dtype(monkeypatch):
    monkeypatch.setattr(filters, "_MAX_CACHED_GAUSSIAN_KERNEL_BYTES", 16)
    filters._cached_gaussian_kernel1d_for_device.cache_clear()
    args = (1.0, 0, 1, cp.dtype(cp.float32).str)
    first = filters._cached_gaussian_kernel1d(*args)
    second = filters._cached_gaussian_kernel1d(*args)
    assert first.nbytes == 12
    assert first is second


def test_large_gaussian_kernels_are_not_cached(monkeypatch):
    kernels = []

    def make_kernel(*args):
        kernel = object()
        kernels.append(kernel)
        return kernel

    monkeypatch.setattr(filters, "_MAX_CACHED_GAUSSIAN_KERNEL_BYTES", 1)
    monkeypatch.setattr(filters, "_gaussian_kernel1d", make_kernel)

    args = (1.0, 1, 1, cp.dtype(cp.float32).str)
    first = filters._cached_gaussian_kernel1d(*args)
    second = filters._cached_gaussian_kernel1d(*args)

    assert first is not second
    assert len(kernels) == 2
