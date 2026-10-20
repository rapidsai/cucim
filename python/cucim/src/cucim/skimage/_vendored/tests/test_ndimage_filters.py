# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pytest
from scipy import ndimage as scipy_ndi

from cucim.skimage._vendored import _ndimage_filters as filters


@pytest.mark.parametrize("order", [0, 1])
@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64"])
def test_cached_gaussian_kernel_is_initialized_synchronously(
    monkeypatch, order, dtype
):
    asarray_calls = []
    original_asarray = cp.asarray

    def asarray_spy(*args, **kwargs):
        asarray_calls.append(kwargs)
        return original_asarray(*args, **kwargs)

    monkeypatch.setattr(filters.cupy, "asarray", asarray_spy)
    filters._cached_gaussian_kernel1d_for_device.cache_clear()

    args = (1.0, order, 4, cp.dtype(dtype).str)
    first = filters._cached_gaussian_kernel1d(*args)
    second = filters._cached_gaussian_kernel1d(*args)

    assert first is second
    assert first.dtype == cp.dtype(dtype)
    assert len(asarray_calls) == 1
    assert asarray_calls[0]["blocking"] is True


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
