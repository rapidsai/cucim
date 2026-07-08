# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import pytest

from cucim.skimage._vendored import _ndimage_filters as filters


@pytest.mark.parametrize("order", [0, 1])
def test_cached_gaussian_kernel_is_initialized_synchronously(
    monkeypatch, order
):
    asarray_calls = []
    original_asarray = cp.asarray

    def asarray_spy(*args, **kwargs):
        asarray_calls.append(kwargs)
        return original_asarray(*args, **kwargs)

    monkeypatch.setattr(filters.cupy, "asarray", asarray_spy)
    filters._cached_gaussian_kernel1d_for_device.cache_clear()

    args = (1.0, order, 4, cp.dtype(cp.float32).str)
    first = filters._cached_gaussian_kernel1d(*args)
    second = filters._cached_gaussian_kernel1d(*args)

    assert first is second
    assert len(asarray_calls) == 1
    assert asarray_calls[0]["blocking"] is True


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
