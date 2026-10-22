# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pytest
from cupyx import allow_synchronize

from cucim.skimage._vendored import _ndimage_filters as filters
from cucim.skimage._vendored import _ndimage_interpolation as interpolation
from cucim.skimage._vendored import _ndimage_util as util


@pytest.fixture
def delay_kernel():
    delay = cp.RawKernel(
        r"""
        extern "C" __global__ void delay(unsigned long long ns) {
            unsigned long long start, now;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
            do {
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
            } while (now - start < ns);
        }
        """,
        "delay",
    )
    delay((1,), (1,), (np.uint64(0),))
    return delay


@pytest.mark.parametrize("kind", ["gaussian", "zoom", "weights"])
@pytest.mark.parametrize("non_blocking", [False, True])
def test_cache_miss_leaves_caller_work_pending(
    kind, non_blocking, delay_kernel
):
    if kind == "gaussian":
        cache = filters._cached_gaussian_kernel1d_for_device

        def get():
            return filters._cached_gaussian_kernel1d(1.75, 0, 7, "<f4")
    elif kind == "weights":
        cache = util._cached_filter_weights_for_device

        def get():
            return util._cached_filter_weights(
                (1, 0, -1), "float32", divisor=np.sqrt(2)
            )
    else:
        cache = interpolation._cached_zoom_factors_for_device

        def get():
            return interpolation._cached_zoom_factors((0.5, 0.25), "float32")

    # Warm allocations and initialization machinery before queuing the delay.
    get()
    cache.cache_clear()
    cp.cuda.Device().synchronize()
    caller = (
        cp.cuda.Stream(non_blocking=True)
        if non_blocking
        else cp.cuda.Stream.null
    )
    pending = cp.cuda.Event()
    try:
        with caller:
            delay_kernel((1,), (1,), (np.uint64(250_000_000),))
            pending.record()
            get()
            assert not pending.done, "cache miss drained queued caller work"
    finally:
        caller.synchronize()


@pytest.mark.parametrize("kind", ["gaussian", "zoom", "weights"])
def test_cache_eviction_preserves_queued_consumers(kind, delay_kernel):
    # Delay both consumer streams so the LRU evicts entries before their reads.
    delay = delay_kernel
    count = 256
    if kind == "gaussian":
        cache = filters._cached_gaussian_kernel1d_for_device

        def get(index):
            return filters._cached_gaussian_kernel1d(
                1 + index / count, 0, 4, "<f4"
            )

        x = np.arange(-4, 5)
        expected = np.exp(
            -0.5
            * x[None, :] ** 2
            / (1 + np.arange(count)[:, None] / count) ** 2
        )
        expected = (expected / expected.sum(axis=1, keepdims=True)).astype(
            np.float32
        )
    else:
        cache = (
            interpolation._cached_zoom_factors_for_device
            if kind == "zoom"
            else util._cached_filter_weights_for_device
        )

        def get(index):
            weights = (0.5 + index / count, 0.25)
            if kind == "zoom":
                return interpolation._cached_zoom_factors(weights, "float32")
            return util._cached_filter_weights(weights, "float32")

        expected = np.array(
            [(0.5 + i / count, 0.25) for i in range(count)], dtype=np.float32
        )

    cache.cache_clear()
    outputs = [cp.empty(expected.shape, dtype=cp.float32) for _ in range(2)]
    # Warm the copy kernel and allocations before delaying.
    outputs[0][0] = get(0)
    cp.cuda.Device().synchronize()
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
    for stream in streams:
        with stream:
            delay((1,), (1,), (np.uint64(200_000_000),))
    for index in range(count):
        for stream, output in zip(streams, outputs):
            with stream:
                output[index] = get(index)
    # Clearing must also preserve reads of the entries that remain in the LRU.
    cache.cache_clear()
    for stream, output in zip(streams, outputs):
        stream.synchronize()
        cp.testing.assert_array_equal(output, expected)
    util._release_completed_cache_allocations(cp.cuda.runtime.getDevice())
    assert not util._retired_cache_allocations[cp.cuda.runtime.getDevice()]


@pytest.mark.filterwarnings(
    "ignore:cupyx.allow_synchronize will be removed:UserWarning"
)
def test_cache_hits_do_not_synchronize():
    gaussian_args = (1.0, 0, 4, "<f4")
    zoom_args = ((0.5, 0.25), "float32")
    gaussian = filters._cached_gaussian_kernel1d(*gaussian_args)
    zoom = interpolation._cached_zoom_factors(*zoom_args)
    weights_args = ((1, 0, -1), "float32", np.sqrt(2))
    weights = util._cached_filter_weights(*weights_args)
    with allow_synchronize(False):
        assert filters._cached_gaussian_kernel1d(*gaussian_args) is gaussian
        assert interpolation._cached_zoom_factors(*zoom_args) is zoom
        assert util._cached_filter_weights(*weights_args) is weights


@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64"])
def test_filter_weight_normalization_does_not_modify_other_entries(dtype):
    coefficients = (1, 0, -1)
    unscaled = util._cached_filter_weights(coefficients, dtype)
    for ndim in (2, 3, 2):
        expected = cp.asarray(coefficients, dtype=dtype)
        expected /= np.sqrt(ndim)
        actual = util._cached_filter_weights(
            coefficients, dtype, divisor=np.sqrt(ndim)
        )
        cp.testing.assert_array_equal(actual, expected)
        cp.testing.assert_array_equal(unscaled, cp.asarray(coefficients))
    other_dtype = "float64" if dtype != "float64" else "float32"
    other = util._cached_filter_weights(coefficients, other_dtype)
    assert other.dtype == cp.dtype(other_dtype)
    assert other is not unscaled
