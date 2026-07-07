# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp

from cucim.skimage._vendored import _ndimage_filters as filters


def test_cached_gaussian_kernel_waits_for_initialization(monkeypatch):
    kernel = object()
    ready = object()

    monkeypatch.setattr(
        filters,
        "_cached_gaussian_kernel1d_for_device",
        lambda *args: (kernel, ready),
    )
    monkeypatch.setattr(filters.cupy.cuda.runtime, "getDevice", lambda: 0)

    class Stream:
        waited_for = None

        def wait_event(self, event):
            self.waited_for = event

    stream = Stream()
    monkeypatch.setattr(filters.cupy.cuda, "get_current_stream", lambda: stream)

    result = filters._cached_gaussian_kernel1d(1.0, 0, 4, "<f8")

    assert result is kernel
    assert stream.waited_for is ready


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
