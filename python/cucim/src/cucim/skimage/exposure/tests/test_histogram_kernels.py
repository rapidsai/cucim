# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pytest

from cucim.skimage.exposure import _adapthist
from cucim.skimage.exposure._histogram_kernels import (
    _bincount_rows,
    _clip_histogram_rows,
)


@pytest.mark.parametrize(
    "dtype",
    ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"],
)
@pytest.mark.parametrize("rows", [1, 3])
def test_bincount_rows_integer_limits(dtype, rows):
    limits = np.iinfo(dtype)
    low = int(limits.min)
    values = np.asarray([low, low + 1, low + 7, low + 7], dtype=dtype)
    image = cp.asarray(np.tile(values, (rows, 1025)))
    result = _bincount_rows(image, 8, low)
    expected = np.tile([1025, 1025, 0, 0, 0, 0, 0, 2050], (rows, 1))
    assert result.dtype == cp.int64
    cp.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("nbins", [1, 256, 5000])
def test_bincount_rows_empty_and_global(nbins):
    empty = _bincount_rows(cp.empty((3, 0), dtype=cp.uint16), nbins)
    cp.testing.assert_array_equal(empty, cp.zeros((3, nbins), dtype=cp.int64))
    image = cp.arange(3 * 4097, dtype=cp.uint16).reshape(3, -1) % nbins
    expected = cp.stack([cp.bincount(row, minlength=nbins) for row in image])
    cp.testing.assert_array_equal(_bincount_rows(image, nbins), expected)


@pytest.mark.skipif(cp.cuda.runtime.is_hip, reason="uses CUDA warp intrinsics")
@pytest.mark.parametrize("nbins", [1, 17, 31, 32, 33, 256, 257, 4096])
@pytest.mark.parametrize("limit", [1, 7, 89])
def test_clip_histogram_rows_matches_host(nbins, limit):
    rng = np.random.default_rng(191)
    values = rng.integers(0, 2 * limit + 1, size=(9, nbins), dtype=np.int64)
    values[0] = 0
    values[1] = limit + 1
    values[2] = 0
    values[2, 0] = nbins * limit // 2 + 1
    # Exercise counts near the limit where the low-stage additions feed the
    # middle-stage cap and the last increment can fill several bins at once.
    values[3] = rng.integers(0, limit + 1, size=nbins)
    values[3, 0] += 3 * limit
    hist = cp.asarray(values)
    expected = np.apply_along_axis(
        _adapthist.clip_histogram, -1, values.copy(), clip_limit=limit
    )
    _clip_histogram_rows(hist, limit)
    cp.testing.assert_array_equal(hist, expected)


@pytest.mark.parametrize(
    "shape, tile", [((31, 37), (8, 7)), ((19, 23, 29), (5, 7, 4))]
)
def test_adapthist_device_matches_host(shape, tile, monkeypatch):
    image = cp.random.default_rng(98).random(shape, dtype=cp.float32)
    actual = _adapthist.equalize_adapthist(
        image, kernel_size=tile, clip_limit=0.2
    )
    monkeypatch.setattr(_adapthist, "_MAX_CLIP_BINS", 0)
    expected = _adapthist.equalize_adapthist(
        image, kernel_size=tile, clip_limit=0.2
    )
    cp.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("clip_limit", [0.0, 1.0, 1.5])
def test_adapthist_bypasses_clipping_when_no_bin_can_exceed_limit(
    clip_limit, monkeypatch
):
    image = cp.random.default_rng(98).random((31, 37), dtype=cp.float32)

    def unexpected_clip(*args, **kwargs):
        pytest.fail("histogram clipping ran for an unreachable limit")

    monkeypatch.setattr(_adapthist, "_clip_histogram_rows", unexpected_clip)
    monkeypatch.setattr(_adapthist, "clip_histogram", unexpected_clip)
    actual = _adapthist.equalize_adapthist(
        image, kernel_size=8, clip_limit=clip_limit
    )
    expected = _adapthist.equalize_adapthist(image, kernel_size=8, clip_limit=0)
    cp.testing.assert_array_equal(actual, expected)
