# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pytest

from cucim.skimage.exposure._histogram_kernels import _bincount_rows


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
