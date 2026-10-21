# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import pytest

from cucim.skimage._vendored import _ndimage_filters as filters


@pytest.mark.parametrize("algorithm", ["elementwise", "shared_memory"])
@pytest.mark.parametrize("partial_overlap", [False, True])
def test_correlate1d_overlapping_output(algorithm, partial_overlap):
    storage = cp.arange(13 * 15, dtype=cp.float32).reshape(13, 15)
    image = storage[:-1] if partial_overlap else storage
    output = storage[1:] if partial_overlap else storage
    weights = cp.asarray([0.25, 0.5, 0.25], dtype=cp.float32)
    expected = filters.correlate1d(
        image.copy(), weights, axis=0, algorithm=algorithm
    )
    result = filters.correlate1d(
        image, weights, axis=0, output=output, algorithm=algorithm
    )
    assert result is output
    cp.testing.assert_array_equal(result, expected)
