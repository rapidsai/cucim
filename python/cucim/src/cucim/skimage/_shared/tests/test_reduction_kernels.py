# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pytest
from cupyx import allow_synchronize

from cucim.skimage._shared._reduction_kernels import _get_minmax

pytestmark = pytest.mark.filterwarnings(
    "ignore:cupyx.allow_synchronize will be removed:UserWarning"
)


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
@pytest.mark.parametrize("shape", [(7, 9), (257, 513)])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("nan_pattern", ["none", "mixed", "all"])
def test_minmax_propagates_nan(dtype, shape, transpose, nan_pattern):
    image = cp.linspace(-1, 2, np.prod(shape), dtype=dtype).reshape(shape)
    if nan_pattern == "mixed":
        image[0, 0] = cp.nan
    elif nan_pattern == "all":
        image[:] = cp.nan
    if transpose:
        image = image.T
    expected = cp.stack((image.min(), image.max()))
    with allow_synchronize(False):
        bounds = _get_minmax(image)
    assert bounds.shape == (2,)
    assert bounds.dtype == image.dtype
    cp.testing.assert_array_equal(bounds, expected)


@pytest.mark.parametrize("dtype", ["int64", "uint64"])
def test_minmax_integer_precision(dtype):
    limits = np.iinfo(dtype)
    image = cp.asarray([limits.min, limits.max], dtype=dtype)
    with allow_synchronize(False):
        bounds = _get_minmax(image)
    assert bounds.get().tolist() == [limits.min, limits.max]
