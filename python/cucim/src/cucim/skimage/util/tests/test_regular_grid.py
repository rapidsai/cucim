# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import cupy as cp

from cucim.skimage.util import regular_grid


def test_regular_grid_full():
    ar = cp.zeros((2, 2))
    g = regular_grid(ar.shape, 25)
    assert g == (slice(None, None, None), slice(None, None, None))
    ar[g] = 1
    assert ar.size == ar.sum()


def test_regular_grid_2d_8():
    ar = cp.zeros((20, 40))
    g = regular_grid(ar.shape, 8)
    assert g == (slice(5.0, None, 10.0), slice(5.0, None, 10.0))
    ar[g] = 1
    assert int(ar.sum()) == 8


def test_regular_grid_2d_32():
    ar = cp.zeros((20, 40))
    g = regular_grid(ar.shape, 32)
    assert g == (slice(2.0, None, 5.0), slice(2.0, None, 5.0))
    ar[g] = 1
    assert int(ar.sum()) == 32


def test_regular_grid_3d_8():
    ar = cp.zeros((3, 20, 40))
    g = regular_grid(ar.shape, 8)
    assert g == (
        slice(1.0, None, 3.0),
        slice(5.0, None, 10.0),
        slice(5.0, None, 10.0),
    )
    ar[g] = 1
    assert int(ar.sum()) == 8
