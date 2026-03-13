# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import cupy as cp
import pytest
from numpy.testing import assert_almost_equal

from cucim.skimage import data
from cucim.skimage._shared.testing import assert_stacklevel


def test_binary_blobs():
    blobs = data.binary_blobs(length=128)
    assert_almost_equal(blobs.mean(), 0.5, decimal=1)
    blobs = data.binary_blobs(length=128, volume_fraction=0.25)
    assert_almost_equal(blobs.mean(), 0.25, decimal=1)
    blobs = data.binary_blobs(length=32, volume_fraction=0.25, n_dim=3)
    assert_almost_equal(blobs.mean(), 0.25, decimal=1)
    other_realization = data.binary_blobs(
        length=32, volume_fraction=0.25, n_dim=3
    )
    assert not cp.all(blobs == other_realization)


def test_binary_blobs_boundary():
    # Assert that `boundary_mode="wrap"` decreases the pixel difference on
    # opposing borders compared to `boundary_mode="nearest"`
    blobs_near = data.binary_blobs(length=300, boundary_mode="nearest", rng=3)
    blobs_wrap = data.binary_blobs(length=300, boundary_mode="wrap", rng=3)

    diff_near_ax0 = blobs_near[0, :] ^ blobs_near[-1, :]
    diff_wrap_ax0 = blobs_wrap[0, :] ^ blobs_wrap[-1, :]
    assert diff_wrap_ax0.sum() < diff_near_ax0.sum()

    diff_near_ax1 = blobs_near[:, 0] ^ blobs_near[:, -1]
    diff_wrap_ax1 = blobs_wrap[:, 0] ^ blobs_wrap[:, -1]
    assert diff_wrap_ax1.sum() < diff_near_ax1.sum()


def test_binary_blobs_small_blob_size():
    # A very small `blob_size_fraction` in relation to `length` will allocate
    # excessive memory and likely leads to unexpected results. Check that this
    # is gracefully handled
    regex = ".* Clamping to .* blob size of 0.1 pixels"
    with pytest.warns(RuntimeWarning, match=regex) as record:
        result = data.binary_blobs(100, rng=3, blob_size_fraction=0.0009)
    assert_stacklevel(record)
    cp.testing.assert_array_equal(result, 1)
