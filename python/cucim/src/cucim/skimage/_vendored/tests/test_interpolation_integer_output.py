# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integer outputs round half away from zero and saturate, as in SciPy."""

import cupy as cp
import numpy as np
import pytest
from scipy import ndimage as scipy_ndi

from cucim.skimage._vendored import ndimage as cndi

_SMALL_INTEGER_TYPES = [
    np.uint8,
    np.int8,
    np.uint16,
    np.int16,
    np.uint32,
    np.int32,
]
_ALL_INTEGER_TYPES = _SMALL_INTEGER_TYPES + [np.uint64, np.int64]


def _with_channels(image, channels):
    # Trailing channels with zero shift use the loop-batch kernel.
    if channels is None:
        return image, ()
    return np.stack([image] * channels, axis=-1), (0,)


def _shift_both(image, shift, **kwargs):
    expected = scipy_ndi.shift(image, shift, **kwargs)
    actual = cndi.shift(cp.asarray(image), shift, **kwargs)
    return actual, expected


@pytest.mark.parametrize("dtype", _SMALL_INTEGER_TYPES)
@pytest.mark.parametrize("order", [2, 3, 5])
@pytest.mark.parametrize("channels", [None, 3])
def test_spline_overshoot_saturates(dtype, order, channels):
    info = np.iinfo(dtype)
    # Spline ringing at a step between the type's limits leaves its range.
    step = np.array([info.min] * 4 + [info.max] * 4, dtype=np.float64)
    image, extra = _with_channels(step, channels)
    actual, expected = _shift_both(
        image, (0.4,) + extra, order=order, mode="nearest", output=dtype
    )
    unclipped = scipy_ndi.shift(
        image, (0.4,) + extra, order=order, mode="nearest", output=np.float64
    )
    assert (unclipped > info.max).any() and (unclipped < info.min).any()
    cp.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "dtype, sign",
    [
        (np.uint8, 1),
        (np.int8, 1),
        (np.int8, -1),
        (np.int16, -1),
        (np.int32, 1),
        (np.int32, -1),
    ],
)
@pytest.mark.parametrize("channels", [None, 3])
def test_halfway_values_round_away_from_zero(dtype, sign, channels):
    ramp = sign * np.arange(2, 8, dtype=np.float64)
    image, extra = _with_channels(ramp, channels)
    # A half-pixel linear shift makes every interior sample a tie.
    actual, expected = _shift_both(
        image, (0.5,) + extra, order=1, mode="nearest", output=dtype
    )
    cp.testing.assert_array_equal(actual, expected)
    first_channel = actual if channels is None else actual[:, 0]
    cp.testing.assert_array_equal(first_channel, sign * np.arange(2, 8))


@pytest.mark.parametrize("dtype", [np.uint8, np.int16])
@pytest.mark.parametrize("mode", ["constant", "grid-constant"])
@pytest.mark.parametrize("order", [0, 1])
@pytest.mark.parametrize("excess", [45.0, -45.0])
@pytest.mark.parametrize("channels", [None, 3])
def test_out_of_range_fill_saturates(dtype, mode, order, excess, channels):
    info = np.iinfo(dtype)
    cval = (info.max if excess > 0 else info.min) + excess
    image, extra = _with_channels(np.arange(6, dtype=np.float64), channels)
    actual, expected = _shift_both(
        image,
        (2.25,) + extra,
        order=order,
        mode=mode,
        cval=cval,
        output=dtype,
        prefilter=False,
    )
    cp.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", _ALL_INTEGER_TYPES)
def test_nan_samples_store_zero(dtype):
    # SciPy gives 0 for unsigned outputs; its signed conversion of NaN is
    # undefined (0 for 8- and 16-bit and the minimum for 32- and 64-bit
    # outputs on x86). cuCIM stores 0 for every integer type.
    actual = cndi.map_coordinates(
        cp.asarray([np.nan, np.nan, 1.0]),
        cp.asarray([[0.5, 1.5]]),
        order=1,
        output=dtype,
    )
    cp.testing.assert_array_equal(actual, np.zeros(2, dtype=dtype))


@pytest.mark.parametrize("dtype", [np.uint8, np.int16])
@pytest.mark.parametrize("cval", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("order", [0, 1, 3])
def test_nonfinite_grid_constant_fill(dtype, cval, order):
    # The fill is weighted in floating point, so a zero weight times an
    # infinite fill gives NaN (stored as 0), as in SciPy.
    image = np.zeros((2, 4))
    actual, expected = _shift_both(
        image,
        (0.5, -2.5),
        order=order,
        mode="grid-constant",
        cval=cval,
        output=dtype,
    )
    cp.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "dtype, values, expected",
    [
        (np.int64, [2.0**63, 2.0**64, -(2.0**63), -1e30], [1, 1, 0, 0]),
        (np.uint64, [2.0**63, 2.0**64, -1.0, -1e30], [2, 1, 0, 0]),
    ],
)
def test_64bit_limits_saturate(dtype, values, expected):
    # SciPy compares with the limits rounded to double, so it wraps 2**63
    # for int64 and 2**64 for uint64; cuCIM saturates exactly.
    info = np.iinfo(dtype)
    limits = {0: info.min, 1: info.max, 2: np.uint64(2**63)}
    actual = cndi.map_coordinates(
        cp.asarray(values),
        cp.arange(len(values), dtype=cp.float64)[None],
        order=1,
        output=dtype,
    )
    expected = np.array([limits[k] for k in expected], dtype=dtype)
    cp.testing.assert_array_equal(actual, expected)
