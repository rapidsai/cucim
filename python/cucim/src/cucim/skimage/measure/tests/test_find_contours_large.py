# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests of the 64-bit index kernels of ``find_contours``.

Images of 2**31 or more pixels (and segment offsets of that size) use kernels
with 64-bit indices. The tests run these kernels on small images (by lowering
the limit) and compare with the 32-bit kernels, which must give the same
contours.
"""

import cupy as cp
import numpy as np
import pytest

from cucim.skimage.measure import _find_contours, find_contours


def _image(shape, seed):
    rng = np.random.default_rng(seed)
    return cp.asarray(rng.random(shape))


@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("fully_connected", ["low", "high"])
@pytest.mark.parametrize("shape", [(40, 37), (300, 310)])
def test_wide_kernels_match_narrow(
    shape, fully_connected, use_mask, monkeypatch
):
    # (300, 310) has enough segments for the assembly on the device
    image = _image(shape, seed=0)
    mask = cp.asarray(np.random.default_rng(1).random(shape) > 0.1)
    kwargs = dict(
        level=0.5,
        fully_connected=fully_connected,
        mask=mask if use_mask else None,
    )
    narrow = find_contours(image, **kwargs)
    narrow_packed = find_contours(image, return_packed=True, **kwargs)

    monkeypatch.setattr(_find_contours, "_WIDE_LIMIT", 0)
    assert _find_contours._wide(1)
    wide = find_contours(image, **kwargs)
    wide_packed = find_contours(image, return_packed=True, **kwargs)
    assert len(wide) == len(narrow)
    for a, b in zip(narrow, wide):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(narrow_packed, wide_packed):
        np.testing.assert_array_equal(a, b)


def test_too_many_segments_raises(monkeypatch):
    # lower the limit so that the 64-bit total of the segments exceeds it
    image = _image((40, 37), seed=0)
    monkeypatch.setattr(_find_contours, "_INT32_LIMIT", 10)
    monkeypatch.setattr(_find_contours, "_WIDE_LIMIT", 0)
    with pytest.raises(ValueError, match="too large"):
        find_contours(image, 0.5)


def test_circle_in_an_image_above_the_32_bit_limit():
    # 46500**2 pixels is above 2**31; the image (float64) and the cell counts
    # and scans need about 35 GB
    n = 46_500
    free, _ = cp.cuda.runtime.memGetInfo()
    if free < 40 * 2**30:
        pytest.skip("needs about 40 GB of free device memory")
    center = (n - 1) / 2
    radius = 0.3 * n
    axis = cp.arange(n, dtype=cp.float64) - center
    image = cp.empty((n, n), dtype=cp.float64)
    for start in range(0, n, 4096):  # a block of rows at a time
        rows = axis[start : start + 4096]
        image[start : start + 4096] = cp.sqrt(
            rows[:, None] ** 2 + axis[None, :] ** 2
        )
    contours = find_contours(image, radius)
    del image
    assert len(contours) == 1
    points = cp.asarray(contours[0])
    distance = cp.sqrt(
        (points[:, 0] - center) ** 2 + (points[:, 1] - center) ** 2
    )
    assert float(cp.abs(distance - radius).max()) < 1e-3
    np.testing.assert_allclose(contours[0][0], contours[0][-1])
