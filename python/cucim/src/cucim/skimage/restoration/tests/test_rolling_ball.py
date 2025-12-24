# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause


"""
Tests for Rolling Ball Filter
(skimage.restoration.rolling_ball)
"""

import inspect

import cupy as cp
import numpy as np
import pytest
from skimage import data
from skimage.restoration import (
    ellipsoid_kernel as skimage_ellipsoid_kernel,
    rolling_ball as skimage_rolling_ball,
)

from cucim.skimage.restoration import (
    ellipsoid_kernel,
    rolling_ball,
)


@pytest.mark.parametrize(
    "dtype", [np.uint8, np.int32, np.float16, np.float32, np.float64]
)
@pytest.mark.parametrize("structure_and_footprint", [False, True])
def test_ellipsoid_const(dtype, structure_and_footprint):
    img = cp.full((100, 100), 155, dtype=dtype)
    if structure_and_footprint:
        structure, footprint = ellipsoid_kernel(
            (25, 53), 50, structure_and_footprint=structure_and_footprint
        )
        assert footprint.dtype == bool
        background = rolling_ball(img, kernel=(structure, footprint))
    else:
        kernel = ellipsoid_kernel((25, 53), 50)
        background = rolling_ball(img, kernel=kernel)
    assert cp.allclose(img - background, np.zeros_like(img))
    assert background.dtype == img.dtype


def test_nan_const():
    img = np.full((100, 100), 123, dtype=float)
    img[20, 20] = np.nan
    img[50, 53] = np.nan
    img = cp.asarray(img)

    kernel_shape = (10, 10)
    x = np.arange(-kernel_shape[1] // 2, kernel_shape[1] // 2 + 1)[
        np.newaxis, :
    ]
    y = np.arange(-kernel_shape[0] // 2, kernel_shape[0] // 2 + 1)[
        :, np.newaxis
    ]
    expected_img = np.zeros_like(img)
    expected_img[y + 20, x + 20] = np.nan
    expected_img[y + 50, x + 53] = np.nan
    expected_img = cp.asarray(expected_img)

    kernel = ellipsoid_kernel(kernel_shape, 100)
    background = rolling_ball(img, kernel=kernel, nansafe=True)
    assert cp.allclose(img - background, expected_img, equal_nan=True)


@pytest.mark.parametrize("radius", [1, 2.5, 10.346, 50])
def test_const_image(radius):
    # infinite plane light source at top left corner
    img = cp.full((100, 100), 23, dtype=np.uint8)
    background = rolling_ball(img, radius=radius)
    assert cp.allclose(img - background, np.zeros_like(img))


def test_radial_gradient():
    # spot light source at top left corner
    spot_radius = 50
    x, y = np.meshgrid(range(5), range(5))
    img = np.sqrt(np.clip(spot_radius**2 - y**2 - x**2, 0, None))
    img = cp.asarray(img)

    background = rolling_ball(img, radius=5)
    assert cp.allclose(img - background, cp.zeros_like(img))


def test_linear_gradient():
    # linear light source centered at top left corner
    x, y = np.meshgrid(range(100), range(100))
    img = cp.asarray(y * 20 + x * 20)

    expected_img = cp.full_like(img, 19)
    expected_img[0, 0] = 0

    background = rolling_ball(img, radius=1)
    assert cp.allclose(img - background, expected_img)


@pytest.mark.parametrize("radius", [2, 10, 12.5, 50])
def test_preserve_peaks(radius):
    x, y = np.meshgrid(range(100), range(100))
    img = 0 * x + 0 * y + 10
    img[10, 10] = 20
    img[20, 20] = 35
    img[45, 26] = 156
    img = cp.asarray(img)

    expected_img = img - 10
    background = rolling_ball(img, radius=radius)
    assert cp.allclose(img - background, expected_img)


@pytest.mark.parametrize("workers", [None, 2])
def test_workers(workers):
    # workers is unused, just verifying that the API allows specifying it
    img = cp.full((100, 100), 23, dtype=np.uint8)
    rolling_ball(img, radius=10, workers=workers)
    rolling_ball(img, radius=10, nansafe=False, workers=workers)


def test_ndim():
    image = data.cells3d()[:5, 1, ...]
    kernel_args = ((3, 50, 50), 50)
    kernel = ellipsoid_kernel(*kernel_args)
    out = rolling_ball(image, kernel=kernel)

    # validate kernel against scikit-image
    skimage_kernel = skimage_ellipsoid_kernel(*kernel_args)
    mask = np.isfinite(skimage_kernel)
    cp.testing.assert_allclose(
        kernel[mask], skimage_kernel[mask], atol=1e-5, rtol=1e-5
    )

    # validate filtering result against scikit-image
    out_cpu = skimage_rolling_ball(cp.asnumpy(image), kernel=skimage_kernel)
    cp.testing.assert_allclose(out, out_cpu)


@pytest.mark.parametrize("num_threads", [None, 2])
def test_deprecated_num_threads(num_threads):
    img = cp.full((100, 100), 23, dtype=np.uint8)
    with pytest.warns(
        FutureWarning, match=".*`num_threads` is deprecated"
    ) as record:
        rolling_ball(img, radius=10, num_threads=num_threads)
        lineno = inspect.currentframe().f_lineno - 1
    assert len(record) == 1
    assert record[0].filename == __file__
    assert record[0].lineno == lineno
