# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import pytest
from cupy.testing import assert_array_equal
from skimage.data import astronaut

from cucim.skimage.color import rgb2gray
from cucim.skimage.filters import gaussian
from cucim.skimage.measure import blur_effect


def test_blur_effect():
    """Test that the blur metric increases with more blurring."""
    image = cp.array(astronaut())
    B0 = blur_effect(image, channel_axis=-1)
    B1 = blur_effect(gaussian(image, sigma=1, channel_axis=-1), channel_axis=-1)
    B2 = blur_effect(gaussian(image, sigma=4, channel_axis=-1), channel_axis=-1)
    assert 0 <= B0 < 1
    assert B0 < B1 < B2


def test_blur_effect_h_size():
    """Test that the blur metric decreases with increasing size of the
    re-blurring filter.
    """
    image = cp.array(astronaut())
    B0 = blur_effect(image, h_size=3, channel_axis=-1)
    B1 = blur_effect(image, channel_axis=-1)  # default h_size is 11
    B2 = blur_effect(image, h_size=30, channel_axis=-1)
    assert 0 <= B0 < 1
    assert B0 > B1 > B2


def test_blur_effect_channel_axis():
    """Test that passing an RGB image is equivalent to passing its grayscale
    version.
    """
    image = cp.array(astronaut())
    B0 = blur_effect(image, channel_axis=-1)
    B1 = blur_effect(rgb2gray(image))
    B0_arr = blur_effect(image, channel_axis=-1, reduce_func=None)
    B1_arr = blur_effect(rgb2gray(image), reduce_func=None)
    assert 0 <= B0 < 1
    assert B0 == B1
    assert_array_equal(B0_arr, B1_arr)


def test_blur_effect_3d():
    """Test that the blur metric works on a 3D image."""
    data = pytest.importorskip("skimage.data")
    if not hasattr(data, "cells3d"):
        pytest.skip(
            "cells3d data not available in this version of scikit-image"
        )
    image_3d = cp.array(data.cells3d()[:, 1, :, :])  # grab just the nuclei
    B0 = blur_effect(image_3d)
    B1 = blur_effect(gaussian(image_3d, sigma=1))
    B2 = blur_effect(gaussian(image_3d, sigma=4))
    assert 0 <= B0 < 1
    assert B0 < B1 < B2
