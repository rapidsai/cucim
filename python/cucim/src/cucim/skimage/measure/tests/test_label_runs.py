# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pytest
from skimage.measure import label as label_cpu

from cucim.skimage.measure import _label, _label_kernels, label


@pytest.mark.parametrize("width", [32, 127, 128, 129, 257])
@pytest.mark.parametrize("density", [0.0, 0.5, 1.0])
def test_face_connected_runs_match_scikit_image(width, density, monkeypatch):
    # Runs can span more than one warp and end at partially filled tiles.
    rng = np.random.default_rng(141)
    image = rng.random((3, 5, width)) < density
    expected, expected_count = label_cpu(image, connectivity=1, return_num=True)

    def unexpected_pixel_path(*args):
        pytest.fail("expected the face-connected run path")

    monkeypatch.setattr(_label_kernels, "_connect", unexpected_pixel_path)
    actual, count = label(cp.asarray(image), connectivity=1, return_num=True)

    assert count == expected_count
    np.testing.assert_array_equal(actual.get(), expected)


def test_face_connected_runs_with_winding_components():
    # Each marked plane contains a long one-voxel winding corridor. One
    # voxel at the end of the corridor joins it to the next marked plane.
    image = np.zeros((9, 17, 513), dtype=bool)
    image[::2, ::2, :] = True
    image[::2, 1::4, -1] = True
    image[::2, 3::4, 0] = True
    image[1::4, -1, -1] = True
    image[3::4, 0, 0] = True
    image[1, 1, 20] = True  # A separate component exercises rank propagation.
    expected, expected_count = label_cpu(image, connectivity=1, return_num=True)
    assert expected_count == 2
    device_image = cp.asarray(image)

    for _ in range(3):
        actual, count = label(device_image, connectivity=1, return_num=True)
        assert count == expected_count
        np.testing.assert_array_equal(actual.get(), expected)


@pytest.mark.parametrize("layout", ["strided", "narrow", "wide_indices"])
def test_face_connected_run_fallbacks_match_scikit_image(layout, monkeypatch):
    rng = np.random.default_rng(142)
    image = rng.random((3, 5, 128)) < 0.5
    device_image = cp.asarray(image)
    if layout == "strided":
        device_image = device_image[:, :, ::2]
    elif layout == "narrow":
        device_image = cp.ascontiguousarray(device_image[:, :, :31])
    else:
        monkeypatch.setattr(
            _label, "_label_dtype_for_size", lambda size: cp.int64
        )

    def unexpected_run_path(*args):
        pytest.fail("expected the general pixel path")

    monkeypatch.setattr(
        _label_kernels, "_label_face_connected_runs", unexpected_run_path
    )
    actual, count = label(device_image, connectivity=1, return_num=True)
    expected, expected_count = label_cpu(
        device_image.get(), connectivity=1, return_num=True
    )

    assert count == expected_count
    np.testing.assert_array_equal(actual.get(), expected)
