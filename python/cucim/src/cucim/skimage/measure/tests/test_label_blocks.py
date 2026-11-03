# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests of the 2 x 2 (x 2) block kernels of label (boolean input, full connectivity)."""

import cupy as cp
import numpy as np
import pytest
from skimage.measure import label as label_cpu

from cucim.skimage.measure import _label_kernels, label


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1),
        (1, 7),
        (7, 1),
        (2, 2),
        (3, 3),
        (5, 8),
        (31, 17),
        (64, 63),
        (1, 1, 1),
        (2, 2, 2),
        (3, 3, 3),
        (1, 5, 6),
        (5, 1, 6),
        (5, 6, 1),
        (7, 9, 5),
        (16, 15, 17),
    ],
)
@pytest.mark.parametrize("density", [0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
def test_label_blocks_match_scikit_image(shape, density):
    # shapes with odd sizes have blocks that are only partly inside the image
    rng = np.random.default_rng(len(shape) + sum(shape))
    for _ in range(3):
        image = rng.random(shape) < density
        expected = label_cpu(image, connectivity=len(shape))
        result = label(cp.asarray(image), connectivity=len(shape))
        np.testing.assert_array_equal(result.get(), expected)


@pytest.mark.parametrize("ndim", [2, 3])
def test_label_blocks_equal_the_pixel_kernels(ndim, monkeypatch):
    rng = np.random.default_rng(ndim)
    shape = (70, 45, 33)[:ndim]
    image = cp.asarray(rng.random(shape) < 0.45)
    structure = np.ones((3,) * ndim, dtype=bool)
    assert _label_kernels._use_blocks(image, structure, False)
    with_blocks = label(image)
    monkeypatch.setattr(_label_kernels, "_use_blocks", lambda *args: False)
    without_blocks = label(image)
    cp.testing.assert_array_equal(with_blocks, without_blocks)


def test_label_blocks_are_not_used_for_other_inputs():
    image2d = cp.zeros((8, 8), dtype=cp.uint8)
    full = np.ones((3, 3), dtype=bool)
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    use = _label_kernels._use_blocks
    assert not use(image2d, full, True)  # greyscale values must be compared
    assert not use(image2d.astype(bool), cross, False)  # not full connectivity
    assert not use(cp.zeros((4, 4, 4, 4), dtype=bool), np.ones((3,) * 4), False)
    assert use(image2d.astype(bool), full, True)


@pytest.mark.parametrize("shape", [(33, 40), (9, 14, 11)])
def test_label_blocks_with_64_bit_labels(shape, monkeypatch):
    # images with 2**31 or more elements use 64-bit labels; check them on a
    # small image
    from cucim.skimage.measure import _label

    monkeypatch.setattr(_label, "_label_dtype_for_size", lambda size: cp.int64)
    rng = np.random.default_rng(3)
    image = rng.random(shape) < 0.55
    result = label(cp.asarray(image))
    assert result.dtype == cp.int64
    np.testing.assert_array_equal(
        result.get(), label_cpu(image, connectivity=len(shape))
    )


@pytest.mark.parametrize("shape", [(33, 40), (9, 14, 11)])
@pytest.mark.parametrize("background", [True, 1])
def test_label_bool_background_true_uses_blocks(shape, background, monkeypatch):
    calls = []
    original = _label_kernels._label_blocks
    monkeypatch.setattr(
        _label_kernels,
        "_label_blocks",
        lambda x, y: calls.append(1) or original(x, y),
    )
    rng = np.random.default_rng(4)
    image = rng.random(shape) < 0.45
    result, num = label(
        cp.asarray(image), background=background, return_num=True
    )
    assert calls
    expected = label_cpu(image, background=1, connectivity=len(shape))
    np.testing.assert_array_equal(result.get(), expected)
    assert num == expected.max()
