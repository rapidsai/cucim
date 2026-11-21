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


def test_label_differences_from_scikit_image_are_as_documented():
    # these are listed in the Notes of the docstring of label
    x = cp.asarray([0.9, 1.1, 1.9])
    assert label(x, background=1).tolist() == [
        0,
        0,
        0,
    ]  # scikit-image: [1, 0, 0]
    x = cp.asarray([False, True, False])
    assert label(x, background=2).tolist() == [
        1,
        2,
        3,
    ]  # scikit-image: [0, 1, 0]
    x = cp.ones((1, 1, 1, 1), dtype=cp.uint8)  # NotImplementedError there
    assert label(x).tolist() == [[[[1]]]]
    x = x.astype(bool)
    np.testing.assert_array_equal(label(x).get(), label_cpu(x.get()))


def test_label_zero_dimensional_connectivity():
    image = np.eye(3, dtype=bool)
    connectivity = np.array(2)
    result = label(cp.asarray(image), connectivity=connectivity)
    np.testing.assert_array_equal(
        result.get(), label_cpu(image, connectivity=connectivity)
    )


@pytest.mark.parametrize(
    "limit_kib, expected_pixel_calls", [(344, 0), (280, 1), (264, 1)]
)
def test_label_blocks_memory_limit(
    limit_kib, expected_pixel_calls, monkeypatch
):
    # Allocate the input outside the test pool. The output needs 256 KiB and
    # occupancy and root arrays need another 16 and 64 KiB respectively.
    image = cp.ones((256, 256), dtype=bool)
    pool = cp.cuda.MemoryPool()
    pool.set_limit(size=limit_kib * 1024)
    calls = []
    connect_pixels = _label_kernels._connect

    def record_pixel_connect(*args, **kwargs):
        calls.append(1)
        return connect_pixels(*args, **kwargs)

    monkeypatch.setattr(_label_kernels, "_connect", record_pixel_connect)
    with cp.cuda.using_allocator(pool.malloc):
        labels, count = label(image, return_num=True)

    assert len(calls) == expected_pixel_calls
    assert count == 1
    cp.testing.assert_array_equal(labels, 1)


def test_label_blocks_reinitializes_output_after_sort_allocation_failure(
    monkeypatch,
):
    image = cp.zeros((65, 33), dtype=bool)
    image[::3, ::3] = True
    sort = cp.sort
    failures = []

    def fail_first_sort(values):
        if not failures:
            failures.append(1)
            raise cp.cuda.memory.OutOfMemoryError(values.nbytes, 0)
        return sort(values)

    monkeypatch.setattr(cp, "sort", fail_first_sort)
    labels, count = label(image, return_num=True)
    expected, expected_count = label_cpu(image.get(), return_num=True)

    assert failures
    assert count == expected_count
    np.testing.assert_array_equal(labels.get(), expected)


def test_label_blocks_wide_axis_metadata():
    # Launch only the first logical block of the wide image. Its two pixels
    # fit in the buffers, and no other block is read by the initializer.
    image = cp.ones((1, 2), dtype=bool)
    labels = cp.full((1, 2), -1, dtype=cp.int64)
    occupancy = cp.empty(1, dtype=cp.uint8)
    _label_kernels._kernel_block_init(2)(
        image, 1, 2**31, 1, 2**30, labels, occupancy, size=1
    )
    cp.testing.assert_array_equal(labels, [[0, 0]])
    cp.testing.assert_array_equal(occupancy, [3])


def test_label_pixels_wide_axis_metadata():
    # Only the first two positions of the logical image are visited. At
    # these coordinates the only preceding neighbor is the first pixel.
    image = cp.ones((1, 2), dtype=cp.uint8)
    shape = cp.array([1, 2**31], dtype=cp.int64)
    labels = cp.full((1, 2), -1, dtype=cp.int64)
    dirs = ((0, -1),)
    _label_kernels._kernel_init_link(True, 2, dirs)(
        image, shape, labels, size=2
    )
    cp.testing.assert_array_equal(labels, [[0, 0]])
    _label_kernels._kernel_connect(True, 2, dirs)(image, shape, labels, size=2)
    cp.testing.assert_array_equal(labels, [[0, 0]])


@pytest.mark.parametrize("dtype", [cp.int32, cp.int64])
@pytest.mark.parametrize("shape", [(16385,), (37, 55, 87)])
def test_label_dense_components_across_cuda_blocks(dtype, shape, monkeypatch):
    from cucim.skimage.measure import _label

    monkeypatch.setattr(_label, "_label_dtype_for_size", lambda size: dtype)
    rng = np.random.default_rng(193)
    image = np.ones(shape, dtype=bool)
    if image.ndim == 3:
        image = rng.random(shape) < 0.93
    expected, count = label_cpu(image, return_num=True)
    device_image = cp.asarray(image)

    for _ in range(3):
        result, n = label(device_image, return_num=True)
        assert n == count
        assert result.dtype == dtype
        np.testing.assert_array_equal(result.get(), expected)
