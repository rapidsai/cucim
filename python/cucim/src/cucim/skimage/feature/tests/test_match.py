# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import math

import cupy as cp
import pytest
from cupy.testing import assert_array_equal
from skimage import data

# TODO: change to cucim.skimage.feature.BRIEF once implemented
from skimage.feature import BRIEF, match_descriptors as match_descriptors_cpu

from cucim.skimage import transform
from cucim.skimage._shared import testing
from cucim.skimage.color import rgb2gray
from cucim.skimage.feature import corner_harris, corner_peaks, match_descriptors


def test_binary_descriptors_unequal_descriptor_sizes_error():
    """Sizes of descriptors of keypoints to be matched should be equal."""
    # fmt: off
    descs1 = cp.array([[True, True, False, True],
                       [False, True, False, True]])
    descs2 = cp.array([[True, False, False, True, False],
                       [False, True, True, True, False]])
    # fmt: on
    with testing.raises(ValueError):
        match_descriptors(descs1, descs2)


def test_binary_descriptors():
    # fmt: off
    descs1 = cp.array([[True, True, False, True, True],
                       [False, True, False, True, True]])
    descs2 = cp.array([[True, False, False, True, False],
                       [False, False, True, True, True]])
    # fmt: on
    matches = match_descriptors(descs1, descs2)
    assert_array_equal(matches, [[0, 0], [1, 1]])


def test_binary_descriptors_rotation_crosscheck_false():
    """Verify matched keypoints and their corresponding masks results between
    image and its rotated version against the CPU reference with
    cross_check disabled."""
    img = cp.array(data.astronaut())
    img = rgb2gray(img)
    tform = transform.SimilarityTransform(
        scale=1, rotation=0.15, translation=(0, 0)
    )
    rotated_img = transform.warp(img, tform, clip=False)

    extractor = BRIEF(descriptor_size=512)

    keypoints1 = corner_peaks(
        corner_harris(img), min_distance=5, threshold_abs=0, threshold_rel=0.1
    )
    extractor.extract(cp.asnumpy(img), cp.asnumpy(keypoints1))
    descriptors1 = cp.array(extractor.descriptors)

    keypoints2 = corner_peaks(
        corner_harris(rotated_img),
        min_distance=5,
        threshold_abs=0,
        threshold_rel=0.1,
    )
    extractor.extract(cp.asnumpy(rotated_img), cp.asnumpy(keypoints2))
    descriptors2 = cp.array(extractor.descriptors)

    matches = match_descriptors(descriptors1, descriptors2, cross_check=False)

    # Image interpolation and BRIEF comparisons can change a descriptor bit
    # across GPUs. Test matching against the CPU using the same descriptors.
    expected = match_descriptors_cpu(
        cp.asnumpy(descriptors1), cp.asnumpy(descriptors2), cross_check=False
    )
    assert len(expected) > 0
    assert_array_equal(matches, expected)

    # minkowski takes a different code path, therefore we test it explicitly
    matches = match_descriptors(
        descriptors1, descriptors2, metric="minkowski", cross_check=False
    )
    assert_array_equal(matches, expected)

    # it also has an extra parameter
    matches = match_descriptors(
        descriptors1, descriptors2, metric="minkowski", p=4, cross_check=False
    )
    assert_array_equal(matches, expected)


def test_binary_descriptors_rotation_crosscheck_true():
    """Verify matched keypoints and their corresponding masks results between
    image and its rotated version against the CPU reference with
    cross_check enabled."""
    img = cp.array(data.astronaut())
    img = rgb2gray(img)
    tform = transform.SimilarityTransform(
        scale=1, rotation=0.15, translation=(0, 0)
    )
    rotated_img = transform.warp(img, tform, clip=False)

    extractor = BRIEF(descriptor_size=512)

    keypoints1 = corner_peaks(
        corner_harris(img), min_distance=5, threshold_abs=0, threshold_rel=0.1
    )
    extractor.extract(cp.asnumpy(img), cp.asnumpy(keypoints1))
    descriptors1 = cp.array(extractor.descriptors)

    keypoints2 = corner_peaks(
        corner_harris(rotated_img),
        min_distance=5,
        threshold_abs=0,
        threshold_rel=0.1,
    )
    extractor.extract(cp.asnumpy(rotated_img), cp.asnumpy(keypoints2))
    descriptors2 = cp.array(extractor.descriptors)

    matches = match_descriptors(descriptors1, descriptors2, cross_check=True)

    expected = match_descriptors_cpu(
        cp.asnumpy(descriptors1), cp.asnumpy(descriptors2), cross_check=True
    )
    assert len(expected) > 0
    assert_array_equal(matches, expected)


@pytest.mark.parametrize("cross_check", [False, True])
@pytest.mark.parametrize(
    "metric, p", [(None, 2), ("minkowski", 2), ("minkowski", 4)]
)
def test_binary_descriptors_ties(cross_check, metric, p):
    descriptors1 = cp.array([[False, False], [False, False], [True, True]])
    descriptors2 = cp.array([[False, True], [True, False], [True, True]])
    matches = match_descriptors(
        descriptors1, descriptors2, metric=metric, p=p, cross_check=cross_check
    )
    # Equal distances choose the first index in both matching directions.
    expected = [[0, 0], [2, 2]] if cross_check else [[0, 0], [1, 0], [2, 2]]
    assert_array_equal(matches, expected)


def test_max_distance():
    descs1 = cp.zeros((10, 128))
    descs2 = cp.zeros((15, 128))

    descs1[0, :] = 1

    matches = match_descriptors(
        descs1, descs2, metric="euclidean", max_distance=0.1, cross_check=False
    )
    assert len(matches) == 9

    matches = match_descriptors(
        descs1,
        descs2,
        metric="euclidean",
        max_distance=math.sqrt(128.1),
        cross_check=False,
    )
    assert len(matches) == 10

    matches = match_descriptors(
        descs1, descs2, metric="euclidean", max_distance=0.1, cross_check=True
    )
    assert_array_equal(matches, [[1, 0]])

    matches = match_descriptors(
        descs1,
        descs2,
        metric="euclidean",
        max_distance=math.sqrt(128.1),
        cross_check=True,
    )
    assert_array_equal(matches, [[1, 0]])


def test_max_ratio():
    descs1 = 10 * cp.arange(10)[:, None].astype(cp.float32)
    descs2 = 10 * cp.arange(15)[:, None].astype(cp.float32)

    descs2[0] = 5.0

    matches = match_descriptors(
        descs1, descs2, metric="euclidean", max_ratio=1.0, cross_check=False
    )
    assert_array_equal(len(matches), 10)

    matches = match_descriptors(
        descs1, descs2, metric="euclidean", max_ratio=0.6, cross_check=False
    )
    assert_array_equal(len(matches), 10)

    matches = match_descriptors(
        descs1, descs2, metric="euclidean", max_ratio=0.5, cross_check=False
    )
    assert_array_equal(len(matches), 9)

    descs1[0] = 7.5

    matches = match_descriptors(
        descs1, descs2, metric="euclidean", max_ratio=0.5, cross_check=False
    )
    assert_array_equal(len(matches), 9)

    descs2 = 10 * cp.arange(1)[:, None].astype(cp.float32)

    matches = match_descriptors(
        descs1, descs2, metric="euclidean", max_ratio=1.0, cross_check=False
    )
    assert_array_equal(len(matches), 10)

    matches = match_descriptors(
        descs1, descs2, metric="euclidean", max_ratio=0.5, cross_check=False
    )
    assert_array_equal(len(matches), 10)

    descs1 = 10 * cp.arange(1)[:, None].astype(cp.float32)

    matches = match_descriptors(
        descs1, descs2, metric="euclidean", max_ratio=1.0, cross_check=False
    )
    assert_array_equal(len(matches), 1)

    matches = match_descriptors(
        descs1, descs2, metric="euclidean", max_ratio=0.5, cross_check=False
    )
    assert_array_equal(len(matches), 1)
