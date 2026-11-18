# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import cupy as cp
import numpy as np
import pytest
from cupy.testing import assert_array_equal, assert_warns

from cucim.skimage._shared._warnings import expected_warnings
from cucim.skimage.morphology import (
    convex_hull_image,
    convex_hull_object,
    diamond,
    octahedron,
)


def test_basic():
    image = cp.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )

    expected = cp.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )

    assert_array_equal(
        convex_hull_image(image, cpu_fallback_threshold=0), expected
    )


def test_empty_image():
    image = cp.zeros((6, 6), dtype=bool)
    with expected_warnings(["entirely zero"]):
        assert_array_equal(
            convex_hull_image(image, cpu_fallback_threshold=0), image
        )


def test_qhull_offset_example():
    nonzeros = (
        (
            [
                1367,
                1368,
                1368,
                1368,
                1369,
                1369,
                1369,
                1369,
                1369,
                1370,
                1370,
                1370,
                1370,
                1370,
                1370,
                1370,
                1371,
                1371,
                1371,
                1371,
                1371,
                1371,
                1371,
                1371,
                1371,
                1372,
                1372,
                1372,
                1372,
                1372,
                1372,
                1372,
                1372,
                1372,
                1373,
                1373,
                1373,
                1373,
                1373,
                1373,
                1373,
                1373,
                1373,
                1374,
                1374,
                1374,
                1374,
                1374,
                1374,
                1374,
                1375,
                1375,
                1375,
                1375,
                1375,
                1376,
                1376,
                1376,
                1377,
                1372,
            ]
        ),
        (
            [
                151,
                150,
                151,
                152,
                149,
                150,
                151,
                152,
                153,
                148,
                149,
                150,
                151,
                152,
                153,
                154,
                147,
                148,
                149,
                150,
                151,
                152,
                153,
                154,
                155,
                146,
                147,
                148,
                149,
                150,
                151,
                152,
                153,
                154,
                146,
                147,
                148,
                149,
                150,
                151,
                152,
                153,
                154,
                147,
                148,
                149,
                150,
                151,
                152,
                153,
                148,
                149,
                150,
                151,
                152,
                149,
                150,
                151,
                150,
                155,
            ]
        ),
    )
    image = np.zeros((1392, 1040), dtype=bool)
    image[nonzeros] = True
    image = cp.asarray(image)
    expected = image.copy()
    assert_array_equal(
        convex_hull_image(image, cpu_fallback_threshold=0), expected
    )


def test_pathological_qhull_example():
    image = cp.array(
        [[0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0, 0]],
        dtype=bool,
    )
    expected = cp.array(
        [[0, 0, 0, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0]],
        dtype=bool,
    )
    assert_array_equal(
        convex_hull_image(image, cpu_fallback_threshold=0), expected
    )


@pytest.mark.skipif(True, reason="include_borders option not implemented")
def test_pathological_qhull_labels():
    image = cp.array(
        [[0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0, 0]],
        dtype=bool,
    )

    expected = cp.array(
        [[0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0]],
        dtype=bool,
    )

    actual = convex_hull_image(
        image, include_borders=False, cpu_fallback_threshold=0
    )
    assert_array_equal(actual, expected)


def test_object():
    image = cp.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )

    expected_conn_1 = cp.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 1, 0, 1],
            [1, 1, 1, 0, 0, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )

    assert_array_equal(
        convex_hull_object(image, connectivity=1, cpu_fallback_threshold=0),
        expected_conn_1,
    )

    expected_conn_2 = cp.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )

    assert_array_equal(
        convex_hull_object(image, connectivity=2, cpu_fallback_threshold=0),
        expected_conn_2,
    )

    with pytest.raises(ValueError):
        convex_hull_object(image, connectivity=3, cpu_fallback_threshold=0)

    out = convex_hull_object(image, connectivity=1, cpu_fallback_threshold=0)
    assert_array_equal(out, expected_conn_1)


def test_non_c_contiguous():
    # 2D Fortran-contiguous
    image = cp.ones((2, 2), order="F", dtype=bool)
    assert_array_equal(
        convex_hull_image(image, cpu_fallback_threshold=0), image
    )
    # 3D Fortran-contiguous
    image = cp.ones((2, 2, 2), order="F", dtype=bool)
    assert_array_equal(
        convex_hull_image(image, cpu_fallback_threshold=0), image
    )
    # 3D non-contiguous
    image = cp.transpose(cp.ones((2, 2, 2), dtype=bool), [0, 2, 1])
    assert_array_equal(
        convex_hull_image(image, cpu_fallback_threshold=0), image
    )


def test_consistent_2d_3d_hulls():
    from cucim.skimage.measure.tests.test_regionprops import SAMPLE as image

    image3d = cp.stack((image, image, image))
    chimage = convex_hull_image(image, cpu_fallback_threshold=0)
    chimage[8, 0] = True  # correct for single point exactly on hull edge
    chimage3d = convex_hull_image(image3d, cpu_fallback_threshold=0)
    assert_array_equal(chimage3d[1], chimage)


def test_few_points():
    image = cp.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    image3d = cp.stack([image, image, image])
    with assert_warns(UserWarning):
        chimage3d = convex_hull_image(
            image3d, offset_coordinates=False, cpu_fallback_threshold=0
        )
        assert_array_equal(chimage3d, cp.zeros(image3d.shape, dtype=bool))

    # non-zero when using offset_coordinates
    # (This is an improvement over skimage v0.25 implementation due to how
    #  initial points are determined without a separate ConvexHull call before
    #  the addistion of the offset coordinates)
    chimage3d = convex_hull_image(
        image3d, offset_coordinates=True, cpu_fallback_threshold=0
    )
    assert chimage3d.sum() > 0


"""
The following test cases are original and could be contributed back upstream
to scikit-image
"""


@pytest.mark.parametrize("radius", [1, 10, 100])
@pytest.mark.parametrize("offset_coordinates", [False, True])
@pytest.mark.parametrize("float64_computation", [False, True])
@pytest.mark.parametrize("omit_empty_coords_check", [False, True])
def test_diamond(
    radius, offset_coordinates, omit_empty_coords_check, float64_computation
):
    expected = diamond(radius)

    # plus sign should become a diamond once convex
    image = cp.zeros_like(expected)
    image[:, radius] = True
    image[radius, :] = True

    chimage = convex_hull_image(
        image,
        offset_coordinates=offset_coordinates,
        omit_empty_coords_check=omit_empty_coords_check,
        float64_computation=float64_computation,
        cpu_fallback_threshold=0,
    )
    if offset_coordinates:
        assert_array_equal(chimage, expected)
    else:
        # may not be an exact match if offset coordinates are not used
        num_mismatch = cp.sum(chimage != expected)
        percent_mismatch = 100 * num_mismatch / expected.sum()
        if float64_computation:
            assert percent_mismatch < 5
        else:
            assert percent_mismatch < 20


@pytest.mark.parametrize("radius", [1, 5, 50])
@pytest.mark.parametrize("offset_coordinates", [False, True])
@pytest.mark.parametrize("float64_computation", [False, True])
@pytest.mark.parametrize("omit_empty_coords_check", [False, True])
def test_octahedron(
    radius, offset_coordinates, omit_empty_coords_check, float64_computation
):
    expected = octahedron(radius)

    # 3D equivalent of the 2D "plus" -> "diamond" test in test_diamond
    image = cp.zeros_like(expected)
    image[:, radius, radius] = True
    image[radius, :, radius] = True
    image[radius, radius, :] = True

    chimage = convex_hull_image(
        image,
        offset_coordinates=offset_coordinates,
        omit_empty_coords_check=omit_empty_coords_check,
        float64_computation=float64_computation,
        cpu_fallback_threshold=0,
    )
    if offset_coordinates:
        assert_array_equal(chimage, expected)
    else:
        # may not be an exact match if offset coordinates are not used
        num_mismatch = cp.sum(chimage != expected)
        percent_mismatch = 100 * num_mismatch / expected.sum()
        if float64_computation:
            assert percent_mismatch < 5
        else:
            assert percent_mismatch < 20


@pytest.mark.parametrize("radius", [1, 10, 100])
@pytest.mark.parametrize("offset_coordinates", [False, True])
@pytest.mark.parametrize("cpu_fallback_threshold", [0, None, 1000000000])
def test_cpu_fallback(radius, offset_coordinates, cpu_fallback_threshold):
    expected = diamond(radius)

    # plus sign should become a diamond once convex
    image = cp.zeros_like(expected)
    image[:, radius] = True
    image[radius, :] = True

    chimage = convex_hull_image(
        image,
        offset_coordinates=offset_coordinates,
        cpu_fallback_threshold=cpu_fallback_threshold,
    )
    if offset_coordinates:
        assert_array_equal(chimage, expected)
    else:
        # may not be an exact match if offset coordinates are not used
        num_mismatch = cp.sum(chimage != expected)
        percent_mismatch = 100 * num_mismatch / expected.sum()
        assert percent_mismatch < 5


# The batch convex-hull kernels (2D and 3D with offset_coordinates=True)
# against scikit-image and against the per-image Qhull path


def _blob_images(ndim, sizes, seeds):
    from skimage import data

    for size in sizes:
        for seed in seeds:
            blobs = data.binary_blobs(
                size,
                blob_size_fraction=0.15,
                n_dim=ndim,
                volume_fraction=0.25,
                rng=seed,
            )
            if blobs.any():
                yield blobs


@pytest.mark.parametrize("ndim, sizes", [(2, (24, 100)), (3, (20, 32))])
def test_convex_hull_image_batch(ndim, sizes):
    morphology_cpu = pytest.importorskip("skimage.morphology")
    from cucim.skimage.morphology.convex_hull import _convex_hull_image

    for blobs in _blob_images(ndim, sizes, range(3)):
        image = cp.asarray(blobs)
        out = convex_hull_image(image, cpu_fallback_threshold=0)
        # the per-image Qhull path
        expected = _convex_hull_image(
            image, cpu_fallback_threshold=0, use_batch_kernels=False
        )
        assert_array_equal(out, expected)
        if ndim == 2:
            # (scikit-image's 3D path takes minutes on these images)
            assert_array_equal(out, morphology_cpu.convex_hull_image(blobs))


@pytest.mark.parametrize("tolerance", [1e-10, 0.3])
def test_convex_hull_image_batch_3d_tolerance(tolerance):
    from cucim.skimage.morphology.convex_hull import _convex_hull_image

    for blobs in _blob_images(3, (24,), range(2)):
        image = cp.asarray(blobs)
        out = convex_hull_image(image, tolerance=tolerance)
        expected = _convex_hull_image(
            image,
            tolerance=tolerance,
            cpu_fallback_threshold=0,
            use_batch_kernels=False,
        )
        assert_array_equal(out, expected)


@pytest.mark.parametrize("connectivity", [1, 2])
def test_convex_hull_object_batch_2d(connectivity):
    morphology_cpu = pytest.importorskip("skimage.morphology")
    for blobs in _blob_images(2, (64, 150), range(3)):
        out = convex_hull_object(cp.asarray(blobs), connectivity=connectivity)
        expected = morphology_cpu.convex_hull_object(
            blobs, connectivity=connectivity
        )
        assert_array_equal(out, expected)


def test_convex_hull_object_batch_3d():
    from cucim.skimage.measure import label

    for blobs in _blob_images(3, (24,), range(2)):
        image = cp.asarray(blobs)
        out = convex_hull_object(image, connectivity=3)
        # the union of the hulls of each object
        labels = label(image, connectivity=3)
        expected = cp.zeros_like(image)
        for i in range(1, int(labels.max()) + 1):
            expected |= convex_hull_image(labels == i)
        assert_array_equal(out, expected)
        # the per-object path (an explicit threshold above the image size)
        assert_array_equal(
            convex_hull_object(
                image, connectivity=3, cpu_fallback_threshold=image.size + 1
            ),
            expected,
        )


def test_convex_hull_object_overlapping_hulls():
    # an L-shaped object (hull: a triangle) and a segment that crosses the
    # triangle without touching the L: the result is the union of the hulls
    first = cp.zeros((12, 12), dtype=bool)
    first[1:10, 1] = True
    first[9, 1:10] = True
    second = cp.zeros((12, 12), dtype=bool)
    second[5, 3:11] = True
    second[4, 10] = True
    hull_first = convex_hull_image(first)
    hull_second = convex_hull_image(second)
    assert bool((hull_first & hull_second).any())
    out = convex_hull_object(first | second)
    assert_array_equal(out, hull_first | hull_second)


@pytest.mark.parametrize("shape", [(1, 9, 11), (9, 1, 11), (5, 1, 9, 7)])
def test_singleton_axes_shape(shape):
    # (the GPU paths: scikit-image, used below cpu_fallback_threshold, returns
    # an empty 3D image with a warning for a single voxel)
    image = cp.zeros(shape, dtype=bool)
    with expected_warnings(["entirely zero"]):
        out = convex_hull_image(image, cpu_fallback_threshold=0)
    assert out.shape == shape
    index = tuple(min(2, s - 1) for s in shape)
    image[index] = True
    out = convex_hull_image(image, cpu_fallback_threshold=0)
    assert out.shape == shape
    assert_array_equal(out, image)
    assert convex_hull_object(image, connectivity=1).shape == shape


@pytest.mark.parametrize("value", [1, 2, 255])
@pytest.mark.parametrize("cpu_fallback_threshold", [0, None])
def test_convex_hull_image_uint8_values(value, cpu_fallback_threshold):
    """Any nonzero uint8 value is foreground (values other than 1 were read
    as labels by the batch path: out-of-bounds atomic writes)."""
    from skimage.morphology import convex_hull_image as convex_hull_image_cpu

    shape = (5, 5) if cpu_fallback_threshold == 0 else (300, 300)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[1, 1] = mask[1, 3] = mask[3, 2] = value
    if shape != (5, 5):
        mask[200, 150] = mask[120, 260] = value
    expected = convex_hull_image_cpu(mask != 0)
    kwargs = dict(cpu_fallback_threshold=cpu_fallback_threshold)
    actual = convex_hull_image(cp.asarray(mask), **kwargs)
    assert_array_equal(actual, cp.asarray(expected))
    assert_array_equal(
        actual, convex_hull_image(cp.asarray(mask != 0), **kwargs)
    )
    assert_array_equal(
        convex_hull_object(cp.asarray(mask), **kwargs),
        convex_hull_object(cp.asarray(mask != 0), **kwargs),
    )


@pytest.mark.parametrize("scale", [1, 16])
@pytest.mark.parametrize("repeat", range(2))
def test_convex_hull_object_overlap_many_blocks(scale, repeat):
    """U-shaped objects enclosing disconnected inner objects: the hulls
    overlap, across many blocks for the larger scale; the result is the
    logical OR of the separately computed hulls."""
    from cucim.skimage.measure import label

    tile = np.zeros((12, 12), dtype=bool)
    tile[1:11, 1] = tile[1:11, 10] = tile[10, 1:11] = True  # a U
    tile[3:7, 4:8] = True  # inside the U, not connected to it
    tile = np.kron(tile, np.ones((scale, scale), dtype=bool))
    image = cp.asarray(np.tile(tile, (4, 5)))
    actual = convex_hull_object(image, cpu_fallback_threshold=0)
    labels = label(image, connectivity=2)
    expected = cp.zeros_like(image)
    for lab in range(1, int(labels.max()) + 1):
        expected |= convex_hull_image(labels == lab, cpu_fallback_threshold=0)
    assert_array_equal(actual, expected)


@pytest.mark.parametrize("empty", [False, True])
def test_convex_hull_object_negative_threshold(empty):
    image = cp.zeros((20, 20), dtype=bool)
    if not empty:
        image[5:8, 5:9] = True
    with pytest.raises(ValueError, match="non-negative"):
        convex_hull_object(image, cpu_fallback_threshold=-1)
