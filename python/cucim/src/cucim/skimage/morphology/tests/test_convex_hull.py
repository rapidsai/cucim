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

    assert_array_equal(convex_hull_image(image), expected)


def test_empty_image():
    image = cp.zeros((6, 6), dtype=bool)
    with expected_warnings(["entirely zero"]):
        assert_array_equal(convex_hull_image(image), image)


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
    assert_array_equal(convex_hull_image(image), expected)


def test_pathological_qhull_example():
    image = cp.array(
        [[0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0, 0]],
        dtype=bool,
    )
    expected = cp.array(
        [[0, 0, 0, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0]],
        dtype=bool,
    )
    assert_array_equal(convex_hull_image(image), expected)


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

    actual = convex_hull_image(image, include_borders=False)
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
        convex_hull_object(image, connectivity=1), expected_conn_1
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
        convex_hull_object(image, connectivity=2), expected_conn_2
    )

    with pytest.raises(ValueError):
        convex_hull_object(image, connectivity=3)

    out = convex_hull_object(image, connectivity=1)
    assert_array_equal(out, expected_conn_1)


def test_non_c_contiguous():
    # 2D Fortran-contiguous
    image = cp.ones((2, 2), order="F", dtype=bool)
    assert_array_equal(convex_hull_image(image), image)
    # 3D Fortran-contiguous
    image = cp.ones((2, 2, 2), order="F", dtype=bool)
    assert_array_equal(convex_hull_image(image), image)
    # 3D non-contiguous
    image = cp.transpose(cp.ones((2, 2, 2), dtype=bool), [0, 2, 1])
    assert_array_equal(convex_hull_image(image), image)


def test_consistent_2d_3d_hulls():
    from cucim.skimage.measure.tests.test_regionprops import SAMPLE as image

    image3d = cp.stack((image, image, image))
    chimage = convex_hull_image(image)
    chimage[8, 0] = True  # correct for single point exactly on hull edge
    chimage3d = convex_hull_image(image3d)
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
        chimage3d = convex_hull_image(image3d, offset_coordinates=False)
        assert_array_equal(chimage3d, cp.zeros(image3d.shape, dtype=bool))

    # non-zero when using offset_coordinates
    # (This is an improvement over skimage v0.25 implementation due to how
    #  initial points are determined without a separate ConvexHull call before
    #  the addistion of the offset coordinates)
    chimage3d = convex_hull_image(image3d, offset_coordinates=True)
    chimage3d.sum() > 0


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
    )
    if offset_coordinates:
        assert_array_equal(chimage, expected)
    else:
        # may not be an exact match if offset coordinates are used
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
    )
    if offset_coordinates:
        assert_array_equal(chimage, expected)
    else:
        # may not be an exact match if offset coordinates are used
        num_mismatch = cp.sum(chimage != expected)
        percent_mismatch = 100 * num_mismatch / expected.sum()
        if float64_computation:
            assert percent_mismatch < 5
        else:
            assert percent_mismatch < 20
