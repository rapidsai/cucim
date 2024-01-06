import math

import cupy as cp
import cupyx.scipy.ndimage as ndi
import numpy as np
import pytest
from cupy.testing import assert_array_almost_equal, assert_array_equal
from numpy.testing import assert_almost_equal, assert_equal
from skimage import data, draw
from skimage.segmentation import slic

from cucim.skimage import transform
from cucim.skimage._vendored import pad
from cucim.skimage.measure import (
    euler_number,
    perimeter,
    perimeter_crofton,
    regionprops,
    regionprops_table,
)
from cucim.skimage.measure._regionprops import (  # noqa
    COL_DTYPES,
    OBJECT_COLUMNS,
    PROPS,
    _inertia_eigvals_to_axes_lengths_3D,
    _parse_docs,
    _props_to_dict,
    _require_intensity_image,
)

# fmt: off
SAMPLE = cp.array(
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
     [1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0],
     [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1],
     [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]
)
# fmt: on
INTENSITY_SAMPLE = SAMPLE.copy()
INTENSITY_SAMPLE[1, 9:11] = 2
INTENSITY_FLOAT_SAMPLE = INTENSITY_SAMPLE.copy().astype(cp.float64) / 10.0

SAMPLE_MULTIPLE = cp.eye(10, dtype=np.int32)
SAMPLE_MULTIPLE[3:5, 7:8] = 2
INTENSITY_SAMPLE_MULTIPLE = SAMPLE_MULTIPLE.copy() * 2.0

SAMPLE_3D = cp.zeros((6, 6, 6), dtype=cp.uint8)
SAMPLE_3D[1:3, 1:3, 1:3] = 1
SAMPLE_3D[3, 2, 2] = 1
INTENSITY_SAMPLE_3D = SAMPLE_3D.copy()


def get_moment_function(img, spacing=(1, 1)):
    rows, cols = img.shape
    Y, X = np.meshgrid(
        cp.linspace(0, rows * spacing[0], rows, endpoint=False),
        cp.linspace(0, cols * spacing[1], cols, endpoint=False),
        indexing="ij",
    )
    return lambda p, q: cp.sum(Y**p * X**q * img)


def get_moment3D_function(img, spacing=(1, 1, 1)):
    slices, rows, cols = img.shape
    Z, Y, X = np.meshgrid(
        cp.linspace(0, slices * spacing[0], slices, endpoint=False),
        cp.linspace(0, rows * spacing[1], rows, endpoint=False),
        cp.linspace(0, cols * spacing[2], cols, endpoint=False),
        indexing="ij",
    )
    return lambda p, q, r: cp.sum(Z**p * Y**q * X**r * img)


def get_central_moment_function(img, spacing=(1, 1)):
    rows, cols = img.shape
    Y, X = np.meshgrid(
        cp.linspace(0, rows * spacing[0], rows, endpoint=False),
        cp.linspace(0, cols * spacing[1], cols, endpoint=False),
        indexing="ij",
    )
    Mpq = get_moment_function(img, spacing=spacing)
    cY = Mpq(1, 0) / Mpq(0, 0)
    cX = Mpq(0, 1) / Mpq(0, 0)
    return lambda p, q: cp.sum((Y - cY) ** p * (X - cX) ** q * img)


def test_all_props():
    region = regionprops(SAMPLE, INTENSITY_SAMPLE)[0]
    for prop in PROPS:
        try:
            # access legacy name via dict
            assert_array_almost_equal(
                region[prop], getattr(region, PROPS[prop])
            )

            # skip property access tests for old CamelCase names
            # (we intentionally do not provide properties for these)
            if prop.lower() == prop:
                # access legacy name via attribute
                assert_array_almost_equal(
                    getattr(region, prop), getattr(region, PROPS[prop])
                )

        except TypeError:  # the `slice` property causes this
            pass


def test_all_props_3d():
    region = regionprops(SAMPLE_3D, INTENSITY_SAMPLE_3D)[0]
    for prop in PROPS:
        try:
            assert_array_almost_equal(
                region[prop], getattr(region, PROPS[prop])
            )

            # skip property access tests for old CamelCase names
            # (we intentionally do not provide properties for these)
            if prop.lower() == prop:
                assert_array_almost_equal(
                    getattr(region, prop), getattr(region, PROPS[prop])
                )

        except (NotImplementedError, TypeError):
            pass


def test_num_pixels():
    num_pixels = regionprops(SAMPLE)[0].num_pixels
    assert num_pixels == 72

    num_pixels = regionprops(SAMPLE, spacing=(2, 1))[0].num_pixels
    assert num_pixels == 72


def test_dtype():
    regionprops(cp.zeros((10, 10), dtype=int))
    regionprops(cp.zeros((10, 10), dtype=cp.uint))
    with pytest.raises(TypeError):
        regionprops(cp.zeros((10, 10), dtype=float))
    with pytest.raises(TypeError):
        regionprops(cp.zeros((10, 10), dtype=cp.float64))
    with pytest.raises(TypeError):
        regionprops(cp.zeros((10, 10), dtype=bool))


def test_ndim():
    regionprops(cp.zeros((10, 10), dtype=int))
    regionprops(cp.zeros((10, 10, 1), dtype=int))
    regionprops(cp.zeros((10, 10, 10), dtype=int))
    regionprops(cp.zeros((1, 1), dtype=int))
    regionprops(cp.zeros((1, 1, 1), dtype=int))
    with pytest.raises(TypeError):
        regionprops(cp.zeros((10, 10, 10, 2), dtype=int))


@pytest.mark.skip("feret_diameter_max not implemented on the GPU")
def test_feret_diameter_max():
    # comparator result is based on SAMPLE from manually-inspected computations
    comparator_result = 18
    test_result = regionprops(SAMPLE)[0].feret_diameter_max
    assert cp.abs(test_result - comparator_result) < 1
    comparator_result_spacing = 10
    test_result_spacing = regionprops(SAMPLE, spacing=[1, 0.1])[
        0
    ].feret_diameter_max  # noqa
    assert cp.abs(test_result_spacing - comparator_result_spacing) < 1
    # square, test that Feret diameter is sqrt(2) * square side
    img = cp.zeros((20, 20), dtype=cp.uint8)
    img[2:-2, 2:-2] = 1
    feret_diameter_max = regionprops(img)[0].feret_diameter_max
    assert cp.abs(feret_diameter_max - 16 * math.sqrt(2)) < 1
    # Due to marching-squares with a level of .5 the diagonal goes
    # from (0, 0.5) to (16, 15.5).
    assert cp.abs(feret_diameter_max - np.sqrt(16**2 + (16 - 1) ** 2)) < 1e-6
    spacing = (2, 1)
    feret_diameter_max = regionprops(img, spacing=spacing)[
        0
    ].feret_diameter_max  # noqa
    # For anisotropic spacing the shift is applied to the smaller spacing.
    assert (
        cp.abs(
            feret_diameter_max
            - cp.sqrt(
                (spacing[0] * 16 - (spacing[0] <= spacing[1])) ** 2
                + (spacing[1] * 16 - (spacing[1] < spacing[0])) ** 2
            )
        )
        < 1e-6
    )


@pytest.mark.skip("feret_diameter_max not implemented on the GPU")
def test_feret_diameter_max_3d():
    img = cp.zeros((20, 20), dtype=cp.uint8)
    img[2:-2, 2:-2] = 1
    img_3d = cp.dstack((img,) * 3)
    feret_diameter_max = regionprops(img_3d)[0].feret_diameter_max
    # Due to marching-cubes with a level of .5 -1=2*0.5 has to be subtracted
    # from two axes. There are three combinations
    # (x-1, y-1, z), (x-1, y, z-1), (x, y-1, z-1).
    # The option yielding the longest diagonal is the computed
    # max_feret_diameter.
    assert (
        cp.abs(
            feret_diameter_max - cp.sqrt((16 - 1) ** 2 + 16**2 + (3 - 1) ** 2)
        )
        < 1e-6
    )  # noqa
    spacing = (1, 2, 3)
    feret_diameter_max = regionprops(img_3d, spacing=spacing)[
        0
    ].feret_diameter_max  # noqa
    # The longest of the three options is the max_feret_diameter
    assert (
        cp.abs(
            feret_diameter_max
            - cp.sqrt(
                (spacing[0] * (16 - 1)) ** 2
                + (spacing[1] * (16 - 0)) ** 2
                + (spacing[2] * (3 - 1)) ** 2
            )
        )
        < 1e-6
    )
    assert (
        cp.abs(
            feret_diameter_max
            - cp.sqrt(
                (spacing[0] * (16 - 1)) ** 2
                + (spacing[1] * (16 - 1)) ** 2
                + (spacing[2] * (3 - 0)) ** 2
            )
        )
        > 1e-6
    )
    assert (
        cp.abs(
            feret_diameter_max
            - cp.sqrt(
                (spacing[0] * (16 - 0)) ** 2
                + (spacing[1] * (16 - 1)) ** 2
                + (spacing[2] * (3 - 1)) ** 2
            )
        )
        > 1e-6
    )


def test_area():
    area = regionprops(SAMPLE)[0].area
    assert area == cp.sum(SAMPLE)
    spacing = (1, 2)
    area = regionprops(SAMPLE, spacing=spacing)[0].area
    assert area == cp.sum(SAMPLE * math.prod(spacing))
    area = regionprops(SAMPLE_3D)[0].area
    assert area == cp.sum(SAMPLE_3D)
    spacing = (2, 1, 3)
    area = regionprops(SAMPLE_3D, spacing=spacing)[0].area
    assert area == cp.sum(SAMPLE_3D * math.prod(spacing))


def test_bbox():
    bbox = regionprops(SAMPLE)[0].bbox
    assert_array_almost_equal(bbox, (0, 0, SAMPLE.shape[0], SAMPLE.shape[1]))

    bbox = regionprops(SAMPLE, spacing=(1, 2))[0].bbox
    assert_array_almost_equal(bbox, (0, 0, SAMPLE.shape[0], SAMPLE.shape[1]))

    SAMPLE_mod = SAMPLE.copy()
    SAMPLE_mod[:, -1] = 0
    bbox = regionprops(SAMPLE_mod)[0].bbox
    assert_array_almost_equal(
        bbox, (0, 0, SAMPLE.shape[0], SAMPLE.shape[1] - 1)
    )
    bbox = regionprops(SAMPLE_mod, spacing=(3, 2))[0].bbox
    assert_array_almost_equal(
        bbox, (0, 0, SAMPLE.shape[0], SAMPLE.shape[1] - 1)
    )

    bbox = regionprops(SAMPLE_3D)[0].bbox
    assert_array_almost_equal(bbox, (1, 1, 1, 4, 3, 3))
    bbox = regionprops(SAMPLE_3D, spacing=(0.5, 2, 7))[0].bbox
    assert_array_almost_equal(bbox, (1, 1, 1, 4, 3, 3))


def test_area_bbox():
    padded = pad(SAMPLE, 5, mode="constant")
    bbox_area = regionprops(padded)[0].area_bbox
    assert_array_almost_equal(bbox_area, SAMPLE.size)

    spacing = (0.5, 3)
    bbox_area = regionprops(padded, spacing=spacing)[0].area_bbox
    assert_array_almost_equal(bbox_area, SAMPLE.size * math.prod(spacing))


def test_moments_central():
    mu = regionprops(SAMPLE)[0].moments_central
    # determined with OpenCV
    assert_almost_equal(mu[2, 0], 436.00000000000045, decimal=4)
    # different from OpenCV results, bug in OpenCV
    assert_almost_equal(mu[3, 0], -737.333333333333, decimal=3)
    assert_almost_equal(mu[1, 1], -87.33333333333303, decimal=3)
    assert_almost_equal(mu[2, 1], -127.5555555555593, decimal=3)
    assert_almost_equal(mu[0, 2], 1259.7777777777774, decimal=2)
    assert_almost_equal(mu[1, 2], 2000.296296296291, decimal=2)
    assert_almost_equal(mu[0, 3], -760.0246913580195, decimal=2)

    # Verify central moment test functions
    centralMpq = get_central_moment_function(SAMPLE, spacing=(1, 1))
    assert_almost_equal(centralMpq(2, 0), mu[2, 0], decimal=3)
    assert_almost_equal(centralMpq(3, 0), mu[3, 0], decimal=3)
    assert_almost_equal(centralMpq(1, 1), mu[1, 1], decimal=3)
    assert_almost_equal(centralMpq(2, 1), mu[2, 1], decimal=3)
    assert_almost_equal(centralMpq(0, 2), mu[0, 2], decimal=3)
    assert_almost_equal(centralMpq(1, 2), mu[1, 2], decimal=3)
    assert_almost_equal(centralMpq(0, 3), mu[0, 3], decimal=3)

    # Test spacing against verified central moment test function
    spacing = (1.8, 0.8)
    centralMpq = get_central_moment_function(SAMPLE, spacing=spacing)

    mu = regionprops(SAMPLE, spacing=spacing)[0].moments_central
    assert_almost_equal(mu[2, 0], centralMpq(2, 0), decimal=3)
    assert_almost_equal(mu[3, 0], centralMpq(3, 0), decimal=2)
    assert_almost_equal(mu[1, 1], centralMpq(1, 1), decimal=3)
    assert_almost_equal(mu[2, 1], centralMpq(2, 1), decimal=2)
    assert_almost_equal(mu[0, 2], centralMpq(0, 2), decimal=3)
    assert_almost_equal(mu[1, 2], centralMpq(1, 2), decimal=2)
    assert_almost_equal(mu[0, 3], centralMpq(0, 3), decimal=2)


def test_centroid():
    centroid = regionprops(SAMPLE)[0].centroid
    # determined with MATLAB
    assert_array_almost_equal(centroid, (5.66666666666666, 9.444444444444444))

    # Verify test moment function with spacing=(1, 1)
    Mpq = get_moment_function(SAMPLE, spacing=(1, 1))
    cY = float(Mpq(1, 0) / Mpq(0, 0))
    cX = float(Mpq(0, 1) / Mpq(0, 0))

    assert_array_almost_equal((cY, cX), centroid)

    spacing = (1.8, 0.8)
    # Moment
    Mpq = get_moment_function(SAMPLE, spacing=spacing)
    cY = float(Mpq(1, 0) / Mpq(0, 0))
    cX = float(Mpq(0, 1) / Mpq(0, 0))

    centroid = regionprops(SAMPLE, spacing=spacing)[0].centroid
    assert_array_almost_equal(centroid, (cY, cX))


def test_centroid_3d():
    centroid = regionprops(SAMPLE_3D)[0].centroid
    # determined by mean along axis 1 of SAMPLE_3D.nonzero()
    assert_array_almost_equal(centroid, (1.66666667, 1.55555556, 1.55555556))

    # Verify moment 3D test function
    Mpqr = get_moment3D_function(SAMPLE_3D, spacing=(1, 1, 1))
    cZ = float(Mpqr(1, 0, 0) / Mpqr(0, 0, 0))
    cY = float(Mpqr(0, 1, 0) / Mpqr(0, 0, 0))
    cX = float(Mpqr(0, 0, 1) / Mpqr(0, 0, 0))
    assert_array_almost_equal((cZ, cY, cX), centroid)

    # Test spacing
    spacing = (2, 1, 0.8)
    Mpqr = get_moment3D_function(SAMPLE_3D, spacing=spacing)
    cZ = float(Mpqr(1, 0, 0) / Mpqr(0, 0, 0))
    cY = float(Mpqr(0, 1, 0) / Mpqr(0, 0, 0))
    cX = float(Mpqr(0, 0, 1) / Mpqr(0, 0, 0))
    centroid = regionprops(SAMPLE_3D, spacing=spacing)[0].centroid
    assert_array_almost_equal(centroid, (cZ, cY, cX))


def test_area_convex():
    area = regionprops(SAMPLE)[0].area_convex
    assert area == 125

    spacing = (1, 4)
    area = regionprops(SAMPLE, spacing=spacing)[0].area_convex
    assert area == 125 * np.prod(spacing)


def test_image_convex():
    img = regionprops(SAMPLE)[0].image_convex
    # fmt: off
    ref = cp.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
         [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    )
    # fmt: on
    assert_array_equal(img, ref)


def test_coordinates():
    sample = cp.zeros((10, 10), dtype=cp.int8)
    coords = cp.array([[3, 2], [3, 3], [3, 4]])
    sample[coords[:, 0], coords[:, 1]] = 1
    prop_coords = regionprops(sample)[0].coords
    assert_array_equal(prop_coords, coords)
    prop_coords = regionprops(sample, spacing=(0.5, 1.2))[0].coords
    assert_array_equal(prop_coords, coords)


def test_coordinates_scaled():
    sample = cp.zeros((10, 10), dtype=np.int8)
    coords = cp.array([[3, 2], [3, 3], [3, 4]])
    sample[coords[:, 0], coords[:, 1]] = 1

    spacing = (1, 1)
    prop_coords = regionprops(sample, spacing=spacing)[0].coords_scaled
    assert_array_equal(prop_coords, coords * cp.array(spacing))

    spacing = (1, 0.5)
    prop_coords = regionprops(sample, spacing=spacing)[0].coords_scaled
    assert_array_equal(prop_coords, coords * cp.array(spacing))

    sample = cp.zeros((6, 6, 6), dtype=cp.int8)
    coords = cp.array([[1, 1, 1], [1, 2, 1], [1, 3, 1]])
    sample[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    prop_coords = regionprops(sample)[0].coords_scaled
    assert_array_equal(prop_coords, coords)

    spacing = (0.2, 3, 2.3)
    prop_coords = regionprops(sample, spacing=spacing)[0].coords_scaled
    assert_array_equal(prop_coords, coords * cp.array(spacing))


def test_slice():
    padded = pad(SAMPLE, ((2, 4), (5, 2)), mode="constant")
    nrow, ncol = SAMPLE.shape
    result = regionprops(padded)[0].slice
    expected = (slice(2, 2 + nrow), slice(5, 5 + ncol))
    assert_array_equal(result, expected)

    spacing = (2, 0.2)
    result = regionprops(padded, spacing=spacing)[0].slice
    assert_equal(result, expected)


def test_eccentricity():
    eps = regionprops(SAMPLE)[0].eccentricity
    assert_almost_equal(eps, 0.814629313427)

    eps = regionprops(SAMPLE, spacing=(1.5, 1.5))[0].eccentricity
    assert_almost_equal(eps, 0.814629313427)

    img = cp.zeros((5, 5), dtype=int)
    img[2, 2] = 1
    eps = regionprops(img)[0].eccentricity
    assert_almost_equal(eps, 0)

    eps = regionprops(img, spacing=(3, 3))[0].eccentricity
    assert_almost_equal(eps, 0)


def test_equivalent_diameter_area():
    diameter = regionprops(SAMPLE)[0].equivalent_diameter_area
    # determined with MATLAB
    assert_almost_equal(diameter, 9.57461472963)

    spacing = (1, 3)
    diameter = regionprops(SAMPLE, spacing=spacing)[0].equivalent_diameter_area
    equivalent_area = cp.pi * (diameter / 2.0) ** 2
    assert_almost_equal(equivalent_area, SAMPLE.sum() * math.prod(spacing))


def test_euler_number():
    for spacing in [(1, 1), (2.1, 0.9)]:
        en = regionprops(SAMPLE, spacing=spacing)[0].euler_number
        assert en == 0

        SAMPLE_mod = SAMPLE.copy()
        SAMPLE_mod[7, -3] = 0
        en = regionprops(SAMPLE_mod, spacing=spacing)[0].euler_number
        assert en == -1

        en = euler_number(SAMPLE, 1)
        assert en == 2

        en = euler_number(SAMPLE_mod, 1)
        assert en == 1

    en = euler_number(SAMPLE_3D, 1)
    assert en == 1

    en = euler_number(SAMPLE_3D, 3)
    assert en == 1

    # for convex body, Euler number is 1
    SAMPLE_3D_2 = cp.zeros((100, 100, 100))
    SAMPLE_3D_2[40:60, 40:60, 40:60] = 1
    en = euler_number(SAMPLE_3D_2, 3)
    assert en == 1

    SAMPLE_3D_2[45:55, 45:55, 45:55] = 0
    en = euler_number(SAMPLE_3D_2, 3)
    assert en == 2


def test_extent():
    extent = regionprops(SAMPLE)[0].extent
    assert_almost_equal(extent, 0.4)
    extent = regionprops(SAMPLE, spacing=(5, 0.2))[0].extent
    assert_almost_equal(extent, 0.4)


def test_moments_hu():
    hu = regionprops(SAMPLE)[0].moments_hu
    # fmt: off
    ref = cp.array([
        3.27117627e-01,
        2.63869194e-02,
        2.35390060e-02,
        1.23151193e-03,
        1.38882330e-06,
        -2.72586158e-05,
        -6.48350653e-06
    ])
    # fmt: on
    # bug in OpenCV caused in Central Moments calculation?
    assert_array_almost_equal(hu, ref)

    with pytest.raises(NotImplementedError):
        regionprops(SAMPLE, spacing=(2, 1))[0].moments_hu


def test_image():
    img = regionprops(SAMPLE)[0].image
    assert_array_equal(img, SAMPLE)

    img = regionprops(SAMPLE_3D)[0].image
    assert_array_equal(img, SAMPLE_3D[1:4, 1:3, 1:3])


def test_label():
    label = regionprops(SAMPLE)[0].label
    assert_array_equal(label, 1)

    label = regionprops(SAMPLE_3D)[0].label
    assert_array_equal(label, 1)


def test_area_filled():
    area = regionprops(SAMPLE)[0].area_filled
    assert area == cp.sum(SAMPLE)

    spacing = (2, 1.2)
    area = regionprops(SAMPLE, spacing=spacing)[0].area_filled
    assert area == cp.sum(SAMPLE) * math.prod(spacing)

    SAMPLE_mod = SAMPLE.copy()
    SAMPLE_mod[7, -3] = 0
    area = regionprops(SAMPLE_mod)[0].area_filled
    assert area == cp.sum(SAMPLE)

    area = regionprops(SAMPLE_mod, spacing=spacing)[0].area_filled
    assert area == cp.sum(SAMPLE) * math.prod(spacing)


def test_image_filled():
    img = regionprops(SAMPLE)[0].image_filled
    assert_array_equal(img, SAMPLE)
    img = regionprops(SAMPLE, spacing=(1, 4))[0].image_filled
    assert_array_equal(img, SAMPLE)


def test_axis_major_length():
    length = regionprops(SAMPLE)[0].axis_major_length
    # MATLAB has different interpretation of ellipse than found in literature,
    # here implemented as found in literature
    target_length = 16.7924234999
    assert_almost_equal(length, target_length, decimal=4)

    length = regionprops(SAMPLE, spacing=(2, 2))[0].axis_major_length
    assert_almost_equal(length, 2 * target_length, decimal=4)

    from skimage.draw import ellipse

    img = cp.zeros((20, 24), dtype=cp.uint8)
    rr, cc = ellipse(11, 11, 7, 9, rotation=np.deg2rad(45))
    img[rr, cc] = 1

    target_length = regionprops(img, spacing=(1, 1))[0].axis_major_length
    length_wo_spacing = regionprops(img[::2], spacing=(1, 1))[
        0
    ].axis_minor_length
    assert abs(length_wo_spacing - target_length) > 0.1
    length = regionprops(img[:, ::2], spacing=(1, 2))[0].axis_major_length
    assert_almost_equal(length, target_length, decimal=0)


def test_intensity_max():
    intensity = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE)[
        0
    ].intensity_max
    assert_almost_equal(intensity, 2)


def test_intensity_mean():
    intensity = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE)[
        0
    ].intensity_mean
    assert_almost_equal(intensity, 1.02777777777777)


def test_intensity_min():
    intensity = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE)[
        0
    ].intensity_min
    assert_almost_equal(intensity, 1)


def test_axis_minor_length():
    length = regionprops(SAMPLE)[0].axis_minor_length
    # MATLAB has different interpretation of ellipse than found in literature,
    # here implemented as found in literature
    target_length = 9.739302807263
    assert_almost_equal(length, target_length, decimal=5)

    length = regionprops(SAMPLE, spacing=(1.5, 1.5))[0].axis_minor_length
    assert_almost_equal(length, 1.5 * target_length, decimal=5)

    from skimage.draw import ellipse

    img = cp.zeros((10, 12), dtype=np.uint8)
    rr, cc = ellipse(5, 6, 3, 5, rotation=np.deg2rad(30))
    img[rr, cc] = 1

    target_length = regionprops(img, spacing=(1, 1))[0].axis_minor_length
    length_wo_spacing = regionprops(img[::2], spacing=(1, 1))[
        0
    ].axis_minor_length
    assert abs(length_wo_spacing - target_length) > 0.1
    length = regionprops(img[::2], spacing=(2, 1))[0].axis_minor_length
    assert_almost_equal(length, target_length, decimal=1)


def test_moments():
    m = regionprops(SAMPLE)[0].moments
    # determined with OpenCV
    assert_almost_equal(m[0, 0], 72.0)
    assert_almost_equal(m[0, 1], 680.0)
    assert_almost_equal(m[0, 2], 7682.0)
    assert_almost_equal(m[0, 3], 95588.0)
    assert_almost_equal(m[1, 0], 408.0)
    assert_almost_equal(m[1, 1], 3766.0)
    assert_almost_equal(m[1, 2], 43882.0)
    assert_almost_equal(m[2, 0], 2748.0)
    assert_almost_equal(m[2, 1], 24836.0)
    assert_almost_equal(m[3, 0], 19776.0)

    # Verify moment test function
    Mpq = get_moment_function(SAMPLE, spacing=(1, 1))
    assert_almost_equal(Mpq(0, 0), m[0, 0])
    assert_almost_equal(Mpq(0, 1), m[0, 1])
    assert_almost_equal(Mpq(0, 2), m[0, 2])
    assert_almost_equal(Mpq(0, 3), m[0, 3])
    assert_almost_equal(Mpq(1, 0), m[1, 0])
    assert_almost_equal(Mpq(1, 1), m[1, 1])
    assert_almost_equal(Mpq(1, 2), m[1, 2])
    assert_almost_equal(Mpq(2, 0), m[2, 0])
    assert_almost_equal(Mpq(2, 1), m[2, 1])
    assert_almost_equal(Mpq(3, 0), m[3, 0])

    # Test moment on spacing
    spacing = (2, 0.3)
    m = regionprops(SAMPLE, spacing=spacing)[0].moments
    Mpq = get_moment_function(SAMPLE, spacing=spacing)
    assert_almost_equal(m[0, 0], Mpq(0, 0), decimal=3)
    assert_almost_equal(m[0, 1], Mpq(0, 1), decimal=3)
    assert_almost_equal(m[0, 2], Mpq(0, 2), decimal=3)
    assert_almost_equal(m[0, 3], Mpq(0, 3), decimal=3)
    assert_almost_equal(m[1, 0], Mpq(1, 0), decimal=3)
    assert_almost_equal(m[1, 1], Mpq(1, 1), decimal=3)
    assert_almost_equal(m[1, 2], Mpq(1, 2), decimal=3)
    assert_almost_equal(m[2, 0], Mpq(2, 0), decimal=3)
    assert_almost_equal(m[2, 1], Mpq(2, 1), decimal=2)
    assert_almost_equal(m[3, 0], Mpq(3, 0), decimal=3)


def test_moments_normalized():
    nu = regionprops(SAMPLE)[0].moments_normalized

    # determined with OpenCV
    assert_almost_equal(nu[0, 2], 0.24301268861454037)
    assert_almost_equal(nu[0, 3], -0.017278118992041805)
    assert_almost_equal(nu[1, 1], -0.016846707818929982)
    assert_almost_equal(nu[1, 2], 0.045473992910668816)
    assert_almost_equal(nu[2, 0], 0.08410493827160502)
    assert_almost_equal(nu[2, 1], -0.002899800614433943)

    spacing = (3, 3)
    nu = regionprops(SAMPLE, spacing=spacing)[0].moments_normalized

    # Normalized moments are scale invariant.
    assert_almost_equal(nu[0, 2], 0.24301268861454037)
    assert_almost_equal(nu[0, 3], -0.017278118992041805)
    assert_almost_equal(nu[1, 1], -0.016846707818929982)
    assert_almost_equal(nu[1, 2], 0.045473992910668816)
    assert_almost_equal(nu[2, 0], 0.08410493827160502)
    assert_almost_equal(nu[2, 1], -0.002899800614433943)


def test_orientation():
    orient = regionprops(SAMPLE)[0].orientation
    # determined with MATLAB
    target_orient = -1.4663278802756865
    assert_almost_equal(orient, target_orient)

    orient = regionprops(SAMPLE, spacing=(2, 2))[0].orientation
    assert_almost_equal(orient, target_orient)

    # test diagonal regions
    diag = cp.eye(10, dtype=int)
    orient_diag = regionprops(diag)[0].orientation
    assert_almost_equal(orient_diag, -math.pi / 4)
    orient_diag = regionprops(diag, spacing=(1, 2))[0].orientation
    assert_almost_equal(orient_diag, np.arccos(0.5 / math.sqrt(1 + 0.5**2)))
    orient_diag = regionprops(cp.flipud(diag))[0].orientation
    assert_almost_equal(orient_diag, math.pi / 4)
    orient_diag = regionprops(cp.flipud(diag), spacing=(1, 2))[0].orientation
    assert_almost_equal(orient_diag, -np.arccos(0.5 / math.sqrt(1 + 0.5**2)))
    orient_diag = regionprops(cp.fliplr(diag))[0].orientation
    assert_almost_equal(orient_diag, math.pi / 4)
    orient_diag = regionprops(cp.fliplr(diag), spacing=(1, 2))[0].orientation
    assert_almost_equal(orient_diag, -np.arccos(0.5 / math.sqrt(1 + 0.5**2)))
    orient_diag = regionprops(cp.fliplr(cp.flipud(diag)))[0].orientation
    assert_almost_equal(orient_diag, -math.pi / 4)
    orient_diag = regionprops(np.fliplr(np.flipud(diag)), spacing=(1, 2))[
        0
    ].orientation
    assert_almost_equal(orient_diag, np.arccos(0.5 / math.sqrt(1 + 0.5**2)))


def test_perimeter():
    per = regionprops(SAMPLE)[0].perimeter
    target_per = 55.2487373415
    assert_almost_equal(per, target_per)
    per = regionprops(SAMPLE, spacing=(2, 2))[0].perimeter
    assert_almost_equal(per, 2 * target_per)

    per = perimeter(SAMPLE.astype(float), neighborhood=8)
    assert_almost_equal(per, 46.8284271247)

    with pytest.raises(NotImplementedError):
        per = regionprops(SAMPLE, spacing=(2, 1))[0].perimeter


def test_perimeter_crofton():
    per = regionprops(SAMPLE)[0].perimeter_crofton
    target_per_crof = 61.0800637973
    assert_almost_equal(per, target_per_crof)
    per = regionprops(SAMPLE, spacing=(2, 2))[0].perimeter_crofton
    assert_almost_equal(per, 2 * target_per_crof)

    per = perimeter_crofton(SAMPLE.astype("double"), directions=2)
    assert_almost_equal(per, 64.4026493985)

    with pytest.raises(NotImplementedError):
        per = regionprops(SAMPLE, spacing=(2, 1))[0].perimeter_crofton


def test_solidity():
    solidity = regionprops(SAMPLE)[0].solidity
    target_solidity = 0.576
    assert_almost_equal(solidity, target_solidity)

    solidity = regionprops(SAMPLE, spacing=(3, 9))[0].solidity
    assert_almost_equal(solidity, target_solidity)


def test_moments_weighted_central():
    wmu = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE)[
        0
    ].moments_weighted_central
    # fmt: off
    ref = cp.array(
        [[7.4000000000e+01, 3.7303493627e-14, 1.2602837838e+03,
          -7.6561796932e+02],
         [-2.1316282073e-13, -8.7837837838e+01, 2.1571526662e+03,
          -4.2385971907e+03],
         [4.7837837838e+02, -1.4801314828e+02, 6.6989799420e+03,
          -9.9501164076e+03],
         [-7.5943608473e+02, -1.2714707125e+03, 1.5304076361e+04,
          -3.3156729271e+04]])
    # fmt: on
    np.set_printoptions(precision=10)
    assert_array_almost_equal(wmu, ref)

    # Verify test function
    centralMpq = get_central_moment_function(INTENSITY_SAMPLE, spacing=(1, 1))
    assert_almost_equal(centralMpq(0, 0), ref[0, 0])
    assert_almost_equal(centralMpq(0, 1), ref[0, 1])
    assert_almost_equal(centralMpq(0, 2), ref[0, 2])
    assert_almost_equal(centralMpq(0, 3), ref[0, 3])
    assert_almost_equal(centralMpq(1, 0), ref[1, 0])
    assert_almost_equal(centralMpq(1, 1), ref[1, 1])
    assert_almost_equal(centralMpq(1, 2), ref[1, 2])
    assert_almost_equal(centralMpq(1, 3), ref[1, 3])
    assert_almost_equal(centralMpq(2, 0), ref[2, 0])
    assert_almost_equal(centralMpq(2, 1), ref[2, 1])
    assert_almost_equal(centralMpq(2, 2), ref[2, 2])
    assert_almost_equal(centralMpq(2, 3), ref[2, 3])
    assert_almost_equal(centralMpq(3, 0), ref[3, 0])
    assert_almost_equal(centralMpq(3, 1), ref[3, 1])
    assert_almost_equal(centralMpq(3, 2), ref[3, 2])
    assert_almost_equal(centralMpq(3, 3), ref[3, 3])

    # Test spacing
    spacing = (3.2, 1.2)
    wmu = regionprops(
        SAMPLE, intensity_image=INTENSITY_SAMPLE, spacing=spacing
    )[0].moments_weighted_central
    centralMpq = get_central_moment_function(INTENSITY_SAMPLE, spacing=spacing)
    assert_almost_equal(wmu[0, 0], centralMpq(0, 0))
    assert_almost_equal(wmu[0, 1], centralMpq(0, 1))
    assert_almost_equal(wmu[0, 2], centralMpq(0, 2))
    assert_almost_equal(wmu[0, 3], centralMpq(0, 3))
    assert_almost_equal(wmu[1, 0], centralMpq(1, 0))
    assert_almost_equal(wmu[1, 1], centralMpq(1, 1))
    assert_almost_equal(wmu[1, 2], centralMpq(1, 2))
    assert_almost_equal(wmu[1, 3], centralMpq(1, 3))
    assert_almost_equal(wmu[2, 0], centralMpq(2, 0))
    assert_almost_equal(wmu[2, 1], centralMpq(2, 1))
    assert_almost_equal(wmu[2, 2], centralMpq(2, 2))
    assert_almost_equal(wmu[2, 3], centralMpq(2, 3))
    assert_almost_equal(wmu[3, 0], centralMpq(3, 0))
    assert_almost_equal(wmu[3, 1], centralMpq(3, 1))
    assert_almost_equal(wmu[3, 2], centralMpq(3, 2))
    assert_almost_equal(wmu[3, 3], centralMpq(3, 3))


def test_centroid_weighted():
    centroid = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE)[
        0
    ].centroid_weighted
    target_centroid = (5.540540540540, 9.445945945945)
    centroid = tuple(float(c) for c in centroid)
    assert_array_almost_equal(centroid, target_centroid)

    # Verify test function
    Mpq = get_moment_function(INTENSITY_SAMPLE, spacing=(1, 1))
    cY = float(Mpq(0, 1) / Mpq(0, 0))
    cX = float(Mpq(1, 0) / Mpq(0, 0))
    assert_almost_equal((cX, cY), centroid)

    # Test spacing
    spacing = (2, 2)
    Mpq = get_moment_function(INTENSITY_SAMPLE, spacing=spacing)
    cY = float(Mpq(0, 1) / Mpq(0, 0))
    cX = float(Mpq(1, 0) / Mpq(0, 0))
    centroid = regionprops(
        SAMPLE, intensity_image=INTENSITY_SAMPLE, spacing=spacing
    )[0].centroid_weighted
    centroid = tuple(float(c) for c in centroid)
    assert_almost_equal(centroid, (cX, cY))
    assert_almost_equal(centroid, tuple(2 * c for c in target_centroid))

    spacing = (1.3, 0.7)
    Mpq = get_moment_function(INTENSITY_SAMPLE, spacing=spacing)
    cY = float(Mpq(0, 1) / Mpq(0, 0))
    cX = float(Mpq(1, 0) / Mpq(0, 0))
    centroid = regionprops(
        SAMPLE, intensity_image=INTENSITY_SAMPLE, spacing=spacing
    )[0].centroid_weighted
    centroid = tuple(float(c) for c in centroid)
    assert_almost_equal(centroid, (cX, cY))


def test_moments_weighted_hu():
    whu = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE)[
        0
    ].moments_weighted_hu
    # fmt: off
    ref = cp.array([
        3.1750587329e-01,
        2.1417517159e-02,
        2.3609322038e-02,
        1.2565683360e-03,
        8.3014209421e-07,
        -3.5073773473e-05,
        -6.7936409056e-06
    ])
    # fmt: on
    assert_array_almost_equal(whu, ref)

    with pytest.raises(NotImplementedError):
        regionprops(SAMPLE, spacing=(2, 1))[0].moments_weighted_hu


def test_moments_weighted():
    wm = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE)[
        0
    ].moments_weighted
    # fmt: off
    ref = cp.array(
        [[7.4000000e+01, 6.9900000e+02, 7.8630000e+03, 9.7317000e+04],
         [4.1000000e+02, 3.7850000e+03, 4.4063000e+04, 5.7256700e+05],
         [2.7500000e+03, 2.4855000e+04, 2.9347700e+05, 3.9007170e+06],
         [1.9778000e+04, 1.7500100e+05, 2.0810510e+06, 2.8078871e+07]]
    )
    # fmt: on
    assert_array_almost_equal(wm, ref)

    # Verify test function
    Mpq = get_moment_function(INTENSITY_SAMPLE, spacing=(1, 1))
    assert_almost_equal(Mpq(0, 0), ref[0, 0])
    assert_almost_equal(Mpq(0, 1), ref[0, 1])
    assert_almost_equal(Mpq(0, 2), ref[0, 2])
    assert_almost_equal(Mpq(0, 3), ref[0, 3])
    assert_almost_equal(Mpq(1, 0), ref[1, 0])
    assert_almost_equal(Mpq(1, 1), ref[1, 1])
    assert_almost_equal(Mpq(1, 2), ref[1, 2])
    assert_almost_equal(Mpq(1, 3), ref[1, 3])
    assert_almost_equal(Mpq(2, 0), ref[2, 0])
    assert_almost_equal(Mpq(2, 1), ref[2, 1])
    assert_almost_equal(Mpq(2, 2), ref[2, 2])
    assert_almost_equal(Mpq(2, 3), ref[2, 3])
    assert_almost_equal(Mpq(3, 0), ref[3, 0])
    assert_almost_equal(Mpq(3, 1), ref[3, 1])
    assert_almost_equal(Mpq(3, 2), ref[3, 2])
    assert_almost_equal(Mpq(3, 3), ref[3, 3])

    # Test spacing
    spacing = (3.2, 1.2)
    wmu = regionprops(
        SAMPLE, intensity_image=INTENSITY_SAMPLE, spacing=spacing
    )[0].moments_weighted
    Mpq = get_moment_function(INTENSITY_SAMPLE, spacing=spacing)
    assert_almost_equal(wmu[0, 0], Mpq(0, 0))
    assert_almost_equal(wmu[0, 1], Mpq(0, 1))
    assert_almost_equal(wmu[0, 2], Mpq(0, 2))
    assert_almost_equal(wmu[0, 3], Mpq(0, 3))
    assert_almost_equal(wmu[1, 0], Mpq(1, 0))
    assert_almost_equal(wmu[1, 1], Mpq(1, 1))
    assert_almost_equal(wmu[1, 2], Mpq(1, 2))
    assert_almost_equal(wmu[1, 3], Mpq(1, 3))
    assert_almost_equal(wmu[2, 0], Mpq(2, 0))
    assert_almost_equal(wmu[2, 1], Mpq(2, 1))
    assert_almost_equal(wmu[2, 2], Mpq(2, 2))
    assert_almost_equal(wmu[2, 3], Mpq(2, 3))
    assert_almost_equal(wmu[3, 0], Mpq(3, 0))
    assert_almost_equal(wmu[3, 1], Mpq(3, 1))
    assert_almost_equal(wmu[3, 2], Mpq(3, 2))
    assert_almost_equal(wmu[3, 3], Mpq(3, 3), decimal=6)


def test_moments_weighted_normalized():
    wnu = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE)[
        0
    ].moments_weighted_normalized
    # fmt: off
    ref = np.array(
        [[np.nan,        np.nan, 0.2301467830, -0.0162529732],         # noqa
         [np.nan, -0.0160405109, 0.0457932622, np.nan],         # noqa
         [0.0873590903, -0.0031421072, np.nan, np.nan],   # noqa
         [-0.0161217406, np.nan, np.nan, np.nan]]  # noqa
    )
    # fmt: on
    assert_array_almost_equal(wnu, ref)

    spacing = (3, 3)
    wnu = regionprops(
        SAMPLE, intensity_image=INTENSITY_SAMPLE, spacing=spacing
    )[0].moments_weighted_normalized

    # Normalized moments are scale invariant
    assert_almost_equal(wnu[0, 2], 0.2301467830)
    assert_almost_equal(wnu[0, 3], -0.0162529732)
    assert_almost_equal(wnu[1, 1], -0.0160405109)
    assert_almost_equal(wnu[1, 2], 0.0457932622)
    assert_almost_equal(wnu[2, 0], 0.0873590903)
    assert_almost_equal(wnu[2, 1], -0.0031421072)
    assert_almost_equal(wnu[3, 0], -0.0161217406)


def test_label_sequence():
    a = cp.empty((2, 2), dtype=int)
    a[:, :] = 2
    ps = regionprops(a)
    assert len(ps) == 1
    assert ps[0].label == 2


def test_pure_background():
    a = cp.zeros((2, 2), dtype=int)
    ps = regionprops(a)
    assert len(ps) == 0


def test_invalid():
    ps = regionprops(SAMPLE)

    def get_intensity_image():
        ps[0].image_intensity

    with pytest.raises(AttributeError):
        get_intensity_image()


def test_invalid_size():
    wrong_intensity_sample = cp.array([[1], [1]])
    with pytest.raises(ValueError):
        regionprops(SAMPLE, wrong_intensity_sample)


def test_equals():
    arr = cp.zeros((100, 100), dtype=int)
    arr[0:25, 0:25] = 1
    arr[50:99, 50:99] = 2

    regions = regionprops(arr)
    r1 = regions[0]

    regions = regionprops(arr)
    r2 = regions[0]
    r3 = regions[1]

    assert_equal(r1 == r2, True, "Same regionprops are not equal")
    assert_equal(r1 != r3, True, "Different regionprops are equal")


def test_iterate_all_props():
    region = regionprops(SAMPLE)[0]
    p0 = {p: region[p] for p in region}

    region = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE)[0]
    p1 = {p: region[p] for p in region}

    assert len(p0) < len(p1)


def test_cache():
    SAMPLE_mod = SAMPLE.copy()
    region = regionprops(SAMPLE_mod)[0]
    f0 = region.image_filled
    region._label_image[:10] = 1
    f1 = region.image_filled

    # Changed underlying image, but cache keeps result the same
    assert_array_equal(f0, f1)

    # Now invalidate cache
    region._cache_active = False
    f1 = region.image_filled

    assert cp.any(f0 != f1)


def test_docstrings_and_props():
    def foo():
        """foo"""

    has_docstrings = bool(foo.__doc__)

    region = regionprops(SAMPLE)[0]

    docs = _parse_docs()
    props = [m for m in dir(region) if not m.startswith("_")]

    nr_docs_parsed = len(docs)
    nr_props = len(props)
    if has_docstrings:
        assert_equal(nr_docs_parsed, nr_props)
        ds = docs["moments_weighted_normalized"]
        assert "iteration" not in ds
        assert len(ds.split("\n")) > 3
    else:
        assert_equal(nr_docs_parsed, 0)


def test_props_to_dict():
    regions = regionprops(SAMPLE)
    out = _props_to_dict(regions)
    assert out == {
        "label": cp.array([1]),
        "bbox-0": cp.array([0]),
        "bbox-1": cp.array([0]),
        "bbox-2": cp.array([10]),
        "bbox-3": cp.array([18]),
    }

    regions = regionprops(SAMPLE)
    out = _props_to_dict(
        regions, properties=("label", "area", "bbox"), separator="+"
    )
    assert out == {
        "label": cp.array([1]),
        "area": cp.array([72]),
        "bbox+0": cp.array([0]),
        "bbox+1": cp.array([0]),
        "bbox+2": cp.array([10]),
        "bbox+3": cp.array([18]),
    }

    regions = regionprops(SAMPLE_MULTIPLE)
    out = _props_to_dict(regions, properties=("coords",))
    coords = np.empty(2, object)
    coords[0] = cp.stack((cp.arange(10),) * 2, axis=-1)
    coords[1] = cp.array([[3, 7], [4, 7]])
    assert out["coords"].shape == coords.shape
    assert_array_equal(out["coords"][0], coords[0])
    assert_array_equal(out["coords"][1], coords[1])


def test_regionprops_table():
    out = regionprops_table(SAMPLE)
    assert out == {
        "label": cp.array([1]),
        "bbox-0": cp.array([0]),
        "bbox-1": cp.array([0]),
        "bbox-2": cp.array([10]),
        "bbox-3": cp.array([18]),
    }

    out = regionprops_table(
        SAMPLE, properties=("label", "area", "bbox"), separator="+"
    )
    assert out == {
        "label": cp.array([1]),
        "area": cp.array([72]),
        "bbox+0": cp.array([0]),
        "bbox+1": cp.array([0]),
        "bbox+2": cp.array([10]),
        "bbox+3": cp.array([18]),
    }

    out = regionprops_table(SAMPLE_MULTIPLE, properties=("coords",))
    coords = np.empty(2, object)
    coords[0] = cp.stack((cp.arange(10),) * 2, axis=-1)
    coords[1] = cp.array([[3, 7], [4, 7]])
    assert out["coords"].shape == coords.shape
    assert_array_equal(out["coords"][0], coords[0])
    assert_array_equal(out["coords"][1], coords[1])


def test_regionprops_table_deprecated_vector_property():
    out = regionprops_table(SAMPLE, properties=("local_centroid",))
    for key in out.keys():
        # key reflects the deprecated name, not its new (centroid_local) value
        assert key.startswith("local_centroid")


def test_regionprops_table_deprecated_scalar_property():
    out = regionprops_table(SAMPLE, properties=("bbox_area",))
    assert list(out.keys()) == ["bbox_area"]


def test_regionprops_table_equal_to_original():
    regions = regionprops(SAMPLE, INTENSITY_FLOAT_SAMPLE)
    out_table = regionprops_table(
        SAMPLE, INTENSITY_FLOAT_SAMPLE, properties=COL_DTYPES.keys()
    )

    for prop, dtype in COL_DTYPES.items():
        for i, reg in enumerate(regions):
            rp = reg[prop]
            if (
                cp.isscalar(rp)
                or (isinstance(rp, cp.ndarray) and rp.ndim == 0)
                or prop in OBJECT_COLUMNS
                or dtype is np.object_
            ):
                assert_array_equal(rp, out_table[prop][i])
            else:
                shape = rp.shape if isinstance(rp, cp.ndarray) else (len(rp),)
                for ind in np.ndindex(shape):
                    modified_prop = "-".join(map(str, (prop,) + ind))
                    loc = ind if len(ind) > 1 else ind[0]
                    assert_array_equal(rp[loc], out_table[modified_prop][i])


def test_regionprops_table_no_regions():
    out = regionprops_table(
        cp.zeros((2, 2), dtype=int),
        properties=("label", "area", "bbox"),
        separator="+",
    )
    assert len(out) == 6
    assert len(out["label"]) == 0
    assert len(out["area"]) == 0
    assert len(out["bbox+0"]) == 0
    assert len(out["bbox+1"]) == 0
    assert len(out["bbox+2"]) == 0
    assert len(out["bbox+3"]) == 0


def test_column_dtypes_complete():
    assert set(COL_DTYPES.keys()).union(OBJECT_COLUMNS) == set(PROPS.values())


def test_column_dtypes_correct():
    msg = "mismatch with expected type,"
    region = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE)[0]
    for col in COL_DTYPES:
        r = region[col]

        if col in OBJECT_COLUMNS:
            assert COL_DTYPES[col] == object
            continue

        # TODO: grlee77: check desired types for returned.
        #       e.g. currently inertia_tensor_eigvals returns a list of 0-dim
        #       arrays
        if isinstance(r, (tuple, list)):
            r0 = r[0]
            if isinstance(r0, cp.ndarray) and r0.ndim == 0:
                r0 = r0.item()
            t = type(r0)
        elif cp.isscalar(r):
            t = type(r)
        else:
            t = type(r.ravel()[0].item())

        if cp.issubdtype(t, cp.floating):
            assert (
                COL_DTYPES[col] == float
            ), f"{col} dtype {t} {msg} {COL_DTYPES[col]}"
        elif cp.issubdtype(t, cp.integer):
            assert (
                COL_DTYPES[col] == int
            ), f"{col} dtype {t} {msg} {COL_DTYPES[col]}"
        else:
            assert False, f"{col} dtype {t} {msg} {COL_DTYPES[col]}"


def pixelcount(regionmask):
    """a short test for an extra property"""
    return cp.sum(regionmask)


def intensity_median(regionmask, image_intensity):
    return cp.median(image_intensity[regionmask])


def bbox_list(regionmask):
    """Extra property whose output shape is dependent on mask shape."""
    return [1] * regionmask.shape[1]


def too_many_args(regionmask, image_intensity, superfluous):
    return 1


def too_few_args():
    return 1


def test_extra_properties():
    region = regionprops(SAMPLE, extra_properties=(pixelcount,))[0]
    assert region.pixelcount == cp.sum(SAMPLE == 1)


def test_extra_properties_intensity():
    region = regionprops(
        SAMPLE,
        intensity_image=INTENSITY_SAMPLE,
        extra_properties=(intensity_median,),
    )[0]
    assert region.intensity_median == cp.median(INTENSITY_SAMPLE[SAMPLE == 1])


@pytest.mark.parametrize("intensity_prop", _require_intensity_image)
def test_intensity_image_required(intensity_prop):
    region = regionprops(SAMPLE)[0]
    with pytest.raises(AttributeError) as e:
        getattr(region, intensity_prop)
    expected_error = (
        f"Attribute '{intensity_prop}' unavailable when `intensity_image` has "
        f"not been specified."
    )
    assert expected_error == str(e.value)


def test_extra_properties_no_intensity_provided():
    with pytest.raises(AttributeError):
        region = regionprops(SAMPLE, extra_properties=(intensity_median,))[0]
        _ = region.intensity_median


def test_extra_properties_nr_args():
    with pytest.raises(AttributeError):
        region = regionprops(SAMPLE, extra_properties=(too_few_args,))[0]
        _ = region.too_few_args
    with pytest.raises(AttributeError):
        region = regionprops(SAMPLE, extra_properties=(too_many_args,))[0]
        _ = region.too_many_args


def test_extra_properties_mixed():
    # mixed properties, with and without intensity
    region = regionprops(
        SAMPLE,
        intensity_image=INTENSITY_SAMPLE,
        extra_properties=(intensity_median, pixelcount),
    )[0]
    assert region.intensity_median == cp.median(INTENSITY_SAMPLE[SAMPLE == 1])
    assert region.pixelcount == cp.sum(SAMPLE == 1)


def test_extra_properties_table():
    out = regionprops_table(
        SAMPLE_MULTIPLE,
        intensity_image=INTENSITY_SAMPLE_MULTIPLE,
        properties=("label",),
        extra_properties=(intensity_median, pixelcount, bbox_list),
    )
    assert_array_almost_equal(out["intensity_median"], np.array([2.0, 4.0]))
    assert_array_equal(out["pixelcount"], np.array([10, 2]))

    assert out['bbox_list'].dtype == np.object_
    assert out["bbox_list"][0] == [1] * 10
    assert out["bbox_list"][1] == [1] * 1


def test_multichannel():
    """Test that computing multichannel properties works."""
    astro = data.astronaut()[::4, ::4]
    labels = slic(astro.astype(float), start_label=1)

    astro = cp.asarray(astro)
    astro_green = astro[..., 1]
    labels = cp.asarray(labels)

    segment_idx = int(cp.max(labels) // 2)
    region = regionprops(
        labels,
        astro_green,
        extra_properties=[intensity_median],
    )[segment_idx]
    region_multi = regionprops(
        labels,
        astro,
        extra_properties=[intensity_median],
    )[segment_idx]

    for prop in list(PROPS.keys()) + ["intensity_median"]:
        p = region[prop]
        p_multi = region_multi[prop]
        if isinstance(p, (list, tuple)):
            p = tuple([cp.asnumpy(p_) for p_ in p])
            p = np.stack(p)
        if isinstance(p_multi, (list, tuple)):
            p_multi = tuple([cp.asnumpy(p_) for p_ in p_multi])
            p_multi = np.stack(p_multi)
        if np.shape(p) == np.shape(p_multi):
            # property does not depend on multiple channels
            assert_array_equal(p, p_multi)
        else:
            # property uses multiple channels, returns props stacked along
            # final axis
            assert_array_equal(p, p_multi[..., 1])


def test_3d_ellipsoid_axis_lengths():
    """Verify that estimated axis lengths are correct.

    Uses an ellipsoid at an arbitrary position and orientation.
    """
    # generate a centered ellipsoid with non-uniform half-lengths (radii)
    half_lengths = (20, 10, 50)
    e = draw.ellipsoid(*half_lengths).astype(int)

    # Pad by asymmetric amounts so the ellipse isn't centered. Also, pad enough
    # that the rotated ellipse will still be within the original volume.
    e = np.pad(e, pad_width=[(30, 18), (30, 12), (40, 20)], mode="constant")
    e = cp.array(e)

    # apply rotations to the ellipsoid
    R = transform.EuclideanTransform(rotation=[0.2, 0.3, 0.4], dimensionality=3)
    e = ndi.affine_transform(e, R.params)

    # Compute regionprops
    rp = regionprops(e)[0]

    # estimate principal axis lengths via the inertia tensor eigenvalues
    evs = rp.inertia_tensor_eigvals
    axis_lengths = _inertia_eigvals_to_axes_lengths_3D(evs)
    expected_lengths = sorted([2 * h for h in half_lengths], reverse=True)
    for ax_len_expected, ax_len in zip(expected_lengths, axis_lengths):
        # verify accuracy to within 1%
        assert abs(ax_len - ax_len_expected) < 0.01 * ax_len_expected

    # verify that the axis length regionprops also agree
    assert abs(rp.axis_major_length - axis_lengths[0]) < 1e-7
    assert abs(rp.axis_minor_length - axis_lengths[-1]) < 1e-7
