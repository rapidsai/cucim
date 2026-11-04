# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pytest
from skimage._shared.testing import assert_array_equal

from cucim.skimage.measure import _find_contours as _fc, find_contours

try:
    from cucim.skimage import _cucim_skimage_cpp_ext as _skimage_cpp
except ImportError:
    _skimage_cpp = None

a = np.ones((8, 8), dtype=np.float32)
a[1:-1, 1] = 0
a[1, 1:-1] = 0

x, y = np.mgrid[-1:1:5j, -1:1:5j]
r = np.sqrt(x**2 + y**2)


def _unpack_contours(points, offsets):
    return [
        points[start:stop] for start, stop in zip(offsets[:-1], offsets[1:])
    ]


def test_binary():
    ref = [
        [6.0, 1.5],
        [5.0, 1.5],
        [4.0, 1.5],
        [3.0, 1.5],
        [2.0, 1.5],
        [1.5, 2.0],
        [1.5, 3.0],
        [1.5, 4.0],
        [1.5, 5.0],
        [1.5, 6.0],
        [1.0, 6.5],
        [0.5, 6.0],
        [0.5, 5.0],
        [0.5, 4.0],
        [0.5, 3.0],
        [0.5, 2.0],
        [0.5, 1.0],
        [1.0, 0.5],
        [2.0, 0.5],
        [3.0, 0.5],
        [4.0, 0.5],
        [5.0, 0.5],
        [6.0, 0.5],
        [6.5, 1.0],
        [6.0, 1.5],
    ]

    contours = find_contours(cp.asarray(a), 0.5, positive_orientation="high")
    assert len(contours) == 1
    assert_array_equal(contours[0][::-1], ref)


# target contour for mask tests
mask_contour = [
    [6.0, 0.5],
    [5.0, 0.5],
    [4.0, 0.5],
    [3.0, 0.5],
    [2.0, 0.5],
    [1.0, 0.5],
    [0.5, 1.0],
    [0.5, 2.0],
    [0.5, 3.0],
    [0.5, 4.0],
    [0.5, 5.0],
    [0.5, 6.0],
    [1.0, 6.5],
    [1.5, 6.0],
    [1.5, 5.0],
    [1.5, 4.0],
    [1.5, 3.0],
    [1.5, 2.0],
    [2.0, 1.5],
    [3.0, 1.5],
    [4.0, 1.5],
    [5.0, 1.5],
    [6.0, 1.5],
]

mask = np.ones((8, 8), dtype=bool)
# Some missing data that should result in a hole in the contour:
mask[7, 0:3] = False


@pytest.mark.parametrize("level", [0.5, None])
def test_nodata(level):
    # Test missing data via NaNs in input array
    b = np.copy(a)
    b[~mask] = np.nan
    contours = find_contours(cp.asarray(b), level, positive_orientation="high")
    assert len(contours) == 1
    assert_array_equal(contours[0], mask_contour)


@pytest.mark.parametrize("level", [0.5, None])
def test_mask(level):
    # Test missing data via explicit masking
    contours = find_contours(
        cp.asarray(a), level, positive_orientation="high", mask=cp.asarray(mask)
    )
    assert len(contours) == 1
    assert_array_equal(contours[0], mask_contour)


@pytest.mark.parametrize("level", [0, None])
def test_mask_shape(level):
    bad_mask = np.ones((8, 7), dtype=bool)
    with pytest.raises(ValueError, match="shape"):
        find_contours(cp.asarray(a), level, mask=cp.asarray(bad_mask))


@pytest.mark.parametrize("level", [0, None])
def test_mask_dtype(level):
    bad_mask = np.ones((8, 8), dtype=np.uint8)
    with pytest.raises(TypeError, match="binary"):
        find_contours(cp.asarray(a), level, mask=cp.asarray(bad_mask))


def test_float():
    contours = find_contours(cp.asarray(r), 0.5)
    assert len(contours) == 1
    assert_array_equal(
        contours[0],
        [[2.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 2.0], [2.0, 3.0]],
    )


@pytest.mark.parametrize("level", [0.5, None])
def test_memory_order(level):
    contours = find_contours(cp.asarray(np.ascontiguousarray(r)), level)
    assert len(contours) == 1

    contours = find_contours(cp.asarray(np.asfortranarray(r)), level)
    assert len(contours) == 1


def test_invalid_input():
    with pytest.raises(ValueError):
        find_contours(cp.asarray(r), 0.5, "foo", "bar")
    with pytest.raises(ValueError):
        find_contours(cp.asarray(r[..., None]), 0.5)


def test_level_default():
    # image with range [0.9, 0.91]
    image = np.random.random((100, 100)) * 0.01 + 0.9
    contours = find_contours(cp.asarray(image))  # use default level
    # many contours should be found
    assert len(contours) > 1


@pytest.mark.parametrize(
    "image",
    [
        [
            [0.13680, 0.11220, 0.0, 0.0, 0.0, 0.19417, 0.19417, 0.33701],
            [0.0, 0.15140, 0.10267, 0.0, np.nan, 0.14908, 0.18158, 0.19178],
            [0.0, 0.06949, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01860],
            [0.0, 0.06949, 0.0, 0.17852, 0.08469, 0.02135, 0.08198, np.nan],
            [0.0, 0.08244, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [0.12342, 0.21330, 0.0, np.nan, 0.01301, 0.04335, 0.0, 0.0],
        ],
        [
            [0.08, -0.03, -0.17, -0.08, 0.24, 0.06, 0.17, -0.02],
            [0.12, 0.0, np.nan, 0.24, 0.0, -0.53, 0.26, 0.16],
            [0.39, 0.0, 0.0, 0.0, 0.0, -0.02, -0.3, 0.01],
            [0.28, -0.04, -0.03, 0.16, 0.12, 0.01, -0.87, 0.16],
            [0.26, 0.08, 0.08, 0.08, 0.12, 0.13, 0.11, 0.19],
            [0.27, 0.24, 0.0, 0.25, 0.32, 0.19, 0.26, 0.22],
        ],
        [
            [
                -0.18,
                np.nan,
                np.nan,
                0.22,
                -0.14,
                -0.23,
                -0.2,
                -0.17,
                -0.19,
                -0.24,
            ],
            [
                0.0,
                np.nan,
                np.nan,
                np.nan,
                -0.1,
                -0.24,
                -0.15,
                -0.02,
                -0.09,
                -0.21,
            ],
            [
                0.43,
                0.19,
                np.nan,
                np.nan,
                -0.01,
                -0.2,
                -0.22,
                -0.18,
                -0.16,
                -0.07,
            ],
            [
                0.23,
                0.0,
                np.nan,
                -0.06,
                -0.07,
                -0.21,
                -0.24,
                -0.25,
                -0.23,
                -0.13,
            ],
            [-0.05, -0.11, 0.0, 0.1, -0.19, -0.23, -0.23, -0.18, -0.19, -0.16],
            [
                -0.19,
                -0.05,
                0.13,
                -0.08,
                -0.22,
                -0.23,
                -0.26,
                -0.15,
                -0.12,
                -0.13,
            ],
            [
                -0.2,
                -0.11,
                -0.11,
                -0.24,
                -0.29,
                -0.27,
                -0.35,
                -0.36,
                -0.27,
                -0.13,
            ],
            [
                -0.28,
                -0.33,
                -0.31,
                -0.36,
                -0.39,
                -0.37,
                -0.38,
                -0.32,
                -0.34,
                -0.2,
            ],
            [
                -0.28,
                -0.33,
                -0.39,
                -0.4,
                -0.42,
                -0.38,
                -0.35,
                -0.39,
                -0.35,
                -0.34,
            ],
            [
                -0.38,
                -0.35,
                -0.41,
                -0.42,
                -0.39,
                -0.36,
                -0.34,
                -0.36,
                -0.28,
                -0.34,
            ],
        ],
    ],
)
def test_keyerror_fix(image):
    """Failing samples from issue #4830"""
    find_contours(cp.asarray(np.array(image, np.float32)), 0)


@pytest.mark.skipif(
    _skimage_cpp is None,
    reason="optional cucim.skimage C++ extension is not available",
)
def test_cpp_assemble_contours_matches_python():
    x, y = np.mgrid[-1:1:32j, -1:1:32j]
    image = (np.sin(8 * x) + np.cos(5 * y)).astype(np.float32)
    image, level, vertex_connect_high, _, mask = (
        _fc._validate_find_contours_inputs(
            cp.asarray(image), 0.0, "low", "low", None
        )
    )
    segments, segment_keys = _fc._get_contour_segments(
        image, level, vertex_connect_high, mask
    )
    segments = cp.asnumpy(segments)
    segment_keys = cp.asnumpy(segment_keys)

    expected = _fc._assemble_contours_python(segments)
    result = _skimage_cpp.assemble_contours(segments, segment_keys)

    assert len(result) == len(expected)
    for result_contour, expected_contour in zip(result, expected):
        assert_array_equal(result_contour, expected_contour)


@pytest.mark.skipif(
    _skimage_cpp is None,
    reason="optional cucim.skimage C++ extension is not available",
)
@pytest.mark.parametrize("positive_orientation", ["low", "high"])
def test_find_contours_return_packed_matches_list(positive_orientation):
    image = (np.sin(8 * x) + np.cos(5 * y)).astype(np.float32)

    expected = find_contours(
        cp.asarray(image),
        0.0,
        positive_orientation=positive_orientation,
    )
    points, offsets = find_contours(
        cp.asarray(image),
        0.0,
        positive_orientation=positive_orientation,
        return_packed=True,
    )
    result = _unpack_contours(points, offsets)

    assert points.flags.c_contiguous
    assert points.dtype == np.float64
    assert offsets.dtype == np.int64
    assert offsets[0] == 0
    assert offsets[-1] == len(points)
    assert len(result) == len(expected)
    for result_contour, expected_contour in zip(result, expected):
        assert_array_equal(result_contour, expected_contour)


def test_find_contours_return_packed_requires_cpp(monkeypatch):
    monkeypatch.setattr(_fc, "_skimage_cpp", None)
    with pytest.raises(RuntimeError, match="requires the optional"):
        find_contours(cp.asarray(r), 0.5, return_packed=True)


def _segments_of(image, level, fully_connected="low", mask=None):
    image, level, vertex_connect_high, _, mask = (
        _fc._validate_find_contours_inputs(
            cp.asarray(image), level, fully_connected, "low", mask
        )
    )
    return _fc._get_contour_segments(image, level, vertex_connect_high, mask)


@pytest.mark.skipif(
    _skimage_cpp is None,
    reason="optional cucim.skimage C++ extension is not available",
)
@pytest.mark.parametrize("fully_connected", ["low", "high"])
@pytest.mark.parametrize("seed", range(6))
def test_linked_contours_match_sequential_assembler(seed, fully_connected):
    # Smooth noise has open contours and many closed ones
    rng = np.random.default_rng(seed)
    shape = rng.integers(8, 80, 2)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    image = np.sin(xx / rng.uniform(1, 6)) * np.cos(yy / rng.uniform(1, 6))
    image = image + 0.1 * rng.standard_normal(shape)
    segments, segment_keys = _segments_of(image, 0.2, fully_connected)

    expected_points, expected_offsets = _skimage_cpp.assemble_contours_packed(
        cp.asnumpy(segments), cp.asnumpy(segment_keys)
    )
    packed = _fc._assemble_linked(segments, segment_keys)

    assert packed is not None
    points, offsets = packed
    assert len(offsets) > 3
    assert_array_equal(offsets, expected_offsets)
    assert_array_equal(points, expected_points)


@pytest.mark.skipif(
    _skimage_cpp is None,
    reason="optional cucim.skimage C++ extension is not available",
)
def test_linked_contours_closed_contour():
    # A one pixel peak gives a diamond, a closed contour of four segments
    image = np.zeros((5, 5))
    image[2, 2] = 1
    segments, segment_keys = _segments_of(image, 0.5)

    points, offsets = _fc._assemble_linked(segments, segment_keys)

    assert_array_equal(offsets, [0, 5])
    assert_array_equal(points[0], points[-1])
    assert len(np.unique(points, axis=0)) == 4


@pytest.mark.skipif(
    _skimage_cpp is None,
    reason="optional cucim.skimage C++ extension is not available",
)
def test_linked_contours_fall_back_when_a_vertex_is_on_the_level():
    # Several segments end at the vertices that are exactly on the level, so
    # the segments cannot be linked by sorting their end points.
    image = np.array([[2, 1, 0], [0, 2, 1], [1, 0, 0]], dtype=float)
    segments, segment_keys = _segments_of(image, 1.0)
    assert _fc._link_segments(segments, segment_keys) is None

    expected = _fc._assemble_contours_python(cp.asnumpy(segments))
    result = _fc._assemble_contours(segments, segment_keys)

    assert len(result) == len(expected)
    for result_contour, expected_contour in zip(result, expected):
        assert_array_equal(result_contour, expected_contour)


@pytest.mark.skipif(
    _skimage_cpp is None,
    reason="optional cucim.skimage C++ extension is not available",
)
def test_linked_contours_long_contours():
    # Spiral arms are single contours with thousands of segments, that the
    # assembly on the device has to cut into many sublists.
    yy, xx = np.mgrid[:300, :300] - 150.0
    image = np.sin(0.35 * np.hypot(yy, xx) + np.arctan2(yy, xx))
    segments, segment_keys = _segments_of(image, 0.0)

    expected_points, expected_offsets = _skimage_cpp.assemble_contours_packed(
        cp.asnumpy(segments), cp.asnumpy(segment_keys)
    )
    points, offsets = _fc._assemble_linked(segments, segment_keys)

    assert np.diff(offsets).max() > 5000
    assert_array_equal(offsets, expected_offsets)
    assert_array_equal(points, expected_points)


@pytest.mark.skipif(
    _skimage_cpp is None,
    reason="optional cucim.skimage C++ extension is not available",
)
def test_plan_linked_contours_rejects_invalid_sublists():
    ints = [np.array(x, dtype=np.int32) for x in ([0, 1], [1, 1], [0, 0])]
    first, length, min_id = ints[0], ints[1], ints[2]
    with pytest.raises(ValueError, match="at most one"):
        # Two sublists cannot both be followed by sublist 0
        _skimage_cpp.plan_linked_contours(
            first, length, np.array([0, 0], np.int32), min_id, first, min_id
        )
    with pytest.raises(ValueError, match="same length"):
        _skimage_cpp.plan_linked_contours(
            first,
            length[:1],
            np.array([-1, -1], np.int32),
            min_id,
            first,
            min_id,
        )


@pytest.mark.skipif(
    _skimage_cpp is None,
    reason="optional cucim.skimage C++ extension is not available",
)
@pytest.mark.parametrize("return_packed", [False, True])
@pytest.mark.parametrize("positive_orientation", ["low", "high"])
def test_find_contours_device_and_host_assembly_agree(
    monkeypatch, positive_orientation, return_packed
):
    image = cp.asarray((np.sin(8 * x) + np.cos(5 * y)).astype(np.float32))
    results = []
    # always assemble on the device, then always on the host
    for min_segments in (0, 10**9):
        monkeypatch.setattr(_fc, "_GPU_ASSEMBLY_MIN_SEGMENTS", min_segments)
        results.append(
            find_contours(
                image,
                0.0,
                positive_orientation=positive_orientation,
                return_packed=return_packed,
            )
        )
    device, host = results
    if return_packed:
        device, host = list(device), list(host)
    else:
        assert len(device) == len(host)
    for device_part, host_part in zip(device, host):
        assert_array_equal(device_part, host_part)


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_level_default_matches_skimage_dtype(dtype):
    # the default level is computed in the dtype of the input
    from skimage.measure import find_contours as sk_find_contours

    image = np.array([[0.9, 0.91], [0.91, 0.9]], dtype=dtype)
    expected = sk_find_contours(image)
    result = find_contours(cp.asarray(image))
    assert len(result) == len(expected)
    for res, exp in zip(result, expected):
        assert_array_equal(res, exp)


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("assembly_min_segments", [0, 10**9])
@pytest.mark.parametrize("negate", [False, True])
def test_endpoint_on_vertex_after_rounding(
    monkeypatch, transpose, assembly_min_segments, negate
):
    # an interpolation fraction in (0, 1) can add up to an exact grid vertex;
    # the segments must then still be linked there (horizontal and vertical
    # edges)
    from skimage.measure import find_contours as sk_find_contours

    image = np.array(
        [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0], [1.0, -1e-16, 1.0]]
    )
    if negate:
        image = -image
    if transpose:
        image = image.T
    monkeypatch.setattr(
        _fc, "_GPU_ASSEMBLY_MIN_SEGMENTS", assembly_min_segments
    )
    expected = sk_find_contours(image, 0.0)
    result = find_contours(cp.asarray(image), 0.0)
    assert len(result) == len(expected)
    for res, exp in zip(result, expected):
        assert_array_equal(res, exp)
