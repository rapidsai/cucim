# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import cupy as cp
import numpy as np
import pytest

from cucim.skimage.morphology import local_maxima, local_minima


class TestLocalMaxima:
    """Some tests for local_minima are included as well."""

    supported_dtypes = [
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.float32,
        np.float64,
    ]
    image = cp.asarray(
        [
            [1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 3, 3, 0, 0, 4, 0, 2, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 4, 4, 0, 3, 0, 0, 0],
            [0, 2, 0, 1, 0, 2, 1, 0, 0, 0, 0, 3, 0, 0, 0],
            [0, 0, 2, 0, 2, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    # Connectivity 2, maxima can touch border, returned with default values
    expected_default = cp.asarray(
        [
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    # Connectivity 1 (cross), maxima can touch border
    expected_cross = cp.asarray(
        [
            [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )

    def test_empty(self):
        """Test result with empty image."""
        result = local_maxima(cp.asarray([[]]), indices=False)
        assert result.size == 0
        assert result.dtype == bool
        assert result.shape == (1, 0)

        result = local_maxima(cp.asarray([]), indices=True)
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert result[0].size == 0
        assert result[0].dtype == np.intp

        result = local_maxima(cp.asarray([[]]), indices=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0].size == 0
        assert result[0].dtype == np.intp
        assert result[1].size == 0
        assert result[1].dtype == np.intp

    def test_dtypes(self):
        """Test results with default configuration for all supported dtypes."""
        for dtype in self.supported_dtypes:
            result = local_maxima(self.image.astype(dtype))
            assert result.dtype == bool
            cp.testing.assert_array_equal(result, self.expected_default)

    def test_dtypes_old(self):
        """
        Test results with default configuration and data copied from old unit
        tests for all supported dtypes.
        """
        data = cp.asarray(
            [
                [10, 11, 13, 14, 14, 15, 14, 14, 13, 11],
                [11, 13, 15, 16, 16, 16, 16, 16, 15, 13],
                [13, 15, 40, 40, 18, 18, 18, 60, 60, 15],
                [14, 16, 40, 40, 19, 19, 19, 60, 60, 16],
                [14, 16, 18, 19, 19, 19, 19, 19, 18, 16],
                [15, 16, 18, 19, 19, 20, 19, 19, 18, 16],
                [14, 16, 18, 19, 19, 19, 19, 19, 18, 16],
                [14, 16, 80, 80, 19, 19, 19, 100, 100, 16],
                [13, 15, 80, 80, 18, 18, 18, 100, 100, 15],
                [11, 13, 15, 16, 16, 16, 16, 16, 15, 13],
            ],
            dtype=np.uint8,
        )
        expected = cp.asarray(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        for dtype in self.supported_dtypes:
            image = data.astype(dtype)
            result = local_maxima(image)
            assert result.dtype == bool
            cp.testing.assert_array_equal(result, expected)

    def test_connectivity(self):
        """Test results if footprint is a scalar."""
        # Connectivity 1: generates cross shaped footprint
        result_conn1 = local_maxima(self.image, connectivity=1)
        assert result_conn1.dtype == bool
        cp.testing.assert_array_equal(result_conn1, self.expected_cross)

        # Connectivity 2: generates square shaped footprint
        result_conn2 = local_maxima(self.image, connectivity=2)
        assert result_conn2.dtype == bool
        cp.testing.assert_array_equal(result_conn2, self.expected_default)

        # Connectivity 3: generates square shaped footprint
        result_conn3 = local_maxima(self.image, connectivity=3)
        assert result_conn3.dtype == bool
        cp.testing.assert_array_equal(result_conn3, self.expected_default)

    def test_footprint(self):
        """Test results if footprint is given."""
        footprint_cross = cp.asarray(
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool
        )
        result_footprint_cross = local_maxima(
            self.image, footprint=footprint_cross
        )
        assert result_footprint_cross.dtype == bool
        cp.testing.assert_array_equal(
            result_footprint_cross, self.expected_cross
        )

        for footprint in [
            ((True,) * 3,) * 3,
            cp.ones((3, 3), dtype=np.float64),
            cp.ones((3, 3), dtype=np.uint8),
            cp.ones((3, 3), dtype=bool),
        ]:
            # Test different dtypes for footprint which expects a boolean array
            # but will accept and convert other types if possible
            result_footprint_square = local_maxima(
                self.image, footprint=footprint
            )
            assert result_footprint_square.dtype == bool
            cp.testing.assert_array_equal(
                result_footprint_square, self.expected_default
            )

        footprint_x = cp.asarray([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool)
        expected_footprint_x = cp.asarray(
            [
                [1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        result_footprint_x = local_maxima(self.image, footprint=footprint_x)
        assert result_footprint_x.dtype == bool
        cp.testing.assert_array_equal(result_footprint_x, expected_footprint_x)

    def test_indices(self):
        """Test output if indices of peaks are desired."""
        # Connectivity 1
        expected_conn1 = cp.nonzero(self.expected_cross)
        result_conn1 = local_maxima(self.image, connectivity=1, indices=True)
        assert len(result_conn1) == len(expected_conn1)
        for r, e in zip(result_conn1, expected_conn1):
            cp.testing.assert_array_equal(r, e)

        # Connectivity 2
        expected_conn2 = cp.nonzero(self.expected_default)
        result_conn2 = local_maxima(self.image, connectivity=2, indices=True)
        assert len(result_conn2) == len(expected_conn2)
        for r, e in zip(result_conn2, expected_conn2):
            cp.testing.assert_array_equal(r, e)

    def test_allow_borders(self):
        """Test maxima detection at the image border."""
        # Use connectivity 1 to allow many maxima, only filtering at border is
        # of interest
        result_with_boder = local_maxima(
            self.image, connectivity=1, allow_borders=True
        )
        assert result_with_boder.dtype == bool
        cp.testing.assert_array_equal(result_with_boder, self.expected_cross)

        expected_without_border = cp.asarray(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        result_without_border = local_maxima(
            self.image, connectivity=1, allow_borders=False
        )
        assert result_with_boder.dtype == bool
        cp.testing.assert_array_equal(
            result_without_border, expected_without_border
        )

    def test_nd(self):
        """Test one- and three-dimensional case."""
        # One-dimension
        x_1d = cp.asarray([1, 1, 0, 1, 2, 3, 0, 2, 1, 2, 0])
        expected_1d = np.asarray([1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0], dtype=bool)
        result_1d = local_maxima(x_1d)
        assert result_1d.dtype == bool
        cp.testing.assert_array_equal(result_1d, expected_1d)

        # 3-dimensions (adapted from old unit test)
        x_3d = cp.zeros((8, 8, 8), dtype=np.uint8)
        expected_3d = cp.zeros((8, 8, 8), dtype=bool)
        # first maximum: only one pixel
        x_3d[1, 1:3, 1:3] = 100
        x_3d[2, 2, 2] = 200
        x_3d[3, 1:3, 1:3] = 100
        expected_3d[2, 2, 2] = 1
        # second maximum: three pixels in z-direction
        x_3d[5:8, 1, 1] = 200
        expected_3d[5:8, 1, 1] = 1
        # third: two maxima in 0 and 3.
        x_3d[0, 5:8, 5:8] = 200
        x_3d[1, 6, 6] = 100
        x_3d[2, 5:7, 5:7] = 200
        x_3d[0:3, 5:8, 5:8] += 50
        expected_3d[0, 5:8, 5:8] = 1
        expected_3d[2, 5:7, 5:7] = 1
        # four : one maximum in the corner of the square
        x_3d[6:8, 6:8, 6:8] = 200
        x_3d[7, 7, 7] = 255
        expected_3d[7, 7, 7] = 1
        result_3d = local_maxima(x_3d)
        assert result_3d.dtype == bool
        cp.testing.assert_array_equal(result_3d, expected_3d)

    def test_constant(self):
        """Test behaviour for 'flat' images."""
        const_image = cp.full((7, 6), 42, dtype=np.uint8)
        expected = cp.zeros((7, 6), dtype=np.uint8)
        for dtype in self.supported_dtypes:
            const_image = const_image.astype(dtype)
            # test for local maxima
            result = local_maxima(const_image)
            assert result.dtype == bool
            cp.testing.assert_array_equal(result, expected)
            # test for local minima
            result = local_minima(const_image)
            assert result.dtype == bool
            cp.testing.assert_array_equal(result, expected)

    def test_extrema_float(self):
        """Specific tests for float type."""
        # Copied from old unit test for local_maxima
        image = cp.asarray(
            [
                [0.10, 0.11, 0.13, 0.14, 0.14, 0.15, 0.14, 0.14, 0.13, 0.11],
                [0.11, 0.13, 0.15, 0.16, 0.16, 0.16, 0.16, 0.16, 0.15, 0.13],
                [0.13, 0.15, 0.40, 0.40, 0.18, 0.18, 0.18, 0.60, 0.60, 0.15],
                [0.14, 0.16, 0.40, 0.40, 0.19, 0.19, 0.19, 0.60, 0.60, 0.16],
                [0.14, 0.16, 0.18, 0.19, 0.19, 0.19, 0.19, 0.19, 0.18, 0.16],
                [0.15, 0.182, 0.18, 0.19, 0.204, 0.20, 0.19, 0.19, 0.18, 0.16],
                [0.14, 0.16, 0.18, 0.19, 0.19, 0.19, 0.19, 0.19, 0.18, 0.16],
                [0.14, 0.16, 0.80, 0.80, 0.19, 0.19, 0.19, 1.0, 1.0, 0.16],
                [0.13, 0.15, 0.80, 0.80, 0.18, 0.18, 0.18, 1.0, 1.0, 0.15],
                [0.11, 0.13, 0.15, 0.16, 0.16, 0.16, 0.16, 0.16, 0.15, 0.13],
            ],
            dtype=np.float32,
        )
        inverted_image = 1.0 - image
        expected_result = cp.asarray(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )

        # Test for local maxima with automatic step calculation
        result = local_maxima(image)
        assert result.dtype == bool
        cp.testing.assert_array_equal(result, expected_result)

        # Test for local minima with automatic step calculation
        result = local_minima(inverted_image)
        assert result.dtype == bool
        cp.testing.assert_array_equal(result, expected_result)

    def test_extrema_small_float(self):
        image = cp.asarray(
            [
                [
                    9.89232736e-20,
                    8.78543302e-20,
                    8.78543302e-20,
                    9.89232736e-20,
                ],
                [
                    8.78543302e-20,
                    6.38842355e-20,
                    6.38842355e-20,
                    8.78543302e-20,
                ],
                [
                    8.78543302e-20,
                    6.38842355e-20,
                    6.38842355e-20,
                    8.78543302e-20,
                ],
                [
                    9.89232736e-20,
                    8.78543302e-20,
                    8.78543302e-20,
                    9.89232736e-20,
                ],
            ]
        )

        result = local_minima(image)

        expected_result = cp.asarray(
            [
                [False, False, False, False],
                [False, True, True, False],
                [False, True, True, False],
                [False, False, False, False],
            ]
        )

        cp.testing.assert_array_equal(result, expected_result)

    def test_exceptions(self):
        """Test if input validation triggers correct exceptions."""
        # Mismatching number of dimensions
        with pytest.raises(ValueError, match="number of dimensions"):
            local_maxima(self.image, footprint=cp.ones((3, 3, 3), dtype=bool))
        with pytest.raises(ValueError, match="number of dimensions"):
            local_maxima(self.image, footprint=cp.ones((3,), dtype=bool))

        # All dimensions in footprint must be of size 3
        with pytest.raises(ValueError, match="dimension size"):
            local_maxima(self.image, footprint=cp.ones((2, 3), dtype=bool))
        with pytest.raises(ValueError, match="dimension size"):
            local_maxima(self.image, footprint=cp.ones((5, 5), dtype=bool))

        with pytest.raises(TypeError, match="float16 which is not supported"):
            local_maxima(cp.empty(1, dtype=np.float16))

    def test_small_array(self):
        """Test output for arrays with dimension smaller 3.

        If any dimension of an array is smaller than 3 and `allow_borders` is
        false a footprint, which has at least 3 elements in each
        dimension, can't be applied. This is an implementation detail so
        `local_maxima` should still return valid output (see gh-3261).

        If `allow_borders` is true the array is padded internally and there is
        no problem.
        """
        warning_msg = "maxima can't exist .* any dimension smaller 3 .*"
        x = cp.asarray([0, 1])
        local_maxima(x, allow_borders=True)  # no warning
        with pytest.warns(UserWarning, match=warning_msg):
            result = local_maxima(x, allow_borders=False)
        cp.testing.assert_array_equal(result, [0, 0])
        assert result.dtype == bool

        x = cp.asarray([[1, 2], [2, 2]])
        local_maxima(x, allow_borders=True, indices=True)  # no warning
        with pytest.warns(UserWarning, match=warning_msg):
            result = local_maxima(x, allow_borders=False, indices=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0].size == 0
        assert result[1].size == 0
        assert result[0].dtype == np.intp
        assert result[1].dtype == np.intp
