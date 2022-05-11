# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import cupy as cp
import pytest

from cucim.core.operations.color import HEStainExtractor, StainNormalizer


class TestHEStainExtractor():
    @pytest.mark.parametrize(
        'image',
        [
            cp.full((3, 2, 4), -1),   # negative value
            cp.full((3, 2, 4), 256),  # out of range value
            None,
            cp.full((3, 2, 4), 240),  # uniformly below the beta threshold
        ]
    )
    def test_transparent_image(self, image):
        """
        Test HE stain extraction on an image that comprises
        only transparent pixels - pixels with absorbance below the
        beta absorbance threshold. A ValueError should be raised,
        since once the transparent pixels are removed, there are no
        remaining pixels to compute eigenvectors.
        """
        if image is None:
            with pytest.raises(TypeError):
                HEStainExtractor()(image)
        else:
            with pytest.raises(ValueError):
                HEStainExtractor()(image)

    @pytest.mark.parametrize(
        'image',
        [
            None,
            cp.full((3, 2, 4), 100),  # uniform, above beta absorbance thresh.
            cp.full((3, 2, 4), 150),  # uniform, above beta absorbance thresh.
        ]
    )
    def test_identical_result_vectors(self, image):
        """
        Test HE stain extraction on input images that are
        uniformly filled with pixels that have absorbance above the
        beta absorbance threshold. Since input image is uniformly filled,
        the two extracted stains should have the same RGB values. So,
        we assert that the first column is equal to the second column
        of the returned stain matrix.
        """
        if image is None:
            with pytest.raises(TypeError):
                HEStainExtractor()(image)
        else:
            result = HEStainExtractor()(image)
            cp.testing.assert_array_equal(result[:, 0], result[:, 1])

    @pytest.mark.parametrize(
        'image, expected',
        [
            (None, None),
            # uniformly zero -> two identical stains extracted
            (
                cp.zeros((3, 2, 4)),
                cp.array(
                    [
                        [0.0, 0.0],
                        [0.70710678, 0.70710678],
                        [0.70710678, 0.70710678]
                    ]
                )
            ),
            # input pixels not uniformly filled, leading to two different
            # stains extracted
            (
                cp.array(
                    [
                        [[100, 0, 0], [0, 0, 0]],
                        [[0, 0, 0], [0, 0, 0]],
                        [[0, 0, 0], [0, 0, 0]],
                    ]
                ),
                cp.array(
                    [
                        [0.70710677, 0.18702291],
                        [0.0, 0.0],
                        [0.70710677, 0.9823556],
                    ]
                ),
            ),
        ]
    )
    def test_result_value(self, image, expected):
        """
        Test that an input image returns an expected stain matrix.

        For test case 4:
        - a uniformly filled input image should result in
          eigenvectors [[1,0,0],[0,1,0],[0,0,1]]
        - phi should be an array containing only values of
          arctan(1) since the ratio between the eigenvectors
          corresponding to the two largest eigenvalues is 1
        - maximum phi and minimum phi should thus be arctan(1)
        - thus, maximum vector and minimum vector should be
          [[0],[0.70710677],[0.70710677]]
        - the resulting extracted stain should be
          [[0,0],[0.70710678,0.70710678],[0.70710678,0.70710678]]

        For test case 5:
        - the non-uniformly filled input image should result in
          eigenvectors [[0,0,1],[1,0,0],[0,1,0]]
        - maximum phi and minimum phi should thus be 0.785 and
          0.188 respectively
        - thus, maximum vector and minimum vector should be
          [[0.18696113],[0],[0.98236734]] and
          [[0.70710677],[0],[0.70710677]] respectively
        - the resulting extracted stain should be
          [[0.70710677,0.18696113],[0,0],[0.70710677,0.98236734]]
        """
        if image is None:
            with pytest.raises(TypeError):
                HEStainExtractor()(image)
        else:
            result = HEStainExtractor()(image)
            cp.testing.assert_allclose(result, expected)


class TestStainNormalizer():
    @pytest.mark.parametrize(
        'image',
        [
            cp.full((3, 2, 4), -1),   # negative value case
            cp.full((3, 2, 4), 256),  # out of range value
            None,
            cp.full((3, 2, 5), 240),  # uniformly below the beta threshold
        ]
    )
    def test_transparent_image(self, image):
        """
        Test HE stain normalization on an image that comprises
        only transparent pixels - pixels with absorbance below the
        beta absorbance threshold. A ValueError should be raised,
        since once the transparent pixels are removed, there are no
        remaining pixels to compute eigenvectors.
        """
        if image is None:
            with pytest.raises(TypeError):
                StainNormalizer()(image)
        else:
            with pytest.raises(ValueError):
                StainNormalizer()(image)

    @pytest.mark.parametrize(
        'kwargs, image, expected',
        [
            # 1.) invalid image
            ({}, None, None),
            # 2.) input uniformly zero, and target stain matrix provided.
            # - The normalized concentration returned for each pixel is the
            #   same as the reference maximum stain concentrations in the case
            #   that the image is uniformly filled, as in this test case. This
            #   is because the maximum concentration for each stain is the same
            #   as each pixel's concentration.
            # - Thus, the normalized concentration matrix should be a (2, 6)
            #   matrix with the first row having all values of 1.9705, second
            #   row all 1.0308.
            # - Taking the matrix product of the target stain matrix and the
            #   concentration matrix, then using the inverse Beer-Lambert
            #   transform to obtain the RGB image from the absorbance image,
            #   and finally converting to uint8, we get that the stain
            #   normalized image should be 12 everywhere.
            [
                {"ref_stain_coeff": cp.full((3, 2), 1)},
                cp.zeros((3, 2, 4)),
                cp.full((3, 2, 4), 12),
            ],
            # 3.) input uniformly zero, and target stain matrix provided.
            # - As in test case 2, the normalized concentration matrix should
            #   be a (2, 6) matrix with the first row having all values of
            #   1.9705, second row all 1.0308.
            # - Taking the matrix product of the target default stain matrix
            #   and the concentration matrix, then using the inverse
            #   Beer-Lambert transform to obtain the RGB image from the
            #   absorbance image, and finally converting to uint8, we get the
            #   expected result listed here.
            [
                {},
                cp.zeros((3, 2, 3)),
                cp.array(
                    [
                        [[63, 63, 63], [63, 63, 63]],
                        [[25, 25, 25], [25, 25, 25]],
                        [[61, 61, 61], [61, 61, 61]],
                    ]
                ),
            ],
            # 4.) input pixels not uniformly filled
            # - For this non-uniformly filled image, the stain extracted should
            #   be [[0.70710677,0.18696113],[0,0],[0.70710677,0.98236734]], as
            #   validated for the HEStainExtractor class. Solving the linear
            #   least squares problem (since absorbance matrix = stain matrix *
            #   concentration matrix), we obtain the concentration matrix that
            #   should be [[-0.3101, 7.7508, 7.7508, 7.7508, 7.7508, 7.7508],
            #   [5.8022, 0, 0, 0, 0, 0]].
            # - Normalizing the concentration matrix, taking the matrix product
            #   of the target stain matrix and the concentration matrix, using
            #   the inverse Beer-Lambert transform to obtain the RGB image from
            #   the absorbance image, and finally converting to uint8, we get
            #   the expected result listed here.
            [
                {"ref_stain_coeff": cp.full((3, 2), 1)},
                cp.array(
                    [
                        [[100, 0, 0], [0, 0, 0]],
                        [[0, 0, 0], [0, 0, 0]],
                        [[0, 0, 0], [0, 0, 0]],
                    ]
                ),
                cp.array(
                    [
                        [[88, 33, 33], [33, 33, 33]],
                        [[88, 33, 33], [33, 33, 33]],
                        [[88, 33, 33], [33, 33, 33]],
                    ]
                ),
            ],
        ]
    )
    def test_result_value(self, kwargs, image, expected):
        """Test that an input image returns an expected normalized image."""

        if image is None:
            with pytest.raises(TypeError):
                StainNormalizer()(image)
        else:
            result = StainNormalizer(**kwargs)(image)
            cp.testing.assert_allclose(result, expected)
