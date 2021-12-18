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

from typing import Union

import cupy as cp


class HEStainExtractor:
    """Extract stain coefficients from an image.

    Parameters
    ----------
        source_intensity: transmitted light intensity.
            Defaults to 240.
        alpha: percentiles to ignore for outliers, so to calculate min and max,
            if only consider (alpha, 100-alpha) percentiles. Defaults to 1.
        beta: absorbance threshold for transparent pixels.
            Defaults to 0.15

    Note:
        Please refer to this paper for further information on the method:
        http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf
    """

    def __init__(
        self,
        source_intensity: float = 240,
        alpha: float = 1,
        beta: float = 0.15,
    ) -> None:
        self.source_intensity = source_intensity
        self.alpha = alpha
        self.beta = beta

    def _stain_decomposition(self, image: cp.ndarray) -> cp.ndarray:
        """Extract the matrix of stain coefficient fromthe image.

        Parameters
        ----------
            image: RGB image to perform stain extraction on

        Returns
        -------
            stain_coeff: stain attenuation coefficient matrix derive from the
                image, where first column is H, second column is E, and
                rows are RGB value

        """
        # calculate absorbance
        image = image.astype(cp.float32, copy=False) + 1.0
        absorbance = -cp.log(
            image.clip(max=self.source_intensity) / self.source_intensity
        )

        # reshape to form a CxN matrix
        c = absorbance.shape[0]
        absorbance = absorbance.reshape((c, -1))

        # remove transparent pixels
        absorbance = absorbance[cp.all(absorbance > self.beta, axis=1)]
        if len(absorbance) == 0:
            raise ValueError(
                "All pixels of the input image are below the threshold."
            )

        # compute eigenvectors
        _, eigvecs = cp.linalg.eigh(
            cp.cov(absorbance).astype(cp.float32, copy=False)
        )

        # project on the plane spanned by the eigenvectors
        # corresponding to the two largest eigenvalues
        projection = cp.dot(eigvecs[:, -2:].T, absorbance)

        # find the vectors that span the whole data (min and max angles)
        phi = cp.arctan2(projection[1], projection[0])
        min_phi = cp.percentile(phi, self.alpha)
        max_phi = cp.percentile(phi, 100 - self.alpha)
        # project back to absorbance space
        v_min = eigvecs[:, -2:].dot(
            cp.array([(cp.cos(min_phi), cp.sin(min_phi))], dtype=cp.float32).T
        )
        v_max = eigvecs[:, -2:].dot(
            cp.array([(cp.cos(max_phi), cp.sin(max_phi))], dtype=cp.float32).T
        )

        # make the vector corresponding to hematoxylin first and eosin second
        # by comparing the R channel value
        if v_min[0] > v_max[0]:
            stain_coeff = cp.array((v_min[:, 0], v_max[:, 0])).T
        else:
            stain_coeff = cp.array((v_max[:, 0], v_min[:, 0])).T

        return stain_coeff

    def __call__(self, image: cp.ndarray) -> cp.ndarray:
        """Perform stain extraction.

        Parameters
        ----------
            image: RGB image to extract stain from

        Returns
        -------
            stain_coeff: stain attenuation coefficient matrix derive from the
                image, where first column is H, second column is E, and
                rows are RGB values
        """
        # check image type and values
        if not isinstance(image, cp.ndarray):
            raise TypeError("Image must be of type cupy.ndarray.")
        if image.min() < 0:
            raise ValueError("Image should not have negative values.")

        return self._stain_decomposition(image)


class StainNormalizer:
    """Normalize images with reference stain attenuation coefficient matrix.

    First, it extracts the stain coefficient matrix from the image using the
    provided stain extractor. Then, it calculates the stain concentrations
    based on Beer-Lamber Law. Next, it reconstructs the image using the
    provided reference stain matrix (stain-normalized image).

    Parameters
    ----------
        source_intensity: transmitted light intensity.
            Defaults to 240.
        alpha: percentiles to ignore for outliers, so to calculate min and max,
            if only consider (alpha, 100-alpha) percentiles. Defaults to 1.
        ref_stain_coeff: reference stain attenuation coefficient matrix.
            Defaults to ((0.5626, 0.2159), (0.7201, 0.8012), (0.4062, 0.5581)).
        ref_max_conc: reference maximum stain concentrations for
            Hematoxylin & Eosin (H&E). Defaults to (1.9705, 1.0308).

    """

    def __init__(
        self,
        source_intensity: float = 240,
        alpha: float = 1,
        ref_stain_coeff: Union[tuple, cp.ndarray] = (
            (0.5626, 0.2159),
            (0.7201, 0.8012),
            (0.4062, 0.5581),
        ),
        ref_max_conc: Union[tuple, cp.ndarray] = (1.9705, 1.0308),
        stain_extractor=None,
    ) -> None:
        self.source_intensity = source_intensity
        self.alpha = alpha
        self.ref_stain_coeff = cp.array(ref_stain_coeff)
        self.ref_max_conc = cp.array(ref_max_conc)
        if stain_extractor is None:
            self.stain_extractor = HEStainExtractor()
        else:
            self.stain_extractor = stain_extractor

    def __call__(self, image: cp.ndarray) -> cp.ndarray:
        """Perform stain normalization.

        Parameters
        ----------
            image: RGB image to be stain normalized,
                pixel values between 0 and 255

        Returns
        -------
            image_norm: stain normalized image/patch
        """
        # check image type and values
        if not isinstance(image, cp.ndarray):
            raise TypeError("Image must be of type cupy.ndarray.")
        if image.min() < 0:
            raise ValueError("Image should not have negative values.")

        if self.source_intensity < 0:
            raise ValueError(
                "Source transmitted light intensity must be a positive value."
            )

        # derive stain coefficient matrix from the image
        stain_coeff = self.stain_extractor(image)

        # calculate absorbance
        image = image.astype(cp.float32, copy=False) + 1.0
        absorbance = -cp.log(
            image.clip(max=self.source_intensity) / self.source_intensity
        )

        # reshape to form a CxN matrix
        c, h, w = absorbance.shape
        absorbance = absorbance.reshape((c, -1))

        # calculate concentrations of the each stain, based on Beer-Lambert Law
        conc_raw = cp.linalg.lstsq(stain_coeff, absorbance, rcond=None)[0]

        # normalize stain concentrations
        max_conc = cp.percentile(conc_raw, 100 - self.alpha, axis=1)
        normalization_factors = self.ref_max_conc / max_conc
        conc_norm = conc_raw * normalization_factors[:, cp.newaxis]

        # reconstruct the image based on the reference stain matrix
        image_norm: cp.ndarray = cp.multiply(
            self.source_intensity,
            cp.exp(-self.ref_stain_coeff.dot(conc_norm)),
            dtype=cp.float32,
        )
        image_norm = cp.reshape(image_norm, (c, h, w)).astype(cp.uint8)
        return image_norm
