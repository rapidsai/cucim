# Copyright (c) 2022, NVIDIA CORPORATION.
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

import math
import warnings
from typing import Union

import cupy as cp
import numpy as np


__all__ = [
    "absorbance_to_image",
    "image_to_absorbance",
    "stain_extraction_pca",
    "normalize_colors_pca",
]


@cp.fuse()
def _image_to_absorbance(image, min_val, max_val):
    image = cp.maximum(cp.minimum(image, max_val), min_val)
    absorbance = -cp.log(image / max_val)
    return absorbance


def image_to_absorbance(image, source_intensity=255.0, dtype=cp.float32):
    """Convert an image to units of absorbance (optical density).

    Parameters
    ----------
    image : ndarray
        The image to convert to absorbance. Can be single or multichannel.
    source_intensity : float, optional
        Reference intensity for `image`.
    dtype : numpy.dtype, optional
        The floating point precision at which to compute the absorbance.

    Returns
    -------
    absorbance : ndarray
        The absorbance computed from image.

    Notes
    -----
    If `image` has an integer dtype it will be clipped to range
    ``[1, source_intensity]``, while float image inputs are clipped to range
    ``[source_intensity/255, source_intensity]. The minimum is to avoid log(0).
    Absorbance is then given by

    .. math::

        absorbance = \\log{\\frac{image}{source_intensity}}.
    """
    dtype = cp.dtype(dtype)
    if dtype.kind != "f":
        raise ValueError("dtype must be a floating point type")

    input_dtype = image.dtype
    image = image.astype(dtype, copy=False)
    if source_intensity < 0:
        raise ValueError(
            "Source transmitted light intensity must be a positive value."
        )
    source_intensity = float(source_intensity)
    if input_dtype == "f":
        min_val = source_intensity / 255.0
        max_val = source_intensity
    else:
        min_val = 1.0
        max_val = source_intensity

    absorbance = _image_to_absorbance(image, min_val, max_val)
    return absorbance


def _image_to_absorbance_matrix(image, source_intensity=240,
                                image_type="intensity", channel_axis=0,
                                dtype=cp.float32):
    """Convert image to an absorbance and reshape to (3, n_pixels).

    See ``image_to_absorbance`` for parameter descriptions
    """
    c = image.shape[channel_axis]
    if c != 3:
        raise ValueError("Expected an RGB image")

    if image_type == "intensity":
        absorbance = image_to_absorbance(
            image, source_intensity=source_intensity, dtype=dtype
        )
    elif image_type == "absorbance":
        absorbance = image.astype(dtype, copy=True)
    else:
        raise ValueError(
            "`image_type` must be either 'intensity' or 'absorbance'."
        )

    # reshape to form a (n_channels, n_pixels) matrix
    if channel_axis != 0:
        absorbance = cp.moveaxis(
            absorbance, source=channel_axis, destination=0
        )
    return absorbance.reshape((c, -1))


@cp.fuse()
def _absorbance_to_image_float(absorbance, source_intensity):
    return cp.exp(-absorbance) * source_intensity


@cp.fuse()
def _absorbance_to_image_int(absorbance, source_intensity, min_val, max_val):
    rgb = cp.exp(-absorbance) * source_intensity
    # prevent overflow/underflow
    rgb = cp.minimum(cp.maximum(rgb, min_val), max_val)
    return cp.round(rgb)


@cp.fuse()
def _absorbance_to_image_uint8(absorbance, source_intensity):
    rgb = cp.exp(-absorbance) * source_intensity
    # prevent overflow/underflow
    rgb = cp.minimum(cp.maximum(rgb, 0), 255)
    return cp.round(rgb).astype(cp.uint8)


def absorbance_to_image(absorbance, source_intensity=255, dtype=cp.uint8):
    """Convert an absorbance (optical density) image back to a standard image.

    Parameters
    ----------
    absorbance : ndarray
        The absorbance image to convert back to a linear intensity range.
    source_intensity : float, optional
        Reference intensity for `image`. This should match what was used with
        ``rgb_to_absorbance`` when creating `absorbance`.
    dtype : numpy.dtype, optional
        The datatype to cast the output image to.

    Returns
    -------
    image : ndarray
        An image computed from the absorbance

    """
    # absorbance must be floating point
    absorbance_dtype = cp.promote_types(absorbance.dtype, cp.float16)
    absorbance = absorbance.astype(absorbance_dtype, copy=False)

    if source_intensity < 0:
        raise ValueError(
            "Source transmitted light intensity must be a positive value."
        )

    # specialized code paths depending on output dtype
    dtype = cp.dtype(dtype)
    if dtype == cp.uint8:
        return _absorbance_to_image_uint8(absorbance, source_intensity)
    if dtype.kind in "iu":
        # round to nearest integer and cast to desired integer dtype
        iinfo = cp.iinfo(dtype)
        image = _absorbance_to_image_int(
            absorbance, source_intensity, iinfo.min, iinfo.max
        )
        return image.astype(dtype, copy=False)
    return _absorbance_to_image_float(absorbance, source_intensity)


def _covariance(a):
    """Returns the covariance matrix of an array.

    This is a modified version of cupy.cov that will not automatically promote
    float32 to float64. It also removes all unused kwargs (`y`, `bias`, etc.).

    Parameters
    ----------
    a : cupy.ndarray
        Array to compute the covariance matrix for. Each row represents a
        variable, with observations in the columns.

    Returns
    -------
    cupy.ndarray
        The covariance matrix of the input array.
    """
    if a.ndim > 2:
        raise ValueError("Input must be <= 2-d")

    dtype = cp.promote_types(a.dtype, cp.float32)
    X = a
    if X.ndim == 0:
        X = X[np.newaxis, :]
    if X.shape[0] == 0:
        return cp.array([]).reshape(0, 0)
    # import to have C-contiguous order for fast mean along last axis
    X = cp.array(X, dtype=dtype, order="C", copy=True)
    ddof = 1
    fact = X.shape[1] - ddof
    if fact <= 0:
        warnings.warn("Degrees of freedom <= 0 for slice",
                      RuntimeWarning, stacklevel=2)
        fact = 0.0

    X -= X.mean(axis=1, keepdims=True)
    if not X.flags.f_contiguous:
        # TODO: Empirically, it is faster to use F order for .dot() call below.
        #       F-order was ~4x faster for float32 when measured, but slightly
        #       slower for float64, which seems odd. Should further validate on
        #       additional hardware.
        X = cp.asfortranarray(X)
    out = X.dot(X.T.conj())
    out *= 1 / float(fact)
    return out.squeeze()


def _validate_image(image):
    if not isinstance(image, cp.ndarray):
        raise TypeError("Image must be of type cupy.ndarray.")
    if image.dtype.kind != "u" and image.min() < 0:
        raise ValueError("Image should not have negative values.")


def _prep_channel_axis(channel_axis, ndim):
    if (channel_axis < -ndim) or (channel_axis > ndim - 1):
        raise ValueError(
            f"`channel_axis={channel_axis}` exceeds image dimensions"
        )
    return channel_axis % ndim


def stain_extraction_pca(image, source_intensity=240, alpha=1, beta=0.345,
                         *, channel_axis=0, image_type="intensity"):
    """Extract the matrix of H & E stain coefficient from an image.

    Uses a method that selects stain vectors based on the angle distribution
    within a best-fit plane determined by principle component analysis (PCA)
    [1]_.

    Parameters
    ----------
    image : cp.ndarray
        RGB image to perform stain extraction on. Intensities should typically
        be within unsigned 8-bit integer intensity range ([0, 255]) when
        ``image_type == "intensity"``.
    source_intensity : float, optional
        Transmitted light intensity. The algorithm will clip image intensities
        above the specified `source_intensity` and then normalize by
        `source_intensity` so that `image` intensities are <= 1.0. Only used
        when `image_type=="intensity"`.
    alpha : float, optional
        Algorithm parameter controlling the ``[alpha, 100 - alpha]``
        percentile range used as a robust [min, max] estimate.
    beta : float, optional
        Absorbance (optical density) threshold below which to consider pixels
        as transparent. Transparent pixels are excluded from the estimation.

    Additional Parameters
    ---------------------
    channel_axis : int, optional
        The axis corresponding to color channels (default is the last axis).
    image_type : {"intensity", "absorbance"}, optional
        With the default `image_type` of `"intensity"`, the image will be
        transformed to `absorbance` units via ``image_to_absorbance``. If
        the input `image` is already an absorbance image, then `image_type`
        should be set to `"absorbance"` instead.

    Returns
    -------
    stain_coeff : cp.ndarray
        Stain attenuation coefficient matrix derived from the image, where
        the first column corresponds to H, the second column is E and the rows
        are RGB values.

    Notes
    -----
    The default `beta` of 0.345 is equivalent to the use of 0.15 in [1]_. The
    difference is due to our use of the natural log instead of a decadic log
    (log10) when computing the absorbance.

    References
    ----------
    .. [1] M. Macenko et al., "A method for normalizing histology slides for
           quantitative analysis," 2009 IEEE International Symposium on
           Biomedical Imaging: From Nano to Macro, 2009, pp. 1107-1110,
           doi: 10.1109/ISBI.2009.5193250.
           http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf
    """

    _validate_image(image)
    channel_axis = _prep_channel_axis(channel_axis, image.ndim)
    if alpha < 0 or alpha > 100:
        raise ValueError("alpha must be a percentile in range [0, 100].")
    if beta < 0:
        raise ValueError("beta must be nonnegative.")

    # convert to absorbance (optical density) matrix of shape (3, n_pixels)
    absorbance = _image_to_absorbance_matrix(
        image,
        source_intensity=source_intensity,
        image_type=image_type,
        channel_axis=channel_axis,
    )

    # remove transparent pixels
    absorbance = absorbance[:, cp.all(absorbance > beta, axis=0)]
    if absorbance.size == 0:
        raise ValueError(
            "All pixels of the input image are below the threshold."
        )

    # compute eigenvectors (do small 3x3 matrix calculations on the host)
    cov = _covariance(absorbance)
    cov = cp.asnumpy(cov).astype(np.float32, copy=False)
    _, ev = np.linalg.eigh(cov)
    ev = ev[:, [2, 1]]
    # flip to ensure positive first coordinate so arctan2 angles are about 0
    if ev[0, 0] < 0:
        ev[:, 0] *= -1
    if ev[0, 1] < 0:
        ev[:, 1] *= -1

    # project on the plane spanned by the eigenvectors
    projection = cp.dot(cp.asarray(ev.T), absorbance)

    # find the vectors that span the whole data (min and max angles)
    phi = cp.arctan2(projection[1], projection[0])
    min_phi, max_phi = cp.percentile(phi, (alpha, 100 - alpha))
    # need these scalars on the host
    min_phi, max_phi = float(min_phi), float(max_phi)

    # project back to absorbance space
    v_min = np.array([math.cos(min_phi), math.sin(min_phi)], dtype=np.float32)
    v_max = np.array([math.cos(max_phi), math.sin(max_phi)], dtype=np.float32)
    v1 = np.dot(ev, v_min)
    v2 = np.dot(ev, v_max)

    # Make Hematoxylin (H) first and eosin (E) second by comparing the
    # R channel value
    if v1[0] < v2[0]:
        v1, v2 = v2, v1
    stain_coeff = np.stack((v1, v2), axis=-1)

    # renormalize columns to reduce numerical error
    stain_coeff /= np.linalg.norm(stain_coeff, axis=0, keepdims=True)
    return cp.asarray(stain_coeff)


def _get_raw_concentrations(src_stain_coeff, absorbance):

    if absorbance.ndim != 2 or absorbance.shape[0] != 3:
        raise ValueError("`absorbance` must be shape (3, n_pixels)")

    # estimate the raw stain concentrations

    # pseudo-inverse
    coeff_pinv = cp.dot(
        cp.linalg.inv(cp.dot(src_stain_coeff.T, src_stain_coeff)),
        src_stain_coeff.T
    )
    if cp.any(cp.isnan(coeff_pinv)):
        # fall back to cp.linalg.lstsq if pseudo-inverse above failed
        conc_raw = cp.linalg.lstsq(
            src_stain_coeff, absorbance, rcond=None
        )[0]
    else:
        conc_raw = cp.dot(cp.asarray(coeff_pinv, order="F"), absorbance)

    return conc_raw


def _normalized_from_concentrations(conc_raw, max_percentile, ref_stain_coeff,
                                    ref_max_conc, source_intensity,
                                    original_shape, channel_axis):
    """Determine normalized image from concentrations."""

    # verify conc_raw is shape (2, n_pixels)
    if conc_raw.ndim != 2 or conc_raw.shape[0] != 2:
        raise ValueError(
            "`conc_raw` must be a 2D array of concentrations with size 2 on "
            "axis 0."
        )
    if ref_stain_coeff.ndim != 2 or ref_stain_coeff.shape[0] != 3:
        raise ValueError(
            "`ref_stain_coeff` must be a shape (3, n) matrix, representing "
            "n stain vectors."
        )
    if len(ref_max_conc) != ref_stain_coeff.shape[1]:
        raise ValueError(
            "`ref_max_conc` must have length equal to the number of stain "
            "coefficient vectors."
        )

    # normalize stain concentrations
    # Note: calling percentile separately for each channel is faster than:
    #       max_conc = cp.percentile(conc_raw, 100 - alpha, axis=1)
    max_conc = cp.concatenate(
        [cp.percentile(ch_raw, max_percentile)[np.newaxis]
         for ch_raw in conc_raw]
    )
    normalization_factors = ref_max_conc / max_conc
    conc_norm = conc_raw * normalization_factors[:, cp.newaxis]

    # reconstruct the image based on the reference stain matrix
    absorbance_norm = ref_stain_coeff.dot(conc_norm)
    image_norm = absorbance_to_image(
        absorbance_norm, source_intensity=source_intensity, dtype=np.uint8
    )

    # restore original shape for each channel
    spatial_shape = (
        original_shape[:channel_axis] + original_shape[channel_axis + 1:]
    )
    image_norm = cp.reshape(image_norm, (3,) + spatial_shape)

    # move channels from axis 0 to channel_axis
    if channel_axis != 0:
        image_norm = cp.moveaxis(
            image_norm, source=0, destination=channel_axis
        )
    # restore original shape
    return image_norm


def normalize_colors_pca(
        image,
        source_intensity: float = 240.0,
        alpha: float = 1.0,
        beta: float = 0.345,
        ref_stain_coeff: Union[tuple, cp.ndarray] = (
            (0.5626, 0.2159),
            (0.7201, 0.8012),
            (0.4062, 0.5581),
        ),
        ref_max_conc: Union[tuple, cp.ndarray] = (1.9705, 1.0308),
        image_type: str = "intensity",
        channel_axis: int = 0,
):
    """Extract the matrix of stain coefficient from the image.

    Parameters
    ----------
    image : np.ndarray
        RGB image to determine concentrations for. Intensities should typically
        be within unsigned 8-bit integer intensity range ([0, 255]) when
        ``image_type == "intensity"``.
    source_intensity : float, optional
        Transmitted light intensity. The algorithm will clip image intensities
        above the specified `source_intensity` and then normalize by
        `source_intensity` so that `image` intensities are <= 1.0. Only used
        when `image_type=="intensity"`.
    alpha : float, optional
        Algorithm parameter controlling the ``[alpha, 100 - alpha]``
        percentile range used as a robust [min, max] estimate.
    beta : float, optional
        Absorbance (optical density) threshold below which to consider pixels
        as transparent. Transparent pixels are excluded from the estimation.
    ref_stain_coeff : array-like
        Reference stain coefficients as determined by the output of
        `stain_extraction_pca` for a reference image.
    ref_max_conc : tuple or cp.ndarray
        The reference maximum concentrations.
    image_type : {"intensity", "absorbance"}, optional
        With the default `image_type` of `"intensity"`, the image will be
        transformed to an `absorbance` using ``image_to_absorbance``. If
        the input `image` is already an absorbance image, then `image_type`
        should be set to `"absorbance"` instead.
    channel_axis : int, optional
        The axis corresponding to color channels (default is the last axis).

    Returns
    -------
    stain_coeff : np.ndarray
        Stain attenuation coefficient matrix derived from the image, where
        the first column corresponds to H, the second column is E and the rows
        are RGB values.

    Notes
    -----
    The default `beta` of 0.345 is equivalent to the use of 0.15 in [1]_. The
    difference is due to our use of the natural log instead of a decadic log
    (log10) when computing the absorbance.

    References
    ----------
    .. [1] M. Macenko et al., "A method for normalizing histology slides for
           quantitative analysis," 2009 IEEE International Symposium on
           Biomedical Imaging: From Nano to Macro, 2009, pp. 1107-1110,
           doi: 10.1109/ISBI.2009.5193250.
           http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf
    """

    _validate_image(image)
    channel_axis = _prep_channel_axis(channel_axis, image.ndim)

    # convert to absorbance (optical density) matrix of shape (n_pixels, 3)
    absorbance = _image_to_absorbance_matrix(
        image,
        source_intensity=source_intensity,
        image_type=image_type,
        channel_axis=channel_axis
    )

    # channels_axis=0 for the shape (3, n_pixels) absorbance matrix
    src_stain_coeff = stain_extraction_pca(
        absorbance,
        beta=beta,
        image_type="absorbance",
        channel_axis=0,
    )

    # get normalized image from raw concentrations
    conc_raw = _get_raw_concentrations(src_stain_coeff, absorbance)

    # get normalized image
    image_norm = _normalized_from_concentrations(
        conc_raw=conc_raw,
        max_percentile=100 - alpha,
        ref_max_conc=cp.asarray(ref_max_conc),
        ref_stain_coeff=cp.asarray(ref_stain_coeff),
        source_intensity=source_intensity,
        channel_axis=channel_axis,
        original_shape=image.shape,
    )
    return image_norm
