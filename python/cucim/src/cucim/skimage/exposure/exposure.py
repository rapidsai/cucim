# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import cupy as cp
import numpy as np

from .._shared import utils
from .._shared._reduction_kernels import _get_minmax
from ..util.dtype import dtype_limits, dtype_range
from ._histogram_kernels import _bincount_rows

__all__ = [
    "histogram",
    "cumulative_distribution",
    "equalize_hist",
    "rescale_intensity",
    "adjust_gamma",
    "adjust_log",
    "adjust_sigmoid",
]


DTYPE_RANGE = dtype_range.copy()
DTYPE_RANGE.update((d.__name__, limits) for d, limits in dtype_range.items())
DTYPE_RANGE.update(
    {
        "uint10": (0, 2**10 - 1),
        "uint12": (0, 2**12 - 1),
        "uint14": (0, 2**14 - 1),
        "bool": dtype_range[bool],
        "float": dtype_range[np.float64],
    }
)


def _offset_array(arr, low_boundary, high_boundary):
    """Offset the array to get the lowest value at 0 if negative."""
    if low_boundary < 0:
        offset = low_boundary
        dyn_range = high_boundary - low_boundary
        # get smallest dtype that can hold both minimum and offset maximum
        offset_dtype = np.promote_types(
            np.min_scalar_type(dyn_range), np.min_scalar_type(low_boundary)
        )
        if arr.dtype != offset_dtype:
            # prevent overflow errors when offsetting
            arr = arr.astype(offset_dtype)
        arr = arr - offset
    return arr


def _bincount_histogram_centers(image, source_range, *, return_range=False):
    """Compute integer bin centers, optionally retaining their host bounds."""
    if source_range not in ["image", "dtype"]:
        raise ValueError(
            f"Incorrect value for `source_range` argument: {source_range}"
        )
    if source_range == "image":
        image_min, image_max = (
            _get_minmax(image).get().astype(np.int64, copy=False).tolist()
        )
    elif source_range == "dtype":
        image_min, image_max = dtype_limits(image, clip_negative=False)
    bin_centers = cp.arange(image_min, image_max + 1)
    if return_range:
        return bin_centers, (image_min, image_max)
    return bin_centers


def _bincount_histogram(
    image, source_range, bin_centers=None, *, bin_range=None
):
    """
    Efficient histogram calculation for an image of integers.

    This function is significantly more efficient than cupy.histogram but
    works only on images of integers. Known compact ranges use shared-memory
    counting; wider ranges retain cupy.bincount. Pass `bin_range` with supplied
    centers to reuse their host bounds.

    Parameters
    ----------
    image : array
        Input image.
    source_range : {'image', 'dtype'}
        'image' determines the range from the input image.
        'dtype' determines the range from the expected range of the images
        of that data type.

    Returns
    -------
    hist : array
        The values of the histogram.
    bin_centers : array
        The values at the center of the bins.
    """
    if bin_centers is None:
        bin_centers, bin_range = _bincount_histogram_centers(
            image, source_range, return_range=True
        )
    if bin_range is None:
        image_min, image_max = bin_centers[0].item(), bin_centers[-1].item()
    else:
        # The caller already determined these bounds before making GPU bins.
        image_min, image_max = bin_range
    nbins = image_max - image_min + 1
    if (
        bin_range is not None
        and 0 < nbins <= 4096
        and -(1 << 63) <= image_min < (1 << 63)
    ):
        hist = _bincount_rows(image.reshape(1, -1), nbins, image_min)[0]
        return hist, bin_centers
    image = _offset_array(image, image_min, image_max)

    # Casting back to unsigned dtype seems necessary to avoid incorrect
    # results for larger integer ranges with CUDA 12.x.
    unsigned_dtype_char = image.dtype.char.upper()
    image = image.astype(unsigned_dtype_char, copy=False)

    hist = cp.bincount(
        image.ravel(), minlength=image_max - min(image_min, 0) + 1
    )
    if source_range == "image":
        idx = max(image_min, 0)
        hist = hist[idx:]
    return hist, bin_centers


def _get_outer_edges(image, hist_range):
    """Determine the outer bin edges to use for `numpy.histogram`.

    These are obtained from either the image or hist_range.

    Parameters
    ----------
    image : ndarray
        Image for which the histogram is to be computed.
    hist_range : 2-tuple of int or None
        Range of values covered by the histogram bins. If None, the minimum
        and maximum values of `image` are used.

    Returns
    -------
    first_edge, last_edge : int
        The range spanned by the histogram bins.

    Notes
    -----
    This function is adapted from ``np.lib.histograms._get_outer_edges``.
    """
    if hist_range is not None:
        first_edge, last_edge = hist_range
        if first_edge > last_edge:
            raise ValueError(
                "max must be larger than min in hist_range parameter."
            )
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError(
                f"supplied hist_range of [{first_edge}, {last_edge}] is "
                f"not finite"
            )
    elif image.size == 0:
        # handle empty arrays. Can't determine hist_range, so use 0-1.
        first_edge, last_edge = 0, 1
    else:
        first_edge, last_edge = map(float, _get_minmax(image).get().tolist())
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError(
                f"autodetected hist_range of [{first_edge}, {last_edge}] is "
                f"not finite"
            )

    # expand empty hist_range to avoid divide by zero
    if first_edge == last_edge:
        first_edge = first_edge - 0.5
        last_edge = last_edge + 0.5

    return first_edge, last_edge


def _get_bin_edges(image, nbins, hist_range):
    """Computes histogram bins for use with `numpy.histogram`.

    Parameters
    ----------
    image : ndarray
        Image for which the histogram is to be computed.
    nbins : int
        The number of bins.
    hist_range : 2-tuple of int
        Range of values covered by the histogram bins.

    Returns
    -------
    bin_edges : ndarray
        The histogram bin edges.

    Notes
    -----
    This function is a simplified version of
    ``np.lib.histograms._get_bin_edges`` that only supports uniform bins.
    """
    first_edge, last_edge = _get_outer_edges(image, hist_range)
    # numpy/gh-10322 means that type resolution rules are dependent on array
    # shapes. To avoid this causing problems, we pick a type now and stick
    # with it throughout.
    bin_type = np.result_type(first_edge, last_edge, image)
    if np.issubdtype(bin_type, np.integer):
        bin_type = np.result_type(bin_type, float)

    # compute bin edges
    bin_edges = np.linspace(
        first_edge, last_edge, nbins + 1, endpoint=True, dtype=bin_type
    )
    return bin_edges


def _get_numpy_hist_range(image, source_range):
    if source_range == "image":
        hist_range = None
    elif source_range == "dtype":
        hist_range = dtype_limits(image, clip_negative=False)
    else:
        raise ValueError(
            f"Incorrect value for `source_range` argument: {source_range}"
        )
    return hist_range


@utils.channel_as_last_axis(multichannel_output=False)
def histogram(
    image,
    nbins=256,
    source_range="image",
    normalize=False,
    *,
    channel_axis=None,
):
    """Return histogram of image.

    Unlike `numpy.histogram`, this function returns the centers of bins and
    does not rebin integer arrays. For integer arrays, each integer value has
    its own bin, which improves speed and intensity-resolution.

    If `channel_axis` is not set, the histogram is computed on the flattened
    image. For color or multichannel images, set ``channel_axis`` to use a
    common binning for all channels. Alternatively, one may apply the function
    separately on each channel to obtain a histogram for each color channel
    with separate binning.

    Parameters
    ----------
    image : array
        Input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    source_range : {'image', 'dtype'}, optional
        'image' (default) determines the range from the input image.
        'dtype' determines the range from the expected range of the images
        of that data type.
    normalize : bool, optional
        If True, normalize the histogram by the sum of its values.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

    Returns
    -------
    hist : array
        The values of the histogram. When ``channel_axis`` is not None, hist
        will be a 2D array where the first axis corresponds to channels.
    bin_centers : array
        The values at the center of the bins.

    See Also
    --------
    cumulative_distribution

    Examples
    --------
    >>> import cupy as cp
    >>> from skimage import data
    >>> from cucim.skimage import exposure, img_as_float
    >>> image = img_as_float(cp.array(data.camera()))
    >>> cp.histogram(image, bins=2)
    (array([ 93585, 168559]), array([0. , 0.5, 1. ]))
    >>> exposure.histogram(image, nbins=2)
    (array([ 93585, 168559]), array([0.25, 0.75]))
    """
    sh = image.shape
    if len(sh) == 3 and sh[-1] < 4 and channel_axis is None:
        utils.warn(
            "This might be a color image. The histogram will be "
            "computed on the flattened image. You can instead "
            "apply this function to each color channel, or set "
            "channel_axis."
        )

    if channel_axis is not None:
        channels = sh[-1]
        hist = []
        bin_range = None

        # compute bins based on the raveled array
        if cp.issubdtype(image.dtype, cp.integer):
            # here bins corresponds to the bin centers
            bins, bin_range = _bincount_histogram_centers(
                image, source_range, return_range=True
            )
        else:
            # determine the bin edges for np.histogram
            hist_range = _get_numpy_hist_range(image, source_range)
            bins = _get_bin_edges(image, nbins, hist_range)

        for chan in range(channels):
            h, bc = _histogram(
                image[..., chan],
                bins,
                source_range,
                normalize,
                bin_range=bin_range,
            )
            hist.append(h)
        # Convert to numpy arrays
        bin_centers = cp.asarray(bc)
        hist = cp.stack(hist, axis=0)
    else:
        hist, bin_centers = _histogram(image, nbins, source_range, normalize)

    return hist, bin_centers


def _histogram(image, bins, source_range, normalize, *, bin_range=None):
    """

    Parameters
    ----------
    image : ndarray
        Image for which the histogram is to be computed.
    bins : int or ndarray
        The number of histogram bins. For images with integer dtype, an array
        containing the bin centers can also be provided. For images with
        floating point dtype, this can be an array of bin_edges for use by
        ``np.histogram``.
    source_range : {'image', 'dtype'}, optional
        'image' (default) determines the range from the input image.
        'dtype' determines the range from the expected range of the images
        of that data type.
    normalize : bool, optional
        If True, normalize the histogram by the sum of its values.
    """

    image = image.ravel()
    # For integer types, histogramming with bincount is more efficient.
    if np.issubdtype(image.dtype, cp.integer):
        bin_centers = bins if isinstance(bins, cp.ndarray) else None
        hist, bin_centers = _bincount_histogram(
            image, source_range, bin_centers, bin_range=bin_range
        )
    else:
        hist_range = _get_numpy_hist_range(image, source_range)
        if (
            isinstance(bins, (int, np.integer))
            and bins > 0
            and hist_range is None
        ):
            hist_range = _get_outer_edges(image, None)
            # Dyadic bins over [0, 1] are exact in the input dtype, so the
            # host-scalar fast path also preserves their rounding.
            exact_unit_bins = image.dtype.kind == "f" and (
                hist_range == (0.0, 1.0)
                and bins & (bins - 1) == 0
                and bins <= 1 << np.finfo(image.dtype).nmant
            )
            if image.size and image.dtype.kind == "f" and not exact_unit_bins:
                # Typed endpoints retain the arithmetic used by CuPy's
                # device-scalar linspace. Python float endpoints instead use
                # its double-precision fast path, which can move bin edges.
                first, last = map(image.dtype.type, hist_range)
                delta = float(last) - float(first)
                tiny = np.finfo(np.float32).tiny
                exceptional = not np.isfinite(delta) or (
                    image.dtype == np.float32
                    and (
                        abs(delta) > np.finfo(np.float32).max
                        or any(
                            0 < abs(v) < tiny
                            for v in (first, last, delta, delta / bins)
                        )
                    )
                )
                if exceptional:
                    # Retain CuPy's overflow and float32 flush-to-zero rules.
                    hist_range = None
                else:
                    bins = np.linspace(first, last, bins + 1, dtype=image.dtype)
        hist, bin_edges = cp.histogram(image, bins=bins, range=hist_range)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    if normalize:
        hist = hist / cp.sum(hist)
    return hist, bin_centers


def cumulative_distribution(image, nbins=256):
    """Return cumulative distribution function (cdf) for the given image.

    Parameters
    ----------
    image : array
        Image array.
    nbins : int, optional
        Number of bins for image histogram.

    Returns
    -------
    img_cdf : array
        Values of cumulative distribution function.
    bin_centers : array
        Centers of bins.

    See Also
    --------
    histogram

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Cumulative_distribution_function

    Examples
    --------
    >>> import cupy as cp
    >>> from skimage import data
    >>> from cucim.skimage import exposure, img_as_float
    >>> image = img_as_float(cp.array(data.camera()))
    >>> hi = exposure.histogram(image)
    >>> cdf = exposure.cumulative_distribution(image)
    >>> cp.all(cdf[0] == cp.cumsum(hi[0])/float(image.size))
    array(True)
    """
    hist, bin_centers = histogram(image, nbins)
    img_cdf = hist.cumsum()
    img_cdf = img_cdf / float(img_cdf[-1])
    return img_cdf, bin_centers


def equalize_hist(image, nbins=256, mask=None):
    """Return image after histogram equalization.

    Parameters
    ----------
    image : array
        Image array.
    nbins : int, optional
        Number of bins for image histogram. Note: this argument is
        ignored for integer images, for which each integer is its own
        bin.
    mask: ndarray of bools or 0s and 1s, optional
        Array of same shape as `image`. Only points at which mask == True
        are used for the equalization, which is applied to the whole image.

    Returns
    -------
    out : float array
        Image array after histogram equalization.

    Notes
    -----
    This function is adapted from [1]_ with the author's permission.

    References
    ----------
    .. [1] http://www.janeriksolem.net/histogram-equalization-with-python-and.html
    .. [2] https://en.wikipedia.org/wiki/Histogram_equalization

    """  # noqa
    if mask is not None:
        mask = mask.astype(bool, copy=False)
        cdf, bin_centers = cumulative_distribution(image[mask], nbins)
    else:
        cdf, bin_centers = cumulative_distribution(image, nbins)
    out = cp.interp(image.ravel(), bin_centers, cdf)
    out = out.reshape(image.shape)
    # Unfortunately, np.interp currently always promotes to float64, so we
    # have to cast back to single precision when float32 output is desired
    return out.astype(utils._supported_float_type(image.dtype), copy=False)


def intensity_range(image, range_values="image", clip_negative=False):
    """Return image intensity range (min, max) based on desired value type.

    Parameters
    ----------
    image : array
        Input image.
    range_values : str or 2-tuple, optional
        The image intensity range is configured by this parameter.
        The possible values for this parameter are enumerated below.

        'image'
            Return image min/max as the range.
        'dtype'
            Return min/max of the image's dtype as the range.
        dtype-name
            Return intensity range based on desired `dtype`. Must be valid key
            in `DTYPE_RANGE`. Note: `image` is ignored for this range type.
        2-tuple
            Return `range_values` as min/max intensities. Note that there's no
            reason to use this function if you just want to specify the
            intensity range explicitly. This option is included for functions
            that use `intensity_range` to support all desired range types.

    clip_negative : bool, optional
        If True, clip the negative range (i.e. return 0 for min intensity)
        even if the image dtype allows negative values.

    Returns
    -------
    i_range : tuple
        A 2-tuple where the first element is the minimum and the second is the
        maximum.
    """
    if range_values == "dtype":
        range_values = image.dtype.type

    if range_values == "image":
        # Preserve host scalar results while transferring both extrema together.
        i_min, i_max = _get_minmax(image).get().tolist()
    elif range_values in DTYPE_RANGE:
        i_min, i_max = DTYPE_RANGE[range_values]
        if clip_negative:
            i_min = 0
    else:
        i_min, i_max = range_values
    return i_min, i_max


def _output_dtype(dtype_or_range, image_dtype):
    """Determine the output dtype for rescale_intensity.

    The dtype is determined according to the following rules:
    - if ``dtype_or_range`` is a dtype, that is the output dtype.
    - if ``dtype_or_range`` is a dtype string, that is the dtype used, unless
      it is not a NumPy data type (e.g. 'uint12' for 12-bit unsigned integers),
      in which case the data type that can contain it will be used
      (e.g. uint16 in this case).
    - if ``dtype_or_range`` is a pair of values, the output data type will be
      ``_supported_float_type(image_dtype)``. This preserves float32 output for
      float32 inputs.

    Parameters
    ----------
    dtype_or_range : type, string, or 2-tuple of int/float
        The desired range for the output, expressed as either a NumPy dtype or
        as a (min, max) pair of numbers.
    image_dtype : np.dtype
        The input image dtype.

    Returns
    -------
    out_dtype : type
        The data type appropriate for the desired output.
    """
    if type(dtype_or_range) in [list, tuple, np.ndarray]:
        # pair of values: always return float.
        return utils._supported_float_type(image_dtype)
    if type(dtype_or_range) == type:
        # already a type: return it
        return dtype_or_range
    if dtype_or_range in DTYPE_RANGE:
        # string key in DTYPE_RANGE dictionary
        try:
            # if it's a canonical numpy dtype, convert
            return np.dtype(dtype_or_range).type
        except TypeError:  # uint10, uint12, uint14
            # otherwise, return uint16
            return np.uint16
    else:
        raise ValueError(
            "Incorrect value for out_range, should be a valid image data "
            f"type or a pair of values, got {dtype_or_range}."
        )


@cp.memoize(for_each_device=True)
def _get_rescale_intensity_kernel(dtype):
    """Fuse clipping and rescaling with separately rounded arithmetic stages."""
    ctype = "float" if dtype == "float32" else "double"
    return cp.ElementwiseKernel(
        f"T image, {dtype} imin, {dtype} imax, {dtype} omin, {dtype} omax, "
        f"{dtype} input_scale, {dtype} output_scale, bool constant",
        "U output",
        f"typedef {ctype} S;"
        + """
        S value = S(image);
        value = value < imin ? imin : value;
        value = value > imax ? imax : value;
        if (imin > imax) value = imax;
        if (isnan(imin) || isnan(imax)) value = imin + imax;
        if (constant) {
            value = value < omin ? omin : value;
            value = value > omax ? omax : value;
            if (omin > omax) value = omax;
            if (isnan(omin) || isnan(omax)) value = omin + omax;
        } else {
            value = value - imin;
            value = value / input_scale;
            value = value * output_scale;
            value = value + omin;
        }
        output = U(value);
        """,
        "cucim_rescale_intensity",
        # Preserve the rounding of the separate multiply and add operations.
        options=("-fmad=false",),
    )


def rescale_intensity(image, in_range="image", out_range="dtype"):
    """Return image after stretching or shrinking its intensity levels.

    The desired intensity range of the input and output, `in_range` and
    `out_range` respectively, are used to stretch or shrink the intensity range
    of the input image. See examples below.

    Parameters
    ----------
    image : array
        Image array.
    in_range, out_range : str or 2-tuple, optional
        Min and max intensity values of input and output image.
        The possible values for this parameter are enumerated below.

        'image'
            Use image min/max as the intensity range.
        'dtype'
            Use min/max of the image's dtype as the intensity range.
        dtype-name
            Use intensity range based on desired `dtype`. Must be valid key
            in `DTYPE_RANGE`.
        2-tuple
            Use `range_values` as explicit min/max intensities.

    Returns
    -------
    out : array
        Image array after rescaling its intensity. This image is the same dtype
        as the input image.

    Notes
    -----
    .. versionchanged:: 0.17
        The dtype of the output array has changed to match the output dtype, or
        float if the output range is specified by a pair of values.

    See Also
    --------
    equalize_hist

    Examples
    --------
    By default, the min/max intensities of the input image are stretched to
    the limits allowed by the image's dtype, since `in_range` defaults to
    'image' and `out_range` defaults to 'dtype':

    >>> image = cp.array([51, 102, 153], dtype=np.uint8)
    >>> rescale_intensity(image)
    array([  0, 127, 255], dtype=uint8)

    It's easy to accidentally convert an image dtype from uint8 to float:

    >>> 1.0 * image
    array([ 51., 102., 153.])

    Use `rescale_intensity` to rescale to the proper range for float dtypes:

    >>> image_float = 1.0 * image
    >>> rescale_intensity(image_float)
    array([0. , 0.5, 1. ])

    To maintain the low contrast of the original, use the `in_range` parameter:

    >>> rescale_intensity(image_float, in_range=(0, 255))
    array([0.2, 0.4, 0.6])

    If the min/max value of `in_range` is more/less than the min/max image
    intensity, then the intensity levels are clipped:

    >>> rescale_intensity(image_float, in_range=(0, 102))
    array([0.5, 1. , 1. ])

    If you have an image with signed integers but want to rescale the image to
    just the positive range, use the `out_range` parameter. In that case, the
    output dtype will be float:

    >>> image = cp.asarray([-10, 0, 10], dtype=np.int8)
    >>> rescale_intensity(image, out_range=(0, 127))
    array([  0. ,  63.5, 127. ])

    To get the desired range with a specific dtype, use ``.astype()``:

    >>> rescale_intensity(image, out_range=(0, 127)).astype(np.int8)
    array([  0,  63, 127], dtype=int8)

    If the input image is constant, the output will be clipped directly to the
    output range:
    >>> image = cp.asarray([130, 130, 130], dtype=np.int32)
    >>> rescale_intensity(image, out_range=(0, 127)).astype(np.int32)
    array([127, 127, 127], dtype=int32)
    """
    if out_range in ["dtype", "image"]:
        out_dtype = _output_dtype(image.dtype.type, image.dtype)
    else:
        out_dtype = _output_dtype(out_range, image.dtype)

    input_range = intensity_range(image, in_range)
    imin, imax = map(float, input_range)
    # Both requests describe the original image, so reuse its extrema.
    if in_range == "image" and out_range == "image":
        output_range = input_range
    else:
        output_range = intensity_range(
            image, out_range, clip_negative=(imin >= 0)
        )
    omin, omax = map(float, output_range)

    if np.any(np.isnan([imin, imax, omin, omax])):
        utils.warn(
            "One or more intensity levels are NaN. Rescaling will broadcast "
            "NaN to the full image. Provide intensity levels yourself to "
            "avoid this. E.g. with np.nanmin(image), np.nanmax(image).",
            stacklevel=2,
        )

    if image.dtype.kind in "biu" or image.dtype in (cp.float32, cp.float64):
        # Python float bounds promote integer clipping to float64. Floating
        # images retain their dtype at each arithmetic step.
        computation_type = (
            image.dtype.type if image.dtype.kind == "f" else np.float64
        )
        output = cp.empty(image.shape, dtype=out_dtype)
        return _get_rescale_intensity_kernel(np.dtype(computation_type).name)(
            image,
            *map(
                computation_type,
                (imin, imax, omin, omax, imax - imin, omax - omin),
            ),
            imin == imax,
            output,
        )

    image = cp.clip(image, imin, imax)

    if imin != imax:
        image = (image - imin) / (imax - imin)
        return cp.asarray(image * (omax - omin) + omin, dtype=out_dtype)
    else:
        return cp.clip(image, omin, omax).astype(out_dtype, copy=False)


def _assert_non_negative(image):
    if cp.any(image < 0):  # synchronize!
        raise ValueError(
            "Image Correction methods work correctly only on "
            "images with non-negative values. Use "
            "skimage.exposure.rescale_intensity."
        )


def _adjust_gamma_u8(image, gamma, gain):
    """LUT based implementation of gamma adjustment."""
    lut = 255 * gain * (np.linspace(0, 1, 256) ** gamma)
    lut = np.minimum(np.rint(lut), 255).astype("uint8")
    lut = cp.asarray(lut)
    return lut[image]


def adjust_gamma(image, gamma=1, gain=1):
    """Perform gamma correction on the input image.

    Gamma correction is a power-law transform [1]_. This function
    transforms the input `image` pixel-wise according to the power law
    ``image**gamma`` after scaling each pixel to the range 0 to 1. Then
    it is rescaled to its original range and multiplied by `gain`.

    Parameters
    ----------
    image : ndarray
        Input image.
    gamma : float, optional
        Non negative real number. Default value is 1.
    gain : float, optional
        The constant multiplier. Default value is 1.

    Returns
    -------
    out : ndarray
        Gamma corrected output image.

    See Also
    --------
    adjust_log

    Notes
    -----
    For gamma greater than 1, the histogram will shift towards left and
    the output image will be darker than the input image.

    For gamma less than 1, the histogram will shift towards right and
    the output image will be brighter than the input image.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Gamma_correction

    Examples
    --------
    >>> import skimage as ski
    >>> from cucim.skimage import exposure, img_as_float
    >>> image = img_as_float(cp.array(ski.data.moon()))
    >>> gamma_corrected = exposure.adjust_gamma(image, 2)
    >>> # Output is darker for gamma > 1
    >>> image.mean() > gamma_corrected.mean()
    array(True)
    """
    if gamma < 0:
        raise ValueError("Gamma should be a non-negative real number.")

    dtype = image.dtype.type

    if dtype is cp.uint8:
        out = _adjust_gamma_u8(image, gamma, gain)
    else:
        _assert_non_negative(image)

        limits = dtype_limits(image, clip_negative=True)
        scale = float(limits[1] - limits[0])

        out = (((image / scale) ** gamma) * scale * gain).astype(dtype)

    return out


def adjust_log(image, gain=1, inv=False):
    """Performs Logarithmic correction on the input image.

    This function transforms the input image pixelwise according to the
    equation ``O = gain*log(1 + I)`` after scaling each pixel to the range
    0 to 1.

    For inverse logarithmic correction, the equation is
    ``O = gain*(2**I - 1)``.

    Parameters
    ----------
    image : ndarray
        Input image.
    gain : float, optional
        The constant multiplier. Default value is 1.
    inv : float, optional
        If True, it performs inverse logarithmic correction,
        else correction will be logarithmic. Defaults to False.

    Returns
    -------
    out : ndarray
        Logarithm corrected output image.

    See Also
    --------
    adjust_gamma

    References
    ----------
    .. [1] http://www.ece.ucsb.edu/Faculty/Manjunath/courses/ece178W03/EnhancePart1.pdf

    """  # noqa
    _assert_non_negative(image)
    dtype = image.dtype.type
    scale = float(dtype_limits(image, True)[1] - dtype_limits(image, True)[0])

    if inv:
        out = (2 ** (image / scale) - 1) * scale * gain
        return out.astype(dtype, copy=False)

    out = cp.log2(1 + image / scale) * scale * gain
    return out.astype(dtype, copy=False)


def adjust_sigmoid(image, cutoff=0.5, gain=10, inv=False):
    """Performs Sigmoid Correction on the input image.

    Also known as Contrast Adjustment.
    This function transforms the input image pixelwise according to the
    equation ``O = 1/(1 + exp*(gain*(cutoff - I)))`` after scaling each pixel
    to the range 0 to 1.

    Parameters
    ----------
    image : ndarray
        Input image.
    cutoff : float, optional
        Cutoff of the sigmoid function that shifts the characteristic curve
        in horizontal direction. Default value is 0.5.
    gain : float, optional
        The constant multiplier in exponential's power of sigmoid function.
        Default value is 10.
    inv : bool, optional
        If True, returns the negative sigmoid correction. Defaults to False.

    Returns
    -------
    out : ndarray
        Sigmoid corrected output image.

    See Also
    --------
    adjust_gamma

    References
    ----------
    .. [1] Gustav J. Braun, "Image Lightness Rescaling Using Sigmoidal Contrast
           Enhancement Functions",
           http://markfairchild.org/PDFs/PAP07.pdf

    """
    _assert_non_negative(image)
    dtype = image.dtype.type
    scale = float(dtype_limits(image, True)[1] - dtype_limits(image, True)[0])

    if inv:
        out = (1 - 1 / (1 + cp.exp(gain * (cutoff - image / scale)))) * scale
        return out.astype(dtype, copy=False)

    out = (1 / (1 + cp.exp(gain * (cutoff - image / scale)))) * scale
    return out.astype(dtype, copy=False)


def is_low_contrast(
    image,
    fraction_threshold=0.05,
    lower_percentile=1,
    upper_percentile=99,
    method="linear",
):
    """Determine if an image is low contrast.

    Parameters
    ----------
    image : array-like
        The image under test.
    fraction_threshold : float, optional
        The low contrast fraction threshold. An image is considered low-
        contrast when its range of brightness spans less than this
        fraction of its data type's full range. [1]_
    lower_percentile : float, optional
        Disregard values below this percentile when computing image contrast.
    upper_percentile : float, optional
        Disregard values above this percentile when computing image contrast.
    method : str, optional
        The contrast determination method.  Right now the only available
        option is "linear".

    Returns
    -------
    out : bool
        True when the image is determined to be low contrast.

    Notes
    -----
    For boolean images, this function returns False only if all values are
    the same (the method, threshold, and percentile arguments are ignored).

    References
    ----------
    .. [1] https://scikit-image.org/docs/dev/user_guide/data_types.html

    Examples
    --------
    >>> import cupy as cp
    >>> image = cp.linspace(0, 0.04, 100)
    >>> is_low_contrast(image)
    array(True)
    >>> image[-1] = 1
    >>> is_low_contrast(image)
    array(True)
    >>> is_low_contrast(image, upper_percentile=100)
    array(False)
    """
    if image.dtype == bool:
        return not ((image.max() == 1) and (image.min() == 0))

    if image.ndim == 3:
        from ..color import rgb2gray, rgba2rgb  # avoid circular import

        if image.shape[2] == 4:
            image = rgba2rgb(image)
        if image.shape[2] == 3:
            image = rgb2gray(image)

    dlimits = dtype_limits(image, clip_negative=False)
    limits = cp.percentile(image, [lower_percentile, upper_percentile])
    ratio = (limits[1] - limits[0]) / (dlimits[1] - dlimits[0])

    return ratio < fraction_threshold
