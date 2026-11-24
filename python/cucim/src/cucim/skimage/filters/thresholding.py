# SPDX-FileCopyrightText: 2009-2015 Board of Regents of the University of Wisconsin-Madison, Broad Institute of MIT and Harvard, and Max Planck Institute of Molecular Cell Biology and Genetics
# SPDX-FileCopyrightText: 2009 Zachary Pincus
# SPDX-FileCopyrightText: 2009 Almar Klein
# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-2-Clause AND BSD-3-Clause

import inspect
import math
from collections import OrderedDict
from collections.abc import Iterable

import cupy as cp
import numpy as np
from skimage.filters import (
    threshold_isodata as _threshold_isodata_cpu,
    threshold_minimum as _threshold_minimum_cpu,
)

import cucim.skimage._vendored.ndimage as ndi

from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type, warn
from .._shared.version_requirements import require
from ..exposure import histogram
from ..util import dtype_limits
from ._sparse import _validate_window_size
from ._thresholding_kernels import (
    _MINIMUM_MAX_BINS,
    _mean_std_kernel,
    _multiotsu_threshold_indices,
    _threshold_isodata_gpu,
    _threshold_minimum_gpu,
    _threshold_otsu_gpu,
    _threshold_triangle_gpu,
    _threshold_yen_gpu,
)

__all__ = [
    "try_all_threshold",
    "threshold_otsu",
    "threshold_yen",
    "threshold_isodata",
    "threshold_li",
    "threshold_local",
    "threshold_minimum",
    "threshold_mean",
    "threshold_niblack",
    "threshold_sauvola",
    "threshold_triangle",
    "apply_hysteresis_threshold",
    "threshold_multiotsu",
]


def _try_all(image, methods=None, figsize=None, num_cols=2, verbose=True):
    """Returns a figure comparing the outputs of different methods.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    methods : dict, optional
        Names and associated functions.
        Functions must take and return an image.
    figsize : tuple, optional
        Figure size (in inches).
    num_cols : int, optional
        Number of columns.
    verbose : bool, optional
        Print function name for each method.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axes.
    """
    from matplotlib import pyplot as plt

    # Compute the image histogram for better performances
    nbins = 256  # Default in threshold functions
    hist = histogram(image.ravel(), nbins, source_range="image")

    # Handle default value
    methods = methods or {}

    num_rows = math.ceil((len(methods) + 1.0) / num_cols)
    fig, ax = plt.subplots(
        num_rows, num_cols, figsize=figsize, sharex=True, sharey=True
    )
    ax = ax.ravel()

    ax[0].imshow(cp.asnumpy(image), cmap=plt.cm.gray)
    ax[0].set_title("Original")

    i = 1
    for name, func in methods.items():
        # Use precomputed histogram for supporting functions
        sig = inspect.signature(func)
        _kwargs = dict(hist=hist) if "hist" in sig.parameters else {}

        ax[i].set_title(name)
        try:
            ax[i].imshow(cp.asnumpy(func(image, **_kwargs)), cmap=plt.cm.gray)
        except Exception as e:
            ax[i].text(
                0.5,
                0.5,
                f"{type(e).__name__}",
                ha="center",
                va="center",
                transform=ax[i].transAxes,
            )
        i += 1
        if verbose:
            print(func.__orifunc__)

    for a in ax:
        a.axis("off")

    fig.tight_layout()
    return fig, ax


@require("matplotlib", ">=3.3")
def try_all_threshold(image, figsize=(8, 5), verbose=True):
    """Returns a figure comparing the outputs of different thresholding methods.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    figsize : tuple, optional
        Figure size (in inches).
    verbose : bool, optional
        Print function name for each method.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axes.

    Notes
    -----
    The following algorithms are used:

    * isodata
    * li
    * mean
    * minimum
    * otsu
    * triangle
    * yen

    Examples
    --------
    >>> from skimage.data import text
    >>> text_img = cp.array(text())
    >>> fig, ax = try_all_threshold(text_img, figsize=(10, 6), verbose=False)
    """

    def thresh(func):
        """
        A wrapper function to return a thresholded image.
        """

        def wrapper(im):
            return im > func(im)

        try:
            wrapper.__orifunc__ = func.__orifunc__
        except AttributeError:
            wrapper.__orifunc__ = func.__module__ + "." + func.__name__
        return wrapper

    # Global algorithms.
    methods = OrderedDict(
        {
            "Isodata": thresh(threshold_isodata),
            "Li": thresh(threshold_li),
            "Mean": thresh(threshold_mean),
            "Minimum": thresh(threshold_minimum),
            "Otsu": thresh(threshold_otsu),
            "Triangle": thresh(threshold_triangle),
            "Yen": thresh(threshold_yen),
        }
    )

    return _try_all(image, figsize=figsize, methods=methods, verbose=verbose)


def threshold_local(
    image,
    block_size=3,
    method="gaussian",
    offset=0,
    mode="reflect",
    param=None,
    cval=0,
):
    """Compute a threshold mask image based on local pixel neighborhood.

    Also known as adaptive or dynamic thresholding. The threshold value is
    the weighted mean for the local neighborhood of a pixel subtracted by a
    constant. Alternatively the threshold can be determined dynamically by a
    given function, using the 'generic' method.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray
        Grayscale input image.
    block_size : int or sequence of int
        Odd size of pixel neighborhood which is used to calculate the
        threshold value (e.g. 3, 5, 7, ..., 21, ...).
    method : {'generic', 'gaussian', 'mean', 'median'}, optional
        Method used to determine adaptive threshold for local neighborhood in
        weighted mean image.

        * 'generic': use custom function (see ``param`` parameter)
        * 'gaussian': apply gaussian filter (see ``param`` parameter for custom\
                      sigma value)
        * 'mean': apply arithmetic mean filter
        * 'median': apply median rank filter

        By default, the 'gaussian' method is used.
    offset : float, optional
        Constant subtracted from weighted mean of neighborhood to calculate
        the local threshold value. Default offset is 0.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
        Default is 'reflect'.
    param : {int, function}, optional
        Either specify sigma for 'gaussian' method or function object for
        'generic' method. This functions takes the flat array of local
        neighborhood as a single argument and returns the calculated
        threshold for the centre pixel.
    cval : float, optional
        Value to fill past edges of input if mode is 'constant'.

    Returns
    -------
    threshold : (M, N[, ...]) ndarray
        Threshold image. All pixels in the input image higher than the
        corresponding pixel in the threshold image are considered foreground.

    References
    ----------
    .. [1] Gonzalez, R. C. and Wood, R. E. "Digital Image Processing
           (2nd Edition)." Prentice-Hall Inc., 2002: 600--612.
           ISBN: 0-201-18075-8

    Examples
    --------
    >>> import cupy as cp
    >>> from skimage.data import camera
    >>> image = cp.array(camera()[:50, :50])
    >>> binary_image1 = image > threshold_local(image, 15, 'mean')
    """  # noqa
    if np.isscalar(block_size):
        block_size = (block_size,) * image.ndim
    elif len(block_size) != image.ndim:
        raise ValueError("len(block_size) must equal image.ndim.")
    block_size = tuple(block_size)
    if any(b % 2 == 0 for b in block_size):
        raise ValueError(
            f"block_size must be odd! Given block_size "
            f"{block_size} contains even values."
        )
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    thresh_image = cp.zeros(image.shape, dtype=float_dtype)
    if method == "generic":
        raise NotImplementedError("TODO: implement generic_filter")
        ndi.generic_filter(
            image, param, block_size, output=thresh_image, mode=mode, cval=cval
        )
    elif method == "gaussian":
        if param is None:
            # automatically determine sigma which covers > 99% of distribution
            sigma = tuple([(b - 1) / 6.0 for b in block_size])
        else:
            sigma = param
        gaussian(image, sigma=sigma, out=thresh_image, mode=mode, cval=cval)
    elif method == "mean":
        ndi.uniform_filter(
            image, block_size, output=thresh_image, mode=mode, cval=cval
        )
    elif method == "median":
        ndi.median_filter(
            image, block_size, output=thresh_image, mode=mode, cval=cval
        )
    else:
        raise ValueError(
            "Invalid method specified. Please use `generic`, "
            "`gaussian`, `mean`, or `median`."
        )

    return thresh_image - offset


def _validate_image_histogram(image, hist, nbins=None, normalize=False):
    """Ensure that either image or hist were given, return valid histogram.

    If hist is given, image is ignored.

    Parameters
    ----------
    image : array or None
        Grayscale image.
    hist : array, 2-tuple of array, or None
        Histogram, either a 1D counts array, or an array of counts together
        with an array of bin centers.
    nbins : int, optional
        The number of bins with which to compute the histogram, if `hist` is
        None.
    normalize : bool
        If hist is not given, it will be computed by this function. This
        parameter determines whether the computed histogram is normalized
        (i.e. entries sum up to 1) or not.

    Returns
    -------
    counts : 1D array of float
        Each element is the number of pixels falling in each intensity bin.
    bin_centers : 1D array
        Each element is the value corresponding to the center of each intensity
        bin.

    Raises
    ------
    ValueError : if image and hist are both None
    """
    if image is None and hist is None:
        raise Exception("Either image or hist must be provided.")

    if hist is not None:
        if isinstance(hist, (tuple, list)):
            counts, bin_centers = hist
        else:
            counts = hist
            bin_centers = cp.arange(counts.size)

        if counts[0] == 0 or counts[-1] == 0:
            # Trim histogram from both ends by removing starting and
            # ending zeroes as in histogram(..., source_range="image")
            cond = counts > 0
            start = int(cp.argmax(cond))
            end = cond.size - int(cp.argmax(cond[::-1]))
            counts, bin_centers = counts[start:end], bin_centers[start:end]

    else:
        counts, bin_centers = histogram(
            image.ravel(), nbins, source_range="image", normalize=normalize
        )
    return counts.astype(cp.float32, copy=False), bin_centers


def _first_pixel_if_constant(image, counts):
    """Return the first pixel if the histogram shows a constant image.

    This avoids a pass over the image to compare it with its first pixel. The
    range of the histogram is the one of the image, so the first and the last
    bin are not empty unless the image has a single value: for an integer image
    the histogram then has a single bin, and for a floating point image a
    single non-empty bin.
    """
    if counts.size == 1 or (
        image.dtype.kind == "f" and int(cp.count_nonzero(counts)) == 1
    ):
        return image.ravel()[0]
    return None


def threshold_otsu(image=None, nbins=256, *, hist=None):
    """Return threshold value based on Otsu's method.

    Either image or hist must be provided. If hist is provided, the actual
    histogram of the image is ignored.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray, optional
        Grayscale input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    hist : array, or 2-tuple of arrays, optional
        Histogram from which to determine the threshold, and optionally a
        corresponding array of bin center intensities. If no hist provided,
        this function will compute it from the image.


    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.

    References
    ----------
    .. [1] Wikipedia, https://en.wikipedia.org/wiki/Otsu's_Method
    .. [2] Otsu, N., "A threshold selection method from gray-level
           histograms", IEEE Transactions on Systems, Man, and Cybernetics
           9(1): 62-66, 1979. :DOI:`10.1109/TSMC.1979.4310076`

    Examples
    --------
    >>> from skimage.data import camera
    >>> image = cp.array(camera())
    >>> thresh = threshold_otsu(image)
    >>> binary = image <= thresh

    Notes
    -----
    The input image must be grayscale.
    """

    if image is not None and image.ndim > 2 and image.shape[-1] in (3, 4):
        warn(
            f"threshold_otsu is expected to work correctly only for "
            f"grayscale images; image shape {image.shape} looks like "
            f"that of an RGB image."
        )

    if image is not None and hist is not None:
        # (the histogram is not the one of the image)
        first_pixel = image.ravel()[0]
        if cp.all(image == first_pixel):  # device synchronization!
            return first_pixel

    counts, bin_centers = _validate_image_histogram(image, hist, nbins)

    # Check if the image has more than one intensity value; if not, return that
    # value
    if image is not None and hist is None:
        first_pixel = _first_pixel_if_constant(image, counts)
        if first_pixel is not None:
            return first_pixel

    return _threshold_otsu_gpu(counts, bin_centers)


def threshold_yen(image=None, nbins=256, *, hist=None):
    """Return threshold value based on Yen's method.
    Either image or hist must be provided. In case hist is given, the actual
    histogram of the image is ignored.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray
        Grayscale input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    hist : array, or 2-tuple of arrays, optional
        Histogram from which to determine the threshold, and optionally a
        corresponding array of bin center intensities.
        An alternative use of this function is to pass it only hist.

    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.

    References
    ----------
    .. [1] Yen J.C., Chang F.J., and Chang S. (1995) "A New Criterion
           for Automatic Multilevel Thresholding" IEEE Trans. on Image
           Processing, 4(3): 370-378. :DOI:`10.1109/83.366472`
    .. [2] Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding
           Techniques and Quantitative Performance Evaluation" Journal of
           Electronic Imaging, 13(1): 146-165, :DOI:`10.1117/1.1631315`
           http://www.busim.ee.boun.edu.tr/~sankur/SankurFolder/Threshold_survey.pdf
    .. [3] ImageJ AutoThresholder code, http://fiji.sc/wiki/index.php/Auto_Threshold

    Examples
    --------
    >>> from skimage.data import camera
    >>> image = cp.array(camera())
    >>> thresh = threshold_yen(image)
    >>> binary = image <= thresh
    """  # noqa

    counts, bin_centers = _validate_image_histogram(image, hist, nbins)

    # On blank images (e.g. filled with 0) with int dtype, `histogram()`
    # returns ``bin_centers`` containing only one value. Speed up with it.
    if bin_centers.size == 1:
        return bin_centers[0]

    return _threshold_yen_gpu(counts, bin_centers)


def threshold_isodata(image=None, nbins=256, return_all=False, *, hist=None):
    """Return threshold value(s) based on ISODATA method.

    Histogram-based threshold, known as Ridler-Calvard method or inter-means.
    Threshold values returned satisfy the following equality::

        threshold = (image[image <= threshold].mean() +
                     image[image > threshold].mean()) / 2.0

    That is, returned thresholds are intensities that separate the image into
    two groups of pixels, where the threshold intensity is midway between the
    mean intensities of these groups.

    For integer images, the above equality holds to within one; for floating-
    point images, the equality holds to within the histogram bin-width.

    Either image or hist must be provided. In case hist is given, the actual
    histogram of the image is ignored.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray
        Grayscale input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    return_all : bool, optional
        If False (default), return only the lowest threshold that satisfies
        the above equality. If True, return all valid thresholds.
    hist : array, or 2-tuple of arrays, optional
        Histogram to determine the threshold from and a corresponding array
        of bin center intensities. Alternatively, only the histogram can be
        passed.

    Returns
    -------
    threshold : float or int or array
        Threshold value(s).

    References
    ----------
    .. [1] Ridler, TW & Calvard, S (1978), "Picture thresholding using an
           iterative selection method"
           IEEE Transactions on Systems, Man and Cybernetics 8: 630-632,
           :DOI:`10.1109/TSMC.1978.4310039`
    .. [2] Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding
           Techniques and Quantitative Performance Evaluation" Journal of
           Electronic Imaging, 13(1): 146-165,
           http://www.busim.ee.boun.edu.tr/~sankur/SankurFolder/Threshold_survey.pdf
           :DOI:`10.1117/1.1631315`
    .. [3] ImageJ AutoThresholder code,
           http://fiji.sc/wiki/index.php/Auto_Threshold

    Examples
    --------
    >>> from skimage.data import coins
    >>> image = cp.array(coins())
    >>> thresh = threshold_isodata(image)
    >>> binary = image > thresh
    """

    counts, bin_centers = _validate_image_histogram(image, hist, nbins)

    # image only contains one unique value
    if len(bin_centers) == 1:
        if return_all:
            return bin_centers
        else:
            return bin_centers[0]

    if not return_all:
        return _threshold_isodata_gpu(counts, bin_centers)

    # small size counts, bin_centers -> faster on the host
    counts = cp.asnumpy(counts)
    bin_centers = cp.asnumpy(bin_centers)
    counts = counts.astype(cp.float32, copy=False)
    return cp.asarray(
        _threshold_isodata_cpu(
            nbins=nbins, return_all=return_all, hist=(counts, bin_centers)
        )
    )


# Computing a histogram using cp.histogram on a uint8 image with bins=256
# doesn't work and results in aliasing problems. We use a fully specified set
# of bins to ensure that each uint8 value false into its own bin.
_DEFAULT_ENTROPY_BINS = tuple(np.arange(-0.5, 255.51, 1))


def _cross_entropy(image, threshold, bins=_DEFAULT_ENTROPY_BINS):
    """Compute cross-entropy between distributions above and below a threshold.

    Parameters
    ----------
    image : array
        The input array of values.
    threshold : float
        The value dividing the foreground and background in ``image``.
    bins : int or array of float, optional
        The number of bins or the bin edges. (Any valid value to the ``bins``
        argument of ``cp.histogram`` will work here.) For an exact calculation,
        each unique value should have its own bin. The default value for bins
        ensures exact handling of uint8 images: ``bins=256`` results in
        aliasing problems due to bin width not being equal to 1.

    Returns
    -------
    nu : float
        The cross-entropy target value as defined in [1]_.

    Notes
    -----
    See Li and Lee, 1993 [1]_; this is the objective function ``threshold_li``
    minimizes. This function can be improved but this implementation most
    closely matches equation 8 in [1]_ and equations 1-3 in [2]_.

    References
    ----------
    .. [1] Li C.H. and Lee C.K. (1993) "Minimum Cross Entropy Thresholding"
           Pattern Recognition, 26(4): 617-625
           :DOI:`10.1016/0031-3203(93)90115-D`
    .. [2] Li C.H. and Tam P.K.S. (1998) "An Iterative Algorithm for Minimum
           Cross Entropy Thresholding" Pattern Recognition Letters, 18(8): 771-776
           :DOI:`10.1016/S0167-8655(98)00057-9`
    """  # noqa
    bins = cp.asarray(bins)  # required for _DEFAULT_ENTROPY_BINS tuple
    histogram, bin_edges = cp.histogram(image, bins=bins, density=True)
    bin_centers = cp.convolve(bin_edges, cp.array([0.5, 0.5]), mode="valid")
    t = cp.flatnonzero(bin_centers > threshold)[0]
    m0a = cp.sum(histogram[:t])  # 0th moment, background
    m0b = cp.sum(histogram[t:])
    m1a = cp.sum(histogram[:t] * bin_centers[:t])  # 1st moment, background
    m1b = cp.sum(histogram[t:] * bin_centers[t:])
    mua = m1a / m0a  # mean value, background
    mub = m1b / m0b
    nu = -m1a * cp.log(mua) - m1b * cp.log(mub)
    return nu


def _threshold_li_integer(image, tolerance, iter_callback):
    """Li's iterative method for an image of integers, from its histogram.

    The same iteration as in `threshold_li`, with the histogram of the image
    (one pass, then one small transfer to the host) instead of passes over the
    image to remove nan and non-finite values (there are none in integer
    images), to look for a constant image, to subtract the minimum and to
    compute the mean.
    """
    hist, bin_centers = histogram(image.reshape(-1), source_range="image")
    if hist.size == 1:
        # constant image: return that value
        return image.ravel()[0]

    counts = cp.asnumpy(hist)  # synchronize
    centers = cp.asnumpy(bin_centers).astype(np.int64)
    image_min = bin_centers[0].astype(image.dtype)
    centers = centers - centers[0]  # the image minus its minimum
    weights = counts.astype(np.float32)  # as in the iteration of threshold_li

    tolerance = tolerance or 0.5
    # initial estimate: the mean of the image (minus its minimum)
    t_next = float((centers * counts).sum()) / int(counts.sum())
    t_curr = -2 * tolerance
    if iter_callback is not None:
        iter_callback(t_next + image_min)

    eps = 100 * np.finfo(float).eps
    while abs(t_next - t_curr) > tolerance:
        t_curr = t_next
        foreground = centers > t_curr
        background = ~foreground
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_fore = float(
                np.multiply(
                    centers[foreground], weights[foreground], dtype=np.float64
                ).sum()
                / weights[foreground].sum(dtype=np.float64)
            )
            mean_back = float(
                np.multiply(
                    centers[background], weights[background], dtype=np.float64
                ).sum()
                / weights[background].sum(dtype=np.float64)
            )

        if mean_back == 0:
            break

        t_next = (mean_back - mean_fore) / (
            math.log(mean_back + eps) - math.log(mean_fore + eps)
        )

        if iter_callback is not None:
            iter_callback(t_next + image_min)

    return t_next + image_min


def threshold_li(
    image, *, tolerance=None, initial_guess=None, iter_callback=None
):
    """Compute threshold value by Li's iterative Minimum Cross Entropy method.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray
        Grayscale input image.

    tolerance : float, optional
        Finish the computation when the change in the threshold in an iteration
        is less than this value. By default, this is half the smallest
        difference between intensity values in ``image``.

    initial_guess : float or Callable[[array[float]], float], optional
        Li's iterative method uses gradient descent to find the optimal
        threshold. If the image intensity histogram contains more than two
        modes (peaks), the gradient descent could get stuck in a local optimum.
        An initial guess for the iteration can help the algorithm find the
        globally-optimal threshold. A float value defines a specific start
        point, while a callable should take in an array of image intensities
        and return a float value. Example valid callables include
        ``numpy.mean`` (default), ``lambda arr: numpy.quantile(arr, 0.95)``,
        or even :func:`skimage.filters.threshold_otsu`.

    iter_callback : Callable[[float], Any], optional
        A function that will be called on the threshold at every iteration of
        the algorithm.

    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.

    References
    ----------
    .. [1] Li C.H. and Lee C.K. (1993) "Minimum Cross Entropy Thresholding"
           Pattern Recognition, 26(4): 617-625
           :DOI:`10.1016/0031-3203(93)90115-D`
    .. [2] Li C.H. and Tam P.K.S. (1998) "An Iterative Algorithm for Minimum
           Cross Entropy Thresholding" Pattern Recognition Letters, 18(8): 771-776
           :DOI:`10.1016/S0167-8655(98)00057-9`
    .. [3] Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding
           Techniques and Quantitative Performance Evaluation" Journal of
           Electronic Imaging, 13(1): 146-165
           :DOI:`10.1117/1.1631315`
    .. [4] ImageJ AutoThresholder code, http://fiji.sc/wiki/index.php/Auto_Threshold

    Examples
    --------
    >>> from skimage.data import camera
    >>> image = cp.array(camera())
    >>> thresh = threshold_li(image)
    >>> binary = image > thresh
    """  # noqa
    if image.dtype.kind in "iu" and initial_guess is None and image.size > 0:
        # integer images: everything is computed from the histogram
        return _threshold_li_integer(image, tolerance, iter_callback)

    # Remove nan:
    image = image[~cp.isnan(image)]
    if image.size == 0:
        return cp.nan

    # Make sure image has more than one value; otherwise, return that value
    # This works even for cp.inf
    val0 = image.ravel()[0]
    if cp.all(image == val0):  # device synchronize!
        return val0

    # At this point, the image only contains cp.inf, -cp.inf, or valid numbers
    image = image[cp.isfinite(image)]
    # if there are no finite values in the image, return 0. This is because
    # at this point we *know* that there are *both* inf and -inf values,
    # because inf == inf evaluates to True. We might as well separate them.
    if image.size == 0:
        return 0.0

    # Li's algorithm requires positive image (because of log(mean))
    image_min = cp.min(image)
    image -= image_min
    if image.dtype.kind in "iu":
        tolerance = tolerance or 0.5
    else:
        tolerance = tolerance or float(cp.min(cp.diff(cp.unique(image))) / 2)

    # Initial estimate for iteration. See "initial_guess" in the parameter list
    if initial_guess is None:
        t_next = cp.mean(image)
    elif callable(initial_guess):
        t_next = initial_guess(image)
    elif cp.isscalar(initial_guess):  # convert to new, positive image range
        image_min = float(image_min)
        t_next = initial_guess - image_min
        image_max = cp.max(image) + image_min
        if not 0 < t_next < cp.max(image):
            msg = (
                f"The initial guess for threshold_li must be within the "
                f"range of the image. Got {initial_guess} for image min "
                f"{image_min} and max {image_max}."
            )
            raise ValueError(msg)
    else:
        raise TypeError(
            "Incorrect type for `initial_guess`; should be "
            "a floating point value, or a function mapping an "
            "array to a floating point value."
        )

    # initial value for t_curr must be different from t_next by at
    # least the tolerance. Since the image is positive, we ensure this
    # by setting to a large-enough negative number
    t_curr = -2 * tolerance

    # Callback on initial iterations
    if iter_callback is not None:
        iter_callback(t_next + image_min)

    # Stop the iterations when the difference between the
    # new and old threshold values is less than the tolerance

    if image.dtype.kind in "iu":
        hist, bin_centers = histogram(image.reshape(-1), source_range="image")
        hist = hist.astype(cp.float32, copy=False)
        while abs(t_next - t_curr) > tolerance:
            t_curr = t_next
            foreground = bin_centers > t_curr
            background = ~foreground

            mean_fore = np.average(
                bin_centers[foreground], weights=hist[foreground]
            )
            mean_back = np.average(
                bin_centers[background], weights=hist[background]
            )

            if mean_back == 0:
                break

            eps = 100 * np.finfo(float).eps
            mean_back = float(mean_back)
            mean_fore = float(mean_fore)
            t_next = (mean_back - mean_fore) / (
                math.log(mean_back + eps) - math.log(mean_fore + eps)
            )

            if iter_callback is not None:
                iter_callback(t_next + image_min)

    else:
        # Stop the iterations when the difference between the
        # new and old threshold values is less than the tolerance
        while abs(t_next - t_curr) > tolerance:
            t_curr = t_next
            foreground = image > t_curr
            eps = 100 * np.finfo(float).eps
            mean_fore = float(cp.mean(image[foreground]))
            mean_back = float(cp.mean(image[~foreground]))

            if mean_back == 0:
                break

            t_next = (mean_back - mean_fore) / (
                math.log(mean_back + eps) - math.log(mean_fore + eps)
            )

            if iter_callback is not None:
                iter_callback(t_next + image_min)

    threshold = t_next + image_min
    return threshold


def threshold_minimum(image=None, nbins=256, max_num_iter=10000, *, hist=None):
    """Return threshold value based on minimum method.

    The histogram of the input ``image`` is computed if not provided and
    smoothed until there are only two maxima. Then the minimum in between is
    the threshold value.

    Either image or hist must be provided. In case hist is given, the actual
    histogram of the image is ignored.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray, optional
        Grayscale input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    max_num_iter : int, optional
        Maximum number of iterations to smooth the histogram.
    hist : array, or 2-tuple of arrays, optional
        Histogram to determine the threshold from and a corresponding array
        of bin center intensities. Alternatively, only the histogram can be
        passed.

    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.

    Raises
    ------
    RuntimeError
        If unable to find two local maxima in the histogram or if the
        smoothing takes more than 1e4 iterations.

    References
    ----------
    .. [1] C. A. Glasbey, "An analysis of histogram-based thresholding
           algorithms," CVGIP: Graphical Models and Image Processing,
           vol. 55, pp. 532-537, 1993.
    .. [2] Prewitt, JMS & Mendelsohn, ML (1966), "The analysis of cell
           images", Annals of the New York Academy of Sciences 128: 1035-1053
           :DOI:`10.1111/j.1749-6632.1965.tb11715.x`

    Examples
    --------
    >>> from skimage.data import camera
    >>> image = cp.array(camera())
    >>> thresh = threshold_minimum(image)
    >>> binary = image > thresh
    """

    counts, bin_centers = _validate_image_histogram(image, hist, nbins)
    if counts.size > _MINIMUM_MAX_BINS:
        # too large for the shared memory of the kernel
        return cp.asarray(
            _threshold_minimum_cpu(
                nbins=nbins,
                max_num_iter=max_num_iter,
                hist=(
                    cp.asnumpy(counts).astype(np.float32, copy=False),
                    cp.asnumpy(bin_centers),
                ),
            )
        )
    return _threshold_minimum_gpu(counts, bin_centers, max_num_iter)


def threshold_mean(image):
    """Return threshold value based on the mean of grayscale values.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray
        Grayscale input image.

    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.

    References
    ----------
    .. [1] C. A. Glasbey, "An analysis of histogram-based thresholding
        algorithms," CVGIP: Graphical Models and Image Processing,
        vol. 55, pp. 532-537, 1993.
        :DOI:`10.1006/cgip.1993.1040`

    Examples
    --------
    >>> from skimage.data import camera
    >>> image = cp.array(camera())
    >>> thresh = threshold_mean(image)
    >>> binary = image > thresh
    """
    return cp.mean(image)


def threshold_triangle(image, nbins=256):
    """Return threshold value based on the triangle algorithm.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray
        Grayscale input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.

    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.

    References
    ----------
    .. [1] Zack, G. W., Rogers, W. E. and Latt, S. A., 1977,
       Automatic Measurement of Sister Chromatid Exchange Frequency,
       Journal of Histochemistry and Cytochemistry 25 (7), pp. 741-753
       :DOI:`10.1177/25.7.70454`
    .. [2] ImageJ AutoThresholder code,
       http://fiji.sc/wiki/index.php/Auto_Threshold

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage.filters import threshold_triangle
    >>> from skimage.data import camera
    >>> image = cp.array(camera())
    >>> thresh = threshold_triangle(image)
    >>> binary = image > thresh
    """
    # nbins is ignored for integer arrays
    # so, we recalculate the effective nbins.
    hist, bin_centers = histogram(image.ravel(), nbins, source_range="image")
    # constant intensity: a single bin (integer-valued image) or a single
    # non-empty bin (floating point image)
    first_pixel = _first_pixel_if_constant(image, hist)
    if first_pixel is not None:
        return first_pixel

    return _threshold_triangle_gpu(hist, bin_centers)


def _mean_std(image, w):
    """Return local mean and standard deviation of each pixel using a
    neighborhood defined by a rectangular window size ``w``.
    The mean and the mean of the squares are computed with box filters in
    float64. This is used by :func:`threshold_niblack` and
    :func:`threshold_sauvola`.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray
        Grayscale input image.
    w : int, or iterable of int
        Window size specified as a single odd integer (3, 5, 7, …),
        or an iterable of length ``image.ndim`` containing only odd
        integers (e.g. ``(1, 5, 5)``).

    Returns
    -------
    m : ndarray of float, same shape as ``image``
        Local mean of the image.
    s : ndarray of float, same shape as ``image``
        Local standard deviation of the image.

    References
    ----------
    .. [1] F. Shafait, D. Keysers, and T. M. Breuel, "Efficient
           implementation of local adaptive thresholding techniques
           using integral images." in Document Recognition and
           Retrieval XV, (San Jose, USA), Jan. 2008.
           :DOI:`10.1117/12.767755`
    """
    if not isinstance(w, Iterable):
        w = (w,) * image.ndim
    _validate_window_size(w)

    float_dtype = _supported_float_type(image.dtype)
    # Box sums of the image and of its square in float64, divided by the number
    # of elements of the window once at the end. The sums of integer-valued
    # images are exact, and the variance (a difference of two nearly equal
    # values) is computed in float64 before it is cast. The mode "mirror" is the
    # "reflect" mode of numpy.pad (the edge value is not repeated).
    values = image.astype(cp.float64)
    sums = _box_sum(values, w)
    values *= values
    sums_sq = _box_sum(values, w)
    del values
    mean = cp.empty(sums.shape, dtype=float_dtype)
    std = cp.empty(sums.shape, dtype=float_dtype)
    # Note: g2 is not guaranteed to be greater than m*m when floating point
    # error is considered, so the variance is clipped at 0
    _mean_std_kernel(sums, sums_sq, float(math.prod(w)), mean, std)
    return mean, std


def _box_sum(values, w):
    """Sums over the windows of sizes ``w`` (separable, mode "mirror")."""
    for axis, size in enumerate(w):
        if size > 1:
            values = ndi.correlate1d(
                values, cp.ones(size), axis=axis, mode="mirror"
            )
    return values


def threshold_niblack(image, window_size=15, k=0.2):
    """Applies Niblack local threshold to an array.

    A threshold T is calculated for every pixel in the image using the
    following formula::

        T = m(x,y) - k * s(x,y)

    where m(x,y) and s(x,y) are the mean and standard deviation of
    pixel (x,y) neighborhood defined by a rectangular window with size w
    times w centered around the pixel. k is a configurable parameter
    that weights the effect of standard deviation.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray
        Grayscale input image.
    window_size : int, or iterable of int, optional
        Window size specified as a single odd integer (3, 5, 7, …),
        or an iterable of length ``image.ndim`` containing only odd
        integers (e.g. ``(1, 5, 5)``).
    k : float, optional
        Value of parameter k in threshold formula.

    Returns
    -------
    threshold : (M, N) ndarray
        Threshold mask. All pixels with an intensity higher than
        this value are assumed to be foreground.

    Notes
    -----
    This algorithm is originally designed for text recognition.

    The Bradley threshold is a particular case of the Niblack
    one, being equivalent to

    >>> from skimage import data
    >>> image = cp.array(data.page())
    >>> q = 1
    >>> threshold_image = threshold_niblack(image, k=0) * q

    for some value ``q``. By default, Bradley and Roth use ``q=1``.


    References
    ----------
    .. [1] W. Niblack, An introduction to Digital Image Processing,
           Prentice-Hall, 1986.
    .. [2] D. Bradley and G. Roth, "Adaptive thresholding using Integral
           Image", Journal of Graphics Tools 12(2), pp. 13-21, 2007.
           :DOI:`10.1080/2151237X.2007.10129236`

    Examples
    --------
    >>> from skimage import data
    >>> image = cp.array(data.page())
    >>> threshold_image = threshold_niblack(image, window_size=7, k=0.1)
    """
    m, s = _mean_std(image, window_size)
    return m - k * s


def threshold_sauvola(image, window_size=15, k=0.2, r=None):
    """Applies Sauvola local threshold to an array. Sauvola is a
    modification of Niblack technique.

    In the original method a threshold T is calculated for every pixel
    in the image using the following formula::

        T = m(x,y) * (1 + k * ((s(x,y) / R) - 1))

    where m(x,y) and s(x,y) are the mean and standard deviation of
    pixel (x,y) neighborhood defined by a rectangular window with size w
    times w centered around the pixel. k is a configurable parameter
    that weights the effect of standard deviation.
    R is the maximum standard deviation of a grayscale image.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray
        Grayscale input image.
    window_size : int, or iterable of int, optional
        Window size specified as a single odd integer (3, 5, 7, …),
        or an iterable of length ``image.ndim`` containing only odd
        integers (e.g. ``(1, 5, 5)``).
    k : float, optional
        Value of the positive parameter k.
    r : float, optional
        Value of R, the dynamic range of standard deviation.
        If None, set to the half of the image dtype range.

    Returns
    -------
    threshold : (M, N) ndarray
        Threshold mask. All pixels with an intensity higher than
        this value are assumed to be foreground.

    Notes
    -----
    This algorithm is originally designed for text recognition.

    References
    ----------
    .. [1] J. Sauvola and M. Pietikainen, "Adaptive document image
           binarization," Pattern Recognition 33(2),
           pp. 225-236, 2000.
           :DOI:`10.1016/S0031-3203(99)00055-2`

    Examples
    --------
    >>> from skimage import data
    >>> image = cp.array(data.page())
    >>> t_sauvola = threshold_sauvola(image, window_size=15, k=0.2)
    >>> binary_image = image > t_sauvola
    """
    if r is None:
        imin, imax = dtype_limits(image, clip_negative=False)
        r = 0.5 * (imax - imin)
    m, s = _mean_std(image, window_size)
    return m * (1 + k * ((s / r) - 1))


def apply_hysteresis_threshold(image, low, high):
    """Apply hysteresis thresholding to ``image``.

    This algorithm finds regions where ``image`` is greater than ``high``
    OR ``image`` is greater than ``low`` *and* that region is connected to
    a region greater than ``high``.

    Parameters
    ----------
    image : array, shape (M,[ N, ..., P])
        Grayscale input image.
    low : float, or array of same shape as ``image``
        Lower threshold.
    high : float, or array of same shape as ``image``
        Higher threshold.

    Returns
    -------
    thresholded : array of bool, same shape as ``image``
        Array in which ``True`` indicates the locations where ``image``
        was above the hysteresis threshold.

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage.filters import apply_hysteresis_threshold
    >>> image = cp.asarray([1, 2, 3, 2, 1, 2, 1, 3, 2])
    >>> apply_hysteresis_threshold(image, 1.5, 2.5).astype(int)
    array([0, 1, 1, 1, 0, 0, 0, 1, 1])

    References
    ----------
    .. [1] J. Canny. A computational approach to edge detection.
           IEEE Transactions on Pattern Analysis and Machine Intelligence.
           1986; vol. 8, pp.679-698.
           :DOI:`10.1109/TPAMI.1986.4767851`
    """
    low = cp.asarray(low)  # asarray to allow scalar low
    # ensure low always below high
    low = cp.clip(low, a_min=None, a_max=high)
    mask_low = image > low
    mask_high = image > high
    # Connected components of mask_low
    labels_low, num_labels = ndi.label(mask_low)
    # Check which connected components contain pixels from mask_high

    # CuPy Backend: refactored in the same style as features.canny to avoid
    #               slow call to cupyx.scipy.ndimage.sum
    nonzero_sums = cp.unique(labels_low[mask_high])
    connected_to_high = cp.zeros((num_labels + 1,), bool)
    connected_to_high[nonzero_sums] = True
    thresholded = connected_to_high[labels_low]
    return thresholded


def threshold_multiotsu(image=None, classes=3, nbins=256, *, hist=None):
    r"""Generate `classes`-1 threshold values to divide gray levels in `image`,
    following Otsu's method for multiple classes.

    The threshold values are chosen to maximize the total sum of pairwise
    variances between the thresholded graylevel classes [3]_. See Notes and [1]_
    for more details.

    Either image or hist must be provided. If hist is provided, the actual
    histogram of the image is ignored.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray, optional
        Grayscale input image.
    classes : int, optional
        Number of classes to be thresholded, i.e. the number of resulting
        regions.
    nbins : int, optional
        Number of bins used to calculate the histogram. This value is ignored
        for integer arrays.
    hist : array, or 2-tuple of arrays, optional
        Histogram from which to determine the threshold, and optionally a
        corresponding array of bin center intensities. If no hist provided,
        this function will compute it from the image (see notes).

    Returns
    -------
    thresh : array
        Array containing the threshold values for the desired classes.

    Raises
    ------
    ValueError
         If ``image`` contains less grayscale value then the desired
         number of classes.

    Notes
    -----
    This implementation finds the thresholds with an exact search by dynamic
    programming on the GPU, whose complexity is :math:`O(C h \log h)`, where
    :math:`h` is the number of histogram bins and :math:`C` is the number of
    classes desired (the exhaustive search of scikit-image is
    :math:`O\left(\frac{Ch^{C-1}}{(C-1)!}\right)`, and the plain dynamic
    programming is :math:`O(C h^2)`). The criterion is the sum
    over the classes of :math:`S^2/P`, the squared sum of the gray levels
    weighted by the histogram divided by the number of pixels of the class [1]_
    (the "modified between-class variance"). It is a sum of terms that each
    depend only on the two thresholds that bound a class, so the best
    thresholds for :math:`k` classes up to each bin follow from those for
    :math:`k - 1` classes by maximizing over the previous threshold, and the
    thresholds are recovered by following back pointers [4]_. The best
    previous threshold does not decrease with the end bin of the class (the
    matrix of each stage is totally monotone [4]_), so each stage is
    searched by divide and conquer instead of over all pairs of thresholds,
    which gives the :math:`h \log h` factor. The search
    goes from the last class to the first, which gives the smallest
    thresholds in lexicographic order among thresholds with exactly the same
    value of the criterion, as in scikit-image. The arithmetic is in float64.

    The thresholds are the same as those of scikit-image for typical images
    (for the sample images of ``skimage.data`` tried, 54 of the 57 results with
    2 to 4 classes are identical and the other 3 differ by one bin). The search
    of scikit-image accumulates in float32 and gives the first bin a weight of
    one instead of zero in the class that contains it, so for some histograms
    (mostly those with several nearly equal maxima) it does not return the
    thresholds that maximize the variance between classes, while this
    implementation does. Thresholds with exactly equal values of the criterion
    can still be chosen differently from scikit-image because of rounding.

    If no hist is given, this function will make use of
    `skimage.exposure.histogram`, which behaves differently than
    `np.histogram`. While both allowed, use the former for consistent
    behaviour.

    The input image must be grayscale.

    References
    ----------
    .. [1] Liao, P-S., Chen, T-S. and Chung, P-C., "A fast algorithm for
           multilevel thresholding", Journal of Information Science and
           Engineering 17 (5): 713-727, 2001. Available at:
           <https://ftp.iis.sinica.edu.tw/JISE/2001/200109_01.pdf>
           :DOI:`10.6688/JISE.2001.17.5.1`
    .. [2] Tosa, Y., "Multi-Otsu Threshold", a java plugin for ImageJ.
           Available at:
           <http://imagej.net/plugins/download/Multi_OtsuThreshold.java>
    .. [3] Otsu, N., "A threshold selection method from gray-level
           histograms", IEEE Transactions on Systems, Man, and Cybernetics
           9(1): 62-66, 1979. :DOI:`10.1109/TSMC.1979.4310076`
    .. [4] Luessi, M., Eichmann, M., Schuster, G. M. and Katsaggelos, A. K.,
           "New results on efficient optimal multilevel image thresholding",
           IEEE International Conference on Image Processing (ICIP) 2006,
           pp. 773-776. :DOI:`10.1109/ICIP.2006.312426`

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage.color import label2rgb
    >>> from skimage import data
    >>> image = cp.asarray(data.camera())
    >>> thresholds = threshold_multiotsu(image)
    >>> regions = cp.digitize(image, bins=thresholds)
    >>> regions_colorized = label2rgb(regions)

    """

    if image is not None and image.ndim > 2 and image.shape[-1] in (3, 4):
        warn(
            f"threshold_multiotsu is expected to work correctly only for "
            f"grayscale images; image shape {image.shape} looks like "
            f"that of an RGB image."
        )

    # the histogram of the gray levels
    counts, bin_centers = _validate_image_histogram(image, hist, nbins)

    if counts.size >= classes:
        thresh_idx, n_nonzero = _multiotsu_threshold_indices(counts, classes)
        nvalues = int(n_nonzero[0])  # device synchronization!
    else:
        nvalues = int(cp.count_nonzero(counts))
    if nvalues < classes:
        msg = (
            f"After discretization into bins, the input image has "
            f"only {nvalues} different values. It cannot be thresholded "
            f"in {classes} classes. If there are more unique values "
            f"before discretization, try increasing the number of bins "
            f"(`nbins`)."
        )
        raise ValueError(msg)
    elif nvalues == classes:
        thresh_idx = cp.flatnonzero(counts)[:-1]
    return bin_centers[thresh_idx]
