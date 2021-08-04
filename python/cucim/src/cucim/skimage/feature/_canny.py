"""
canny.py - Canny Edge detector

Reference: Canny, J., A Computational Approach To Edge Detection, IEEE Trans.
    Pattern Analysis and Machine Intelligence, 8:679-714, 1986

Originally part of CellProfiler, code licensed under both GPL and BSD licenses.
Website: http://www.cellprofiler.org
Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.
Original author: Lee Kamentsky
"""
import functools

import cupy as cp
import cupyx.scipy.ndimage as ndi
from cupyx.scipy.ndimage import binary_erosion, generate_binary_structure

from .. import dtype_limits, img_as_float
from .._shared.utils import check_nD
from ..filters import gaussian


# fuse several commonly paired ufunc operations into a single kernel call
@cp.fuse(kernel_name='cucim_skimage_feature_fused_comparison')
def _fused_comparison(w, c1, c2, m):
    return c2 * w + c1 * (1.0 - w) <= m


def smooth_with_function_and_mask(image, function, mask):
    """Smooth an image with a linear function, ignoring masked pixels.

    Parameters
    ----------
    image : array
        Image you want to smooth.
    function : callable
        A function that does image smoothing.
    mask : array
        Mask with 1's for significant pixels, 0's for masked pixels.

    Notes
    ------
    This function calculates the fractional contribution of masked pixels
    by applying the function to the mask (which gets you the fraction of
    the pixel data that's due to significant points). We then mask the image
    and apply the function. The resulting values will be lower by the
    bleed-over fraction, so you can recalibrate by dividing by the function
    on the mask to recover the effect of smoothing from just the significant
    pixels.
    """
    bleed_over = function(mask.astype(cp.float32))
    masked_image = cp.zeros(image.shape, image.dtype)
    masked_image[mask] = image[mask]
    smoothed_image = function(masked_image)
    output_image = smoothed_image / (bleed_over + cp.finfo(cp.float32).eps)
    return output_image


def canny(image, sigma=1., low_threshold=None, high_threshold=None, mask=None,
          use_quantiles=False):
    """Edge filter an image using the Canny algorithm.

    Parameters
    -----------
    image : 2D array
        Grayscale input image to detect edges on; can be of any dtype.
    sigma : float, optional
        Standard deviation of the Gaussian filter.
    low_threshold : float, optional
        Lower bound for hysteresis thresholding (linking edges).
        If None, low_threshold is set to 10% of dtype's max.
    high_threshold : float, optional
        Upper bound for hysteresis thresholding (linking edges).
        If None, high_threshold is set to 20% of dtype's max.
    mask : array, dtype=bool, optional
        Mask to limit the application of Canny to a certain area.
    use_quantiles : bool, optional
        If True then treat low_threshold and high_threshold as quantiles of the
        edge magnitude image, rather than absolute edge magnitude values. If
        True, then the thresholds must be in the range [0, 1].

    Returns
    -------
    output : 2D array (image)
        The binary edge map.

    See also
    --------
    skimage.sobel

    Notes
    -----
    The steps of the algorithm are as follows:

    * Smooth the image using a Gaussian with ``sigma`` width.

    * Apply the horizontal and vertical Sobel operators to get the gradients
      within the image. The edge strength is the norm of the gradient.

    * Thin potential edges to 1-pixel wide curves. First, find the normal
      to the edge at each point. This is done by looking at the
      signs and the relative magnitude of the X-Sobel and Y-Sobel
      to sort the points into 4 categories: horizontal, vertical,
      diagonal and antidiagonal. Then look in the normal and reverse
      directions to see if the values in either of those directions are
      greater than the point in question. Use interpolation to get a mix of
      points instead of picking the one that's the closest to the normal.

    * Perform a hysteresis thresholding: first label all points above the
      high threshold as edges. Then recursively label any point above the
      low threshold that is 8-connected to a labeled point as an edge.

    References
    -----------
    .. [1] Canny, J., A Computational Approach To Edge Detection, IEEE Trans.
           Pattern Analysis and Machine Intelligence, 8:679-714, 1986
           :DOI:`10.1109/TPAMI.1986.4767851`
    .. [2] William Green's Canny tutorial
           https://en.wikipedia.org/wiki/Canny_edge_detector

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage import feature
    >>> # Generate noisy image of a square
    >>> im = cp.zeros((256, 256))
    >>> im[64:-64, 64:-64] = 1
    >>> im += 0.2 * cp.random.rand(*im.shape)
    >>> # First trial with the Canny filter, with the default smoothing
    >>> edges1 = feature.canny(im)
    >>> # Increase the smoothing for better results
    >>> edges2 = feature.canny(im, sigma=3)
    """

    #
    # The steps involved:
    #
    # * Smooth using the Gaussian with sigma above.
    #
    # * Apply the horizontal and vertical Sobel operators to get the gradients
    #   within the image. The edge strength is the sum of the magnitudes
    #   of the gradients in each direction.
    #
    # * Find the normal to the edge at each point using the arctangent of the
    #   ratio of the Y sobel over the X sobel - pragmatically, we can
    #   look at the signs of X and Y and the relative magnitude of X vs Y
    #   to sort the points into 4 categories: horizontal, vertical,
    #   diagonal and antidiagonal.
    #
    # * Look in the normal and reverse directions to see if the values
    #   in either of those directions are greater than the point in question.
    #   Use interpolation to get a mix of points instead of picking the one
    #   that's the closest to the normal.
    #
    # * Label all points above the high threshold as edges.
    # * Recursively label any point above the low threshold that is 8-connected
    #   to a labeled point as an edge.
    #
    # Regarding masks, any point touching a masked point will have a gradient
    # that is "infected" by the masked point, so it's enough to erode the
    # mask by one and then mask the output. We also mask out the border points
    # because who knows what lies beyond the edge of the image?
    #
    check_nD(image, 2)
    dtype_max = dtype_limits(image, clip_negative=False)[1]

    if low_threshold is None:
        low_threshold = 0.1
    elif use_quantiles:
        if not (0.0 <= low_threshold <= 1.0):
            raise ValueError("Quantile thresholds must be between 0 and 1.")
    else:
        low_threshold = low_threshold / dtype_max

    if high_threshold is None:
        high_threshold = 0.2
    elif use_quantiles:
        if not (0.0 <= high_threshold <= 1.0):
            raise ValueError("Quantile thresholds must be between 0 and 1.")
    else:
        high_threshold = high_threshold / dtype_max

    _gaussian = functools.partial(gaussian, sigma=sigma)

    def fsmooth(x, mode='constant'):
        return img_as_float(_gaussian(x, mode=mode))

    if mask is None:
        smoothed = fsmooth(image, mode='reflect')
        # mask that is ones everywhere except the borders
        eroded_mask = cp.ones(image.shape, dtype=bool)
        eroded_mask[:1, :] = 0
        eroded_mask[-1:, :] = 0
        eroded_mask[:, :1] = 0
        eroded_mask[:, -1:] = 0
    else:
        smoothed = smooth_with_function_and_mask(image, fsmooth, mask)
        #
        # Make the eroded mask. Setting the border value to zero will wipe
        # out the image edges for us.
        #
        s = generate_binary_structure(2, 2)
        eroded_mask = binary_erosion(mask, s, border_value=0)

    jsobel = ndi.sobel(smoothed, axis=1)
    isobel = ndi.sobel(smoothed, axis=0)
    abs_isobel = cp.abs(isobel)
    abs_jsobel = cp.abs(jsobel)
    magnitude = cp.hypot(isobel, jsobel)
    eroded_mask = eroded_mask & (magnitude > 0)
    # TODO: implement custom kernel to compute local maxima

    #
    # --------- Find local maxima --------------
    #
    # Assign each point to have a normal of 0-45 degrees, 45-90 degrees,
    # 90-135 degrees and 135-180 degrees.
    #
    local_maxima = cp.zeros(image.shape, bool)

    isobel_gt_0 = isobel >= 0
    jsobel_gt_0 = jsobel >= 0
    isobel_lt_0 = isobel <= 0
    jsobel_lt_0 = jsobel <= 0
    abs_isobel_lt_jsobel = abs_isobel <= abs_jsobel
    abs_isobel_gt_jsobel = abs_isobel >= abs_jsobel

    # ----- 0 to 45 degrees ------
    pts_plus = isobel_gt_0 & jsobel_gt_0
    pts_minus = isobel_lt_0 & jsobel_lt_0
    pts_tmp = (pts_plus | pts_minus) & eroded_mask
    pts = pts_tmp & abs_isobel_gt_jsobel
    # Get the magnitudes shifted left to make a matrix of the points to the
    # right of pts. Similarly, shift left and down to get the points to the
    # top right of pts.

    c1 = magnitude[1:, :][pts[:-1, :]]
    c2 = magnitude[1:, 1:][pts[:-1, :-1]]
    m = magnitude[pts]
    w = abs_jsobel[pts] / abs_isobel[pts]
    c_plus = _fused_comparison(w, c1, c2, m)
    c1 = magnitude[:-1, :][pts[1:, :]]
    c2 = magnitude[:-1, :-1][pts[1:, 1:]]
    c_minus = _fused_comparison(w, c1, c2, m)
    local_maxima[pts] = c_plus & c_minus
    # ----- 45 to 90 degrees ------
    # Mix diagonal and vertical
    #
    pts = pts_tmp & abs_isobel_lt_jsobel
    c1 = magnitude[:, 1:][pts[:, :-1]]
    c2 = magnitude[1:, 1:][pts[:-1, :-1]]
    m = magnitude[pts]
    w = abs_isobel[pts] / abs_jsobel[pts]
    c_plus = _fused_comparison(w, c1, c2, m)
    c1 = magnitude[:, :-1][pts[:, 1:]]
    c2 = magnitude[:-1, :-1][pts[1:, 1:]]
    c_minus = _fused_comparison(w, c1, c2, m)
    local_maxima[pts] = c_plus & c_minus
    # ----- 90 to 135 degrees ------
    # Mix anti-diagonal and vertical
    #
    pts_plus = isobel_lt_0 & jsobel_gt_0
    pts_minus = isobel_gt_0 & jsobel_lt_0
    pts_tmp = (pts_plus | pts_minus) & eroded_mask
    pts = pts_tmp & abs_isobel_lt_jsobel
    c1a = magnitude[:, 1:][pts[:, :-1]]
    c2a = magnitude[:-1, 1:][pts[1:, :-1]]
    m = magnitude[pts]
    w = abs_isobel[pts] / abs_jsobel[pts]
    c_plus = _fused_comparison(w, c1a, c2a, m)
    c1 = magnitude[:, :-1][pts[:, 1:]]
    c2 = magnitude[1:, :-1][pts[:-1, 1:]]
    c_minus = _fused_comparison(w, c1, c2, m)
    local_maxima[pts] = c_plus & c_minus
    # ----- 135 to 180 degrees ------
    # Mix anti-diagonal and anti-horizontal
    #
    pts = pts_tmp & abs_isobel_gt_jsobel
    c1 = magnitude[:-1, :][pts[1:, :]]
    c2 = magnitude[:-1, 1:][pts[1:, :-1]]
    m = magnitude[pts]
    w = abs_jsobel[pts] / abs_isobel[pts]
    c_plus = _fused_comparison(w, c1, c2, m)
    c1 = magnitude[1:, :][pts[:-1, :]]
    c2 = magnitude[1:, :-1][pts[:-1, 1:]]
    c_minus = _fused_comparison(w, c1, c2, m)
    local_maxima[pts] = c_plus & c_minus

    #
    # ---- If use_quantiles is set then calculate the thresholds to use
    #
    if use_quantiles:
        high_threshold = cp.percentile(magnitude, 100.0 * high_threshold)
        low_threshold = cp.percentile(magnitude, 100.0 * low_threshold)

    #
    # ---- Create two masks at the two thresholds.
    #
    high_mask = local_maxima & (magnitude >= high_threshold)
    low_mask = local_maxima & (magnitude >= low_threshold)

    #
    # Segment the low-mask, then only keep low-segments that have
    # some high_mask component in them
    #
    labels, count = ndi.label(low_mask, structure=cp.ones((3, 3), bool))
    if count == 0:
        return low_mask

    nonzero_sums = cp.unique(labels[high_mask])
    good_label = cp.zeros((count + 1,), bool)
    good_label[nonzero_sums] = True
    output_mask = good_label[labels]
    return output_mask
