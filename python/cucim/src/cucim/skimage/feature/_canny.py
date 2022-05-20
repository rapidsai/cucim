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
import cupy as cp
import cupyx.scipy.ndimage as ndi

from cucim.skimage.util import dtype_limits

from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type, check_nD


def _preprocess(image, mask, sigma, mode, cval):
    """Generate a smoothed image and an eroded mask.

    The image is smoothed using a gaussian filter ignoring masked
    pixels and the mask is eroded.

    Parameters
    ----------
    image : array
        Image to be smoothed.
    mask : array
        Mask with 1's for significant pixels, 0's for masked pixels.
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel. The standard
        deviations of the Gaussian filter are given for each axis as a
        sequence, or as a single number, in which case it is equal for
        all axes.
    mode : str, {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}
        The ``mode`` parameter determines how the array borders are
        handled, where ``cval`` is the value when mode is equal to
        'constant'.
    cval : float, optional
        Value to fill past edges of input if `mode` is 'constant'.

    Returns
    -------
    smoothed_image : ndarray
        The smoothed array
    eroded_mask : ndarray
        The eroded mask.

    Notes
    -----
    This function calculates the fractional contribution of masked pixels
    by applying the function to the mask (which gets you the fraction of
    the pixel data that's due to significant points). We then mask the image
    and apply the function. The resulting values will be lower by the
    bleed-over fraction, so you can recalibrate by dividing by the function
    on the mask to recover the effect of smoothing from just the significant
    pixels.
    """

    gaussian_kwargs = dict(sigma=sigma, mode=mode, cval=cval,
                           preserve_range=False)
    compute_bleedover = (mode == 'constant' or mask is not None)
    float_type = _supported_float_type(image.dtype)
    if mask is None:
        if compute_bleedover:
            mask = cp.ones(image.shape, dtype=float_type)
        masked_image = image

        # mask that is ones everywhere except the borders
        eroded_mask = cp.ones(image.shape, dtype=bool)
        eroded_mask[:1, :] = 0
        eroded_mask[-1:, :] = 0
        eroded_mask[:, :1] = 0
        eroded_mask[:, -1:] = 0
    else:
        mask = mask.astype(bool, copy=False)
        masked_image = cp.zeros_like(image)
        masked_image[mask] = image[mask]

        # Make the eroded mask. Setting the border value to zero will wipe
        # out the image edges for us.
        s = ndi.generate_binary_structure(2, 2)
        eroded_mask = ndi.binary_erosion(mask, s, border_value=0)

    if compute_bleedover:
        # Compute the fractional contribution of masked pixels by applying
        # the function to the mask (which gets you the fraction of the
        # pixel data that's due to significant points)
        bleed_over = gaussian(mask.astype(cp.float32), **gaussian_kwargs)
        bleed_over += cp.finfo(cp.float32).eps

    # Smooth the masked image
    smoothed_image = gaussian(masked_image, **gaussian_kwargs)

    if compute_bleedover:
        # Lower the result by the bleed-over fraction, so you can
        # recalibrate by dividing by the function on the mask to recover
        # the effect of smoothing from just the significant pixels.
        smoothed_image /= bleed_over

    return smoothed_image, eroded_mask


def _generate_nonmaximum_suppression_op(large_int=False):
    """CUDA inner loop code for non-maximum suppression

    Parameters
    ----------
    large_int : bool, optional
        If True, `size_t` is used rather than `unsigned int` for
        strides/indexing.

    Returns
    -------
    ops: str
        inner loop operation for use with cupy.ElementwiseKernel
    """

    if large_int:
        uint_t = 'size_t'
    else:
        uint_t = 'unsigned int'

    ops = f"""
    // determine strides (in number of elements) along each axis
    const {uint_t} stride_x = magnitude.shape()[1];
    const {uint_t} stride_y = 1;
    """

    ops += """
    T w, m, neigh1_1, neigh1_2, neigh2_1, neigh2_2;

    out = static_cast<T>(0.);
    m = magnitude[i];
    if (!((eroded_mask[i] > 0) && (m >= low_threshold))) {
        // return out
        break;
    }

    T isob = isobel[i];
    T jsob = jsobel[i];

    bool is_down = (isob <= 0);
    bool is_up = !is_down;

    bool is_left = (jsob <= 0);
    bool is_right = !is_left;

    bool cond1 = (is_up && is_right) || (is_down && is_left);
    bool cond2 = (is_down && is_right) || (is_up && is_left);
    if (cond1 || cond2) {
        isob = fabs(isob);
        jsob = fabs(jsob);
        if (cond1) {
            if (isob > jsob) {
                w = jsob / isob;
                neigh1_1 = magnitude[i + stride_x];             // x + 1, y
                neigh1_2 = magnitude[i + stride_x + stride_y];  // x + 1, y + 1
                neigh2_1 = magnitude[i - stride_x];             // x - 1, y
                neigh2_2 = magnitude[i - stride_x - stride_y];  // x - 1, y - 1
            } else {
                w = isob / jsob;
                neigh1_1 = magnitude[i + stride_y];             // x    , y + 1
                neigh1_2 = magnitude[i + stride_x + stride_y];  // x + 1, y + 1
                neigh2_1 = magnitude[i - stride_y];             // x    , y - 1
                neigh2_2 = magnitude[i - stride_x - stride_y];  // x - 1, y - 1
            }
        } else if (cond2) {
            if (isob < jsob) {
                w = isob / jsob;
                neigh1_1 = magnitude[i + stride_y];             // x    , y + 1
                neigh1_2 = magnitude[i - stride_x + stride_y];  // x - 1, y + 1
                neigh2_1 = magnitude[i - stride_y];             // x    , y - 1
                neigh2_2 = magnitude[i + stride_x - stride_y];  // x + 1, y - 1
            } else {
                w = jsob / isob;
                neigh1_1 = magnitude[i - stride_x];             // x - 1, y
                neigh1_2 = magnitude[i - stride_x + stride_y];  // x - 1, y + 1
                neigh2_1 = magnitude[i + stride_x];             // x + 1, y
                neigh2_2 = magnitude[i + stride_x - stride_y];  // x + 1, y - 1
            }
        }
        // linear interpolation
        bool c_plus = (neigh1_2 * w + neigh1_1 * (1.0 - w)) <= m;
        if (c_plus) {
            bool c_minus = (neigh2_2 * w + neigh2_1 * (1.0 - w)) <= m;
            if (c_minus) {
                out = m;
            }
        }
    }
    """
    return ops


@cp.memoize(for_each_device=True)
def _get_nonmax_kernel(large_int=False):
    in_params = ('raw T isobel, raw T jsobel, raw T magnitude, '
                 'raw uint8 eroded_mask, T low_threshold')
    out_params = 'T out'
    name = 'cupyx_skimage_canny_nonmaximum_suppression'
    if large_int:
        name += '_large'
    return cp.ElementwiseKernel(
        in_params,
        out_params,
        operation=_generate_nonmaximum_suppression_op(large_int),
        name=name,
    )


def _nonmaximum_suppression(
    isobel, jsobel, magnitude, eroded_mask, low_threshold
):

    # make sure inputs are C-contiguous (stride calculations assume this)
    isobel = cp.ascontiguousarray(isobel)
    jsobel = cp.ascontiguousarray(jsobel)
    magnitude = cp.ascontiguousarray(magnitude)
    eroded_mask = cp.ascontiguousarray(eroded_mask)
    if eroded_mask.dtype == cp.bool_:
        # uint8 view of boolean array
        eroded_mask = eroded_mask.view(cp.uint8)
    elif eroded_mask.dtype != cp.uint8:
        raise ValueError("eroded_mask must be boolean (or uint8)")

    # generate the Elementwise kernel (with size dependent index type)
    large_int = magnitude.size > 1 << 31
    kernel = _get_nonmax_kernel(large_int=large_int)

    out = cp.empty_like(magnitude)
    kernel(isobel, jsobel, magnitude, eroded_mask, low_threshold, out)
    return out


def canny(image, sigma=1., low_threshold=None, high_threshold=None, mask=None,
          use_quantiles=False, *, mode='constant', cval=0.0):
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
        If ``True`` then treat low_threshold and high_threshold as
        quantiles of the edge magnitude image, rather than absolute
        edge magnitude values. If ``True`` then the thresholds must be
        in the range [0, 1].
    mode : str, {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}
        The ``mode`` parameter determines how the array borders are
        handled during Gaussian filtering, where ``cval`` is the value when
        mode is equal to 'constant'.
    cval : float, optional
        Value to fill past edges of input if `mode` is 'constant'.

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
    ----------
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

    check_nD(image, 2)
    dtype_max = dtype_limits(image, clip_negative=False)[1]

    if low_threshold is None:
        low_threshold = 0.1
    elif use_quantiles:
        if not (0.0 <= low_threshold <= 1.0):
            raise ValueError("Quantile thresholds must be between 0 and 1.")
    else:
        low_threshold /= dtype_max

    if high_threshold is None:
        high_threshold = 0.2
    elif use_quantiles:
        if not (0.0 <= high_threshold <= 1.0):
            raise ValueError("Quantile thresholds must be between 0 and 1.")
    else:
        high_threshold /= dtype_max

    if high_threshold < low_threshold:
        raise ValueError("low_threshold should be lower then high_threshold")

    # Image filtering
    smoothed, eroded_mask = _preprocess(image, mask, sigma, mode, cval)

    # Gradient magnitude estimation
    jsobel = ndi.sobel(smoothed, axis=1)
    isobel = ndi.sobel(smoothed, axis=0)
    magnitude = cp.hypot(isobel, jsobel)

    if use_quantiles:
        low_threshold, high_threshold = cp.percentile(
            magnitude, [100.0 * low_threshold, 100.0 * high_threshold]
        )

    # Non-maximum suppression
    low_masked = _nonmaximum_suppression(
        isobel, jsobel, magnitude, eroded_mask, low_threshold
    )

    # Double thresholding and edge tracking
    #
    # Segment the low-mask, then only keep low-segments that have
    # some high_mask component in them
    #
    low_mask = low_masked > 0
    strel = cp.ones((3, 3), bool)
    labels, count = ndi.label(low_mask, strel)
    if count == 0:
        return low_mask

    high_mask = low_mask & (low_masked >= high_threshold)
    nonzero_sums = cp.unique(labels[high_mask])
    good_label = cp.zeros((count + 1,), bool)
    good_label[nonzero_sums] = True
    output_mask = good_label[labels]
    return output_mask
