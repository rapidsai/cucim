"""
Adapted code from "Contrast Limited Adaptive Histogram Equalization" by Karel
Zuiderveld <karel@cv.ruu.nl>, Graphics Gems IV, Academic Press, 1994.

http://tog.acm.org/resources/GraphicsGems/

Relicensed with permission of the author under the Modified BSD license.
"""
import functools
import itertools
import math
import numbers
import operator

import cupy as cp
import numpy as np

# TODO: replace _misc.prod with math.prod once minimum Python >= 3.88
from cucim import _misc
from cucim.skimage.exposure.exposure import rescale_intensity

from .._shared.utils import _supported_float_type
from .._vendored import pad
from ..color.adapt_rgb import adapt_rgb, hsv_value
from ..util import img_as_uint

NR_OF_GRAY = 2 ** 14  # number of grayscale levels to use in CLAHE algorithm


@adapt_rgb(hsv_value)
def equalize_adapthist(image, kernel_size=None,
                       clip_limit=0.01, nbins=256):
    """Contrast Limited Adaptive Histogram Equalization (CLAHE).

    An algorithm for local contrast enhancement, that uses histograms computed
    over different tile regions of the image. Local details can therefore be
    enhanced even in regions that are darker or lighter than most of the image.

    Parameters
    ----------
    image : (N1, ...,NN[, C]) ndarray
        Input image.
    kernel_size : int or array_like, optional
        Defines the shape of contextual regions used in the algorithm. If
        iterable is passed, it must have the same number of elements as
        ``image.ndim`` (without color channel). If integer, it is broadcasted
        to each `image` dimension. By default, ``kernel_size`` is 1/8 of
        ``image`` height by 1/8 of its width.
    clip_limit : float, optional
        Clipping limit, normalized between 0 and 1 (higher values give more
        contrast).
    nbins : int, optional
        Number of gray bins for histogram ("data range").

    Returns
    -------
    out : (N1, ...,NN[, C]) ndarray
        Equalized image with float64 dtype.

    See Also
    --------
    equalize_hist, rescale_intensity

    Notes
    -----
    * For color images, the following steps are performed:
       - The image is converted to HSV color space
       - The CLAHE algorithm is run on the V (Value) channel
       - The image is converted back to RGB space and returned
    * For RGBA images, the original alpha channel is removed.

    .. versionchanged:: 0.17
        The values returned by this function are slightly shifted upwards
        because of an internal change in rounding behavior.

    References
    ----------
    .. [1] http://tog.acm.org/resources/GraphicsGems/
    .. [2] https://en.wikipedia.org/wiki/CLAHE#CLAHE
    """

    float_dtype = _supported_float_type(image.dtype)
    image = img_as_uint(image)
    image = cp.around(
        rescale_intensity(image, out_range=(0, NR_OF_GRAY - 1))
    ).astype(cp.min_scalar_type(NR_OF_GRAY))

    if kernel_size is None:
        kernel_size = tuple([max(s // 8, 1) for s in image.shape])
    elif isinstance(kernel_size, numbers.Number):
        kernel_size = (kernel_size,) * image.ndim
    elif len(kernel_size) != image.ndim:
        raise ValueError(f'Incorrect value of `kernel_size`: {kernel_size}')

    kernel_size = [int(k) for k in kernel_size]

    image = _clahe(image, kernel_size, clip_limit, nbins)
    image = image.astype(float_dtype, copy=False)
    return rescale_intensity(image)


def _clahe(image, kernel_size, clip_limit, nbins):
    """Contrast Limited Adaptive Histogram Equalization.

    Parameters
    ----------
    image : (N1,...,NN) ndarray
        Input image.
    kernel_size : int or N-tuple of int
        Defines the shape of contextual regions used in the algorithm.
    clip_limit : float
        Normalized clipping limit between 0 and 1 (higher values give more
        contrast).
    nbins : int
        Number of gray bins for histogram ("data range").

    Returns
    -------
    out : (N1,...,NN) ndarray
        Equalized image.

    The number of "effective" graylevels in the output image is set by `nbins`;
    selecting a small value (e.g. 128) speeds up processing and still produces
    an output image of good quality. A clip limit of 0 or larger than or equal
    to 1 results in standard (non-contrast limited) AHE.
    """
    ndim = image.ndim
    dtype = image.dtype

    # pad the image such that the shape in each dimension
    # - is a multiple of the kernel_size and
    # - is preceded by half a kernel size
    pad_start_per_dim = [k // 2 for k in kernel_size]

    pad_end_per_dim = [(k - s % k) % k + math.ceil(k / 2.)
                       for k, s in zip(kernel_size, image.shape)]

    image = pad(
        image,
        [[p_i, p_f] for p_i, p_f in zip(pad_start_per_dim, pad_end_per_dim)],
        mode='reflect',
    )

    # determine gray value bins
    bin_size = 1 + NR_OF_GRAY // nbins
    lut = cp.arange(NR_OF_GRAY, dtype=cp.min_scalar_type(NR_OF_GRAY))
    lut //= bin_size

    image = lut[image]

    # calculate graylevel mappings for each contextual region
    # rearrange image into flattened contextual regions
    ns_hist = [int(s / k) - 1 for s, k in zip(image.shape, kernel_size)]
    hist_blocks_shape = functools.reduce(
        operator.add, [(s, k) for s, k in zip(ns_hist, kernel_size)]
    )
    hist_blocks_axis_order = (tuple(range(0, ndim * 2, 2)) +
                              tuple(range(1, ndim * 2, 2)))
    hist_slices = [
        slice(k // 2, k // 2 + n * k) for k, n in zip(kernel_size, ns_hist)
    ]
    hist_blocks = image[tuple(hist_slices)].reshape(hist_blocks_shape)
    hist_blocks = hist_blocks.transpose(hist_blocks_axis_order)
    hist_block_assembled_shape = hist_blocks.shape
    hist_blocks = hist_blocks.reshape((_misc.prod(ns_hist), -1))

    # Calculate actual clip limit
    kernel_elements = _misc.prod(kernel_size)
    if clip_limit > 0.0:
        clim = int(max(clip_limit * kernel_elements, 1))
    else:
        # largest possible value, i.e., do not clip (AHE)
        clim = kernel_elements

    # Note: for 4096, 4096 input and default args, shapes are:
    #    hist_blocks.shape = (64, 262144)
    #    hist.shape = (64, 256)
    hist = cp.apply_along_axis(cp.bincount, -1, hist_blocks, minlength=nbins)
    if isinstance(hist_blocks, cp.ndarray):
        # CuPy Backend:
        #    faster to loop over the arrays on the host
        #    (hist is small and clip_histogram has too much overhead)
        # TODO: implement clip_histogram kernel to avoid synchronization?
        hist = cp.asarray(np.apply_along_axis(  # synchronize!
            clip_histogram, -1, cp.asnumpy(hist), clip_limit=clim
        ))
    else:
        hist = cp.apply_along_axis(clip_histogram, -1, hist, clip_limit=clim)
    hist = map_histogram(hist, 0, NR_OF_GRAY - 1, kernel_elements)
    hist = hist.reshape(hist_block_assembled_shape[:ndim] + (-1,))

    # duplicate leading mappings in each dim
    map_array = pad(
        hist, [(1, 1) for _ in range(ndim)] + [(0, 0)], mode='edge'
    )

    # Perform multilinear interpolation of graylevel mappings
    # using the convention described here:
    # https://en.wikipedia.org/w/index.php?title=Adaptive_histogram_
    # equalization&oldid=936814673#Efficient_computation_by_interpolation

    # rearrange image into blocks for vectorized processing
    ns_proc = [int(s / k) for s, k in zip(image.shape, kernel_size)]
    blocks_shape = functools.reduce(
        operator.add, [(s, k) for s, k in zip(ns_proc, kernel_size)]
    )
    blocks_axis_order = hist_blocks_axis_order

    blocks = image.reshape(blocks_shape)
    blocks = blocks.transpose(blocks_axis_order)
    blocks_flattened_shape = blocks.shape
    blocks = blocks.reshape((_misc.prod(ns_proc),
                             _misc.prod(blocks.shape[ndim:])))

    # calculate interpolation coefficients
    coeffs = cp.meshgrid(*tuple([cp.arange(k) / k
                                 for k in kernel_size[::-1]]), indexing='ij')
    coeffs = [cp.transpose(c).flatten() for c in coeffs]
    inv_coeffs = [1 - c for c in coeffs]

    # sum over contributions of neighboring contextual
    # regions in each direction
    result = cp.zeros(blocks.shape, dtype=cp.float32)
    for iedge, edge in enumerate(itertools.product(*((range(2),) * ndim))):

        edge_maps = map_array[tuple(slice(e, e + n)
                                    for e, n in zip(edge, ns_proc))]
        edge_maps = edge_maps.reshape((_misc.prod(ns_proc), -1))

        # apply map
        edge_mapped = cp.take_along_axis(edge_maps, blocks, axis=-1)

        # interpolate
        edge_coeffs = functools.reduce(
            operator.mul,
            [[inv_coeffs, coeffs][e][d] for d, e in enumerate(edge[::-1])],
        )

        result += (edge_mapped * edge_coeffs).astype(result.dtype)

    result = result.astype(dtype)

    # rebuild result image from blocks
    result = result.reshape(blocks_flattened_shape)
    blocks_axis_rebuild_order = functools.reduce(
        operator.add,
        [(s, k) for s, k in zip(range(0, ndim), range(ndim, ndim * 2))],
    )
    result = result.transpose(blocks_axis_rebuild_order)
    result = result.reshape(image.shape)

    # undo padding
    unpad_slices = tuple([slice(p_i, s - p_f) for p_i, p_f, s in
                          zip(pad_start_per_dim, pad_end_per_dim,
                              image.shape)])
    result = result[unpad_slices]

    return result


# TODO: refactor this clip_histogram bottleneck.
def clip_histogram(hist, clip_limit):
    """Perform clipping of the histogram and redistribution of bins.

    The histogram is clipped and the number of excess pixels is counted.
    Afterwards the excess pixels are equally redistributed across the
    whole histogram (providing the bin count is smaller than the cliplimit).

    Parameters
    ----------
    hist : ndarray
        Histogram array.
    clip_limit : int
        Maximum allowed bin count.

    Returns
    -------
    hist : ndarray
        Clipped histogram.
    """
    # calculate total number of excess pixels
    excess_mask = hist > clip_limit
    excess = hist[excess_mask]
    n_excess = excess.sum() - excess.size * clip_limit
    hist[excess_mask] = clip_limit

    # Second part: clip histogram and redistribute excess pixels in each bin
    bin_incr = n_excess // hist.size  # average binincrement
    xp = cp.get_array_module(hist)
    upper = clip_limit - bin_incr  # Bins larger than upper set to cliplimit

    low_mask = hist < upper
    n_excess -= hist[low_mask].size * bin_incr
    hist[low_mask] += bin_incr

    mid_mask = xp.logical_and(hist >= upper, hist < clip_limit)
    mid = hist[mid_mask]
    n_excess += mid.sum() - mid.size * clip_limit
    hist[mid_mask] = clip_limit

    while n_excess > 0:  # Redistribute remaining excess
        prev_n_excess = n_excess
        for index in range(hist.size):
            under_mask = hist < clip_limit
            step_size = max(1, xp.count_nonzero(under_mask) // n_excess)
            under_mask = under_mask[index::step_size]
            hist[index::step_size][under_mask] += 1
            n_excess -= xp.count_nonzero(under_mask)
            if n_excess <= 0:
                break
        if prev_n_excess == n_excess:
            break

    return hist


def map_histogram(hist, min_val, max_val, n_pixels):
    """Calculate the equalized lookup table (mapping).

    It does so by cumulating the input histogram.
    Histogram bins are assumed to be represented by the last array dimension.

    Parameters
    ----------
    hist : ndarray
        Clipped histogram.
    min_val : int
        Minimum value for mapping.
    max_val : int
        Maximum value for mapping.
    n_pixels : int
        Number of pixels in the region.

    Returns
    -------
    out : ndarray
       Mapped intensity LUT.
    """
    xp = cp.get_array_module(hist)
    out = xp.cumsum(hist, axis=-1).astype(float)
    out *= (max_val - min_val) / n_pixels
    out += min_val
    cp.clip(out, a_min=None, a_max=max_val, out=out)

    return out.astype(int)
