# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import math
from warnings import warn

import cupy as cp
import numpy as np

import cucim.skimage._vendored.ndimage as ndi

from ._median_hist import KernelResourceError, _can_use_histogram, _median_hist


def median(
    image,
    footprint=None,
    out=None,
    mode="nearest",
    cval=0.0,
    behavior="ndimage",
    *,
    algorithm="auto",
    algorithm_kwargs={},
):
    """Return local median of an image.

    Parameters
    ----------
    image : array-like
        Input image.
    footprint : ndarray, tuple of int, or None
        If ``None``, ``footprint`` will be a N-D array with 3 elements for each
        dimension (e.g., vector, square, cube, etc.). If `footprint` is a
        tuple of integers, it will be an array of ones with the given shape.
        Otherwise, ``footprint`` is an N-D array of 1's and 0's with the same
        number of dimensions as ``image``.
        Note that upstream scikit-image currently does not support supplying
        a tuple for `footprint`. It is added here to avoid overhead of
        generating a small weights array in cases where it is not needed.
    out : ndarray, (same dtype as image), optional
        If None, a new array is allocated.
    mode : {'reflect', 'constant', 'nearest', 'mirror','‘wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        ``cval`` is the value when mode is equal to 'constant'. ``mode`` is
        only used when ``behavior='ndimage'``.
        Default is 'nearest'.
    cval : scalar, optional
        Value to fill past edges of input if mode is 'constant'. ``cval`` is
        only used when ``behavior='ndimage'``. Default is 0.0.
    behavior : {'ndimage', 'rank'}, optional
        Behavior 'ndimage' behaves like `cupyx.scipy.ndimage.median_filter`,
        while 'rank' uses :func:`cucim.skimage.filters.rank.median`. cuCIM rank
        filters use reflected boundary extension with a constant footprint
        size and support N-D images. This differs from scikit-image rank
        filters, which crop neighborhoods near image boundaries and support
        only 2-D and 3-D images. Default is 'ndimage'.

    Other Parameters
    ----------------
    algorithm : {'auto', 'histogram', 'sorting'}
        Determines which algorithm is used to compute the median. The default
        of 'auto' will attempt to use a histogram-based algorithm for 2D
        images with 8 or 16-bit integer data types. Otherwise a sorting-based
        algorithm will be used. Note: this parameter is cuCIM-specific and does
        not exist in upstream scikit-image.
    algorithm_kwargs : dict
        Any additional algorithm-specific keywords. Currently can only be used
        to set the number of parallel partitions for the 'histogram' algorithm.
        (e.g. ``algorithm_kwargs={'partitions': 256}``). Note: this parameter is
        cuCIM-specific and does not exist in upstream scikit-image.

    Returns
    -------
    out : N-D array
        Output image with the same shape as the input image.

    See also
    --------
    skimage.filters.rank.median : Rank-based implementation of the median
        filtering offering more flexibility with additional parameters but
        dedicated for unsigned integer images.

    Notes
    -----
    An efficient, histogram-based median filter as described in [1]_ is faster
    than the sorting based approach for larger kernel sizes (e.g. greater than
    13x13 or so in 2D). It has near-constant run time regardless of the kernel
    size. The algorithm presented in [1]_ has been adapted to additional bit
    depths here. When algorithm='auto', the histogram-based algorithm will be
    chosen for integer-valued images with sufficiently large footprint size.
    Otherwise, the sorting-based approach is used.

    References
    ----------
    .. [1] O. Green, "Efficient Scalable Median Filtering Using Histogram-Based
       Operations," in IEEE Transactions on Image Processing, vol. 27, no. 5,
       pp. 2217-2228, May 2018, https://doi.org/10.1109/TIP.2017.2781375.

    Examples
    --------
    >>> import cupy as cp
    >>> from skimage import data
    >>> from cucim.skimage.morphology import disk
    >>> from cucim.skimage.filters import median
    >>> img = cp.array(data.camera())
    >>> med = median(img, disk(5))

    """
    if behavior == "rank":
        if mode != "nearest" or not np.isclose(cval, 0.0):
            warn(
                "Change 'behavior' to 'ndimage' if you want to use the "
                "parameters 'mode' or 'cval'. They will be discarded "
                "otherwise.",
                stacklevel=2,
            )
        from .rank import median as rank_median

        if isinstance(footprint, tuple):
            if len(footprint) != image.ndim:
                raise ValueError(
                    "tuple footprint must have ndim matching image"
                )
            footprint = cp.ones(footprint, dtype=bool)
        return rank_median(image, footprint=footprint, out=out)

    if footprint is None:
        footprint_shape = (3,) * image.ndim
    elif isinstance(footprint, tuple):
        if len(footprint) != image.ndim:
            raise ValueError("tuple footprint must have ndim matching image")
        footprint_shape = footprint
        footprint = None
    else:
        footprint_shape = footprint.shape

    if algorithm == "sorting":
        can_use_histogram = False
    elif algorithm in ["auto", "histogram"]:
        can_use_histogram, reason = _can_use_histogram(
            image, footprint, footprint_shape
        )
    else:
        raise ValueError(f"unknown algorithm: {algorithm}")

    if algorithm == "histogram" and not can_use_histogram:
        raise ValueError(
            "The histogram-based algorithm was requested, but it cannot "
            f"be used for this image and footprint (reason: {reason})."
        )

    # The sorting-based implementation in CuPy is faster for small footprints.
    # Empirically, shapes above (13, 13) and above on RTX A6000 have faster
    # execution for the histogram-based approach.
    use_histogram = can_use_histogram
    if algorithm == "auto":
        # prefer sorting-based algorithm if footprint shape is small
        use_histogram = use_histogram and math.prod(footprint_shape) > 150

    if use_histogram:
        try:
            # as in SciPy, a user-provided `out` can be an array or a dtype
            output_array_provided = False
            out_dtype = None
            if out is not None:
                output_array_provided = isinstance(out, cp.ndarray)
                if not output_array_provided:
                    try:
                        out_dtype = cp.dtype(out)
                    except TypeError:
                        raise TypeError(
                            "out must be either a cupy.array or a valid input "
                            "to cupy.dtype"
                        )

            # TODO: Can't currently pass an output array into _median_hist as a
            #       new array currently needs to be created during padding.

            # pass shape if explicit footprint isn't needed
            # (use new variable name in case KernelResourceError occurs)
            temp = _median_hist(
                image,
                footprint_shape if footprint is None else footprint,
                mode=mode,
                cval=cval,
                **algorithm_kwargs,
            )
            if output_array_provided:
                out[:] = temp
            else:
                if out_dtype is not None:
                    temp = temp.astype(out_dtype, copy=False)
                out = temp
            return out
        except KernelResourceError as e:
            # Fall back to sorting-based implementation if we encounter a
            # resource limit (e.g. insufficient shared memory per block).
            warn(
                "Kernel resource error encountered in histogram-based "
                f"median kernel: {e}\n"
                "Falling back to sorting-based median instead."
            )

    if algorithm_kwargs:
        warn(
            f"algorithm_kwargs={algorithm_kwargs} ignored for sorting-based "
            f"algorithm"
        )

    size = footprint_shape if footprint is None else None
    return ndi.median_filter(
        image, size=size, footprint=footprint, output=out, mode=mode, cval=cval
    )
