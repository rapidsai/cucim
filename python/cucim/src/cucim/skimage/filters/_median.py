from warnings import warn

import cupy as cp
import numpy as np

import cucim.skimage._vendored.ndimage as ndi
from .._shared.utils import deprecate_kwarg
from ._median_hist import _can_use_histogram, _median_hist, KernelResourceError

try:
    from math import prod
except ImportError:
    from functools import reduce
    from operator import mul

    def prod(x):
        return reduce(mul, x)


@deprecate_kwarg(kwarg_mapping={'selem': 'footprint'},
                 removed_version="23.02.00",
                 deprecated_version="22.02.00")
def median(image, footprint=None, out=None, mode='nearest', cval=0.0,
           behavior='ndimage', *, algorithm='auto', algorithm_kwargs={}):
    """Return local median of an image.

    Parameters
    ----------
    image : array-like
        Input image.
    footprint : ndarray, optional
        If ``behavior=='rank'``, ``footprint`` is a 2-D array of 1's and 0's.
        If ``behavior=='ndimage'``, ``footprint`` is a N-D array of 1's and 0's
        with the same number of dimension than ``image``.
        If None, ``footprint`` will be a N-D array with 3 elements for each
        dimension (e.g., vector, square, cube, etc.)
    out : ndarray, (same dtype as image), optional
        If None, a new array is allocated.
    mode : {'reflect', 'constant', 'nearest', 'mirror','‘wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        ``cval`` is the value when mode is equal to 'constant'.
        Default is 'nearest'.

        .. versionadded:: 0.15
           ``mode`` is used when ``behavior='ndimage'``.
    cval : scalar, optional
        Value to fill past edges of input if mode is 'constant'. Default is 0.0

        .. versionadded:: 0.15
           ``cval`` was added in 0.15 is used when ``behavior='ndimage'``.
    behavior : {'ndimage', 'rank'}, optional
        Either to use the old behavior (i.e., < 0.15) or the new behavior.
        The old behavior will call the :func:`skimage.filters.rank.median`.
        The new behavior will call the :func:`scipy.ndimage.median_filter`.
        Default is 'ndimage'.

        .. versionadded:: 0.15
           ``behavior`` is introduced in 0.15
        .. versionchanged:: 0.16
           Default ``behavior`` has been changed from 'rank' to 'ndimage'

    Other Parameters
    ----------------
    algorithm : {'auto', 'histogram', 'sorting'}
        Determines which algorithm is used to compute the median. The default
        of 'auto' will attempt to use a histogram-based algorithm for 2D
        images with 8 or 16-bit integer data types. Otherwise a sorting-based
        algorithm will be used. Note: this paramter is cuCIM-specific and does
        not exist in upstream scikit-image.
    algorithm_kwargs : dict
        Any additional algorithm-specific keywords. Currently can only be used
        to set the number of parallel partitions for the 'histogram' algorithm.
        (e.g. ``algorithm_kwargs={'partitions': 256}``). Note: this paramter is
        cuCIM-specific and does not exist in upstream scikit-image.

    Returns
    -------
    out : 2-D array (same dtype as input image)
        Output image.

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
    if behavior == 'rank':
        if mode != 'nearest' or not np.isclose(cval, 0.0):
            warn("Change 'behavior' to 'ndimage' if you want to use the "
                 "parameters 'mode' or 'cval'. They will be discarded "
                 "otherwise.")
        raise NotImplementedError("rank behavior not currently implemented")
        # TODO: implement median rank filter
        # return generic.median(image, selem=selem, out=out)

    if footprint is None:
        footprint = ndi.generate_binary_structure(image.ndim, image.ndim)

    if algorithm == 'sorting':
        can_use_histogram = False
    elif algorithm in ['auto', 'histogram']:
        can_use_histogram, reason = _can_use_histogram(image, footprint)
    else:
        raise ValueError(f"unknown algorithm: {algorithm}")

    if algorithm == 'histogram' and not can_use_histogram:
        raise ValueError(
            "The histogram-based algorithm was requested, but it cannot "
            f"be used for this image and footprint (reason: {reason})."
        )

    # The sorting-based implementation in CuPy is faster for small footprints.
    # Empirically, shapes above (13, 13) and above on RTX A6000 have faster
    # execution for the histogram-based approach.
    use_histogram = can_use_histogram
    if algorithm == 'auto':
        # prefer sorting-based algorithm if footprint shape is small
        use_histogram = use_histogram and prod(footprint.shape) > 150

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
            temp = _median_hist(image, footprint, mode=mode, cval=cval,
                                **algorithm_kwargs)
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
            warn("Kernel resource error encountered in histogram-based "
                 f"median kerne: {e}\n"
                 "Falling back to sorting-based median instead.")

    if algorithm_kwargs:
        warn(f"algorithm_kwargs={algorithm_kwargs} ignored for sorting-based "
             f"algorithm")

    return ndi.median_filter(image, footprint=footprint, output=out, mode=mode,
                             cval=cval)
