# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

"""Inferior and superior ranks, provided by the user, are passed to the kernel
function to provide a softer version of the rank filters. E.g.
``autolevel_percentile`` will stretch image levels between percentile [p0, p1]
instead of using [min, max]. It means that isolated bright or dark pixels will
not produce halos.

This GPU implementation uses CuPy and CUDA kernels for accelerated processing.
The kernels do not currently take advantage of the sliding window approach
used by scikit-image (described in [1]_).

Input images can be any numeric dtype and N-dimensional (not restricted to
8-bit or 16-bit, 2D like the CPU implementation).

Result image has the same dtype as the input image.

References
----------

.. [1] Huang, T. ,Yang, G. ;  Tang, G.. "A fast two-dimensional
       median filtering algorithm", IEEE Transactions on Acoustics, Speech and
       Signal Processing, Feb 1979. Volume: 27 , Issue: 1, Page(s): 13 - 18.

"""

import cupy as cp

from cucim.skimage._vendored._ndimage_filters import _percentile_range_filter

__all__ = [
    "autolevel_percentile",
    "gradient_percentile",
    "mean_percentile",
    # "sum_percentile",
    "subtract_mean_percentile",
    "enhance_contrast_percentile",
    "percentile",
    "pop_percentile",
    "threshold_percentile",
]


def _preprocess_input(
    image,
    footprint=None,
    out=None,
    mask=None,
    out_dtype=None,
    shifts=None,
):
    """Preprocess and verify input for filters.rank methods (GPU version).

    Parameters
    ----------
    image : cupy.ndarray
        Input image (N-dimensional, any numeric dtype).
    footprint : cupy.ndarray, optional
        The neighborhood expressed as an array of 1's and 0's.
    out : cupy.ndarray, optional
        If None, a new array is allocated.
    mask : cupy.ndarray, optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    out_dtype : data-type, optional
        Desired output data-type. Default is None, which preserves input dtype.
    shifts : sequence of int, optional
        Offset added to the footprint center point along each axis. The length
        must match image.ndim. Each shift is bounded to the footprint size
        (center must be inside the given footprint).

    Returns
    -------
    image : cupy.ndarray
        Input image as CuPy array.
    footprint : cupy.ndarray
        The neighborhood expressed as a boolean array.
    out : cupy.ndarray
        Output array with same shape as input.
    mask : cupy.ndarray or None
        Mask array as boolean CuPy array, or None.
    origin : tuple of int
        Origin offset for the footprint (converted from shifts).

    """
    # Convert to CuPy array if needed
    if not isinstance(image, cp.ndarray):
        image = cp.asarray(image)

    input_dtype = image.dtype
    if input_dtype == bool or out_dtype == bool:
        raise ValueError("dtype cannot be bool.")

    # Convert footprint to boolean CuPy array
    if footprint is not None:
        if not isinstance(footprint, cp.ndarray):
            footprint = cp.asarray(footprint)
        footprint = cp.ascontiguousarray(footprint > 0, dtype=bool)
        if footprint.ndim != image.ndim:
            raise ValueError(
                "Image dimensions and footprint dimensions do not match"
            )

    # Ensure image is contiguous
    if not image.flags.c_contiguous:
        image = cp.ascontiguousarray(image)

    # Handle mask
    if mask is not None:
        if not isinstance(mask, cp.ndarray):
            mask = cp.asarray(mask)
        mask = cp.ascontiguousarray(mask > 0, dtype=bool)
        if mask.shape != image.shape:
            raise ValueError("Mask shape must match image shape")

    # Handle output array
    if image is out:
        raise NotImplementedError("Cannot perform rank operation in place.")

    if out is None:
        if out_dtype is None:
            out_dtype = image.dtype
        out = cp.empty(image.shape, dtype=out_dtype)
    else:
        if not isinstance(out, cp.ndarray):
            raise ValueError("out must be a CuPy array")
        if out.shape != image.shape:
            raise ValueError("out shape must match image shape")

    # Handle shifts parameter
    origin = 0  # Default origin
    if shifts is not None:
        if not hasattr(shifts, "__len__"):
            raise ValueError("shifts must be a sequence")
        if len(shifts) != image.ndim:
            raise ValueError(
                f"shifts length ({len(shifts)}) must match image.ndim "
                f"({image.ndim})"
            )
        # Convert shifts to origin (shifts are offsets from center)
        # Note: In ndimage, origin shifts the filter in the opposite
        # direction. For now, map shifts directly to origin
        if any(s != 0 for s in shifts):
            # origin = tuple(-s for s in shifts)  # Negate for opposite
            origin = tuple(shifts)  # Or use directly
            # TODO: Verify the sign convention matches scikit-image

    return image, footprint, out, mask, origin


def _apply(
    operation,
    image,
    footprint,
    out,
    mask,
    shift_x,
    shift_y,
    p0,
    p1,
    out_dtype=None,
    shifts=None,
):
    """Apply percentile range filter with specified operation.

    Parameters
    ----------
    operation : str
        Operation to perform: 'mean', 'sum', 'gradient', etc.
    image : cupy.ndarray
        Input image.
    footprint : cupy.ndarray
        Footprint defining neighborhood.
    out : cupy.ndarray or None
        Output array.
    mask : cupy.ndarray or None
        Mask array.
    shift_x, shift_y : int
        Footprint shifts for 2D images (scikit-image compatibility).
    p0, p1 : float
        Percentile range [0, 1].
    out_dtype : dtype or None
        Output dtype.
    shifts : sequence of int or None
        N-dimensional footprint shifts. If provided, shift_x and shift_y
        must be 0.

    Returns
    -------
    out : cupy.ndarray
        Filtered image.
    """
    # Handle shift_x, shift_y vs shifts
    if shifts is not None:
        if shift_x != 0 or shift_y != 0:
            raise ValueError(
                "shift_x and shift_y must be 0 when shifts is specified"
            )
    else:
        # Convert shift_x, shift_y to shifts for 2D compatibility
        if image.ndim >= 2:
            # For 2D+: shift_y applies to axis 0, shift_x to axis 1
            shifts = [0] * image.ndim
            shifts[0] = shift_y
            shifts[1] = shift_x
            shifts = tuple(shifts)
        elif shift_x != 0 or shift_y != 0:
            raise ValueError(
                "shift_x and shift_y are only valid for 2D or higher "
                "dimensional images"
            )

    image, footprint, out, mask, origin = _preprocess_input(
        image,
        footprint,
        out,
        mask,
        out_dtype,
        shifts=shifts,
    )

    # Convert percentiles from [0, 1] to [0, 100] for our implementation
    p0_pct = p0 * 100.0
    p1_pct = p1 * 100.0

    # Call the GPU implementation
    result = _percentile_range_filter(
        image,
        p0=p0_pct,
        p1=p1_pct,
        operation=operation,
        footprint=footprint,
        output=out,
        mode="reflect",
        cval=0.0,
        origin=origin,
        axes=None,
        mask=mask,
    )

    return result


def autolevel_percentile(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=0,
    shift_y=0,
    p0=0,
    p1=1,
    *,
    shifts=None,
):
    """Return grayscale local autolevel of an image.

    This filter locally stretches the histogram of grayvalues to cover the
    entire range of values from "white" to "black".

    Only grayvalues between percentiles [p0, p1] are considered in the filter.

    Parameters
    ----------
    image : cupy.ndarray
        Input image (N-dimensional, any numeric dtype).
    footprint : cupy.ndarray
        The neighborhood expressed as an array of 1's and 0's.
    out : cupy.ndarray, optional
        If None, a new array is allocated.
    mask : cupy.ndarray, optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
        Offset added to the footprint center point (for 2D images).
        Default is 0.
    p0, p1 : float, optional, in interval [0, 1]
        Define the [p0, p1] percentile interval to be considered for computing
        the value.
    shifts : sequence of int, optional (keyword-only)
        N-dimensional offsets. If provided, shift_x and shift_y must be 0.
        Length must match image.ndim.

    Returns
    -------
    out : cupy.ndarray
        Output image with same shape and dtype as input.

    """
    return _apply(
        "autolevel",
        image,
        footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        p0=p0,
        p1=p1,
        shifts=shifts,
    )


def gradient_percentile(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=0,
    shift_y=0,
    p0=0,
    p1=1,
    *,
    shifts=None,
):
    """Return local gradient of an image (i.e. local maximum - local minimum).

    Only grayvalues between percentiles [p0, p1] are considered in the filter.

    Parameters
    ----------
    image : cupy.ndarray
        Input image (N-dimensional, any numeric dtype).
    footprint : cupy.ndarray
        The neighborhood expressed as an array of 1's and 0's.
    out : cupy.ndarray, optional
        If None, a new array is allocated.
    mask : cupy.ndarray, optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
        Offset added to the footprint center point (for 2D images).
        Default is 0.
    p0, p1 : float, optional, in interval [0, 1]
        Define the [p0, p1] percentile interval to be considered for computing
        the value.
    shifts : sequence of int, optional (keyword-only)
        N-dimensional offsets. If provided, shift_x and shift_y must be 0.
        Length must match image.ndim.

    Returns
    -------
    out : cupy.ndarray
        Output image with same shape and dtype as input.

    """
    return _apply(
        "gradient",
        image,
        footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        p0=p0,
        p1=p1,
        shifts=shifts,
    )


def mean_percentile(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=0,
    shift_y=0,
    p0=0,
    p1=1,
    *,
    shifts=None,
):
    """Return local mean of an image.

    Only grayvalues between percentiles [p0, p1] are considered in the filter.

    Parameters
    ----------
    image : cupy.ndarray
        Input image (N-dimensional, any numeric dtype).
    footprint : cupy.ndarray
        The neighborhood expressed as an array of 1's and 0's.
    out : cupy.ndarray, optional
        If None, a new array is allocated.
    mask : cupy.ndarray, optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
        Offset added to the footprint center point (for 2D images).
        Default is 0.
    p0, p1 : float, optional, in interval [0, 1]
        Define the [p0, p1] percentile interval to be considered for computing
        the value.
    shifts : sequence of int, optional (keyword-only)
        N-dimensional offsets. If provided, shift_x and shift_y must be 0.
        Length must match image.ndim.

    Returns
    -------
    out : cupy.ndarray
        Output image with same shape and dtype as input.

    """

    return _apply(
        "mean",
        image,
        footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        p0=p0,
        p1=p1,
        shifts=shifts,
    )


def subtract_mean_percentile(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=0,
    shift_y=0,
    p0=0,
    p1=1,
    *,
    shifts=None,
):
    """Return image subtracted from its local mean.

    Only grayvalues between percentiles [p0, p1] are considered in the filter.

    Parameters
    ----------
    image : cupy.ndarray
        Input image (N-dimensional, any numeric dtype).
    footprint : cupy.ndarray
        The neighborhood expressed as an array of 1's and 0's.
    out : cupy.ndarray, optional
        If None, a new array is allocated.
    mask : cupy.ndarray, optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
        Offset added to the footprint center point (for 2D images).
        Default is 0.
    p0, p1 : float, optional, in interval [0, 1]
        Define the [p0, p1] percentile interval to be considered for computing
        the value.
    shifts : sequence of int, optional (keyword-only)
        N-dimensional offsets. If provided, shift_x and shift_y must be 0.
        Length must match image.ndim.

    Returns
    -------
    out : cupy.ndarray
        Output image with same shape and dtype as input.

    """
    return _apply(
        "subtract_mean",
        image,
        footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        p0=p0,
        p1=p1,
        shifts=shifts,
    )


def enhance_contrast_percentile(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=0,
    shift_y=0,
    p0=0,
    p1=1,
    *,
    shifts=None,
):
    """Enhance contrast of an image.

    This replaces each pixel by the local maximum if the pixel grayvalue is
    closer to the local maximum than the local minimum. Otherwise it is
    replaced by the local minimum.

    Only grayvalues between percentiles [p0, p1] are considered in the filter.

    Parameters
    ----------
    image : cupy.ndarray
        Input image (N-dimensional, any numeric dtype).
    footprint : cupy.ndarray
        The neighborhood expressed as an array of 1's and 0's.
    out : cupy.ndarray, optional
        If None, a new array is allocated.
    mask : cupy.ndarray, optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
        Offset added to the footprint center point (for 2D images).
        Default is 0.
    p0, p1 : float, optional, in interval [0, 1]
        Define the [p0, p1] percentile interval to be considered for computing
        the value.
    shifts : sequence of int, optional (keyword-only)
        N-dimensional offsets. If provided, shift_x and shift_y must be 0.
        Length must match image.ndim.

    Returns
    -------
    out : cupy.ndarray
        Output image with same shape and dtype as input.

    """
    return _apply(
        "enhance_contrast",
        image,
        footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        p0=p0,
        p1=p1,
        shifts=shifts,
    )


def percentile(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=0,
    shift_y=0,
    p0=0,
    *,
    shifts=None,
):
    """Return local percentile of an image.

    Returns the value of the p0 lower percentile of the local grayvalue
    distribution.

    Parameters
    ----------
    image : cupy.ndarray
        Input image (N-dimensional, any numeric dtype).
    footprint : cupy.ndarray
        The neighborhood expressed as an array of 1's and 0's.
    out : cupy.ndarray, optional
        If None, a new array is allocated.
    mask : cupy.ndarray, optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
        Offset added to the footprint center point (for 2D images).
        Default is 0.
    p0 : float, optional, in interval [0, 1]
        Set the percentile value.
    shifts : sequence of int, optional (keyword-only)
        N-dimensional offsets. If provided, shift_x and shift_y must be 0.
        Length must match image.ndim.

    Returns
    -------
    out : cupy.ndarray
        Output image with same shape and dtype as input.

    """
    return _apply(
        "percentile",
        image,
        footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        p0=p0,
        p1=p0,  # p1 not used for single percentile
        shifts=shifts,
    )


def pop_percentile(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=0,
    shift_y=0,
    p0=0,
    p1=1,
    *,
    shifts=None,
):
    """Return the local number (population) of pixels.

    The number of pixels is defined as the number of pixels which are included
    in the footprint and the mask.

    Only grayvalues between percentiles [p0, p1] are considered in the filter.

    Parameters
    ----------
    image : cupy.ndarray
        Input image (N-dimensional, any numeric dtype).
    footprint : cupy.ndarray
        The neighborhood expressed as an array of 1's and 0's.
    out : cupy.ndarray, optional
        If None, a new array is allocated.
    mask : cupy.ndarray, optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
        Offset added to the footprint center point (for 2D images).
        Default is 0.
    p0, p1 : float, optional, in interval [0, 1]
        Define the [p0, p1] percentile interval to be considered for computing
        the value.
    shifts : sequence of int, optional (keyword-only)
        N-dimensional offsets. If provided, shift_x and shift_y must be 0.
        Length must match image.ndim.

    Returns
    -------
    out : cupy.ndarray
        Output image with same shape and dtype as input.

    """
    return _apply(
        "pop",
        image,
        footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        p0=p0,
        p1=p1,
        shifts=shifts,
    )


def sum_percentile(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=0,
    shift_y=0,
    p0=0,
    p1=1,
    *,
    shifts=None,
):
    """Return the local sum of pixels.

    Only grayvalues between percentiles [p0, p1] are considered in the filter.

    Note that the sum may overflow depending on the data type of the input
    array.

    Parameters
    ----------
    image : cupy.ndarray
        Input image (N-dimensional, any numeric dtype).
    footprint : cupy.ndarray
        The neighborhood expressed as an array of 1's and 0's.
    out : cupy.ndarray, optional
        If None, a new array is allocated.
    mask : cupy.ndarray, optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
        Offset added to the footprint center point (for 2D images).
        Default is 0.
    p0, p1 : float, optional, in interval [0, 1]
        Define the [p0, p1] percentile interval to be considered for computing
        the value.
    shifts : sequence of int, optional (keyword-only)
        N-dimensional offsets. If provided, shift_x and shift_y must be 0.
        Length must match image.ndim.

    Returns
    -------
    out : cupy.ndarray
        Output image with same shape and dtype as input.

    """

    return _apply(
        "sum",
        image,
        footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        p0=p0,
        p1=p1,
        shifts=shifts,
    )


def threshold_percentile(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=0,
    shift_y=0,
    p0=0,
    *,
    shifts=None,
):
    """Local threshold of an image.

    The resulting binary mask is True if the grayvalue of the center pixel is
    greater than the local mean.

    Only grayvalues between percentiles [p0, p1] are considered in the filter.

    Parameters
    ----------
    image : cupy.ndarray
        Input image (N-dimensional, any numeric dtype).
    footprint : cupy.ndarray
        The neighborhood expressed as an array of 1's and 0's.
    out : cupy.ndarray, optional
        If None, a new array is allocated.
    mask : cupy.ndarray, optional
        Mask array that defines (>0) area of the image included in the local
        neighborhood. If None, the complete image is used (default).
    shift_x, shift_y : int, optional
        Offset added to the footprint center point (for 2D images).
        Default is 0.
    p0 : float, optional, in interval [0, 1]
        Set the percentile value.
    shifts : sequence of int, optional (keyword-only)
        N-dimensional offsets. If provided, shift_x and shift_y must be 0.
        Length must match image.ndim.

    Returns
    -------
    out : cupy.ndarray
        Output image with same shape and dtype as input.

    """
    return _apply(
        "threshold",
        image,
        footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        p0=p0,
        p1=p0,  # p1 not used for threshold
        shifts=shifts,
    )
