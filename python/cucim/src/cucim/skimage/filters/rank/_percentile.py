# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

"""Percentile rank filters for GPU (CuPy/CUDA).

Inferior and superior ranks, provided by the user, are passed to the kernel
function to provide a softer version of the rank filters. E.g.
``autolevel_percentile`` will stretch image levels between percentile [p0, p1]
instead of using [min, max]. It means that isolated bright or dark pixels will
not produce halos.

See ``cucim.skimage.filters.rank`` for a summary of differences between the
cuCIM and scikit-image implementations.

Dtype notes
-----------

Some operations use a ``dtype_max`` value that affects output scaling
(``autolevel_percentile``, ``threshold_percentile``,
``subtract_mean_percentile``):

- **Unsigned integers** (uint8, uint16, etc.): ``dtype_max`` is the type
  maximum (255, 65535, ...). This matches scikit-image's behavior.
- **Float** (float32, float64): ``dtype_max = 1.0``, assuming normalized
  images in [0, 1]. Operations like ``autolevel_percentile`` will scale
  output to [0.0, 1.0] and ``threshold_percentile`` will output 0.0 or 1.0.
- **Signed integers** (int8, int16, etc.): ``dtype_max`` uses the positive
  maximum (127 for int8, 32767 for int16). This means operations like
  ``autolevel_percentile`` scale to [0, 127] (positive half only) and
  ``subtract_mean_percentile`` centers at 64 (not 0). Signed integer inputs
  are accepted but may not give intuitive results for these operations.

"""

import cupy as cp

from cucim.skimage._vendored._ndimage_filters import _percentile_range_filter

__all__ = [
    "autolevel_percentile",
    "enhance_contrast_percentile",
    "gradient_percentile",
    "mean_percentile",
    "percentile",
    "pop_percentile",
    "subtract_mean_percentile",
    "sum_percentile",
    "threshold_percentile",
]

# --- Common docstring fragments ---

_doc_common_params = """
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
        Default is 0."""

_doc_p0_p1_params = """
    p0, p1 : float, optional, in interval [0, 1]
        Define the [p0, p1] percentile interval to be considered for computing
        the value."""

_doc_p0_only_param = """
    p0 : float, optional, in interval [0, 1]
        Set the percentile value."""

_doc_shifts_param = """
    shifts : sequence of int, optional (keyword-only)
        N-dimensional offsets. If provided, shift_x and shift_y must be 0.
        Length must match image.ndim."""

_doc_returns = """
    Returns
    -------
    out : cupy.ndarray
        Output image with same shape and dtype as input.
"""


def _build_docstring(summary, *, p0_only=False):
    """Build a docstring from common fragments."""
    pct_params = _doc_p0_only_param if p0_only else _doc_p0_p1_params
    return (
        summary
        + "\n\n    Parameters\n    ----------"
        + _doc_common_params
        + pct_params
        + _doc_shifts_param
        + "\n"
        + _doc_returns
    )


def _preprocess_input(
    image,
    footprint=None,
    out=None,
    mask=None,
    out_dtype=None,
    shifts=None,
):
    """Preprocess and verify input for filters.rank methods (GPU version)."""
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
    s0=0,
    s1=0,
):
    """Apply percentile range filter with specified operation."""
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
        s0=s0,
        s1=s1,
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


autolevel_percentile.__doc__ = _build_docstring(
    """Return grayscale local autolevel of an image.

    This filter locally stretches the histogram of grayvalues to cover the
    entire range of values from "white" to "black".

    Only grayvalues between percentiles [p0, p1] are considered in the
    filter. The output is::

        out = dtype_max * (clamp(g, v_p0, v_p1) - v_p0) / (v_p1 - v_p0)

    where ``v_p0`` and ``v_p1`` are the local values at percentiles p0 and
    p1, ``g`` is the center pixel value, and ``dtype_max`` is the maximum
    value for the output dtype (255 for uint8, 65535 for uint16, 1.0 for
    float). See the module-level dtype notes for signed integer behavior.""",
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


gradient_percentile.__doc__ = _build_docstring(
    """Return local gradient of an image (i.e. local maximum - local minimum).

    Only grayvalues between percentiles [p0, p1] are considered in the
    filter. The output is::

        out = v_p1 - v_p0

    where ``v_p0`` and ``v_p1`` are the local values at percentiles p0 and
    p1.""",
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


mean_percentile.__doc__ = _build_docstring(
    """Return local mean of an image.

    Only grayvalues between percentiles [p0, p1] are considered in the
    filter. The output is the arithmetic mean of all neighborhood values
    whose sorted position falls within the [p0, p1] percentile range.

    .. note::

        scikit-image's histogram-based implementation can produce spurious
        zero outputs in low-variance neighborhoods where no histogram bin
        falls entirely within the percentile window. This GPU implementation
        uses a sorted-array approach that always has values in the percentile
        range, avoiding such artifacts.""",
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


subtract_mean_percentile.__doc__ = _build_docstring(
    """Return image subtracted from its local mean.

    Only grayvalues between percentiles [p0, p1] are considered in the
    filter. The output is::

        out = (g - mean_p) * 0.5 + mid_bin

    where ``mean_p`` is the mean of neighborhood values in the [p0, p1]
    percentile range, ``g`` is the center pixel value, and ``mid_bin`` is
    ``(dtype_max + 1) / 2`` (128 for uint8).

    .. note::

        scikit-image's histogram-based implementation can produce spurious
        zero outputs in low-variance neighborhoods where no histogram bin
        falls entirely within the percentile window. This GPU implementation
        uses a sorted-array approach that always has values in the percentile
        range, avoiding such artifacts.

    .. note::

        This function uses an output offset of ``(dtype_max + 1) / 2`` (128
        for uint8), matching scikit-image's ``subtract_mean_percentile``. The
        non-percentile ``subtract_mean`` in ``filters.rank`` uses an offset of
        ``(dtype_max + 1) / 2 - 1`` (127 for uint8), matching scikit-image's
        ``subtract_mean``.""",
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


enhance_contrast_percentile.__doc__ = _build_docstring(
    """Enhance contrast of an image.

    This replaces each pixel by the local maximum if the pixel grayvalue is
    closer to the local maximum than the local minimum. Otherwise it is
    replaced by the local minimum.

    Only grayvalues between percentiles [p0, p1] are considered in the
    filter. The output is::

        out = v_p1  if (v_p1 - g) < (g - v_p0)  else  v_p0

    where ``v_p0`` and ``v_p1`` are the local values at percentiles p0 and
    p1, and ``g`` is the center pixel value.""",
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


percentile.__doc__ = _build_docstring(
    """Return local percentile of an image.

    Returns the value of the p0 lower percentile of the local grayvalue
    distribution. The output is the value at position
    ``floor(p0 * N)`` in the sorted neighborhood, where N is the
    neighborhood population.""",
    p0_only=True,
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


pop_percentile.__doc__ = _build_docstring(
    """Return the local number (population) of pixels.

    The number of pixels is defined as the number of pixels which are included
    in the footprint and the mask.

    Only grayvalues between percentiles [p0, p1] are considered in the
    filter. The output is the count of neighborhood pixels whose values fall
    in histogram bins where the cumulative count is in [p0 * N, p1 * N],
    where N is the neighborhood population.""",
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


sum_percentile.__doc__ = _build_docstring(
    """Return the local sum of pixels.

    Only grayvalues between percentiles [p0, p1] are considered in the filter.

    Note that the sum may overflow depending on the data type of the input
    array. The output dtype matches the input dtype, so for full-range uint8
    images with large footprints, the input should be promoted to a wider
    dtype (e.g. ``image.astype(cupy.int32)``) to prevent overflow.

    .. note::

        scikit-image's rank filters internally convert all inputs to uint8,
        so ``sum_percentile`` on scikit-image always overflows for non-trivial
        footprints. The GPU implementation preserves the input dtype,
        giving correct results when a wider dtype is used.""",
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


threshold_percentile.__doc__ = _build_docstring(
    """Local threshold of an image.

    The resulting binary mask is True if the grayvalue of the center pixel is
    greater than or equal to the value at the p0 percentile. The output is::

        out = dtype_max  if g >= v_p0  else  0

    where ``v_p0`` is the local value at percentile p0, ``g`` is the center
    pixel value, and ``dtype_max`` is the maximum value for the output dtype
    (255 for uint8, 65535 for uint16, 1.0 for float). See the module-level
    dtype notes for signed integer behavior.

    .. note::

        This is different from the (not yet implemented) generic
        ``threshold``, which compares to the local **mean** and outputs 0
        or 1 (not 0 or ``dtype_max``). ``threshold_percentile`` compares to
        the **p0-th percentile** value and outputs 0 or ``dtype_max``.""",
    p0_only=True,
)
