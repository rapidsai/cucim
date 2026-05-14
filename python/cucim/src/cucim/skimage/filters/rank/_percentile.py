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

import warnings

import cupy as cp

from ...util import img_as_ubyte
from ._percentile_range_filter import _skimage_rank_filter

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

_ZERO_FOR_EMPTY_FOOTPRINT_OPS = {
    "geometric_mean",
    "maximum",
    "mean",
    "minimum",
}

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

_doc_backend_param = """
    backend : {'auto', 'histogram', 'elementwise'}, optional (keyword-only)
        Algorithm backend. ``'auto'`` selects the best compatible backend,
        ``'histogram'`` requires the uint8 2D rectangular histogram backend,
        and ``'elementwise'`` forces the generic per-output-pixel backend."""

_doc_boundary_note = """

    Notes
    -----
    Rank filters use reflected boundary extension. The name ``reflect`` can be
    confusing because padding libraries use different conventions. Here it
    follows the SciPy ``ndimage`` convention, which repeats the edge value
    (equivalent to ``numpy.pad(..., mode='symmetric')``)::

        d c b a | a b c d | d c b a

    This also differs from scikit-image's rank filters, which do not extend
    the image at the boundary. scikit-image uses cropped neighborhoods at
    edges and corners, so the effective footprint population can be smaller
    near the image border."""

_doc_cast_to_uint8_param = """
    cast_to_uint8 : bool, optional (keyword-only)
        If True, non-uint8 image inputs are converted to uint8 with
        ``img_as_ubyte`` before backend selection. This can more closely
        match scikit-image's rank filter behavior and can enable the uint8
        histogram backend for compatible inputs. Default is True."""

_doc_returns = """
    Returns
    -------
    out : cupy.ndarray
        Output image with same shape as input. The default output dtype is
        uint8 for non-uint8 inputs converted with ``cast_to_uint8=True``;
        otherwise it follows the input dtype unless ``out`` controls it.
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
        + _doc_backend_param
        + _doc_cast_to_uint8_param
        + "\n"
        + _doc_returns
        + _doc_boundary_note
    )


def _preprocess_input(
    image,
    footprint=None,
    out=None,
    mask=None,
    out_dtype=None,
    shifts=None,
    cast_to_uint8=True,
):
    """Preprocess and verify input for filters.rank methods (GPU version)."""
    if not isinstance(image, cp.ndarray):
        raise ValueError("image must be a CuPy array")

    if image is out:
        raise NotImplementedError("Cannot perform rank operation in place.")

    input_dtype = image.dtype
    if input_dtype == bool or out_dtype == bool:
        raise ValueError("dtype cannot be bool.")

    if cast_to_uint8 and image.dtype != cp.dtype(cp.uint8):
        if image.dtype.kind == "f":
            warnings.warn(
                f"Possible precision loss converting image of type "
                f"{image.dtype} to uint8 as required by rank filters. "
                f"Convert manually using cucim.skimage.util.img_as_ubyte to "
                f"silence this warning.",
                stacklevel=3,
            )
            image = img_as_ubyte(image)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=(
                        r"Downcasting .* to uint8 without scaling because "
                        r"max value .* fits in uint8"
                    ),
                    category=UserWarning,
                )
                image = img_as_ubyte(image)
        input_dtype = image.dtype

    # Convert footprint to boolean CuPy array
    if footprint is not None:
        if not isinstance(footprint, cp.ndarray):
            raise ValueError("footprint must be a CuPy array")
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
            raise ValueError("mask must be a CuPy array")
        mask = cp.ascontiguousarray(mask > 0, dtype=bool)
        if mask.shape != image.shape:
            raise ValueError("Mask shape must match image shape")

    # Handle output array
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
    backend="auto",
    cast_to_uint8=True,
):
    """Apply percentile range filter with specified operation."""
    if not isinstance(image, cp.ndarray):
        raise ValueError("image must be a CuPy array")

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
        cast_to_uint8=cast_to_uint8,
    )

    if (
        operation in _ZERO_FOR_EMPTY_FOOTPRINT_OPS
        and footprint is not None
        and not bool(cp.any(footprint))
    ):
        out.fill(0)
        return out

    # Convert percentiles from [0, 1] to [0, 100] for our implementation
    p0_pct = p0 * 100.0
    p1_pct = p1 * 100.0

    # Call the GPU implementation
    result = _skimage_rank_filter(
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
        backend=backend,
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
    backend="auto",
    cast_to_uint8=True,
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
        backend=backend,
        cast_to_uint8=cast_to_uint8,
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
    backend="auto",
    cast_to_uint8=True,
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
        backend=backend,
        cast_to_uint8=cast_to_uint8,
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
    backend="auto",
    cast_to_uint8=True,
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
        backend=backend,
        cast_to_uint8=cast_to_uint8,
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
    backend="auto",
    cast_to_uint8=True,
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
        backend=backend,
        cast_to_uint8=cast_to_uint8,
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
    backend="auto",
    cast_to_uint8=True,
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
        backend=backend,
        cast_to_uint8=cast_to_uint8,
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
    backend="auto",
    cast_to_uint8=True,
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
        backend=backend,
        cast_to_uint8=cast_to_uint8,
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
    backend="auto",
    cast_to_uint8=True,
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
        backend=backend,
        cast_to_uint8=cast_to_uint8,
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
    backend="auto",
    cast_to_uint8=True,
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
        backend=backend,
        cast_to_uint8=cast_to_uint8,
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
    backend="auto",
    cast_to_uint8=True,
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
        backend=backend,
        cast_to_uint8=cast_to_uint8,
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
