# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

"""Generic rank filters for GPU.

These are equivalent to the corresponding ``*_percentile`` functions called
with ``p0=0, p1=1`` (full range). This GPU implementation supports any numeric
dtype and N-dimensional images (scikit-image is restricted to uint8/uint16 and
2D/3D).

"""

import numpy as np

from ._percentile import _apply, _doc_common_params

__all__ = [
    "autolevel",
    "enhance_contrast",
    "entropy",
    "equalize",
    "geometric_mean",
    "gradient",
    "majority",
    "maximum",
    "mean",
    "median",
    "minimum",
    "modal",
    "noise_filter",
    "pop",
    "subtract_mean",
    "sum",
    "threshold",
]

# --- Docstring fragments for generic (no p0/p1) functions ---

_doc_shifts_param_generic = """
    shifts : sequence of int, optional (keyword-only)
        N-dimensional offsets. If provided, shift_x, shift_y and shift_z
        must be 0. Length must match image.ndim."""

_doc_returns = """
    Returns
    -------
    out : cupy.ndarray
        Output image with same shape and dtype as input.
"""


def _build_generic_docstring(summary):
    """Build a docstring for a generic (no p0/p1) rank filter."""
    return (
        summary
        + "\n\n    Parameters\n    ----------"
        + _doc_common_params
        + _doc_shifts_param_generic
        + "\n"
        + _doc_returns
    )


def _apply_generic(
    operation,
    image,
    footprint,
    out,
    mask,
    shift_x,
    shift_y,
    shift_z,
    shifts,
    p0=0,
    p1=1,
):
    """Apply a generic rank filter (defaults to full range p0=0, p1=1)."""
    # Convert shift_z into the N-D shifts parameter
    if shifts is not None:
        if shift_x != 0 or shift_y != 0 or shift_z != 0:
            raise ValueError(
                "shift_x, shift_y and shift_z must be 0 when shifts "
                "is specified"
            )
    elif shift_z != 0:
        if image.ndim < 3:
            raise ValueError(
                "shift_z is only valid for 3D or higher dimensional images"
            )
        shifts_list = [0] * image.ndim
        shifts_list[0] = shift_z
        shifts_list[1] = shift_y
        shifts_list[2] = shift_x
        shifts = tuple(shifts_list)

    return _apply(
        operation,
        image,
        footprint,
        out=out,
        mask=mask,
        shift_x=shift_x if shifts is None else 0,
        shift_y=shift_y if shifts is None else 0,
        p0=p0,
        p1=p1,
        shifts=shifts,
    )


def autolevel(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=0,
    shift_y=0,
    shift_z=0,
    *,
    shifts=None,
):
    return _apply_generic(
        "autolevel",
        image,
        footprint,
        out,
        mask,
        shift_x,
        shift_y,
        shift_z,
        shifts,
    )


autolevel.__doc__ = _build_generic_docstring(
    """Auto-level image using local histogram.

    This filter locally stretches the histogram of gray values to cover the
    entire range of values from "white" to "black".""",
)


def gradient(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=0,
    shift_y=0,
    shift_z=0,
    *,
    shifts=None,
):
    return _apply_generic(
        "gradient",
        image,
        footprint,
        out,
        mask,
        shift_x,
        shift_y,
        shift_z,
        shifts,
    )


gradient.__doc__ = _build_generic_docstring(
    """Return local gradient of an image (i.e. local maximum - local
    minimum).""",
)


def mean(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=0,
    shift_y=0,
    shift_z=0,
    *,
    shifts=None,
):
    return _apply_generic(
        "mean",
        image,
        footprint,
        out,
        mask,
        shift_x,
        shift_y,
        shift_z,
        shifts,
    )


mean.__doc__ = _build_generic_docstring(
    """Return local mean of an image.

    .. note::

        scikit-image's histogram-based implementation can produce spurious
        zero outputs in low-variance neighborhoods where no histogram bin
        falls entirely within the percentile window. This GPU implementation
        uses a sorted-array approach that always has values in the
        neighborhood, avoiding such artifacts.""",
)


def subtract_mean(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=0,
    shift_y=0,
    shift_z=0,
    *,
    shifts=None,
):
    result = _apply_generic(
        "subtract_mean",
        image,
        footprint,
        out,
        mask,
        shift_x,
        shift_y,
        shift_z,
        shifts,
    )
    # The generic version uses an offset of mid_bin - 1 (127 for uint8),
    # while the percentile version uses mid_bin (128 for uint8). Adjust by
    # subtracting 1 for integer dtypes to match scikit-image's generic
    # subtract_mean.
    if np.issubdtype(result.dtype, np.integer):
        result -= 1
    return result


subtract_mean.__doc__ = _build_generic_docstring(
    """Return image subtracted from its local mean.

    The output is::

        out = (g - mean) * 0.5 + mid_bin - 1

    where ``mean`` is the local neighborhood mean, ``g`` is the center pixel
    value, and ``mid_bin`` is ``(dtype_max + 1) / 2`` (128 for uint8), so the
    effective offset is 127 for uint8.

    .. note::

        scikit-image's histogram-based implementation can produce spurious
        zero outputs in low-variance neighborhoods where no histogram bin
        falls entirely within the percentile window. This GPU implementation
        uses a sorted-array approach that always has values in the
        neighborhood, avoiding such artifacts.

    .. note::

        This function uses an output offset of ``(dtype_max + 1) / 2 - 1``
        (127 for uint8), matching scikit-image's ``subtract_mean``. The
        percentile variant ``subtract_mean_percentile`` uses an offset of
        ``(dtype_max + 1) / 2`` (128 for uint8), matching scikit-image's
        ``subtract_mean_percentile``.""",
)


def enhance_contrast(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=0,
    shift_y=0,
    shift_z=0,
    *,
    shifts=None,
):
    return _apply_generic(
        "enhance_contrast",
        image,
        footprint,
        out,
        mask,
        shift_x,
        shift_y,
        shift_z,
        shifts,
    )


enhance_contrast.__doc__ = _build_generic_docstring(
    """Enhance contrast of an image.

    This replaces each pixel by the local maximum if the pixel gray value is
    closer to the local maximum than the local minimum. Otherwise it is
    replaced by the local minimum.""",
)


def pop(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=0,
    shift_y=0,
    shift_z=0,
    *,
    shifts=None,
):
    return _apply_generic(
        "pop",
        image,
        footprint,
        out,
        mask,
        shift_x,
        shift_y,
        shift_z,
        shifts,
    )


pop.__doc__ = _build_generic_docstring(
    """Return the local number (population) of pixels.

    The number of pixels is defined as the number of pixels which are included
    in the footprint and the mask.

    .. note::

        The output is constant across the entire image (equal to the footprint
        size), except when a mask is provided. Unlike scikit-image, the GPU
        implementation does not reduce the count at image borders because the
        underlying kernel uses reflected boundary extension (the neighborhood
        is always fully populated). In scikit-image, the population decreases
        at borders because the sliding window excludes out-of-bounds pixels.""",
)


def sum(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=0,
    shift_y=0,
    shift_z=0,
    *,
    shifts=None,
):
    return _apply_generic(
        "sum",
        image,
        footprint,
        out,
        mask,
        shift_x,
        shift_y,
        shift_z,
        shifts,
    )


sum.__doc__ = _build_generic_docstring(
    """Return the local sum of pixels.

    Note that the sum may overflow depending on the data type of the input
    array. The output dtype matches the input dtype, so for full-range uint8
    images with large footprints, the input should be promoted to a wider
    dtype (e.g. ``image.astype(cupy.int32)``) to prevent overflow.

    .. note::

        scikit-image's rank filters internally convert all inputs to uint8,
        so ``sum`` on scikit-image always overflows for non-trivial
        footprints. The GPU implementation preserves the input dtype,
        giving correct results when a wider dtype is used.""",
)


def minimum(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=0,
    shift_y=0,
    shift_z=0,
    *,
    shifts=None,
):
    return _apply_generic(
        "minimum",
        image,
        footprint,
        out,
        mask,
        shift_x,
        shift_y,
        shift_z,
        shifts,
    )


minimum.__doc__ = _build_generic_docstring(
    """Return the local minimum of an image.

    .. note::

        This uses a streaming reduction over the neighborhood. If mask support
        is not needed, ``cupyx.scipy.ndimage.minimum_filter`` may be faster.""",
)


def maximum(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=0,
    shift_y=0,
    shift_z=0,
    *,
    shifts=None,
):
    return _apply_generic(
        "maximum",
        image,
        footprint,
        out,
        mask,
        shift_x,
        shift_y,
        shift_z,
        shifts,
    )


maximum.__doc__ = _build_generic_docstring(
    """Return the local maximum of an image.

    .. note::

        This uses a streaming reduction over the neighborhood. If mask support
        is not needed, ``cupyx.scipy.ndimage.maximum_filter`` may be faster.""",
)


def median(
    image,
    footprint=None,
    out=None,
    mask=None,
    shift_x=0,
    shift_y=0,
    shift_z=0,
    *,
    shifts=None,
):
    return _apply_generic(
        "percentile",
        image,
        footprint,
        out,
        mask,
        shift_x,
        shift_y,
        shift_z,
        shifts,
        p0=0.5,
    )


median.__doc__ = _build_generic_docstring(
    """Return the local median of an image.

    .. note::

        This is implemented via ``percentile(p0=0.5)`` to ensure consistent
        neighborhood-level mask handling with other rank filters. If mask
        support is not needed, ``cupyx.scipy.ndimage.median_filter`` may
        be faster.""",
)


def threshold(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=0,
    shift_y=0,
    shift_z=0,
    *,
    shifts=None,
):
    return _apply_generic(
        "threshold_mean",
        image,
        footprint,
        out,
        mask,
        shift_x,
        shift_y,
        shift_z,
        shifts,
    )


threshold.__doc__ = _build_generic_docstring(
    """Local threshold of an image.

    The resulting binary mask is True if the grayvalue of the center pixel
    is greater than the local mean. The output is::

        out = 1  if g > mean  else  0

    where ``g`` is the center pixel value and ``mean`` is the mean of all
    neighborhood values.

    .. note::

        This differs from ``threshold_percentile``, which compares to the
        p0-th percentile value and outputs 0 or ``dtype_max`` (e.g. 0/255
        for uint8). The generic ``threshold`` compares to the local **mean**
        and outputs 0 or 1.""",
)


def equalize(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=0,
    shift_y=0,
    shift_z=0,
    *,
    shifts=None,
):
    return _apply_generic(
        "equalize",
        image,
        footprint,
        out,
        mask,
        shift_x,
        shift_y,
        shift_z,
        shifts,
    )


equalize.__doc__ = _build_generic_docstring(
    """Equalize image using local histogram.

    The output is the rank of the center pixel within its local neighborhood,
    scaled to the full output range::

        out = dtype_max * rank(g) / N

    where ``rank(g)`` is the number of neighborhood values <= ``g``, ``N``
    is the neighborhood population, and ``dtype_max`` is the maximum value
    for the output dtype (255 for uint8, 1.0 for float).""",
)


def geometric_mean(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=0,
    shift_y=0,
    shift_z=0,
    *,
    shifts=None,
):
    return _apply_generic(
        "geometric_mean",
        image,
        footprint,
        out,
        mask,
        shift_x,
        shift_y,
        shift_z,
        shifts,
    )


geometric_mean.__doc__ = _build_generic_docstring(
    """Return the local geometric mean of an image.

    The output is::

        out = round(exp(mean(log(values + 1))) - 1)

    The ``+1`` / ``-1`` offset ensures that zero-valued pixels are handled
    correctly (``log(0)`` is undefined).""",
)


def noise_filter(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=0,
    shift_y=0,
    shift_z=0,
    *,
    shifts=None,
):
    return _apply_generic(
        "noise_filter",
        image,
        footprint,
        out,
        mask,
        shift_x,
        shift_y,
        shift_z,
        shifts,
    )


noise_filter.__doc__ = _build_generic_docstring(
    """Noise feature filter.

    Returns 0 if the center pixel value appears among its neighbors (i.e. it
    is not isolated noise). Otherwise returns the minimum absolute distance
    to the nearest neighbor value. Higher values indicate more isolated
    pixels.""",
)


def modal(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=0,
    shift_y=0,
    shift_z=0,
    *,
    shifts=None,
):
    return _apply_generic(
        "modal",
        image,
        footprint,
        out,
        mask,
        shift_x,
        shift_y,
        shift_z,
        shifts,
    )


modal.__doc__ = _build_generic_docstring(
    """Return the local modal (most frequent) value of an image.""",
)


def majority(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=0,
    shift_y=0,
    shift_z=0,
    *,
    shifts=None,
):
    return _apply_generic(
        "modal",
        image,
        footprint,
        out,
        mask,
        shift_x,
        shift_y,
        shift_z,
        shifts,
    )


majority.__doc__ = _build_generic_docstring(
    """Return the local majority value of an image.

    This is an alias for ``modal`` — it returns the most frequent value
    in the local neighborhood.""",
)


def entropy(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=0,
    shift_y=0,
    shift_z=0,
    *,
    shifts=None,
):
    return _apply_generic(
        "entropy",
        image,
        footprint,
        out,
        mask,
        shift_x,
        shift_y,
        shift_z,
        shifts,
    )


entropy.__doc__ = _build_generic_docstring(
    """Return the local Shannon entropy of an image.

    The output is the entropy in bits of the local grayvalue distribution::

        out = -sum(p * log2(p))

    where the sum is over unique values in the neighborhood and
    ``p = count / N`` is the probability of each value.

    .. note::

        The output is a floating-point quantity (entropy in bits) cast to the
        output dtype. For integer output dtypes, fractional entropy values
        are truncated. Using a float input dtype preserves full precision.""",
)
