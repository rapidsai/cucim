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
    "gradient",
    "mean",
    "pop",
    "subtract_mean",
    "sum",
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
):
    """Apply a generic rank filter (full percentile range p0=0, p1=1)."""
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
        p0=0,
        p1=1,
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
    """Return local mean of an image.""",
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

        The output is constant across the image interior (equal to the
        footprint size). It only varies at image borders (where the footprint
        extends beyond the image) or when a mask is provided.""",
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
    array.""",
)
