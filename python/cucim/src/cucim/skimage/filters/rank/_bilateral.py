# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bilateral rank filters for GPU (CuPy/CUDA).

Bilateral filters include only neighborhood pixels whose grayvalue is within
a specified range around the center pixel: ``g - s1 < value < g + s0``, where
``g`` is the center pixel grayvalue and ``s0``, ``s1`` define the range.

See ``cucim.skimage.filters.rank`` for a summary of differences between the
cuCIM and scikit-image implementations.
"""

from ._percentile import (
    _apply,
    _doc_boundary_note,
    _doc_cast_to_uint8_param,
    _doc_common_params,
)

__all__ = [
    "mean_bilateral",
    "pop_bilateral",
    "sum_bilateral",
]

# --- Docstring fragments ---

_doc_s0_s1_params = """
    s0, s1 : int, optional
        Define the bilateral range ``(g - s1, g + s0)`` around the center
        pixel grayvalue ``g``. Only neighborhood pixels with values strictly
        inside this interval are included. Default is 10."""

_doc_shifts_param = """
    shifts : sequence of int, optional (keyword-only)
        N-dimensional offsets. If provided, shift_x and shift_y must be 0.
        Length must match image.ndim."""

_doc_backend_param = """
    backend : {'auto', 'histogram', 'elementwise'}, optional (keyword-only)
        Algorithm backend. ``'auto'`` selects the best compatible backend,
        ``'histogram'`` requires the uint8 2D rectangular histogram backend,
        and ``'elementwise'`` forces the generic per-output-pixel backend."""

_doc_returns = """
    Returns
    -------
    out : cupy.ndarray
        Output image with same shape and dtype as input.
"""


def _build_bilateral_docstring(summary):
    """Build a docstring for a bilateral rank filter."""
    return (
        summary
        + "\n\n    Parameters\n    ----------"
        + _doc_common_params
        + _doc_s0_s1_params
        + _doc_shifts_param
        + _doc_backend_param
        + _doc_cast_to_uint8_param
        + "\n"
        + _doc_returns
        + _doc_boundary_note
    )


def mean_bilateral(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=0,
    shift_y=0,
    s0=10,
    s1=10,
    *,
    shifts=None,
    backend="auto",
    cast_to_uint8=False,
):
    return _apply(
        "bilateral_mean",
        image,
        footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        p0=0,
        p1=0,
        shifts=shifts,
        s0=s0,
        s1=s1,
        backend=backend,
        cast_to_uint8=cast_to_uint8,
    )


mean_bilateral.__doc__ = _build_bilateral_docstring(
    """Apply a flat kernel bilateral filter.

    This is an edge-preserving and noise reducing denoising filter. It averages
    pixels based on their spatial closeness and radiometric similarity.

    Spatial closeness is measured by considering only the local pixel
    neighborhood given by a footprint (structuring element).

    Radiometric similarity is defined by the graylevel interval ``(g-s1, g+s0)``
    where ``g`` is the current pixel graylevel.

    Only pixels belonging to the footprint and having a graylevel inside this
    interval are averaged.""",
)


def pop_bilateral(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=0,
    shift_y=0,
    s0=10,
    s1=10,
    *,
    shifts=None,
    backend="auto",
    cast_to_uint8=False,
):
    return _apply(
        "bilateral_pop",
        image,
        footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        p0=0,
        p1=0,
        shifts=shifts,
        s0=s0,
        s1=s1,
        backend=backend,
        cast_to_uint8=cast_to_uint8,
    )


pop_bilateral.__doc__ = _build_bilateral_docstring(
    """Return the local number (population) of pixels in the bilateral range.

    The number of pixels is defined as the number of pixels which are included
    in the footprint and the mask, and additionally have a graylevel inside
    the interval ``(g-s1, g+s0)`` where ``g`` is the grayvalue of the center
    pixel.""",
)


def sum_bilateral(
    image,
    footprint,
    out=None,
    mask=None,
    shift_x=0,
    shift_y=0,
    s0=10,
    s1=10,
    *,
    shifts=None,
    backend="auto",
    cast_to_uint8=False,
):
    return _apply(
        "bilateral_sum",
        image,
        footprint,
        out=out,
        mask=mask,
        shift_x=shift_x,
        shift_y=shift_y,
        p0=0,
        p1=0,
        shifts=shifts,
        s0=s0,
        s1=s1,
        backend=backend,
        cast_to_uint8=cast_to_uint8,
    )


sum_bilateral.__doc__ = _build_bilateral_docstring(
    """Return the local sum of pixels in the bilateral range.

    Only pixels belonging to the footprint AND having a graylevel inside the
    interval ``(g-s1, g+s0)`` are summed, where ``g`` is the current pixel
    graylevel.

    Note that the sum may overflow depending on the data type of the input
    array. The output dtype matches the input dtype, so for full-range uint8
    images with large footprints, the input should be promoted to a wider
    dtype (e.g. ``image.astype(cupy.int32)``) to prevent overflow.

    .. note::

        scikit-image's rank filters internally convert all inputs to uint8,
        so ``sum_bilateral`` on scikit-image always overflows for non-trivial
        footprints. The GPU implementation preserves the input dtype,
        giving correct results when a wider dtype is used.""",
)
