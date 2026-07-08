# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

"""GPU-accelerated rank filters.

This module provides GPU (CuPy/CUDA) implementations of most of the local
rank filters from ``skimage.filters.rank``, including generic, percentile, and
bilateral variants. The only unimplemented functions are:

1. ``otsu`` (local Otsu thresholding via between-class variance maximization)
2. ``windowed_histogram`` (returns the full local histogram per pixel)

These two operations did not map cleanly to the common patterns shared across
the other filters implemented here.

Implementation approach
-----------------------

The general elementwise backend computes output pixels independently: each
output pixel is assigned to a single GPU thread that gathers its local
neighborhood and applies the requested operation. Within this backend,
operations that do not require sorted values use streaming reductions, while
operations requiring rank ordering use a sorted-neighborhood kernel.

For compatible 2-D uint8 inputs with fully populated, odd-sized rectangular
footprints (including square footprints), selected filters can instead use a
cooperative sliding-window histogram fast path. This backend partitions the
output rows among CUDA blocks, whose threads cooperatively maintain and
evaluate local histograms while traversing multiple output pixels.

scikit-image, by contrast, always uses a sliding-window histogram approach that
incrementally updates a histogram as it moves across the image. Its current CPU
implementation relies on sequential neighborhood updates and is restricted to
2-D inputs (with 3-D support for a subset of filters). The elementwise backend
in cuCIM does not have this dimensionality restriction and all filters are
available in N-D, although the histogram-based GPU fast path is restricted to
2-D uint8 inputs.

For small window sizes, the general elementwise approach is faster on the GPU,
but a histogram-based approach becomes much faster at large window sizes. The
default setting of ``backend='auto'`` uses thresholds derived from performance
measurements on an RTX A6000 to choose the best approach for a given window
size. This choice can be overridden by explicitly choosing
``backend='elementwise'`` or ``backend='histogram'``.

Note that the elementwise approach is more general and supports features such
as arbitrary footprint shape and an image ``mask``, which are not available
for the histogram-based approach.

Note that behavior at image boundaries differs from scikit-image. The GPU
implementations in cuCIM extend the boundary by reflection, so the footprint
does not shrink at image edges. A supplied mask can still reduce the number of
samples contributing to a neighborhood. The scikit-image implementation does
not extend the image and instead crops the footprint to remain within the image
edges.

cuCIM vs scikit-image
---------------------

The table below summarizes known behavioral differences. Results are otherwise
expected to match.

| Feature                  | scikit-image (CPU)                                              | cuCIM (GPU) |
|--------------------------|-----------------------------------------------------------------|-------------|
| Dimensions               | 2-D (3-D for generic filters)                                   | N-dimensional |
| Supported dtypes         | uint8 and uint16 natively; other real inputs are converted to uint8 | Integer and real floating-point dtypes; non-uint8 inputs are converted to uint8 by default |
| Output dtype             | Usually the processed input dtype; entropy returns float64      | Usually the processed input dtype; entropy defaults to float32 |
| Algorithm                | Sliding-window histogram                                        | Streaming reductions, sorted neighborhoods, or uint8 2-D histogram fast path |
| Boundary handling        | Excludes out-of-bounds pixels (population decreases at borders) | SciPy ``ndimage``-style reflected boundary extension; no boundary-induced population decrease |

By default, non-uint8 inputs are converted to uint8 with
``img_as_ubyte`` before rank filtering. Set ``cast_to_uint8=False`` to opt
out of this conversion and use the elementwise kernels on the native dtype.
These kernels have the following behavioral differences. The cuCIM behavior is
generally preferable in these cases.

| Feature                  | scikit-image (CPU)                                              | cuCIM (GPU) |
|--------------------------|-----------------------------------------------------------------|-------------|
| ``mean_percentile``      | Can return zero when no histogram bin lies in the percentile interval | Selects samples by rank, avoiding this empty-bin artifact |
| ``subtract_mean_percentile`` | Can return zero when no histogram bin lies in the percentile interval | Selects samples by rank, avoiding this empty-bin artifact |
| ``sum``                  | uint8/uint16 output can overflow                                 | Preserves native input dtype; use a wider dtype to avoid overflow |
| ``sum_bilateral``        | uint8/uint16 output can overflow                                 | Preserves native input dtype; use a wider dtype to avoid overflow |
| ``sum_percentile``       | uint8/uint16 output can overflow                                 | Preserves native input dtype; use a wider dtype to avoid overflow |

See the ``_percentile``, ``_generic``, and ``_bilateral`` modules for
additional per-function notes on dtype handling and behavioral differences.

Histogram fast path
-------------------

At larger window sizes, when the input is uint8 (or converted to uint8 via
``cast_to_uint8``, which is enabled by default), a histogram-based approach is
often beneficial.

A uint8 2-D sliding-histogram backend is selected automatically for these rank
filters when all compatibility conditions below are met and the fully
populated rectangular footprint (including square footprints) is at least the
operation-specific benchmark-derived cutoff size. Sparse or arbitrarily shaped
footprints require the elementwise backend.

* ``percentile``
* ``median`` (implemented as ``percentile(p0=0.5)``)
* ``threshold_percentile``
* ``mean_percentile`` with a non-full percentile range
* ``sum_percentile`` with a non-full percentile range
* ``pop_percentile`` with a non-full percentile range
* ``gradient_percentile`` with a non-full percentile range
* ``autolevel_percentile`` with a non-full percentile range
* ``enhance_contrast_percentile`` with a non-full percentile range
* ``subtract_mean_percentile`` with a non-full percentile range
* ``equalize``
* ``geometric_mean``
* ``mean_bilateral``
* ``modal``
* ``majority`` (alias for ``modal``)
* ``pop_bilateral``
* ``sum_bilateral``
* ``entropy``

The compatibility conditions are:

* input image is 2-D and either has dtype ``uint8`` or is converted to
  ``uint8`` before backend selection with ``cast_to_uint8=True``, the default
* footprint is a fully populated rectangular footprint with odd side lengths
  greater than 1, for example ``cupy.ones((15, 15), dtype=bool)``
* output dtype is one of uint8, uint16, float32, or float64; ``entropy``
  specifically requires float32 or float64 output
* no ``mask`` is provided
* no footprint shift is requested (``shift_x == shift_y == 0`` and no nonzero
  ``shifts``)
* the internal boundary mode is ``reflect`` (the public rank wrappers use this
  mode). This is the SciPy ``ndimage`` meaning of ``reflect``, which repeats
  the edge value and is equivalent to ``numpy.pad(..., mode='symmetric')``:
  ``d c b a | a b c d | d c b a``. The naming differs across libraries, so
  this should not be confused with NumPy's ``mode='reflect'``. It also differs
  from scikit-image rank filters, which do not extend the image and therefore
  use smaller cropped neighborhoods near edges and corners.
* footprint half-width does not exceed the corresponding image extent

With ``backend='auto'``, any unsupported case falls back to the generic
elementwise GPU implementation. For compatible calls, automatic dispatch also
requires a benchmark-derived minimum footprint area. Smaller footprints stay
on the generic per-output-pixel backend unless ``backend='histogram'`` is
requested explicitly.

The automatic selection can be overridden with the keyword-only ``backend``
parameter accepted by rank filters:

* ``backend='auto'`` keeps the automatic selection behavior.
* ``backend='histogram'`` requires the histogram backend and raises
  ``ValueError`` if the call is not compatible.
* ``backend='elementwise'`` forces the generic per-output-pixel backend.

"""  # noqa: E501

from ._generic import (
    autolevel,
    enhance_contrast,
    entropy,
    equalize,
    geometric_mean,
    gradient,
    majority,
    maximum,
    mean,
    median,
    minimum,
    modal,
    noise_filter,
    pop,
    subtract_mean,
    sum,
    threshold,
)
from ._percentile import (
    autolevel_percentile,
    enhance_contrast_percentile,
    gradient_percentile,
    mean_percentile,
    percentile,
    pop_percentile,
    subtract_mean_percentile,
    sum_percentile,
    threshold_percentile,
)
from ._bilateral import mean_bilateral, pop_bilateral, sum_bilateral

__all__ = [
    'autolevel',
    'autolevel_percentile',
    'enhance_contrast',
    'enhance_contrast_percentile',
    'entropy',
    'equalize',
    'geometric_mean',
    'gradient',
    'gradient_percentile',
    'majority',
    'maximum',
    'mean',
    'mean_bilateral',
    'mean_percentile',
    'median',
    'minimum',
    'modal',
    'noise_filter',
    'percentile',
    'pop',
    'pop_bilateral',
    'pop_percentile',
    'subtract_mean',
    'subtract_mean_percentile',
    'sum',
    'sum_bilateral',
    'sum_percentile',
    'threshold',
    'threshold_percentile',
    # --- Not yet implemented ---
    # 'otsu',
    # 'windowed_histogram',
]
