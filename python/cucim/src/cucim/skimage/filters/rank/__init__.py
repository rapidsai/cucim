# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

"""GPU-accelerated rank filters.

This module provides GPU (CuPy/CUDA) implementations of most of the local
rank filters from ``skimage.filters.rank``, including all generic, percentile,
and bilateral variants. The only unimplemented functions are ``otsu`` (local Otsu thresholding via
between-class variance maximization) and ``windowed_histogram`` (returns the
full local histogram per pixel), as these do not map cleanly to the
sort-and-reduce pattern used by all other kernels.

Implementation approach
-----------------------

Most GPU rank filters operate independently on a per-pixel basis: each output
pixel is computed by a single GPU thread that gathers its local neighborhood
and applies the requested operation. Operations that do not require sorted
values use streaming reductions. Operations that need rank ordering use either
a sorted-neighborhood kernel or, for a restricted high-value subset, a
sliding-window histogram fast path.

scikit-image, by contrast, uses a sliding-window histogram approach that
incrementally updates a histogram as it moves across the image. This is
efficient on CPU but inherently sequential and restricted to 2D (or 3D for
some generic filters).

Histogram fast path
-------------------

A uint8 2D sliding-histogram backend is selected automatically for these rank
filters when all compatibility conditions below are met:

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
* ``entropy``

Additional histogram implementations are available for profiling with
``backend='histogram'`` but are not selected automatically until benchmark
cutoffs are established:

* ``equalize``
* ``mean_bilateral``
* ``pop_bilateral``
* ``sum_bilateral``

The compatibility conditions are:

* input image is 2D and has dtype ``uint8``
* output is either omitted or has dtype ``uint8``
* footprint is a fully populated rectangular footprint with odd side lengths
  greater than 1, for example ``cupy.ones((15, 15), dtype=bool)``
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

Any unsupported case falls back to the generic GPU implementation. For
compatible calls, automatic dispatch also requires a benchmark-derived minimum
footprint area. Smaller footprints stay on the generic per-output-pixel
backend unless ``backend='histogram'`` is requested explicitly.

The automatic selection can be overridden with the keyword-only ``backend``
parameter accepted by rank filters:

* ``backend='auto'`` keeps the automatic selection behavior.
* ``backend='histogram'`` requires the histogram backend and raises
  ``ValueError`` if the call is not compatible.
* ``backend='elementwise'`` forces the generic per-output-pixel backend.

cuCIM vs scikit-image
---------------------

The table below summarizes known behavioral differences. Results are otherwise
expected to match.

| Feature                  | scikit-image (CPU)                                              | cuCIM (GPU) |
|--------------------------|-----------------------------------------------------------------|-------------|
| Dimensions               | 2D (3D for generic filters)                                     | N-dimensional |
| Supported dtypes         | uint8, uint16 only                                              | Any numeric dtype |
| Output dtype             | Same as input                                                   | Same as input (preserves wider types) |
| Algorithm                | Sliding-window histogram                                        | Streaming reductions, sorted neighborhoods, or uint8 2D histogram fast path |
| Boundary handling        | Excludes out-of-bounds pixels (population decreases at borders) | SciPy ``ndimage``-style reflected boundary extension, with repeated edge values (always fully populated) |
| ``mean``                 | Spurious zero outputs in low-variance neighborhoods             | No zero artifacts (sorted-array always has values) |
| ``subtract_mean``        | Spurious zero outputs in low-variance neighborhoods             | No zero artifacts (sorted-array always has values) |
| ``sum``                  | Input forced to uint8; overflows                    | Preserves input dtype; use int32 to avoid overflow |
| ``sum_bilateral``        | Input forced to uint8; overflows                    | Preserves input dtype; use int32 to avoid overflow |
| ``sum_percentile``       | Input forced to uint8; overflows                    | Preserves input dtype; use int32 to avoid overflow |
| ``threshold``            | Outputs 0 or 1 (comparison to local mean)                       | Same (0 or 1) |
| ``threshold_percentile`` | Outputs 0 or ``dtype_max`` (comparison to p0-th percentile)     | Same (0 or ``dtype_max``) |

See the ``_percentile``, ``_generic``, and ``_bilateral`` modules for
additional per-function notes on dtype handling and behavioral differences.
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
