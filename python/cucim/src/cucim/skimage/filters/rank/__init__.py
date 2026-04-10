# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

"""GPU-accelerated rank filters.

This module provides GPU (CuPy/CUDA) implementations of the local rank
filters from ``skimage.filters.rank``.

cuCIM vs scikit-image
---------------------

| Feature                     | scikit-image (CPU)                                              | cuCIM (GPU) |
|-----------------------------|-----------------------------------------------------------------|-------------|
| Dimensions                  | 2D (3D for generic filters)                                     | N-dimensional |
| Supported dtypes            | uint8, uint16 only                                              | Any numeric dtype |
| Output dtype                | Same as input                                                   | Same as input (preserves wider types) |
| Algorithm                   | Sliding-window histogram                                        | Sort-based per-neighborhood |
| Boundary handling           | Excludes out-of-bounds pixels (population decreases at borders) | Reflected boundary extension (always fully populated) |
| ``mean``, ``subtract_mean`` | Spurious zero outputs in low-variance neighborhoods             | No zero artifacts (sorted-array always has values) |
| ``sum``, ``sum_percentile`` | Input forced to uint8; overflows                                | Preserves input dtype; use int32 to avoid overflow |

See the ``_percentile`` and ``_generic`` modules for additional per-function
notes on dtype handling and behavioral differences.
"""  # noqa: E501

from ._generic import (
    autolevel,
    enhance_contrast,
    gradient,
    maximum,
    mean,
    median,
    minimum,
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
# from .bilateral import mean_bilateral, pop_bilateral, sum_bilateral

# from .generic import (
#     equalize,
#     majority,
#     maximum,
#     mean,
#     geometric_mean,
#     median,
#     minimum,
#     modal,
#     threshold,
#     noise_filter,
#     entropy,
#     otsu,
#     windowed_histogram,
# )

__all__ = [
    'autolevel',
    'autolevel_percentile',
    'enhance_contrast',
    'enhance_contrast_percentile',
    'gradient',
    'gradient_percentile',
    'maximum',
    'mean',
    'mean_percentile',
    # 'mean_bilateral',
    'median',
    'minimum',
    'pop',
    'pop_percentile',
    # 'pop_bilateral',
    'subtract_mean',
    'subtract_mean_percentile',
    'sum',
    'sum_percentile',
    # 'sum_bilateral',
    'percentile',
    'threshold',
    'threshold_percentile',
    # --- Not yet implemented ---
    # 'equalize',
    # 'geometric_mean',
    # 'majority',
    # 'maximum',
    # 'median',
    # 'minimum',
    # 'modal',
    # 'threshold',
    # 'noise_filter',
    # 'entropy',
    # 'otsu',
    # 'windowed_histogram',
]
