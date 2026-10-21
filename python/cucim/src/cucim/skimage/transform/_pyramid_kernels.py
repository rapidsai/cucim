# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strided Gaussian correlations for factor-two, linear pyramid reduction.

On even axes, grid-mode linear resize averages adjacent smoothed samples.
Convolving the Gaussian with [0.5, 0.5] allows each separable pass to compute
only those averages. Boundary extension is applied to original sample indices,
before weighting, so reflected endpoints use the same samples as smoothing
followed by resize. The reassociation changes floating-point rounding.
"""

import math

import cupy as cp

from .._vendored._ndimage_filters import _cached_gaussian_kernel1d
from .._vendored._ndimage_util import _generate_boundary_condition_ops


@cp.memoize(for_each_device=True)
def _get_gaussian_decimate_kernel(mode, radius):
    """Generate a one-axis Gaussian/adjacent-average pass at stride two."""
    if mode == "constant":
        load = "T value = (ix < 0 || ix >= axis_length) ? cval : image[base + ix * inner_size];"
    else:
        boundary = _generate_boundary_condition_ops(
            "grid-wrap" if mode == "wrap" else mode,
            "ix",
            "axis_length",
            int_t="int",
        )
        load = f"if (ix < 0 || ix >= axis_length) {{ {boundary} }}"
        load += "T value = image[base + ix * inner_size];"
    # The two shifted Gaussians share all but their endpoints. Combine their
    # coefficients while loading each contributing image sample once.
    return cp.ElementwiseKernel(
        "raw T image, raw T weights, int32 axis_length, int32 inner_size, T cval",
        "T output",
        f"""
        int out_length = axis_length / 2;
        int position = (i / inner_size) % out_length;
        int base = (i / (inner_size * out_length)) * inner_size * axis_length
                         + i % inner_size;
        T sum = 0;
        #pragma unroll
        for (int k = 0; k < {2 * radius + 2}; ++k) {{
            int ix = 2 * position - {radius} + k;
            {load}
            T weight = 0;
            if (k < {2 * radius + 1}) weight += weights[k] * T(0.5);
            if (k > 0) weight += weights[k - 1] * T(0.5);
            sum += value * weight;
        }}
        output = sum;
        """,
        name=f"cucim_gaussian_decimate_{mode}_{radius}",
    )


def _gaussian_decimate(image, sigma, axes, mode, cval):
    """Smooth and halve each selected even axis, keeping intermediates small."""
    radius = int(4.0 * sigma + 0.5) if sigma > 1e-15 else 0
    weights = _cached_gaussian_kernel1d(
        sigma if radius else 1.0, 0, radius, image.dtype.str
    )
    kernel = _get_gaussian_decimate_kernel(mode, radius)
    for axis in axes:
        shape = list(image.shape)
        shape[axis] //= 2
        output = cp.empty(shape, image.dtype)
        kernel(
            image,
            weights,
            image.shape[axis],
            math.prod(image.shape[axis + 1 :]),
            image.dtype.type(cval),
            output,
        )
        image = output
    return image
