# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conversion arithmetic after the public dtype/range checks have completed."""

import cupy as cp


@cp.memoize(for_each_device=True)
def _get_float_to_integer_kernel(dtype, signed, uniform):
    """Scale, round, clip, and cast without a full-sized floating temporary."""
    ctype = "float" if dtype == "float32" else "double"
    rounding = "value = floor(value);" if uniform else "value = rint(value);"
    if uniform and not signed:
        rounding = ""
    offset = "value = value - S(0.5);" if signed and not uniform else ""
    return cp.ElementwiseKernel(
        f"T image, {dtype} scale, {dtype} low, {dtype} high",
        "U output",
        f"""
        typedef {ctype} S;
        S value = S(image) * scale;
        {offset}
        {rounding}
        value = value < low ? low : value;
        value = value > high ? high : value;
        output = U(value);
        """,
        "cucim_float_to_integer",
        options=("-fmad=false",),
    )
