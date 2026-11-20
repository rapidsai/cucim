# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for the element-wise CUDA kernels of the color conversions."""

import re

import cupy as cp

# A floating point literal without a type suffix (``0.04045``, ``16.``,
# ``.5``, ``1.5e-3``, ``1e-5``) that is not part of an identifier or a longer
# number and is not already cast to ``X``.
_FLOAT_LITERAL = re.compile(
    r"""
    (?<![\w.])
    (?<!\(X\))
    (
        (?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?
        |
        \d+[eE][+-]?\d+
    )
    (?![\w.])
    """,
    re.VERBOSE,
)


def typed_float_literals(code, type_name="X"):
    """Cast the floating point literals in CUDA code to ``type_name``.

    A literal such as ``0.04045`` has type ``double`` in C++, so an
    expression that combines it with a ``float`` value is evaluated in double
    precision (and then rounded when it is stored in a ``float``). That runs
    at the FP64 rate of the GPU, which is 1/32 to 1/64 of the single
    precision rate on most GPUs. With the literals cast to the type of the
    data (``float`` or ``double``), the arithmetic is in the precision of the
    data.
    """
    return _FLOAT_LITERAL.sub(lambda m: f"(({type_name}){m.group(1)})", code)


def elementwise_kernel(in_params, out_params, operation, name, **kwargs):
    """``cupy.ElementwiseKernel`` with the literals of the code cast to ``X``.

    The kernel code (``operation`` and, if given, ``preamble``) must use ``X``
    as the type placeholder of the data.
    """
    if kwargs.get("preamble"):
        kwargs["preamble"] = typed_float_literals(kwargs["preamble"])
    return cp.ElementwiseKernel(
        in_params,
        out_params,
        typed_float_literals(operation),
        name=name,
        **kwargs,
    )
