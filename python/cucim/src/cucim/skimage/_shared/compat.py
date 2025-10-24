# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility helpers for dependencies."""

import cupy as cp
import numpy as np
from packaging.version import parse

__all__ = [
    "NP_COPY_IF_NEEDED",
    "SCIPY_CG_TOL_PARAM_NAME",
]


NUMPY_LT_2_0_0 = parse(np.__version__) < parse("2.0.0.dev0")

# With NumPy 2.0.0, `copy=False` now raises a ValueError if the copy cannot be
# made. The previous behavior to only copy if needed is provided with
# `copy=None`. During the transition period, use this symbol instead.
# Remove once NumPy 2.0.0 is the minimal required version.
# https://numpy.org/devdocs/release/2.0.0-notes.html#new-copy-keyword-meaning-for-array-and-asarray-constructors  # noqa: E501
# https://github.com/numpy/numpy/pull/25168
NP_COPY_IF_NEEDED = False if NUMPY_LT_2_0_0 else None


# check CuPy instead of SciPy
CUPY_LT_14 = parse(cp.__version__) < parse("14.0.0a1")

# Starting in SciPy v1.12, 'scipy.sparse.linalg.cg' keyword argument `tol` is
# deprecated in favor of `rtol`. The corresponding change in cupyx.scipy.sparse
# was made in v14.0
SCIPY_CG_TOL_PARAM_NAME = "tol" if CUPY_LT_14 else "rtol"
