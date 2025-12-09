# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

"""Generic utilities.

This module contains a number of utility functions to work with images in general.
"""

from ._invert import invert
from ._map_array import map_array
from ._montage import montage
from .arraycrop import crop
from .compare import compare_images
from .dtype import (
    dtype_limits,
    img_as_bool,
    img_as_float,
    img_as_float32,
    img_as_float64,
    img_as_int,
    img_as_ubyte,
    img_as_uint,
)
from .noise import random_noise
from .shape import view_as_blocks, view_as_windows

__all__ = [
    "compare_images",
    "img_as_float32",
    "img_as_float64",
    "img_as_float",
    "img_as_int",
    "img_as_uint",
    "img_as_ubyte",
    "img_as_bool",
    "dtype_limits",
    "montage",
    "view_as_blocks",
    "view_as_windows",
    "crop",
    "map_array",
    "random_noise",
    "invert",
]
