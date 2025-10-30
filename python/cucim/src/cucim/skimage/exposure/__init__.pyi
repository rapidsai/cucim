# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

# Explicitly setting `__all__` is necessary for type inference engines
# to know which symbols are exported. See
# https://peps.python.org/pep-0484/#stub-files

__all__ = [
    "histogram",
    "equalize_hist",
    "equalize_adapthist",
    "rescale_intensity",
    "cumulative_distribution",
    "adjust_gamma",
    "adjust_sigmoid",
    "adjust_log",
    "is_low_contrast",
    "match_histograms",
]

from ._adapthist import equalize_adapthist
from .exposure import (
    adjust_gamma,
    adjust_log,
    adjust_sigmoid,
    cumulative_distribution,
    equalize_hist,
    histogram,
    is_low_contrast,
    rescale_intensity,
)
from .histogram_matching import match_histograms
