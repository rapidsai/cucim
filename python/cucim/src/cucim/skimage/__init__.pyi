# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

submodules = [
    "color",
    "data",
    "exposure",
    "feature",
    "filters",
    "measure",
    "metrics",
    "morphology",
    "registration",
    "restoration",
    "segmentation",
    "transform",
    "util",
]

__all__ = submodules

from . import (  # noqa: F401, E402
    color,
    data,
    exposure,
    feature,
    filters,
    measure,
    metrics,
    morphology,
    registration,
    restoration,
    segmentation,
    transform,
    util,
)
