import lazy_loader as lazy  # noqa: F401

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
