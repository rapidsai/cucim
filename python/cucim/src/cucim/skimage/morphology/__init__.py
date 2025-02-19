"""Morphological algorithms, e.g., closing, opening, skeletonization."""

from ._skeletonize import medial_axis, thin
from .binary import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_opening,
)
from .convex_hull import (
    convex_hull_image,
    convex_hull_object,
)
from .footprints import (
    ball,
    cube,
    diamond,
    disk,
    footprint_from_sequence,
    footprint_rectangle,
    octagon,
    octahedron,
    rectangle,
    square,
    star,
)
from .gray import (
    black_tophat,
    closing,
    dilation,
    erosion,
    opening,
    white_tophat,
)
from .grayreconstruct import reconstruction
from .isotropic import (
    isotropic_closing,
    isotropic_dilation,
    isotropic_erosion,
    isotropic_opening,
)
from .misc import remove_small_holes, remove_small_objects

__all__ = [
    "convex_hull_image",
    "convex_hull_object",
    "binary_erosion",
    "binary_dilation",
    "binary_opening",
    "binary_closing",
    "isotropic_dilation",
    "isotropic_erosion",
    "isotropic_opening",
    "isotropic_closing",
    "erosion",
    "dilation",
    "opening",
    "closing",
    "white_tophat",
    "black_tophat",
    "footprint_from_sequence",
    "footprint_rectangle",
    "diamond",
    "disk",
    "octahedron",
    "ball",
    "octagon",
    "star",
    "reconstruction",
    "remove_small_objects",
    "remove_small_holes",
    "thin",
    "medial_axis",
]
