"""Utilities that operate on shapes in images.

These operations are particularly suited for binary images,
although some may be useful for images of other types as well.

Basic morphological operations include dilation and erosion.
"""

from ._skeletonize import medial_axis, thin
from .binary import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_opening,
)
from .footprints import (
    ball,
    cube,
    diamond,
    disk,
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
    "square",
    "rectangle",
    "diamond",
    "disk",
    "cube",
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
