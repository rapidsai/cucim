import math
import warnings
from collections.abc import Sequence

import cupy as cp

from cucim.skimage._shared.distance import pdist_max_blockwise
from cucim.skimage._shared.utils import _ndarray_argwhere
from cucim.skimage._vendored import ndimage as ndi

# Store information on which other properties a given property depends on
# This information will be used by `regionprops_dict` to make sure that when
# a particular property is requested any dependent properties are computed
# first.
convex_deps = dict()
convex_deps["image_convex"] = ["image"]  # computed by regionprops_image
convex_deps["area_convex"] = ["image_convex"]
convex_deps["feret_diameter_max"] = ["image_convex"]
convex_deps["solidity"] = ["area", "area_convex"]


def regionprops_area_convex(
    images_convex,
    max_label=None,
    spacing=None,
    area_dtype=cp.float64,
    props_dict=None,
):
    """Compute the area of each convex image.

    writes "area_convex" to props_dict

    Parameters
    ----------
    images_convex : sequence of cupy.ndarray
        Convex images for each region as produced by ``regionprops_image`` with
        ``compute_convex=True``.
    """
    if max_label is None:
        max_label = len(images_convex)
    if not isinstance(images_convex, Sequence):
        raise ValueError("Expected `images_convex` to be a sequence of images")
    area_convex = cp.zeros((max_label,), dtype=area_dtype)
    for i in range(max_label):
        area_convex[i] = images_convex[i].sum()
    if spacing is not None:
        if isinstance(spacing, cp.ndarray):
            pixel_area = cp.product(spacing)
        else:
            pixel_area = math.prod(spacing)
        area_convex *= pixel_area
    if props_dict is not None:
        props_dict["area_convex"] = area_convex
    return area_convex


def _regionprops_coords_perimeter(
    image,
    connectivity=1,
):
    """
    Takes an image of a single labeled region (e.g. one element of the tuple
    resulting from regionprops_image) and returns the coordinates of the voxels
    at the edge of that region.
    """

    # remove non-boundary pixels
    binary_image = image > 0
    footprint = ndi.generate_binary_structure(
        binary_image.ndim, connectivity=connectivity
    )
    binary_image_eroded = ndi.binary_erosion(binary_image, footprint)
    binary_edges = binary_image * ~binary_image_eroded
    edge_coords = _ndarray_argwhere(binary_edges)
    return edge_coords


def _feret_diameter_max(image_convex, spacing=None, return_argmax=False):
    """Compute the maximum Feret diameter of a single convex image region."""
    if image_convex.size == 1:
        warnings.warn(
            "single element image, returning 0 for feret diameter", UserWarning
        )
        return 0
    coords = _regionprops_coords_perimeter(image_convex, connectivity=1)
    coords = coords.astype(cp.float32)

    if spacing is not None:
        if all(s == 1.0 for s in spacing):
            spacing = None
        else:
            spacing = cp.asarray(spacing, dtype=cp.float32).reshape(1, -1)
            coords *= spacing

    out = pdist_max_blockwise(
        coords,
        metric="sqeuclidean",
        compute_argmax=return_argmax,
        coords_per_block=4000,
    )
    if return_argmax:
        return math.sqrt(out[0]), out[1]
    return math.sqrt(out[0])


def regionprops_feret_diameter_max(
    images_convex, spacing=None, props_dict=None
):
    """Compute the maximum Feret diameter of the convex hull of each image in
    images_convex.

    writes "feret_diameter_max" to props_dict

    Parameters
    ----------
    images_convex : sequence of cupy.ndarray
        Convex images for each region as produced by ``regionprops_image`` with
        ``compute_convex=True``.
    spacing : tuple of float, optional
        The pixel spacing of the image.
    props_dict : dict, optional
        A dictionary to store the computed properties.

    Notes
    -----
    The maximum Feret diameter is the maximum distance between any two
    points on the convex hull of the region. The implementation here is based
    on pairwise distances of all boundary coordinates rather than using
    marching squares or marching cubes as in scikit-image. The implementation
    here is n-dimensional.

    The distance is between pixel centers and so may be approximately one pixel
    width less than the one computed by scikit-image.
    """
    if not isinstance(images_convex, Sequence):
        raise ValueError("Expected `images_convex` to be a sequence of images")
    diameters = cp.asarray(
        tuple(
            _feret_diameter_max(
                image_convex, spacing=spacing, return_argmax=False
            )
            for image_convex in images_convex
        )
    )
    if props_dict is not None:
        props_dict["feret_diameter_max"] = diameters
    return diameters
