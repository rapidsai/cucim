# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

from itertools import product

import cupy as cp
import numpy as np

try:
    from scipy.spatial import ConvexHull, QhullError

    scipy_available = True
except ImportError:
    scipy_available = False

try:
    from skimage.util import unique_rows

    unique_rows_available = True
except ImportError:
    unique_rows_available = False


from cucim.skimage._shared.utils import warn
from cucim.skimage._vendored import ndimage as ndi
from cucim.skimage.measure._label import label

__all__ = [
    "convex_hull_image",
    "convex_hull_object",
]


# Note: in a future MR, _unravel_loop_index will move from this file to a
# common location where it will also be reused across multiple other
# regionprops kernels.
def _unravel_loop_index_declarations(var_name, ndim, uint_t="unsigned int"):
    if ndim == 1:
        code = f"""
        {uint_t} in_coord[1];"""
        return code

    code = f"""
        // variables for unraveling a linear index to a coordinate array
        {uint_t} in_coord[{ndim}];
        {uint_t} temp_floor;"""
    for d in range(ndim):
        code += f"""
        {uint_t} dim{d}_size = {var_name}.shape()[{d}];"""
    return code


def _unravel_loop_index(
    var_name,
    ndim,
    uint_t="unsigned int",
    raveled_index="i",
    omit_declarations=False,
):
    """
    declare a multi-index array in_coord and unravel the 1D index, i into it.
    This code assumes that the array is a C-ordered array.
    """
    code = (
        ""
        if omit_declarations
        else _unravel_loop_index_declarations(var_name, ndim, uint_t)
    )
    if ndim == 1:
        code = f"""
        in_coord[0] = {raveled_index};\n"""
        return code

    code += f"{uint_t} temp_idx = {raveled_index};"
    for d in range(ndim - 1, 0, -1):
        code += f"""
        temp_floor = temp_idx / dim{d}_size;
        in_coord[{d}] = temp_idx - temp_floor * dim{d}_size;
        temp_idx = temp_floor;"""
    code += """
        in_coord[0] = temp_idx;"""
    return code


@cp.memoize(for_each_device=True)
def _offsets_diamond(ndim):
    offsets = np.zeros((2 * ndim, ndim))
    for vertex, (axis, offset) in enumerate(product(range(ndim), (-0.5, 0.5))):
        offsets[vertex, axis] = offset
    return cp.asarray(offsets)


@cp.memoize(for_each_device=True)
def get_coords_in_hull_kernel(coord_dtype, float_dtype, ndim):
    """Keep this kernel for n-dimensional support as the raw_moments kernels
    currently only support 2D and 3D data.
    """
    coord_dtype = cp.dtype(coord_dtype)
    float_dtype = cp.dtype(float_dtype)
    uint_t = (
        "unsigned int" if coord_dtype.itemsize <= 4 else "unsigned long long"
    )
    float_t = "float" if float_dtype.itemsize <= 4 else "double"

    source = """
      if (image[i]) {
        convex_image = true;
      } else {"""
    source += _unravel_loop_index("image", ndim, uint_t=uint_t)
    source += """
        bool in_hull = true;
        int n_hull_equations = hull_equations.shape()[0];
        for (int i_eq = 0; i_eq < n_hull_equations; i_eq++) {"""
    source += f"""
          {float_t} v = 0.0;"""
    for d in range(ndim):
        source += f"""
          v += hull_equations[i_eq * {ndim + 1} + {d}] * in_coord[{d}];"""
    source += f"""
          v += hull_equations[i_eq * {ndim + 1} + {ndim}];"""
    source += """
          if (v > tol) {
            in_hull = false;
            break;
          }
        }
        convex_image = in_hull;
      }\n"""
    inputs = (
        f"raw bool image, raw {float_dtype.name} hull_equations, float64 tol"
    )
    outputs = "bool convex_image"
    name = f"cucim_convex_hull_{ndim}d_{coord_dtype.char}_{float_dtype.char}"
    return cp.ElementwiseKernel(inputs, outputs, source, name=name)


def convex_hull_image(
    image,
    offset_coordinates=True,
    tolerance=1e-10,
    include_borders=True,
    *,
    omit_empty_coords_check=False,
    float64_computation=True,
    cpu_fallback_threshold=None,
):
    """Compute the convex hull image of a binary image.

    The convex hull is the set of pixels included in the smallest convex
    polygon that surround all white pixels in the input image.

    Parameters
    ----------
    image : array
        Binary input image. This array is cast to bool before processing.
    offset_coordinates : bool, optional
        If ``True``, a pixel at coordinate, e.g., (4, 7) will be represented
        by coordinates (3.5, 7), (4.5, 7), (4, 6.5), and (4, 7.5). This adds
        some "extent" to a pixel when computing the hull.
    tolerance : float, optional
        Tolerance when determining whether a point is inside the hull. Due
        to numerical floating point errors, a tolerance of 0 can result in
        some points erroneously being classified as being outside the hull.
    include_borders : bool, optional
        If ``False``, vertices/edges are excluded from the final hull mask.
        Note that the ``True`` case is not currently implemented in cuCIM and
        will raise a ``NotImplementedError``.

    Extra Parameters
    ----------------
    omit_empty_coords_check : bool, optional
        If ``True``, skip check that there are not any True values in `image`.
    float64_computation : bool, optional
        If False, allow use of 32-bit float during the postprocessing stage
        that determines whether each pixel falls within the convex hull.
    cpu_fallback_threshold : non-negative int or None
        Number of pixels in an image before convex_hull_image will fallback
        to pure CPU implementation.

    Returns
    -------
    hull : (M, N) array of bool
        Binary image with pixels in convex hull set to True.

    Notes
    -----
    The parameters listed under "Extra Parameters" above are present only
    in cuCIM and not in scikit-image.

    In 2D and 3D with ``offset_coordinates=True``, the hull is computed from
    the first and last pixel of each row on the GPU (the batch convex-hull
    kernels of ``regionprops``): exactly in 2D, and in 3D with one Qhull call
    on the host for a reduced set of candidate vertices. Other cases use
    Qhull on the boundary pixels, as scikit-image does.

    References
    ----------
    .. [1] https://blogs.mathworks.com/steve/2011/10/04/binary-image-convex-hull-algorithm-notes/

    """
    return _convex_hull_image(
        image,
        offset_coordinates=offset_coordinates,
        tolerance=tolerance,
        include_borders=include_borders,
        omit_empty_coords_check=omit_empty_coords_check,
        float64_computation=float64_computation,
        cpu_fallback_threshold=cpu_fallback_threshold,
    )


def _use_batch_kernels(ndim, offset_coordinates, include_borders):
    """Whether the batch convex-hull kernels of regionprops apply.

    They compute the hull of the pixels' diamond points
    (``offset_coordinates=True``) with the pixels on the hull included
    (``include_borders=True``), in 2D and 3D.
    """
    return ndim in (2, 3) and offset_coordinates and include_borders


def _convex_hull_image(
    image,
    offset_coordinates=True,
    tolerance=1e-10,
    include_borders=True,
    *,
    omit_empty_coords_check=False,
    float64_computation=True,
    cpu_fallback_threshold=None,
    use_batch_kernels=True,
):
    """`convex_hull_image`; `use_batch_kernels=False` forces the Qhull path.

    The batch kernels fall back to the Qhull path for a 3D region whose
    candidate vertices Qhull rejects, through this function with
    ``use_batch_kernels=False``.
    """
    if cpu_fallback_threshold is not None and cpu_fallback_threshold < 0:
        raise ValueError("cpu_fallback_threshold must be non-negative")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if image.ndim < 2:
        raise ValueError("image must be at least 2D")

    # singleton sizes can cause numeric problems in QHull, so squeeze out
    # these dimensions first (and restore original shape at end)
    # Note: This fix is not yet in upstream scikit-image as of v0.25.
    original_shape = image.shape
    image = cp.squeeze(image)
    if image.ndim < 2:
        return image.reshape(original_shape)

    batch = use_batch_kernels and _use_batch_kernels(
        image.ndim, offset_coordinates, include_borders
    )
    if cpu_fallback_threshold is None:
        # Fallback to scikit-image implementation of total number of pixels
        # is less than this. Values here are approximately when GPU computation
        # became faster on an NVIDIA RTX A6000 (batch kernels: about 4000
        # pixels in 2D and 3D, 0.5 ms in 2D and 2.5 ms in 3D).
        if batch:
            cpu_fallback_threshold = 4000
        else:
            cpu_fallback_threshold = 30000 if image.ndim == 2 else 13000

    if image.size < cpu_fallback_threshold:
        # Fallback to pure CPU implementation
        from skimage import morphology as morphology_cpu

        convex_image = cp.asarray(
            morphology_cpu.convex_hull_image(
                cp.asnumpy(image),
                offset_coordinates=offset_coordinates,
                tolerance=tolerance,
                include_borders=include_borders,
            )
        )
        return convex_image.reshape(original_shape)

    if not scipy_available:
        raise ImportError(
            "This function requires SciPy, but it could not import: "
            "scipy.spatial.ConvexHull, scipy.spatial.QhullError"
        )
    if not unique_rows_available:
        raise ImportError(
            "This function requires skimage.util.unique_rows, but it could "
            "not be imported."
        )

    if not include_borders:
        raise NotImplementedError(
            "Only the `include_borders=True` case is implemented"
        )

    ndim = image.ndim
    if not omit_empty_coords_check and cp.count_nonzero(image) == 0:
        warn(
            "Input image is entirely zero, no valid convex hull. "
            "Returning empty image",
            UserWarning,
        )
        return cp.zeros(original_shape, dtype=bool)

    if image.dtype != cp.dtype(bool):
        # canonical 0/1 values: a bool view of a uint8 image would keep
        # values such as 255, which the batch path reads as labels
        image = image != 0

    if batch:
        # the image as a label image with a single label (the hull kernels
        # need the label's tight bounding box)
        from cucim.skimage.measure._regionprops_gpu_basic_kernels import (
            _regionprops_label_pass,
        )
        from cucim.skimage.measure._regionprops_gpu_convex import (
            _convex_hull_union,
        )

        labels = cp.ascontiguousarray(image).view(cp.uint8)
        bbox = _regionprops_label_pass(labels, 1, compute_bbox=True)["bbox"]
        # In 3D, the hull of the one label is computed with qhull on the
        # host: for a large round region the GPU hull kernel is slower (2.2x
        # for a ball of radius 120), while for blobs qhull costs < 1 ms more.
        convex_image = _convex_hull_union(
            labels, bbox, 1, tolerance=tolerance, use_qhull=True
        )
        return convex_image.reshape(original_shape)

    # xor with eroded version to keep only edge pixels of the binary image
    image_boundary = cp.bitwise_xor(image, ndi.binary_erosion(image, 3))

    coords = cp.stack(cp.nonzero(image_boundary), axis=-1)

    # Add a vertex for the middle of each pixel edge
    if offset_coordinates:
        offsets = _offsets_diamond(image.ndim)
        coords = (coords[:, np.newaxis, :] + offsets).reshape(-1, ndim)

    coords = cp.asnumpy(coords)

    # repeated coordinates can *sometimes* cause problems in
    # scipy.spatial.ConvexHull, so we remove them.
    coords = unique_rows(coords)

    # Find the convex hull
    try:
        hull = ConvexHull(coords)
    except QhullError as err:
        warn(
            f"Failed to get convex hull image. "
            f"Returning empty image, see error message below:\n"
            f"{err}"
        )
        return cp.zeros(original_shape, dtype=bool)

    coord_dtype = cp.min_scalar_type(max(image.shape))
    if float64_computation:
        float_dtype = cp.float64
    else:
        # coord_dtype will be an unsigned integer type from min_scalar_type.
        # The promote_types call below will give the following behavior:
        # - `uint8` or `uint16` dtype will become `float32`
        # - `uint32` or `uint64` dtype will become `float64`
        float_dtype = cp.promote_types(cp.float32, coord_dtype)

    kernel = get_coords_in_hull_kernel(coord_dtype, float_dtype, ndim)

    if not image.flags.c_contiguous:
        image = cp.ascontiguousarray(image)
    convex_image = cp.empty_like(image)
    hull_equations = cp.asarray(hull.equations, dtype=float_dtype)
    kernel(image, hull_equations, tolerance, convex_image)
    return convex_image.reshape(original_shape)


def convex_hull_object(
    image,
    *,
    connectivity=2,
    float64_computation=False,
    cpu_fallback_threshold=None,
):
    r"""Compute the convex hull image of individual objects in a binary image.

    The convex hull is the set of pixels included in the smallest convex
    polygon that surround all white pixels in the input image.

    Parameters
    ----------
    image : (M, N) ndarray
        Binary input image.
    connectivity : {1, 2}, int, optional
        Determines the neighbors of each pixel. Adjacent elements
        within a squared distance of ``connectivity`` from pixel center
        are considered neighbors.::

            1-connectivity      2-connectivity
                  [ ]           [ ]  [ ]  [ ]
                   |               \  |  /
             [ ]--[x]--[ ]      [ ]--[x]--[ ]
                   |               /  |  \
                  [ ]           [ ]  [ ]  [ ]

    Extra Parameters
    ----------------
    float64_computation : bool, optional
        If False, allow use of 32-bit float during the postprocessing stage
    cpu_fallback_threshold : non-negative int or None
        Number of pixels in an image before convex_hull_image will fallback
        to pure CPU implementation.

    Returns
    -------
    hull : ndarray of bool
        Binary image with pixels inside convex hull set to ``True``.

    Notes
    -----
    This function uses ``skimage.morphology.label`` to define unique objects,
    finds the convex hull of each using ``convex_hull_image``, and combines
    these regions with logical OR. Be aware the convex hulls of unconnected
    objects may overlap in the result. If this is suspected, consider using
    convex_hull_image separately on each object or adjust ``connectivity``.

    The parameters listed under "Extra Parameters" above are present only
    in cuCIM and not in scikit-image. Unlike scikit-image, images with more
    than two dimensions are supported.

    In 2D and 3D, the convex hulls of all objects are computed at once with
    the batch convex-hull kernels of ``regionprops`` and written into the
    output in one pass (`float64_computation` has no effect there). Other
    images, and images smaller than an explicitly given
    `cpu_fallback_threshold`, use ``convex_hull_image`` once per object.
    """
    if connectivity not in tuple(range(1, image.ndim + 1)):
        raise ValueError("`connectivity` must be between 1 and image.ndim.")
    # (as convex_hull_image, whatever the image's contents and dispatch)
    if cpu_fallback_threshold is not None and cpu_fallback_threshold < 0:
        raise ValueError("cpu_fallback_threshold must be non-negative")

    labeled_im = label(image, connectivity=connectivity, background=0)
    max_label = int(labeled_im.max())

    # singleton axes are squeezed, as convex_hull_image does for each object
    squeezed = cp.squeeze(labeled_im)
    if (
        squeezed.ndim in (2, 3)
        and max_label > 0
        and (
            cpu_fallback_threshold is None
            or image.size >= cpu_fallback_threshold
        )
    ):
        # all objects at once
        from cucim.skimage.measure._regionprops_gpu_basic_kernels import (
            _regionprops_label_pass,
        )
        from cucim.skimage.measure._regionprops_gpu_convex import (
            _convex_hull_union,
        )

        squeezed = cp.ascontiguousarray(squeezed)
        bbox = _regionprops_label_pass(squeezed, max_label, compute_bbox=True)[
            "bbox"
        ]
        # (in 3D, the hulls are computed on the GPU: with many objects this
        # is much faster than one qhull call per object)
        convex_img = _convex_hull_union(squeezed, bbox, max_label)
        return convex_img.reshape(image.shape)

    convex_img = cp.zeros(image.shape, dtype=bool)

    for i in range(1, max_label + 1):
        convex_obj = convex_hull_image(
            labeled_im == i,
            omit_empty_coords_check=True,
            float64_computation=float64_computation,
            cpu_fallback_threshold=cpu_fallback_threshold,
        )
        convex_img = cp.logical_or(convex_img, convex_obj)

    return convex_img
