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


def _offsets_diamond(ndim):
    offsets = np.zeros((2 * ndim, ndim))
    for vertex, (axis, offset) in enumerate(product(range(ndim), (-0.5, 0.5))):
        offsets[vertex, axis] = offset
    return cp.asarray(offsets)


def _check_coords_in_hull(
    gridcoords, hull_equations, tolerance, dtype=cp.float64, batch=None
):
    ndim, n_coords = gridcoords.shape
    n_hull_equations = hull_equations.shape[0]

    float_dtype = cp.promote_types(dtype, gridcoords.dtype)
    if batch is None:
        # rough determination if batch mode should be used
        # If memory required for intermediate do_out array is less than some
        # threshold use a single batched kernel call.
        # The specific choice of 1/3 of the free memory available is arbitrary.
        mem_free_bytes = cp.cuda.Device().mem_info[0]
        dot_output_mem = (
            cp.dtype(float_dtype).itemsize * n_coords * n_hull_equations
        )
        batch = dot_output_mem <= 0.33 * mem_free_bytes

    if batch:
        # apply all hull equations at once
        dot_array = cp.dot(hull_equations[:, :ndim], gridcoords)
        dot_array += hull_equations[:, ndim:]
        mask = cp.min(dot_array < tolerance, axis=0)
    else:
        # loop over hull equations to preserve memory
        mask = cp.ones((n_coords,), dtype=bool)
        mask_temp = cp.ones((n_coords,), dtype=bool)
        dot_out = cp.empty((n_coords,), dtype=float_dtype)
        for idx in range(n_hull_equations):
            cp.dot(hull_equations[idx, :ndim], gridcoords, out=dot_out)
            dot_out += hull_equations[idx, ndim:]
            cp.less(dot_out, tolerance, out=mask_temp)
            mask *= mask_temp

    return mask


def convex_hull_image(
    image,
    offset_coordinates=True,
    tolerance=1e-10,
    include_borders=True,
    *,
    omit_empty_coords_check=False,
    float64_computation=True,
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
    include_borders: bool, optional
        If ``False``, vertices/edges are excluded from the final hull mask.
    omit_empty_coords_check : bool, optional
        If ``True``, skip check that there are not any True values in `image`.

    Returns
    -------
    hull : (M, N) array of bool
        Binary image with pixels in convex hull set to True.

    References
    ----------
    .. [1] https://blogs.mathworks.com/steve/2011/10/04/binary-image-convex-hull-algorithm-notes/

    """
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
        return np.zeros(image.shape, dtype=bool)

    if image.dtype != cp.dtype(bool):
        if image.dtype == cp.uint8:
            # bool are actually already stored as uint8 so can use a view to
            # avoid a copy
            image = image.view(bool)
        else:
            image = image.astype(bool)

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
        return cp.zeros(image.shape, dtype=bool)

    gridcoords = cp.reshape(
        cp.mgrid[tuple(map(slice, image.shape))], (ndim, -1)
    )
    coord_dtype = cp.min_scalar_type(max(image.shape))
    if gridcoords.dtype != coord_dtype:
        gridcoords = gridcoords.astype(coord_dtype)
    if float64_computation:
        float_dtype = cp.float64
    else:
        # float32 will be used if coord_dtype is <= 16-bit
        # otherwise, use float64
        float_dtype = cp.promote_types(cp.float32, coord_dtype)

    hull_equations = cp.asarray(hull.equations, dtype=float_dtype)
    coords_in_hull = _check_coords_in_hull(
        gridcoords, hull_equations, tolerance, dtype=float_dtype
    )
    mask = cp.reshape(coords_in_hull, image.shape)
    return mask


def convex_hull_object(image, *, connectivity=2):
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
    """
    if connectivity not in tuple(range(1, image.ndim + 1)):
        raise ValueError("`connectivity` must be between 1 and image.ndim.")

    labeled_im = label(image, connectivity=connectivity, background=0)
    convex_obj = cp.zeros(image.shape, dtype=bool)
    convex_img = cp.zeros(image.shape, dtype=bool)

    max_label = int(labeled_im.max())
    for i in range(1, max_label + 1):
        convex_obj = convex_hull_image(labeled_im == i)
        convex_img = cp.logical_or(convex_img, convex_obj)

    return convex_img
