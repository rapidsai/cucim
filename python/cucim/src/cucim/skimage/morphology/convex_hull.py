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
from cucim.skimage.measure._regionprops_gpu_utils import _unravel_loop_index

__all__ = [
    "convex_hull_image",
    "convex_hull_object",
]


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
    include_borders: bool, optional
        If ``False``, vertices/edges are excluded from the final hull mask.


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

    References
    ----------
    .. [1] https://blogs.mathworks.com/steve/2011/10/04/binary-image-convex-hull-algorithm-notes/

    """
    if cpu_fallback_threshold is None:
        # Fallback to scikit-image implementation of total number of pixels
        # is less than this
        cpu_fallback_threshold = 30000 if image.ndim == 2 else 13000

    if image.size < cpu_fallback_threshold:
        # Fallback to pure CPU implementation
        from skimage import morphology as morphology_cpu

        return cp.asarray(
            morphology_cpu.convex_hull_image(
                cp.asnumpy(image),
                offset_coordinates=offset_coordinates,
                tolerance=tolerance,
                include_borders=include_borders,
            )
        )

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

    coord_dtype = cp.min_scalar_type(max(image.shape))
    if float64_computation:
        float_dtype = cp.float64
    else:
        # float32 will be used if coord_dtype is <= 16-bit
        # otherwise, use float64
        float_dtype = cp.promote_types(cp.float32, coord_dtype)

    kernel = get_coords_in_hull_kernel(coord_dtype, float_dtype, ndim)

    convex_image = cp.empty_like(image)
    hull_equations = cp.asarray(hull.equations, dtype=float_dtype)
    kernel(image, hull_equations, tolerance, convex_image)
    return convex_image


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
    in cuCIM and not in scikit-image.
    """
    if connectivity not in tuple(range(1, image.ndim + 1)):
        raise ValueError("`connectivity` must be between 1 and image.ndim.")

    labeled_im = label(image, connectivity=connectivity, background=0)
    convex_obj = cp.zeros(image.shape, dtype=bool)
    convex_img = cp.zeros(image.shape, dtype=bool)

    max_label = int(labeled_im.max())
    for i in range(1, max_label + 1):
        convex_obj = convex_hull_image(
            labeled_im == i,
            omit_empty_coords_check=True,
            float64_computation=float64_computation,
            cpu_fallback_threshold=cpu_fallback_threshold,
        )
        convex_img = cp.logical_or(convex_img, convex_obj)

    return convex_img
