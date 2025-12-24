# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

"""
Rolling ball background subtraction algorithm.

This module provides GPU-accelerated implementations of the rolling ball
algorithm for background estimation and subtraction.
"""

import cupy as cp
import numpy as np

from cucim.skimage._shared.utils import DEPRECATED, deprecate_parameter
from cucim.skimage._vendored import ndimage as ndi
from cucim.skimage.morphology import disk, erosion, opening

__all__ = ["rolling_ball", "ball_kernel", "ellipsoid_kernel"]


def _ball_kernel_reference(radius, ndim, dtype=np.float64):
    """Simple numpy-based reference implementation of the ball kernel.

    used to validate ball_kernel elementwise kernel implementation
    """
    dtype = np.dtype(dtype)
    if dtype.kind != "f":
        raise ValueError("dtype must be a floating-point type")

    coords = np.meshgrid(
        *[np.arange(-radius, radius + 1, dtype=dtype) for _ in range(ndim)],
        indexing="ij",
    )
    sum_of_squares = sum(c**2 for c in coords)
    distance_from_center = np.sqrt(sum_of_squares)
    kernel = np.sqrt(np.clip(radius**2 - sum_of_squares, 0, None))
    kernel[distance_from_center > radius] = np.inf

    return cp.asarray(kernel)


def _ellipsoid_kernel_reference(shape, intensity, dtype=np.float64):
    """Simple numpy-based reference implementation of the ellipsoid kernel.

    Used to validate ellipsoid_kernel elementwise kernel implementation.
    """
    shape = np.asarray(shape)
    semi_axis = np.clip(shape // 2, 1, None)
    dtype = np.dtype(dtype)
    if dtype.kind != "f":
        raise ValueError("dtype must be a floating-point type")

    grids = [np.arange(-x, x + 1, dtype=dtype) for x in semi_axis]
    kernel_coords = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1)

    intensity_scaling = 1 - np.sum((kernel_coords / semi_axis) ** 2, axis=-1)
    kernel = intensity * np.sqrt(np.clip(intensity_scaling, 0, None))
    kernel[intensity_scaling < 0] = np.inf

    return cp.asarray(kernel)


# =============================================================================
# GPU-based ball kernel implementations (experimental)
# =============================================================================


@cp.memoize(for_each_device=True)
def _get_ball_kernel_elementwise(ndim, structure_and_footprint):
    """Get or create a cached ElementwiseKernel for the given ndim."""
    # Generate index computation (similar to _generate_indices_ops)
    # Convert flat index i to n-dimensional indices ind_0, ind_1, ..., ind_{n-1}
    index_lines = ["int _i = i;"]
    for j in range(ndim - 1, 0, -1):
        index_lines.append(f"int ind_{j} = _i % size; _i /= size;")
    index_lines.append("int ind_0 = _i;")

    # Generate coordinate computation relative to center (int half_size)
    coord_lines = [f"T c{j} = (T)(ind_{j} - half_size);" for j in range(ndim)]

    # Generate sum of squares
    sum_sq_terms = " + ".join(f"c{j} * c{j}" for j in range(ndim))

    # Join lines before f-string (backslashes not allowed in f-strings)
    index_code = "\n    ".join(index_lines)
    coord_code = "\n    ".join(coord_lines)

    if structure_and_footprint:
        # Compute center_val - height, where center_val = radius
        # This gives how much lower each position is than the apex
        # Also output footprint (True where valid, False where inf)
        # radius is T (float) to support non-integer radii
        input_params = "T radius, int32 size, int32 half_size, T center_val"
        output_params = "T out, bool footprint_out"
        kernel_name = f"ball_kernel_diff_{ndim}d"

        code = f"""
        // Convert flat index (i) to {ndim}D coordinates
        {index_code}

        // Convert to coordinates relative to center
        {coord_code}

        // Compute distance squared
        T sum_sq = {sum_sq_terms};
        T r_sq = radius * radius;

        // Compute ball surface height and footprint
        if (sum_sq <= r_sq) {{
            out = sqrt(r_sq - sum_sq) - center_val;
            footprint_out = true;
        }} else {{
            out = CUDART_INF;
            footprint_out = false;
        }}
        """
    else:
        # Compute raw ball surface height
        # radius is T (float) to support non-integer radii
        input_params = "T radius, int32 size, int32 half_size"
        output_params = "T out"
        kernel_name = f"ball_kernel_{ndim}d"

        code = f"""
        // Convert flat index (i) to {ndim}D coordinates
        {index_code}

        // Convert to coordinates relative to center
        {coord_code}

        // Compute distance squared
        T sum_sq = {sum_sq_terms};
        T r_sq = radius * radius;

        // Compute ball surface height
        if (sum_sq <= r_sq) {{
            out = sqrt(r_sq - sum_sq);
        }} else {{
            out = CUDART_INF;
        }}
        """

    return cp.ElementwiseKernel(input_params, output_params, code, kernel_name)


def ball_kernel(
    radius, ndim=2, dtype=cp.float64, structure_and_footprint=False
):
    """Create a ball-shaped kernel using ElementwiseKernel (GPU).

    This version generates the kernel directly on the GPU using
    cupy.ElementwiseKernel. Supports arbitrary dimensions.

    Parameters
    ----------
    radius : int or float
        Radius of the ball.
    ndim : int, optional
        Number of dimensions. Default is 2.
    dtype : dtype, optional
        The data type of the kernel. Default is ``cp.float64``.
    structure_and_footprint : bool, optional
        If True, directly return the non-flat structuring element and
        boolean footprint for the rolling ball algorithm. The structure
        is the intensity difference from the center (apex), giving how
        much lower each position is than the apex (center = 0).
        Default is False.

    Returns
    -------
    kernel : cupy.ndarray
        The kernel containing the surface intensity of the top half
        of the ball (sphere). If ``structure_and_footprint=True``, this
        is the non-flat structuring element for grey erosion.
    footprint : cupy.ndarray (only if ``structure_and_footprint=True``)
        Boolean array that is True where the kernel is valid (inside
        the ball) and False elsewhere.
    """
    dtype = cp.dtype(dtype)
    half_size = int(radius)  # integer center offset
    size = 2 * half_size + 1
    radius_val = dtype.type(radius)  # float radius for distance calculation

    # Get or create the kernel for this ndim and mode
    kernel_func = _get_ball_kernel_elementwise(ndim, structure_and_footprint)

    # Allocate output
    shape = (size,) * ndim
    kernel = cp.empty(size**ndim, dtype=dtype)

    # Run elementwise kernel (uses built-in `i` as linear index)
    if structure_and_footprint:
        # Center value is the ball height at the apex (distance = 0)
        center_val = radius_val
        # Also allocate footprint output
        footprint = cp.empty(size**ndim, dtype=cp.bool_)
        kernel_func(radius_val, size, half_size, center_val, kernel, footprint)
        return kernel.reshape(shape), footprint.reshape(shape)
    else:
        kernel_func(radius_val, size, half_size, kernel)
        return kernel.reshape(shape)


# =============================================================================
# GPU-based ellipsoid kernel implementations
# =============================================================================


@cp.memoize(for_each_device=True)
def _get_ellipsoid_kernel_elementwise(ndim, structure_and_footprint):
    """Get or create a cached ElementwiseKernel for ellipsoid with given ndim.

    Unlike ball_kernel which has uniform size, ellipsoid has different sizes
    per dimension. We pass sizes and semi_axes as separate scalar parameters
    for each dimension to avoid array indexing overhead.
    """
    # Generate index computation for non-uniform sizes
    # We need size_0, size_1, ... as separate parameters
    index_lines = ["int _i = i;"]
    for j in range(ndim - 1, 0, -1):
        index_lines.append(f"int ind_{j} = _i % size_{j}; _i /= size_{j};")
    index_lines.append("int ind_0 = _i;")

    # Generate coordinate computation relative to center (using semi_axis)
    coord_lines = [
        f"T c{j} = (T)(ind_{j}) - (T)(semi_{j});" for j in range(ndim)
    ]

    # Generate normalized sum of squares: sum((c{j} / semi_{j})^2)
    norm_sq_terms = " + ".join(
        f"(c{j} / (T)(semi_{j})) * (c{j} / (T)(semi_{j}))" for j in range(ndim)
    )

    # Join lines
    index_code = "\n    ".join(index_lines)
    coord_code = "\n    ".join(coord_lines)

    # Build input parameters: size_*, semi_*, intensity
    size_params = ", ".join(f"int32 size_{j}" for j in range(ndim))
    semi_params = ", ".join(f"int32 semi_{j}" for j in range(ndim))

    if structure_and_footprint:
        input_params = f"{size_params}, {semi_params}, T intensity"
        output_params = "T out, bool footprint_out"
        kernel_name = f"ellipsoid_kernel_diff_{ndim}d"

        code = f"""
        // Convert flat index (i) to {ndim}D coordinates
        {index_code}

        // Convert to coordinates relative to center
        {coord_code}

        // Compute normalized sum of squares
        T norm_sq = {norm_sq_terms};

        // Compute ellipsoid surface height and footprint
        if (norm_sq <= (T)1.0) {{
            out = intensity * sqrt((T)1.0 - norm_sq) - intensity;
            footprint_out = true;
        }} else {{
            out = CUDART_INF;
            footprint_out = false;
        }}
        """
    else:
        input_params = f"{size_params}, {semi_params}, T intensity"
        output_params = "T out"
        kernel_name = f"ellipsoid_kernel_{ndim}d"

        code = f"""
        // Convert flat index (i) to {ndim}D coordinates
        {index_code}

        // Convert to coordinates relative to center
        {coord_code}

        // Compute normalized sum of squares
        T norm_sq = {norm_sq_terms};

        // Compute ellipsoid surface height
        if (norm_sq <= (T)1.0) {{
            out = intensity * sqrt((T)1.0 - norm_sq);
        }} else {{
            out = CUDART_INF;
        }}
        """

    return cp.ElementwiseKernel(input_params, output_params, code, kernel_name)


def ellipsoid_kernel(
    shape, intensity, dtype=cp.float64, structure_and_footprint=False
):
    """Create an ellipsoid kernel for rolling_ball.

    This version generates the kernel directly on the GPU using
    cupy.ElementwiseKernel. Supports arbitrary dimensions.

    Parameters
    ----------
    shape : array-like
        Length of the principal axis of the ellipsoid (excluding
        the intensity axis). The kernel needs to have the same
        dimensionality as the image it will be applied to.
    intensity : int or float
        Length of the intensity axis of the ellipsoid.
    dtype : dtype, optional
        The data type of the kernel. Default is ``cp.float64``.
    structure_and_footprint : bool, optional
        If True, directly return the non-flat structuring element and
        boolean footprint for the rolling ball algorithm. The structure
        is the intensity difference from the center (apex), giving how
        much lower each position is than the apex (center = 0).
        Default is False.

    Returns
    -------
    kernel : cupy.ndarray
        The kernel containing the surface intensity of the top half
        of the ellipsoid. If ``structure_and_footprint=True``, this
        is the non-flat structuring element for grey erosion.
    footprint : cupy.ndarray (only if ``structure_and_footprint=True``)
        Boolean array that is True where the kernel is valid (inside
        the ellipsoid) and False elsewhere.

    See Also
    --------
    rolling_ball
    ball_kernel
    """
    shape = np.asarray(shape)
    semi_axis = np.clip(shape // 2, 1, None).astype(np.int32)
    ndim = len(semi_axis)
    dtype = cp.dtype(dtype)

    # Compute output shape: 2 * semi_axis + 1 for each dimension
    out_shape = tuple(2 * s + 1 for s in semi_axis)
    total_size = int(np.prod(out_shape))

    # Get or create the kernel for this ndim
    kernel_func = _get_ellipsoid_kernel_elementwise(
        ndim, structure_and_footprint
    )

    # Allocate output
    kernel = cp.empty(total_size, dtype=dtype)

    # Build arguments: size_*, semi_*, intensity
    sizes = [int(s) for s in out_shape]
    semis = [int(s) for s in semi_axis]
    intensity_val = dtype.type(intensity)

    if structure_and_footprint:
        footprint = cp.empty(total_size, dtype=cp.bool_)
        kernel_func(*sizes, *semis, intensity_val, kernel, footprint)
        return kernel.reshape(out_shape), footprint.reshape(out_shape)
    else:
        kernel_func(*sizes, *semis, intensity_val, kernel)
        return kernel.reshape(out_shape)


def _rolling_ball_exact(image, radius, kernel, mode="constant", nansafe=False):
    """Exact rolling ball using grey_erosion with non-flat structuring element.

    The result matches the scikit-image implementation exactly.
    """

    image = cp.asarray(image)
    float_type = cp.float32 if image.dtype.itemsize <= 4 else cp.float64
    img = image.astype(float_type, copy=False)

    if kernel is None:
        # kernel = ball_kernel(radius, image.ndim, float_type, False)
        structure, footprint = ball_kernel(radius, image.ndim, float_type, True)
    elif isinstance(kernel, tuple) and len(kernel) == 2:
        structure, footprint = kernel
    else:
        kernel_center = tuple(s // 2 for s in kernel.shape)
        center_intensity = kernel[kernel_center]

        # Intensity difference: how much lower is each position than apex
        intensity_difference = center_intensity - kernel
        intensity_difference[kernel == cp.inf] = cp.inf

        # Create footprint (True where kernel is valid)
        footprint = kernel != cp.inf

        # Grey erosion: min(img + intensity_diff) = min(img - structure)
        # where structure = -intensity_difference
        structure = -intensity_difference
        # Structure values outside footprint don't matter

    # For nansafe mode, create mask of valid (non-NaN) pixels
    # NaN pixels will retain their value in the output
    valid_mask = ~cp.isnan(img) if nansafe else None

    background = ndi.grey_erosion(
        img,
        footprint=footprint,
        structure=structure,
        mode=mode,
        cval=cp.inf,
        mask=valid_mask,
    )

    return background.astype(image.dtype, copy=False)


def _rolling_ball_layered(image, radius, num_layers):
    """Approximate rolling ball using layered flat disks.

    A sphere is approximated by stacking flat disks at different heights.
    At distance h from apex: disk radius r = sqrt(h * (2R - h)), offset = h.
    """

    image = cp.asarray(image)
    float_type = np.float32 if image.dtype.itemsize <= 4 else np.float64
    img = image.astype(float_type, copy=False)

    if num_layers is None:
        # Use roughly one layer per four pixels of radius
        # Cap at 32 layers for very large radii
        num_layers = max(5, min(radius // 4, 32))

    # Sample distances from apex (h=0) to equator (h=R)
    heights = np.linspace(0, radius, num_layers + 1)[1:]

    background = cp.full_like(img, cp.inf)

    prev_disk_radius = -1
    for h in heights:
        # Disk radius at distance h from apex:
        # r = sqrt(R² - (R-h)²) = sqrt(h * (2R - h))
        disk_radius = int(np.round(np.sqrt(h * (2 * radius - h))))

        if disk_radius < 1:
            continue

        # Skip if same radius as previous (optimization)
        if disk_radius == prev_disk_radius:
            continue
        prev_disk_radius = disk_radius

        # Intensity offset = distance from apex
        offset = float(h)

        # Flat erosion with disk footprint
        # Use decomposition for efficiency with larger disks
        if disk_radius <= 2:
            fp = disk(disk_radius, decomposition=None)
        else:
            fp = disk(disk_radius, decomposition="sequence")

        eroded = erosion(img, footprint=fp, mode="constant", cval=float(cp.inf))

        # Add offset and take minimum
        background = cp.minimum(background, eroded + offset)

    return background.astype(image.dtype, copy=False)


def _rolling_ball_tophat(image, radius):
    """Fast approximation using morphological opening with a flat disk.

    This is the fastest but least accurate method. It uses a flat disk
    structuring element instead of a spherical ball.
    """

    image = cp.asarray(image)

    # Use disk with decomposition for efficiency
    fp = disk(radius, decomposition="sequence")

    # Opening approximates the background
    background = opening(image, footprint=fp)

    return background


@deprecate_parameter(
    "num_threads",
    new_name="workers",
    start_version="26.02",
    stop_version="26.08",
)
def rolling_ball(
    image,
    *,
    radius=100,
    kernel=None,
    nansafe=False,
    num_threads=DEPRECATED,
    workers=None,
    algorithm="exact",
    num_layers=None,
):
    """Estimate background intensity using the rolling ball algorithm.

    This function estimates the background intensity of a grayscale image
    by conceptually "rolling a ball" under the image surface. At each
    position, the ball's apex gives the resulting background intensity.

    Parameters
    ----------
    image : cupy.ndarray
        The image to be filtered.
    radius : int, optional
        Radius of the ball-shaped kernel to be rolled under the
        image landscape. Used only if ``kernel`` is None. Default is 100.
    kernel : ndarray, or tuple of (ndarray, ndarray), optional
        An alternative way to specify the rolling ball, as an arbitrary
        kernel. It must have the same number of axes as ``image``.
        Use ``ball_kernel`` or ``ellipsoid_kernel`` to create custom
        kernels. If a tuple, the first element is the structure (non-flat
        structuring element) and the second element is the footprint
        (boolean array that is True where the kernel is valid and False
        elsewhere). Only used when ``algorithm="exact"``.
    nansafe : bool, optional
        If True, NaN values in the image are handled safely.
        **Note**: The behavior of this mode is improved over the implementation
        in scikit-image. For example, in scikit-image, any NaN pixels
        contaminate a radius around them. In cuCIM, NaN pixels will retain
        their NaN value in the output, and neighbors of NaN pixels will only
        use valid (non-NaN) values for their computation.

    Other Parameters
    ----------------
    workers : int, optional
        cuCIM ignores this parameter (it is used by scikit-image for the number
        of CPU threads to use).
    algorithm : {"exact", "layered", "tophat"}, optional
        **cuCIM-specific parameter (not available in scikit-image).**
        The algorithm to use for background estimation. Only used when
        ``kernel=None``; if a kernel is provided, the "exact" algorithm
        is always used.

        - ``"exact"``: Uses grey_erosion with a non-flat (spherical)
          structuring element. This matches scikit-image's implementation
          exactly but may be slow for large radii.
        - ``"layered"``: Approximates the ball using multiple flat disk
          erosions at different heights. Faster than exact for large radii
          with controllable accuracy via ``num_layers``.
        - ``"tophat"``: Uses morphological opening with a flat disk.
          Fastest method but least accurate as it doesn't account for
          the ball's curvature.

        Default is ``"exact"``.
    num_layers : int, optional
        **cuCIM-specific parameter (not available in scikit-image).**
        Number of layers for the ``"layered"`` algorithm. More layers
        give better accuracy but slower computation. If None, uses
        ``max(5, min(radius // 4, 32))``. Ignored for other algorithms.

    Returns
    -------
    background : cupy.ndarray
        The estimated background of the image.

    Notes
    -----
    cuCIM and scikit-image implement a general version of this rolling-ball
    algorithm, which allows you to not just use balls, but arbitrary shapes
    as kernel and works on n-dimensional ndimages. This allows you to directly
    filter RGB images or filter image stacks along any (or all) spatial
    dimensions.

    This implementation assumes that dark pixels correspond to the
    background. If you have a bright background, invert the image before
    passing it to this function, then invert the result. For example::

        from cucim.skimage import util
        image_inverted = util.invert(image)
        background_inverted = rolling_ball(image_inverted, radius=radius)
        background = util.invert(background_inverted)
        filtered_image = image - background

    For best results, the radius should be larger than the typical size
    of the foreground features of interest.

    This algorithm is sensitive to noise (particularly salt-and-pepper
    noise). Consider applying mild Gaussian smoothing before using this
    function if noise is present.

    References
    ----------
    .. [1] Sternberg, Stanley R. "Biomedical image processing." Computer 1
           (1983): 22-34. :DOI:`10.1109/MC.1983.1654163`

    Examples
    --------
    >>> import cupy as cp
    >>> from skimage import data
    >>> from cucim.skimage.restoration import rolling_ball
    >>> image = cp.asarray(data.coins())
    >>> background = rolling_ball(image, radius=100)
    >>> filtered_image = image - background

    Use the layered approximation for faster processing:

    >>> background = rolling_ball(image, radius=100, algorithm="layered")

    Use the top-hat approximation for fastest (but less accurate) results:

    >>> background = rolling_ball(image, radius=100, algorithm="tophat")

    See Also
    --------
    ball_kernel : Create a spherical kernel
    ellipsoid_kernel : Create an ellipsoidal kernel
    """
    valid_algorithms = ("exact", "layered", "tophat")
    if algorithm not in valid_algorithms:
        raise ValueError(
            f"algorithm must be one of {valid_algorithms}, got {algorithm!r}"
        )

    if algorithm == "exact":
        return _rolling_ball_exact(image, radius, kernel, nansafe=nansafe)
    elif algorithm == "layered":
        if kernel is not None:
            raise ValueError(
                "kernel parameter is only supported with algorithm='exact'"
            )
        if nansafe:
            raise NotImplementedError(
                "nansafe=True is not yet implemented for algorithm='layered'. "
                "Use algorithm='exact' for NaN-safe filtering."
            )
        return _rolling_ball_layered(image, radius, num_layers)
    else:  # algorithm == "tophat"
        if kernel is not None:
            raise ValueError(
                "kernel parameter is only supported with algorithm='exact'"
            )
        if nansafe:
            raise NotImplementedError(
                "nansafe=True is not yet implemented for algorithm='tophat'. "
                "Use algorithm='exact' for NaN-safe filtering."
            )
        return _rolling_ball_tophat(image, radius)
