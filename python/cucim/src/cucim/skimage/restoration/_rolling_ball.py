# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

"""
Rolling ball background subtraction algorithm.

This module provides a GPU-accelerated implementation of the rolling ball
algorithm for background estimation and subtraction.
"""

import cupy as cp
import numpy as np

from cucim.skimage._vendored import ndimage as ndi
from cucim.skimage.transform import resize

__all__ = ["rolling_ball", "ball_kernel", "ellipsoid_kernel"]


# =====================================
# GPU-based ball kernel implementations
# =====================================


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

    code = f"""
        // Convert flat index (i) to {ndim}D coordinates
        {index_code}

        // Convert to coordinates relative to center
        {coord_code}

        // Compute distance squared
        T sum_sq = {sum_sq_terms};
        T r_sq = radius * radius;
    """

    if structure_and_footprint:
        # Compute center_val - height, where center_val = radius
        # This gives how much lower each position is than the apex
        # Also output footprint (True where valid, False where inf)
        # radius is T (float) to support non-integer radii
        input_params = "T radius, int32 size, int32 half_size, T center_val"
        output_params = "T out, bool footprint_out"
        kernel_name = f"ball_kernel_diff_{ndim}d"

        code += """
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

        code += """
        // Compute ball surface height
        if (sum_sq <= r_sq) {{
            out = sqrt(r_sq - sum_sq);
        }} else {{
            out = CUDART_INF;
        }}
        """

    return cp.ElementwiseKernel(input_params, output_params, code, kernel_name)


def ball_kernel(
    radius, ndim=2, *, dtype=cp.float64, structure_and_footprint=False
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

    Other Parameters
    ----------------
    dtype : dtype, optional
        The data type of the kernel. Default is ``cp.float64``.
        This parameter does not exist in scikit-image.
    structure_and_footprint : bool, optional
        If True, directly return the non-flat structuring element and
        boolean footprint for the rolling ball algorithm. The structure
        is the intensity difference from the center (apex), giving how
        much lower each position is than the apex (center = 0).
        Default is False. This parameter does not exist in scikit-image.

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

    input_params = f"{size_params}, {semi_params}, T intensity"

    code = f"""
        // Convert flat index (i) to {ndim}D coordinates
        {index_code}

        // Convert to coordinates relative to center
        {coord_code}

        // Compute normalized sum of squares
        T norm_sq = {norm_sq_terms};
    """

    if structure_and_footprint:
        output_params = "T out, bool footprint_out"
        kernel_name = f"ellipsoid_kernel_diff_{ndim}d"

        code += """
        // Compute ellipsoid surface height and footprint
        if (norm_sq <= (T)1.0) {{
            out = intensity * sqrt((T)1.0 - norm_sq) - intensity;
            footprint_out = true;
        }} else {{
            out = CUDART_INF;
            footprint_out = false;
        }} """
    else:
        output_params = "T out"
        kernel_name = f"ellipsoid_kernel_{ndim}d"

        code += """
        // Compute ellipsoid surface height
        if (norm_sq <= (T)1.0) {{
            out = intensity * sqrt((T)1.0 - norm_sq);
        }} else {{
            out = CUDART_INF;
        }}"""

    return cp.ElementwiseKernel(input_params, output_params, code, kernel_name)


def ellipsoid_kernel(
    shape, intensity, *, dtype=cp.float64, structure_and_footprint=False
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

    Other Parameters
    ----------------
    dtype : dtype, optional
        The data type of the kernel. Default is ``cp.float64``.
        This parameter does not exist in scikit-image.
    structure_and_footprint : bool, optional
        If True, directly return the non-flat structuring element and
        boolean footprint for the rolling ball algorithm. The structure
        is the intensity difference from the center (apex), giving how
        much lower each position is than the apex (center = 0).
        Default is False. This parameter does not exist in scikit-image.

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


def _rolling_ball_exact(
    image, radius, kernel, mode="constant", nansafe=False, downscale=None
):
    """Exact rolling ball using grey_erosion with non-flat structuring element.

    The result matches the scikit-image implementation exactly.
    """

    image = cp.asarray(image)
    float_type = cp.float32 if image.dtype.itemsize <= 4 else cp.float64
    img = image.astype(float_type, copy=False)

    if downscale is not None:
        # Scale the radius proportionally (minimum 1)
        new_radius = max(1, int(round(radius / downscale)))
        structure_scale_factor = radius / new_radius
    else:
        new_radius = radius
        structure_scale_factor = 1

    if kernel is None:
        # kernel = ball_kernel(radius, image.ndim, float_type, False)
        structure, footprint = ball_kernel(
            new_radius,
            image.ndim,
            dtype=float_type,
            structure_and_footprint=True,
        )
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

    if structure_scale_factor != 1:
        structure *= structure_scale_factor

    background = ndi.grey_erosion(
        img,
        footprint=footprint,
        structure=structure,
        mode=mode,
        cval=cp.inf,
    )

    return background.astype(image.dtype, copy=False)


def rolling_ball(
    image,
    *,
    radius=100,
    kernel=None,
    nansafe=False,
    workers=None,
    downscale=None,
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
        kernels.
        For cuCIM, this also accepts a tuple of cupy.ndarray where the
        first element is the structure (non-flat structuring element)
        and the second element is the footprint (boolean array that is
        True where the kernel is valid and False elsewhere). The
        `ball_kernel` and `ellipsoid_kernel` functions return a tuple of
        (structure, footprint) when `structure_and_footprint` is ``True``.
        This avoids the overhead of a separate CUDA kernel just to rescale
        the structure.
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
    downscale : float, optional
        **cuCIM-specific parameter (not available in scikit-image).**
        If provided and greater than 1, the image is downscaled by this
        factor before processing, then the result is upscaled back to the
        original size. This can significantly speed up processing for large
        images, especially with large radii. The radius is automatically
        scaled proportionally. For example, ``downscale=2`` will halve the
        image dimensions and the radius before processing. Default is None
        (no downscaling).

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

    Use the layered approximation for faster (but less accurate) processing:

    >>> background = rolling_ball(image, radius=100, algorithm="layered")

    Speed up processing by downscaling the image first:

    >>> background = rolling_ball(image, radius=100, downscale=2)

    See Also
    --------
    ball_kernel : Create a spherical kernel
    ellipsoid_kernel : Create an ellipsoidal kernel
    """

    image = cp.asarray(image)
    original_shape = image.shape
    original_dtype = image.dtype

    if nansafe:
        raise NotImplementedError("nansafe mode is not yet supported by cuCIM")

    # Handle downscaling
    if downscale is None:
        downscale = 1
    if downscale < 1:
        raise ValueError(f"downscale must be >= 1, got {downscale}")
    elif downscale > 1:
        if kernel is not None:
            raise ValueError(
                "downscale parameter cannot be used with a custom kernel. "
                "Use radius parameter instead."
            )
        # Compute downscaled shape
        downscaled_shape = tuple(
            max(1, int(round(s / downscale))) for s in original_shape
        )
        # Downscale the image (anti_aliasing smooths before downsampling)
        image = resize(
            image,
            downscaled_shape,
            order=0,
            mode="reflect",
            anti_aliasing=False,
            preserve_range=True,
        )

    background = _rolling_ball_exact(
        image, radius, kernel, nansafe=nansafe, downscale=downscale
    )
    # Handle upscaling back to original size
    if downscale > 1:
        background = resize(
            background,
            original_shape,
            order=1,
            mode="reflect",
            anti_aliasing=False,
            preserve_range=True,
        )
        background = background.astype(original_dtype, copy=False)

    return background
