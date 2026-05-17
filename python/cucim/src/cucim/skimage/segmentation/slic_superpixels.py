# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import os
from collections.abc import Iterable
from warnings import warn

import cupy as cp
import numpy as np
from numpy import random
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import pdist, squareform

from cucim.skimage.color import rgb2lab
from cucim.skimage.filters import gaussian
from cucim.skimage.util import img_as_float, regular_grid

slic_available = True
try:
    from skimage.segmentation.slic_superpixels import (
        _enforce_label_connectivity_cython,
    )
except ImportError:
    slic_available = False


def _get_mask_centroids(mask, n_centroids):
    """Find regularly spaced centroids on a mask."""
    coord = np.array(np.nonzero(mask), dtype=float).T
    rng = random.RandomState(123)

    idx_full = np.arange(len(coord), dtype=int)
    idx = np.sort(
        rng.choice(idx_full, min(n_centroids, len(coord)), replace=False)
    )

    dense_factor = 10
    ndim_spatial = mask.ndim
    n_dense = int((dense_factor**ndim_spatial) * n_centroids)
    if len(coord) > n_dense:
        idx_dense = np.sort(rng.choice(idx_full, n_dense, replace=False))
    else:
        idx_dense = Ellipsis
    centroids, _ = kmeans2(coord[idx_dense], coord[idx], iter=5)

    dist = squareform(pdist(centroids))
    np.fill_diagonal(dist, np.inf)
    closest_pts = dist.argmin(-1)
    steps = abs(centroids - centroids[closest_pts, :]).mean(0)

    return cp.asarray(centroids), tuple(float(s) for s in steps)


def _get_grid_centroids(spatial_shape, n_centroids):
    """Find regularly spaced centroids on the image.

    Parameters
    ----------
    image : 2D, 3D or 4D ndarray
        Input image, which can be 2D or 3D, and grayscale or
        multichannel.
    n_centroids : int
        The (approximate) number of centroids to be returned.

    Returns
    -------
    centroids : 2D ndarray
        The coordinates of the centroids with shape (~n_centroids, 3).
    steps : 1D ndarray
        The approximate distance between two seeds in all dimensions.

    """
    # Approximate when it is faster to compute the grid points on the CPU
    xp = np if n_centroids < 20000 else cp

    slices = regular_grid(spatial_shape, n_centroids)
    grid_vecs = tuple(
        xp.arange(
            sl.start, spatial_shape[i], sl.step if sl.step is not None else 1.0
        )
        for i, sl in enumerate(slices)
    )
    grids_1d = xp.meshgrid(*grid_vecs, indexing="ij")
    centroids = xp.stack(tuple(g.ravel() for g in grids_1d), axis=-1)
    steps = tuple(float(s.step) if s.step is not None else 1.0 for s in slices)
    grid_shape = tuple(int(g.size) for g in grid_vecs)
    if xp != cp:
        centroids = cp.asarray(centroids)
    return centroids, steps, grid_shape


def line_kernel_config(threads_total, block_size=64):
    block = (block_size, 1, 1)
    grid = ((threads_total + block_size - 1) // block_size, 1, 1)
    return block, grid


def box_kernel_config(im_shape, block=None):
    """determine launch parameters"""
    if len(im_shape) == 2:
        if block is None:
            # The 2D CUDA kernel maps threadIdx.x -> image y and
            # threadIdx.y -> image x. Keep threadIdx.y wide so each warp
            # touches consecutive x locations in memory.
            block = (1, 64)
        grid = (
            (im_shape[0] + block[0] - 1) // block[0],
            (im_shape[1] + block[1] - 1) // block[1],
            1,
        )
    else:
        if block is None:
            # block = (z=2,y=4,x=32) was hand tested to be very fast
            # on the Quadro P2000, might not be the fastest config for other
            # cards
            block = (2, 4, 32)
        grid = (
            (im_shape[0] + block[0] - 1) // block[0],
            (im_shape[1] + block[1] - 1) // block[1],
            (im_shape[2] + block[2] - 1) // block[2],
        )
    return block, grid


def _slic(
    image,
    sp_shape,
    sp_grid,
    spacing,
    compactness,
    max_num_iter,
    centers_gpu,
    max_step,
    start_label,
    maximization_algorithm,
    slic_zero,
    mask=None,
    ignore_color=False,
):
    shape_spatial = image.shape[:-1]

    spatial_weight = float(max_step)
    n_centers = centers_gpu.shape[0]
    n_features = image.shape[-1]
    use_mask = mask is not None
    use_atomic_maximization = maximization_algorithm == "atomic"
    maximization_pixels_per_thread = int(
        os.environ.get("CUCIM_SLIC_MAXIMIZATION_PIXELS_PER_THREAD", "8")
    )

    __dirname__ = os.path.dirname(__file__)
    if len(shape_spatial) == 2:
        module_path = os.path.join(__dirname__, "cuda", "slic2d.cu")
    else:
        module_path = os.path.join(__dirname__, "cuda", "slic3d.cu")
    with open(module_path) as f:
        cuda_source = f.read()

    center_block, center_grid = line_kernel_config(n_centers)
    image_block, image_grid = box_kernel_config(shape_spatial)

    ss = spatial_weight * spatial_weight

    cuda_source_defines = f"""
#define N_PIXEL_FEATURES {n_features}
#define START_LABEL {start_label}
#define FLOAT_DTYPE {"double" if image.dtype == np.float64 else "float"}
#define INTERNAL_FLOAT_DTYPE {"double" if image.dtype == np.float64 else "float"}
#define PIXELS_PER_THREAD {maximization_pixels_per_thread}
#define SLIC_ZERO {1 if slic_zero else 0}
#define USE_MASK {1 if use_mask else 0}
#define IGNORE_COLOR {1 if ignore_color else 0}
"""
    cuda_source = (
        '#include "cupy/atomics.cuh"\n'
        + 'extern "C" { '
        + cuda_source_defines
        + cuda_source
        + " }"
    )
    module = cp.RawModule(code=cuda_source, options=("-std=c++11",))
    # gpu_slic_init = module.get_function("init_clusters")
    gpu_slic_expectation = module.get_function(
        "expectation_mask" if use_mask else "expectation"
    )
    if slic_zero:
        gpu_slic_update_max_dist_color = module.get_function(
            "update_max_dist_color"
        )
    if use_atomic_maximization:
        gpu_slic_reset_accumulators = module.get_function("reset_accumulators")
        gpu_slic_accumulate_centers = module.get_function("accumulate_centers")
        gpu_slic_normalize_centers = module.get_function("normalize_centers")
    else:
        gpu_slic_maximization = module.get_function("maximization")

    mask_label = start_label - 1
    labels_gpu = cp.full(shape_spatial, mask_label, dtype=cp.int32)
    max_dist_color_gpu = cp.ones(n_centers, dtype=image.dtype)
    if use_atomic_maximization:
        center_sums_gpu = cp.empty(
            (n_centers, n_features + len(shape_spatial)), dtype=image.dtype
        )
        center_counts_gpu = cp.empty(n_centers, dtype=cp.uint32)
    if use_mask:
        mask = cp.ascontiguousarray(mask, dtype=cp.uint8)

    float_dtype = image.dtype
    spacing = cp.asarray(spacing, dtype=float_dtype)

    # device scalar (passing Python float did not work, so changed to
    # float* in the kernel)
    ss = cp.asarray(ss, dtype=float_dtype)
    if use_atomic_maximization or slic_zero:
        n_pixels = image.size // n_features
        accumulation_block, accumulation_grid = line_kernel_config(
            (n_pixels + maximization_pixels_per_thread - 1)
            // maximization_pixels_per_thread,
            block_size=256,
        )

    for _ in range(max_num_iter):
        expectation_args = (
            image,
            centers_gpu,
            labels_gpu,
            *shape_spatial,
        )
        if use_mask:
            expectation_args += (
                *sp_shape,
                spacing,
                ss,
                max_dist_color_gpu,
                mask,
                n_centers,
            )
        else:
            expectation_args += (
                *sp_shape,
                *sp_grid,
                spacing,
                ss,
                max_dist_color_gpu,
            )
        gpu_slic_expectation(
            image_grid,
            image_block,
            expectation_args,
        )

        if use_atomic_maximization:
            gpu_slic_reset_accumulators(
                center_grid,
                center_block,
                (center_sums_gpu, center_counts_gpu, n_centers),
            )

            gpu_slic_accumulate_centers(
                accumulation_grid,
                accumulation_block,
                (
                    image,
                    labels_gpu,
                    centers_gpu,
                    *shape_spatial,
                    *sp_shape,
                    n_centers,
                    center_sums_gpu,
                    center_counts_gpu,
                    *((mask,) if use_mask else ()),
                ),
            )

            gpu_slic_normalize_centers(
                center_grid,
                center_block,
                (
                    centers_gpu,
                    center_sums_gpu,
                    center_counts_gpu,
                    n_centers,
                ),
            )
        else:
            gpu_slic_maximization(
                center_grid,
                center_block,
                (
                    image,
                    labels_gpu,
                    centers_gpu,
                    *shape_spatial,
                    *sp_shape,
                    n_centers,
                    *((mask,) if use_mask else ()),
                ),
            )
        if slic_zero:
            gpu_slic_update_max_dist_color(
                accumulation_grid,
                accumulation_block,
                (
                    image,
                    labels_gpu,
                    centers_gpu,
                    *shape_spatial,
                    n_centers,
                    max_dist_color_gpu,
                    *((mask,) if use_mask else ()),
                ),
            )

    # TODO (grelee): may want to keep the final centroids for use
    # in GPU-based connectivity enforcement.
    # centroids = centers_gpu[:, -len(shape_spatial):]

    return labels_gpu  # , centroids


def slic(
    image,
    n_segments=100,
    compactness=10.0,
    max_num_iter=10,
    sigma=0,
    spacing=None,
    convert2lab=None,
    enforce_connectivity=True,
    min_size_factor=0.5,
    max_size_factor=3.0,
    slic_zero=False,
    start_label=1,
    mask=None,
    *,
    channel_axis=-1,
    check_finite_and_constant=False,
    maximization_algorithm="atomic",
):
    """Segments image using k-means clustering in Color-(x,y,z) space.
    Parameters
    ----------
    image : 2D, 3D or 4D ndarray
        Input image, which can be 2D or 3D, and grayscale or multichannel
        (see `multichannel` parameter).
    n_segments : int, optional
        The (approximate) number of labels in the segmented output image.
    compactness : float, optional
        Balances color proximity and space proximity. Higher values give
        more weight to space proximity, making superpixel shapes more
        square/cubic.
        This parameter depends strongly on image contrast and on the
        shapes of objects in the image. We recommend exploring possible
        values on a log scale, e.g., 0.01, 0.1, 1, 10, 100, before
        refining around a chosen value.
    max_num_iter : int, optional
        Maximum number of iterations of k-means.
    sigma : float or array-like of floats, optional
        Width of Gaussian smoothing kernel for pre-processing for each
        dimension of the image. The same sigma is applied to each dimension in
        case of a scalar value. Zero means no smoothing.
        Note that `sigma` is automatically scaled if it is scalar and
        if a manual voxel spacing is provided (see Notes section). If
        sigma is array-like, its size must match ``image``'s number
        of spatial dimensions.
    spacing : array-like of floats, optional
        The voxel spacing along each spatial dimension. By default,
        `slic` assumes uniform spacing (same voxel resolution along
        each spatial dimension).
        This parameter controls the weights of the distances along the
        spatial dimensions during k-means clustering.
    multichannel : bool, optional
        Whether the last axis of the image is to be interpreted as multiple
        channels or another spatial dimension.
    convert2lab : bool, optional
        Whether the input should be converted to Lab colorspace prior to
        segmentation. The input image *must* be RGB. Highly recommended.
        This option defaults to ``True`` when ``multichannel=True`` *and*
        ``image.shape[-1] == 3``.
    enforce_connectivity : bool, optional
        Whether the generated segments are connected or not
    min_size_factor : float, optional
        Proportion of the minimum segment size to be removed with respect
        to the supposed segment size ```depth*width*height/n_segments```
    max_size_factor : float, optional
        Proportion of the maximum connected segment size. A value of 3 works
        in most of the cases.
    start_label : int, optional
        The labels' index start. Should be 0 or 1.
    slic_zero: bool, optional
        Run SLIC-zero, the zero-parameter mode of SLIC. [2]_.
    mask : ndarray, optional
        If provided, superpixels are computed only where ``mask`` is ``True``.
        Pixels outside the mask are assigned ``start_label - 1``. The initial
        maskSLIC centroids are currently computed on the host using SciPy's
        ``kmeans2`` implementation, then copied to the device.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

    Extra Parameters
    ----------------
    check_finite_and_constant : bool, optional
        Whether to raise an error if any NaN or infinite values are present in
        the image. This check is always done in the scikit-image implementation
        (for regions inside any provided mask), but has device synchronization
        overhead for CuPy, so is disabled by default in cuCIM. When True, also
        checks for the case where all values in the image are constant.
    maximization_algorithm : {"scan", "atomic"}, optional
        Algorithm used to update cluster centers after assigning pixels to
        their closest center. ``"atomic"`` uses a faster pixel-parallel
        accumulation implementation and is the default for performance.
        Floating point atomic accumulation order can make the exact
        segmentation non-deterministic across repeated runs. Use ``"scan"``
        when exact deterministic results are required.

    Returns
    -------
    labels : 2D or 3D array
        Integer mask indicating segment labels.

    Raises
    ------
    ValueError
        If ``convert2lab`` is set to ``True`` but the last array
        dimension is not of length 3.
    ValueError
        If ``start_label`` is not 0 or 1.
    ValueError
        If ``image.ndim`` is not 2, 3 or 4.
    ValueError
        If ``image`` is 2D but ``channel_axis`` is -1 (the default).

    Notes
    -----
    * If `sigma > 0`, the image is smoothed using a Gaussian kernel prior to
      segmentation.

    * If `sigma` is scalar and `spacing` is provided, the kernel width is
      divided along each dimension by the spacing. For example, if ``sigma=1``
      and ``spacing=[5, 1, 1]``, the effective `sigma` is ``[0.2, 1, 1]``. This
      ensures sensible smoothing for anisotropic images.

    * The image is rescaled to be in [0, 1] prior to processing.

    * Images of shape (M, N, 3) are interpreted as 2D RGB images by default. To
      interpret them as 3D with the last dimension having length 3, use
      `channel_axis=None`.

    * `start_label` is introduced to handle the issue [3]_. Label indexing
      starts at 1 by default.

    References
    ----------
    .. [1] Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi,
        Pascal Fua, and Sabine Süsstrunk, SLIC Superpixels Compared to
        State-of-the-art Superpixel Methods, TPAMI, May 2012.
        :DOI:`10.1109/TPAMI.2012.120`
    .. [2] https://www.epfl.ch/labs/ivrl/research/slic-superpixels/#SLICO
    .. [3] Irving, Benjamin. "maskSLIC: regional superpixel generation with
           application to local pathology characterisation in medical images.",
           2016, :arXiv:`1606.09518`
    .. [4] https://github.com/scikit-image/scikit-image/issues/3722

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage import data
    >>> from cucim.skimage.segmentation import slic
    >>> from skimage import data
    >>> img = cp.asarray(data.astronaut()) # 2D RGB image
    >>> segments = slic(img, n_segments=100, compactness=10)
    Increasing the compactness parameter yields more square regions:
    >>> segments = slic(img, n_segments=100, compactness=20)

    To segment single channel 3D volumes
    >>> vol = data.binary_blobs(length=50, n_dim=3, rng=2)
    >>> labels = slic(vol, n_segments=100, multichannel=False, compactness=0.1)
    """
    if not slic_available:
        raise ImportError(
            "Could not import the private _enforce_label_connectivity_cython "
            "function from scikit-image version '{skimage.__version__}', so "
            "the `slic` algorithm is unavailable."
        )
    if maximization_algorithm not in {"scan", "atomic"}:
        raise ValueError(
            "maximization_algorithm must be either 'scan' or 'atomic'"
        )
    if image.ndim not in [2, 3, 4]:
        raise ValueError(
            "input image must be either 2, 3, or 4 dimensional.\n"
            f"The input image.ndim is {image.ndim}"
        )
    if image.ndim == 2 and channel_axis is not None:
        raise ValueError(
            f"channel_axis={channel_axis} indicates multichannel, which is not "
            "supported for a two-dimensional image; use channel_axis=None if "
            "the image is grayscale"
        )

    image = img_as_float(image)
    float_dtype = image.dtype

    # copy=True so subsequent in-place operations do not modify the
    # function input
    image = image.astype(float_dtype, copy=True)
    use_mask = mask is not None
    if use_mask:
        mask = cp.ascontiguousarray(mask, dtype=cp.bool_)
        if channel_axis is None:
            if mask.shape != image.shape:
                raise ValueError("image and mask should have the same shape.")
            mask_values = mask
        else:
            channel_axis_norm = channel_axis % image.ndim
            spatial_shape = (
                image.shape[:channel_axis_norm]
                + image.shape[channel_axis_norm + 1 :]
            )
            if mask.shape != spatial_shape:
                raise ValueError("image and mask should have the same shape.")
            mask_values = cp.expand_dims(mask, axis=channel_axis_norm)
            mask_values = cp.broadcast_to(mask_values, image.shape)
        image_values = image[mask_values]
    else:
        image_values = image

    # Rescale image to [0, 1] to make choice of compactness insensitive to
    # input image scale.
    imin = image_values.min()
    imax = image_values.max()
    if check_finite_and_constant:
        imin_host = float(imin)
        imax_host = float(imax)
        if np.isnan(imin_host):
            raise ValueError("unmasked NaN values in image are not supported")
        if np.isinf(imin_host) or np.isinf(imax_host):
            raise ValueError(
                "unmasked infinite values in image are not supported"
            )
        constant_valued = imax_host == imin_host
    else:
        constant_valued = False
    image -= imin
    if not constant_valued:
        image /= imax - imin

    dtype = image.dtype

    is_2d = False
    multichannel = channel_axis is not None
    if image.ndim == 2:
        # 2D grayscale image: add channel axis
        image = image[..., cp.newaxis]
        is_2d = True
    elif image.ndim == 3 and multichannel:
        is_2d = True
    elif image.ndim == 3 and not multichannel:
        # Add channel as single last dimension
        image = image[..., cp.newaxis]

    if multichannel and (convert2lab or convert2lab is None):
        if image.shape[-1] != 3 and convert2lab:
            raise ValueError("Lab colorspace conversion requires a RGB image.")
        elif image.shape[-1] == 3:
            image = rgb2lab(image)

    if start_label not in [0, 1]:
        raise ValueError("start_label should be 0 or 1.")

    # omit the channel dimension
    spatial_shape = image.shape[:-1]
    ndim_spatial = len(spatial_shape)
    if use_mask and mask.shape != spatial_shape:
        raise ValueError("image and mask should have the same shape.")

    # initialize cluster centroids for desired number of segments
    update_centroids = False
    if use_mask:
        centroids, steps = _get_mask_centroids(cp.asnumpy(mask), n_segments)
        sp_grid = ()
        update_centroids = True
    else:
        centroids, steps, sp_grid = _get_grid_centroids(
            spatial_shape, n_segments
        )

    n_centroids = centroids.shape[0]
    segments = cp.ascontiguousarray(
        cp.concatenate(
            [cp.zeros((n_centroids, image.shape[-1])), centroids], axis=-1
        ),
        dtype=float_dtype,
    )

    # Scaling of ratio in the same way as in the SLIC paper so the
    # values have the same meaning
    max_step = max(steps)
    ratio = 1.0 / compactness
    image *= ratio

    # TODO (grelee):
    #   check step and ratio parameters and grid generation to make it closely
    #   match scikit-image

    sp_shape = tuple(max(1, math.ceil(step)) for step in steps)
    n_centers = n_centroids

    # TODO(grelee): spacing currently on CPU for use with jinja2.Template
    #   may make this a device array and kernel argument later
    if spacing is None:
        spacing = np.ones(ndim_spatial, dtype=dtype)
    elif isinstance(spacing, Iterable):
        spacing = np.asarray(spacing, dtype=dtype)
        if is_2d:
            if spacing.size != 2:
                if spacing.size == 3:
                    warn(
                        "Input image is 2D: spacing number of elements must "
                        "be 2. In the future, a ValueError will be raised.",
                        FutureWarning,
                        stacklevel=2,
                    )
                    # drop channel dimensions
                    spacing = spacing[:-1]
                else:
                    raise ValueError(
                        f"Input image is 2D, but spacing has {spacing.size} "
                        "elements (expected 2)."
                    )
        elif spacing.size != 3:
            raise ValueError(
                f"Input image is 3D, but spacing has {spacing.size} elements "
                "(expected 3)."
            )
        spacing = np.ascontiguousarray(spacing, dtype=dtype)
    else:
        raise TypeError("spacing must be None or iterable.")

    if np.isscalar(sigma):
        sigma = np.array((sigma,) * ndim_spatial, dtype=dtype)
        sigma /= spacing
    elif isinstance(sigma, Iterable):
        sigma = np.asarray(sigma, dtype=dtype)
        if is_2d:
            if sigma.size != 2:
                if sigma.size == 3:
                    warn(
                        "Input image is 2D: sigma number of elements must be "
                        "2. In the future, a ValueError will be raised.",
                        FutureWarning,
                        stacklevel=2,
                    )
                    # drop channel dimensions
                    sigma = sigma[:-1]
                else:
                    raise ValueError(
                        f"Input image is 2D, but sigma has {sigma.size} "
                        "elements (expected 2)."
                    )
        elif sigma.size != 3:
            raise ValueError(
                f"Input image is 3D, but sigma has {sigma.size} elements "
                "(expected 3)."
            )

    if (sigma > 0).any():
        # add zero smoothing for channel dimension
        # TODO (grelee): Fix scikit-image bug:
        #    does not respect user-provided channel_axis!
        sigma = list(sigma) + [0]
        image = gaussian(image, sigma=sigma, mode="reflect")

    if update_centroids:
        _slic(
            image,
            sp_shape,
            sp_grid,
            spacing,
            compactness,
            max_num_iter,
            segments,
            max_step,
            start_label,
            maximization_algorithm,
            slic_zero,
            mask=mask,
            ignore_color=True,
        )

    labels = _slic(
        image,
        sp_shape,
        sp_grid,
        spacing,
        compactness,
        max_num_iter,
        segments,
        max_step,
        start_label,
        maximization_algorithm,
        slic_zero,
        mask=mask,
    )
    if enforce_connectivity:
        if use_mask:
            segment_size = int(mask.sum()) / n_centers
        else:
            segment_size = math.prod(spatial_shape) / n_centers
        min_size = int(min_size_factor * segment_size)
        max_size = int(max_size_factor * segment_size)

        labels = cp.asnumpy(labels).astype(cp.intp, copy=False)

        if is_2d:
            # prepend singleton axis for 2D case
            # (Cython function only supports 3D spatial images)
            labels = labels[cp.newaxis, ...]
        labels = _enforce_label_connectivity_cython(
            labels, min_size, max_size, start_label=start_label
        )
        if is_2d:
            labels = labels[0]
        labels = cp.asarray(labels)

    return labels
