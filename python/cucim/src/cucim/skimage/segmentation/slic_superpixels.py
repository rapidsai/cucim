import math
import os
from collections.abc import Iterable
from warnings import warn

import cupy as cp
import numpy as np

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
    if xp != cp:
        centroids = cp.asarray(centroids)
    return centroids, steps


def line_kernel_config(threads_total, block_size=64):
    block = (block_size, 1, 1)
    grid = ((threads_total + block_size - 1) // block_size, 1, 1)
    return block, grid


def box_kernel_config(im_shape, block=None):
    """determine launch parameters"""
    if len(im_shape) == 2:
        if block is None:
            block = (1, 1)  # (8, 32)
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
):
    shape_spatial = image.shape[:-1]

    spatial_weight = float(max(sp_shape))
    n_centers = int(math.prod(sp_grid))
    n_features = image.shape[-1]

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
#define N_PIXEL_FEATURES { n_features }
#define START_LABEL { start_label }
#define FLOAT_DTYPE { "double" if image.dtype == np.float64 else "float"}
"""
    cuda_source = 'extern "C" { ' + cuda_source_defines + cuda_source + " }"
    module = cp.RawModule(code=cuda_source, options=("-std=c++11",))
    # gpu_slic_init = module.get_function("init_clusters")
    gpu_slic_expectation = module.get_function("expectation")
    gpu_slic_maximization = module.get_function("maximization")

    labels_gpu = cp.zeros(shape_spatial, dtype=cp.uint32)

    float_dtype = image.dtype
    spacing = cp.asarray(spacing, dtype=float_dtype)

    # device scalar (passing Python float did not work, so changed to
    # float* in the kernel)
    ss = cp.asarray(ss, dtype=float_dtype)

    for _ in range(max_num_iter):
        gpu_slic_expectation(
            image_grid,
            image_block,
            (
                image,
                centers_gpu,
                labels_gpu,
                *shape_spatial,
                *sp_shape,
                *sp_grid,
                spacing,
                ss,
            ),
        )
        cp.cuda.runtime.deviceSynchronize()

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
            ),
        )
        cp.cuda.runtime.deviceSynchronize()

    # TODO (grelee): may want to keep the final centroids for use
    # in GPU-based connectivity enforcement.
    # centroids = centers_gpu[:, -len(shape_spatial):]

    return labels_gpu  # , centroids


# change default to spacing=None
#
# change multichannel -> channel_axis
#
# update order of kwargs to match skimage
#
# Possibly add support for the following options
#     sigma=0,
#
#     slic_zero=False,
#     start_label=1,
#     mask=None,
#     *,
#     channel_axis=-1,


# * added sigma parameter
# * added start_label parameter
# * remove pycuda code paths
# * changed argument default values to match current scikit-image


def slic(
    image,
    n_segments=100,
    compactness=1.0,
    max_num_iter=10,
    sigma=0,
    spacing=None,
    convert2lab=None,
    enforce_connectivity=True,
    min_size_factor=0.5,
    max_size_factor=3.0,
    start_label=1,
    mask=None,
    *,
    channel_axis=-1,
    check_finite_and_constant=False,
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
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.
    mask : ndarray, optional
        Masked SLIC is currently unimplemented in CuPy. A
        ``NotImplementedError`` will be raised if `mask` is not ``None``.

    Extra Parameters
    ----------------
    check_finite_and_constant : bool, optional
        Whether to raise an error if any NaN or infinite values are present in
        the image. This check is always done in the scikit-image implementation
        (for regions inside any provided mask), but has device synchronization
        overhead for CuPy, so is disabled by default in cuCIM. When True, also
        checks for the case where all values in the image are constant.

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
        raise NotImplementedError("masked SLIC not implemented")
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

    # initialize cluster centroids for desired number of segments
    # update_centroids = False
    centroids, steps = _get_grid_centroids(spatial_shape, n_segments)

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

    ndim_spatial = 2 if is_2d else 3
    power = 1 / ndim_spatial
    sp_size = int(math.ceil((math.prod(spatial_shape) / n_segments) ** power))
    # don't allow sp_shape to be larger than image sides
    sp_shape = tuple(min(s, sp_size) for s in spatial_shape)
    sp_grid = tuple(
        (im_sz + sz - 1) // sz for im_sz, sz in zip(spatial_shape, sp_shape)
    )
    n_centers = math.prod(sp_grid)

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
    )
    if enforce_connectivity:
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
