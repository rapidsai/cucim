import math
import os

import cupy as cp
from cupyx.scipy.ndimage import gaussian_laplace
from scipy import spatial

from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type, check_nD
from ..transform import integral_image
from ..util import img_as_float
from ._hessian_det_appx import _hessian_matrix_det
from .peak import peak_local_max

# This basic blob detection algorithm is based on:
# http://www.cs.utah.edu/~jfishbau/advimproc/project1/ (04.04.2013)
# Theory behind: https://en.wikipedia.org/wiki/Blob_detection (04.04.2013)


def _dtype_to_cuda_float_type(dtype):
    """Maps a float data type from cupy to cuda.

    Returns a cuda (c++) data type.

    Parameters
    ----------
    dtype : cupy dtype
        A cupy dtype from type float.

    Returns
    -------
    cpp_float_type : cuda (c++) data type
        Supported cuda data type
    """
    cpp_float_types = {
        cp.float32: 'float',
        cp.float64: 'double',
    }
    dtype = cp.dtype(dtype)
    if dtype.type not in cpp_float_types:
        raise ValueError(f"unrecognized dtype: {dtype.type}")
    return cpp_float_types[dtype.type]


@cp.memoize()
def _get_prune_blob_rawmodule(dtype, large_int) -> cp.RawModule:
    """Loads the kernel according to dtype /cuda/blob.cu
    Returns a cupy RawModule.

    Parameters
    ----------
    dtype : cupy dtype
        Only the cupy dtypes float32 and float64 are supported.

    Returns
    -------
    RawModule : cupy RawModule
        A cupy RawModule containing the __global__ functions `_prune_blobs`
        and `_prune_blobs_kdtree`.
    """
    blob_t = _dtype_to_cuda_float_type(dtype)
    int_t = 'long long' if large_int else 'int'

    _preamble = f"""
#define BLOB_T {blob_t}
#define INT_T {int_t}
    """

    kernel_directory = os.path.join(
        os.path.normpath(os.path.dirname(__file__)), 'cuda'
    )
    with open(os.path.join(kernel_directory, "blob.cu"), 'rt') as f:
        _code = f.read()

    return cp.RawModule(
        code=_preamble + _code,
        options=('--std=c++11',),
        name_expressions=["_prune_blobs", "_prune_blobs_kdtree"]
    )


def _prune_blobs(blobs_array, overlap, *, sigma_dim=1):
    """Eliminated blobs with area overlap.

    Parameters
    ----------
    blobs_array : ndarray
        A 2d array with each row representing 3 (or 4) values,
        ``(row, col, sigma)`` or ``(pln, row, col, sigma)`` in 3D,
        where ``(row, col)`` (``(pln, row, col)``) are coordinates of the blob
        and ``sigma`` is the standard deviation of the Gaussian kernel which
        detected the blob.
        This array must not have a dimension of size 0.
    overlap : float
        A value between 0 and 1. If the fraction of area overlapping for 2
        blobs is greater than `overlap` the smaller blob is eliminated.
    sigma_dim : int, optional
        The number of columns in ``blobs_array`` corresponding to sigmas rather
        than positions.

    Returns
    -------
    A : ndarray
        `blobs_array` with overlapping blobs removed.
    """

    # from here, the kernel does the calculation
    blobs_module = _get_prune_blob_rawmodule(blobs_array.dtype,
                                             max(blobs_array.shape) > 2**31)
    _prune_blobs_kernel = blobs_module.get_function("_prune_blobs")

    block_size = 64
    grid_size = int(math.ceil(blobs_array.shape[0] / block_size))
    _prune_blobs_kernel((grid_size,), (block_size,),
                        (blobs_array.ravel(),
                         int(blobs_array.shape[0]),
                         int(blobs_array.shape[1]),
                         float(overlap),
                         int(sigma_dim))
                        )
    return blobs_array[blobs_array[:, -1] > 0, :]


def _prune_blobs_kdtree(blobs_array, overlap, *, sigma_dim=1):
    """Eliminated blobs with area overlap.

    Parameters
    ----------
    blobs_array : ndarray
        A 2d array with each row representing 3 (or 4) values,
        ``(row, col, sigma)`` or ``(pln, row, col, sigma)`` in 3D,
        where ``(row, col)`` (``(pln, row, col)``) are coordinates of the blob
        and ``sigma`` is the standard deviation of the Gaussian kernel which
        detected the blob.
        This array must not have a dimension of size 0.
    overlap : float
        A value between 0 and 1. If the fraction of area overlapping for 2
        blobs is greater than `overlap` the smaller blob is eliminated.
    sigma_dim : int, optional
        The number of columns in ``blobs_array`` corresponding to sigmas rather
        than positions.

    Returns
    -------
    A : ndarray
        `array` with overlapping blobs removed.

    Notes
    -----
    This function is using scipy's `spatial.cKDTree`, so the blobs_array will
    be copied to CPU RAM and the CPU does the job. For larger blobs_array the
    GPU does this faster. This function will be updated as soon as the function
    `cupyx.scipy.spatial.cKDTree` is implemented.
    """

    sigma = blobs_array[:, -sigma_dim:].max()
    distance = 2 * sigma * math.sqrt(blobs_array.shape[1] - sigma_dim)
    # TODO: replace with cupyx.scipy.spatial.cKDTree as soon as the function is
    #       implemented
    tree = spatial.cKDTree(cp.asnumpy(blobs_array[:, :-sigma_dim]))
    pairs = cp.array(list(tree.query_pairs(distance)))
    if len(pairs) == 0:
        return blobs_array
    else:
        # from here, the kernel does the calculation
        blobs_module = _get_prune_blob_rawmodule(
            blobs_array.dtype,
            max(blobs_array.shape + (pairs.shape[0],)) >> 2**31
        )
        _prune_blobs_kernel = blobs_module.get_function("_prune_blobs_kdtree")

        block_size = 64
        grid_size = int(math.ceil(blobs_array.shape[0] / block_size))
        _prune_blobs_kernel((grid_size,), (block_size,),
                            (pairs.ravel(), pairs.shape[0],
                             blobs_array.ravel(),
                             int(blobs_array.shape[0]),
                             int(blobs_array.shape[1]),
                             float(overlap),
                             int(sigma_dim))
                            )
    return blobs_array[blobs_array[:, -1] > 0, :]


def _format_exclude_border(img_ndim, exclude_border):
    """Format an ``exclude_border`` argument as a tuple of ints for calling
    ``peak_local_max``.
    """
    if isinstance(exclude_border, tuple):
        if len(exclude_border) != img_ndim:
            raise ValueError(
                "`exclude_border` should have the same length as the "
                "dimensionality of the image.")
        for exclude in exclude_border:
            if not isinstance(exclude, int):
                raise ValueError(
                    "exclude border, when expressed as a tuple, must only "
                    "contain ints.")
        return exclude_border
    elif isinstance(exclude_border, int):
        return (exclude_border,) * img_ndim + (0,)
    elif exclude_border is True:
        raise ValueError("exclude_border cannot be True")
    elif exclude_border is False:
        return (0,) * (img_ndim + 1)
    else:
        raise ValueError(
            f'Unsupported value ({exclude_border}) for exclude_border'
        )


def blob_dog(image, min_sigma=1, max_sigma=50, sigma_ratio=1.6, threshold=0.5,
             overlap=.5, *, threshold_rel=None, exclude_border=False):
    r"""Finds blobs in the given grayscale image.
    Blobs are found using the Difference of Gaussian (DoG) method [1]_, [2]_.
    For each blob found, the method returns its coordinates and the standard
    deviation of the Gaussian kernel that detected the blob.
    Parameters
    ----------
    image : ndarray
        Input grayscale image, blobs are assumed to be light on dark
        background (white on black).
    min_sigma : scalar or sequence of scalars, optional
        The minimum standard deviation for Gaussian kernel. Keep this low to
        detect smaller blobs. The standard deviations of the Gaussian filter
        are given for each axis as a sequence, or as a single number, in
        which case it is equal for all axes.
    max_sigma : scalar or sequence of scalars, optional
        The maximum standard deviation for Gaussian kernel. Keep this high to
        detect larger blobs. The standard deviations of the Gaussian filter
        are given for each axis as a sequence, or as a single number, in
        which case it is equal for all axes.
    sigma_ratio : float, optional
        The ratio between the standard deviation of Gaussian Kernels used for
        computing the Difference of Gaussians
    threshold : float or None, optional
        The absolute lower bound for scale space maxima. Local maxima smaller
        than `threshold` are ignored. Reduce this to detect blobs with lower
        intensities. If `threshold_rel` is also specified, whichever threshold
        is larger will be used. If None, `threshold_rel` is used instead.
    overlap : float, optional
        A value between 0 and 1. If the area of two blobs overlaps by a
        fraction greater than `threshold`, the smaller blob is eliminated.
    threshold_rel : float or None, optional
        Minimum intensity of peaks, calculated as
        ``max(dog_space) * threshold_rel``, where ``dog_space`` refers to the
        stack of Difference-of-Gaussian (DoG) images computed internally. This
        should have a value between 0 and 1. If None, `threshold` is used
        instead.
    exclude_border : tuple of ints, int, or False, optional
        If tuple of ints, the length of the tuple must match the input array's
        dimensionality.  Each element of the tuple will exclude peaks from
        within `exclude_border`-pixels of the border of the image along that
        dimension.
        If nonzero int, `exclude_border` excludes peaks from within
        `exclude_border`-pixels of the border of the image.
        If zero or False, peaks are identified regardless of their
        distance from the border.

    Returns
    -------
    A : (n, image.ndim + sigma) ndarray
        A 2d array with each row representing 2 coordinate values for a 2D
        image, or 3 coordinate values for a 3D image, plus the sigma(s) used.
        When a single sigma is passed, outputs are:
        ``(r, c, sigma)`` or ``(p, r, c, sigma)`` where ``(r, c)`` or
        ``(p, r, c)`` are coordinates of the blob and ``sigma`` is the standard
        deviation of the Gaussian kernel which detected the blob. When an
        anisotropic gaussian is used (sigmas per dimension), the detected sigma
        is returned for each dimension.

    See also
    --------
    cucim.skimage.filters.difference_of_gaussians

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Blob_detection#The_difference_of_Gaussians_approach  # noqa
    .. [2] Lowe, D. G. "Distinctive Image Features from Scale-Invariant
        Keypoints." International Journal of Computer Vision 60, 91–110 (2004).
        https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
        :DOI:`10.1023/B:VISI.0000029664.99615.94`

    Examples
    --------
    >>> from cucim.skimage import data, feature
    >>> coins = data.coins()
    >>> feature.blob_dog(coins, threshold=.05, min_sigma=10, max_sigma=40)
    array([[128., 155.,  10.],
           [198., 155.,  10.],
           [124., 338.,  10.],
           [127., 102.,  10.],
           [193., 281.,  10.],
           [126., 208.,  10.],
           [267., 115.,  10.],
           [197., 102.,  10.],
           [198., 215.,  10.],
           [123., 279.,  10.],
           [126.,  46.,  10.],
           [259., 247.,  10.],
           [196.,  43.,  10.],
           [ 54., 276.,  10.],
           [267., 358.,  10.],
           [ 58., 100.,  10.],
           [259., 305.,  10.],
           [185., 347.,  16.],
           [261., 174.,  16.],
           [ 46., 336.,  16.],
           [ 54., 217.,  10.],
           [ 55., 157.,  10.],
           [ 57.,  41.,  10.],
           [260.,  47.,  16.]])

    Notes
    -----
    The radius of each blob is approximately :math:`\sqrt{2}\sigma` for
    a 2-D image and :math:`\sqrt{3}\sigma` for a 3-D image.
    """
    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)

    # if both min and max sigma are scalar, function returns only one sigma
    scalar_sigma = cp.isscalar(max_sigma) and cp.isscalar(min_sigma)

    # Gaussian filter requires that sequence-type sigmas have same
    # dimensionality as image. This broadcasts scalar kernels
    if cp.isscalar(max_sigma):
        max_sigma = cp.full(image.ndim, max_sigma, dtype=float_dtype)
    if cp.isscalar(min_sigma):
        min_sigma = cp.full(image.ndim, min_sigma, dtype=float_dtype)

    # Convert sequence types to array
    min_sigma = cp.asarray(min_sigma, dtype=float_dtype)
    max_sigma = cp.asarray(max_sigma, dtype=float_dtype)

    if sigma_ratio <= 1.0:
        raise ValueError('sigma_ratio must be > 1.0')

    # k such that min_sigma*(sigma_ratio**k) > max_sigma
    k = int(cp.mean(cp.log(max_sigma / min_sigma) / cp.log(sigma_ratio) + 1))

    # a geometric progression of standard deviations for gaussian kernels
    sigma_list = cp.array([min_sigma * (sigma_ratio ** i)
                           for i in range(k + 1)])

    gaussian_images = [gaussian(image, s, mode='reflect') for s in sigma_list]

    # normalization factor for consistency in DoG magnitude
    sf = 1 / (sigma_ratio - 1)

    # computing difference between two successive Gaussian blurred images
    # to obtain an approximation of the scale invariant Laplacian of the
    # Gaussian operator
    dog_images = [
        (gaussian_images[i] - gaussian_images[i + 1]) * sf for i in range(k)
    ]

    image_cube = cp.stack(dog_images, axis=-1)

    exclude_border = _format_exclude_border(image.ndim, exclude_border)
    local_maxima = peak_local_max(
        image_cube,
        threshold_abs=threshold,
        threshold_rel=threshold_rel,
        exclude_border=exclude_border,
        footprint=cp.ones((3,) * (image.ndim + 1)),
    )

    # Catch no peaks
    if local_maxima.size == 0:
        return cp.empty((0, image.ndim + (1 if scalar_sigma else image.ndim)))

    # Convert local_maxima to float64
    lm = local_maxima.astype(float_dtype)

    # translate final column of lm, which contains the index of the
    # sigma that produced the maximum intensity value, into the sigma
    sigmas_of_peaks = sigma_list[local_maxima[:, -1]]

    if scalar_sigma:
        # select one sigma column, keeping dimension
        sigmas_of_peaks = sigmas_of_peaks[:, 0:1]

    # Remove sigma index and replace with sigmas
    lm = cp.hstack([lm[:, :-1], sigmas_of_peaks])

    sigma_dim = sigmas_of_peaks.shape[1]

    return _prune_blobs(lm, overlap, sigma_dim=sigma_dim)


def blob_log(image, min_sigma=1, max_sigma=50, num_sigma=10, threshold=.2,
             overlap=.5, log_scale=False, *, threshold_rel=None,
             exclude_border=False):
    r"""Finds blobs in the given grayscale image.
    Blobs are found using the Laplacian of Gaussian (LoG) method [1]_.
    For each blob found, the method returns its coordinates and the standard
    deviation of the Gaussian kernel that detected the blob.
    Parameters
    ----------
    image : ndarray
        Input grayscale image, blobs are assumed to be light on dark
        background (white on black).
    min_sigma : scalar or sequence of scalars, optional
        the minimum standard deviation for Gaussian kernel. Keep this low to
        detect smaller blobs. The standard deviations of the Gaussian filter
        are given for each axis as a sequence, or as a single number, in
        which case it is equal for all axes.
    max_sigma : scalar or sequence of scalars, optional
        The maximum standard deviation for Gaussian kernel. Keep this high to
        detect larger blobs. The standard deviations of the Gaussian filter
        are given for each axis as a sequence, or as a single number, in
        which case it is equal for all axes.
    num_sigma : int, optional
        The number of intermediate values of standard deviations to consider
        between `min_sigma` and `max_sigma`.
    threshold : float or None, optional
        The absolute lower bound for scale space maxima. Local maxima smaller
        than `threshold` are ignored. Reduce this to detect blobs with lower
        intensities. If `threshold_rel` is also specified, whichever threshold
        is larger will be used. If None, `threshold_rel` is used instead.
    overlap : float, optional
        A value between 0 and 1. If the area of two blobs overlaps by a
        fraction greater than `threshold`, the smaller blob is eliminated.
    log_scale : bool, optional
        If set intermediate values of standard deviations are interpolated
        using a logarithmic scale to the base `10`. If not, linear
        interpolation is used.
    threshold_rel : float or None, optional
        Minimum intensity of peaks, calculated as
        ``max(log_space) * threshold_rel``, where ``log_space`` refers to the
        stack of Laplacian-of-Gaussian (LoG) images computed internally. This
        should have a value between 0 and 1. If None, `threshold` is used
        instead.
    exclude_border : tuple of ints, int, or False, optional
        If tuple of ints, the length of the tuple must match the input array's
        dimensionality.  Each element of the tuple will exclude peaks from
        within `exclude_border`-pixels of the border of the image along that
        dimension.
        If nonzero int, `exclude_border` excludes peaks from within
        `exclude_border`-pixels of the border of the image.
        If zero or False, peaks are identified regardless of their
        distance from the border.

    Returns
    -------
    A : (n, image.ndim + sigma) ndarray
        A 2d array with each row representing 2 coordinate values for a 2D
        image, or 3 coordinate values for a 3D image, plus the sigma(s) used.
        When a single sigma is passed, outputs are:
        ``(r, c, sigma)`` or ``(p, r, c, sigma)`` where ``(r, c)`` or
        ``(p, r, c)`` are coordinates of the blob and ``sigma`` is the standard
        deviation of the Gaussian kernel which detected the blob. When an
        anisotropic gaussian is used (sigmas per dimension), the detected sigma
        is returned for each dimension.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Blob_detection#The_Laplacian_of_Gaussian  # noqa

    Examples
    --------
    >>> from cucim.skimage import data, feature, exposure
    >>> img = data.coins()
    >>> img = exposure.equalize_hist(img)  # improves detection
    >>> feature.blob_log(img, threshold = .3)
    array([[124.        , 336.        ,  11.88888889],
           [198.        , 155.        ,  11.88888889],
           [194.        , 213.        ,  17.33333333],
           [121.        , 272.        ,  17.33333333],
           [263.        , 244.        ,  17.33333333],
           [194.        , 276.        ,  17.33333333],
           [266.        , 115.        ,  11.88888889],
           [128.        , 154.        ,  11.88888889],
           [260.        , 174.        ,  17.33333333],
           [198.        , 103.        ,  11.88888889],
           [126.        , 208.        ,  11.88888889],
           [127.        , 102.        ,  11.88888889],
           [263.        , 302.        ,  17.33333333],
           [197.        ,  44.        ,  11.88888889],
           [185.        , 344.        ,  17.33333333],
           [126.        ,  46.        ,  11.88888889],
           [113.        , 323.        ,   1.        ]])
    Notes
    -----
    The radius of each blob is approximately :math:`\sqrt{2}\sigma` for
    a 2-D image and :math:`\sqrt{3}\sigma` for a 3-D image.
    """
    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)

    # if both min and max are scalar, function returns only one sigma
    scalar_sigma = (
        True if cp.isscalar(max_sigma) and cp.isscalar(min_sigma) else False
    )

    # Gaussian filter requires that sequence-type sigmas have same
    # dimensionality as image. This broadcasts scalar kernels
    if cp.isscalar(max_sigma):
        max_sigma = cp.full(image.ndim, max_sigma, dtype=float_dtype)
    if cp.isscalar(min_sigma):
        min_sigma = cp.full(image.ndim, min_sigma, dtype=float_dtype)

    # Convert sequence types to array
    min_sigma = cp.asarray(min_sigma, dtype=float_dtype)
    max_sigma = cp.asarray(max_sigma, dtype=float_dtype)

    if log_scale:
        start = cp.log10(min_sigma)
        stop = cp.log10(max_sigma)
        sigma_list = cp.logspace(start, stop, num_sigma)
    else:
        sigma_list = cp.linspace(min_sigma, max_sigma, num_sigma)

    # computing gaussian laplace
    # average s**2 provides scale invariance
    gl_images = [-gaussian_laplace(image, s) * cp.mean(s) ** 2
                 for s in sigma_list]

    image_cube = cp.stack(gl_images, axis=-1)

    exclude_border = _format_exclude_border(image.ndim, exclude_border)
    local_maxima = peak_local_max(
        image_cube,
        threshold_abs=threshold,
        threshold_rel=threshold_rel,
        exclude_border=exclude_border,
        footprint=cp.ones((3,) * (image.ndim + 1))
    )

    # Catch no peaks
    if local_maxima.size == 0:
        return cp.empty((0, image.ndim + (1 if scalar_sigma else image.ndim)))

    # Convert local_maxima to float64
    lm = local_maxima.astype(float_dtype)

    # translate final column of lm, which contains the index of the
    # sigma that produced the maximum intensity value, into the sigma
    sigmas_of_peaks = sigma_list[local_maxima[:, -1]]

    if scalar_sigma:
        # select one sigma column, keeping dimension
        sigmas_of_peaks = sigmas_of_peaks[:, 0:1]

    # Remove sigma index and replace with sigmas
    lm = cp.hstack([lm[:, :-1], sigmas_of_peaks])

    sigma_dim = sigmas_of_peaks.shape[1]

    return _prune_blobs(lm, overlap, sigma_dim=sigma_dim)


def blob_doh(image, min_sigma=1, max_sigma=30, num_sigma=10, threshold=0.01,
             overlap=.5, log_scale=False, *, threshold_rel=None):
    """Finds blobs in the given grayscale image.

    Blobs are found using the Determinant of Hessian method [1]_. For each blob
    found, the method returns its coordinates and the standard deviation
    of the Gaussian Kernel used for the Hessian matrix whose determinant
    detected the blob. Determinant of Hessians is approximated using [2]_.

    Parameters
    ----------
    image : 2D ndarray
        Input grayscale image.Blobs can either be light on dark or vice versa.
    min_sigma : float, optional
        The minimum standard deviation for Gaussian Kernel used to compute
        Hessian matrix. Keep this low to detect smaller blobs.
    max_sigma : float, optional
        The maximum standard deviation for Gaussian Kernel used to compute
        Hessian matrix. Keep this high to detect larger blobs.
    num_sigma : int, optional
        The number of intermediate values of standard deviations to consider
        between `min_sigma` and `max_sigma`.
    threshold : float or None, optional
        The absolute lower bound for scale space maxima. Local maxima smaller
        than `threshold` are ignored. Reduce this to detect blobs with lower
        intensities. If `threshold_rel` is also specified, whichever threshold
        is larger will be used. If None, `threshold_rel` is used instead.
    overlap : float, optional
        A value between 0 and 1. If the area of two blobs overlaps by a
        fraction greater than `threshold`, the smaller blob is eliminated.
    log_scale : bool, optional
        If set intermediate values of standard deviations are interpolated
        using a logarithmic scale to the base `10`. If not, linear
        interpolation is used.
    threshold_rel : float or None, optional
        Minimum intensity of peaks, calculated as
        ``max(doh_space) * threshold_rel``, where ``doh_space`` refers to the
        stack of Determinant-of-Hessian (DoH) images computed internally. This
        should have a value between 0 and 1. If None, `threshold` is used
        instead.

    Returns
    -------
    A : (n, 3) ndarray
        A 2d array with each row representing 3 values, ``(y,x,sigma)``
        where ``(y,x)`` are coordinates of the blob and ``sigma`` is the
        standard deviation of the Gaussian kernel of the Hessian Matrix whose
        determinant detected the blob.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Blob_detection#The_determinant_of_the_Hessian  # noqa
    .. [2] Herbert Bay, Andreas Ess, Tinne Tuytelaars, Luc Van Gool,
           "SURF: Speeded Up Robust Features"
           ftp://ftp.vision.ee.ethz.ch/publications/articles/eth_biwi_00517.pdf

    Examples
    --------
    >>> from cucim.skimage import data, feature
    >>> img = data.coins()
    >>> feature.blob_doh(img)
    array([[197.        , 153.        ,  20.33333333],
           [124.        , 336.        ,  20.33333333],
           [126.        , 153.        ,  20.33333333],
           [195.        , 100.        ,  23.55555556],
           [192.        , 212.        ,  23.55555556],
           [121.        , 271.        ,  30.        ],
           [126.        , 101.        ,  20.33333333],
           [193.        , 275.        ,  23.55555556],
           [123.        , 205.        ,  20.33333333],
           [270.        , 363.        ,  30.        ],
           [265.        , 113.        ,  23.55555556],
           [262.        , 243.        ,  23.55555556],
           [185.        , 348.        ,  30.        ],
           [156.        , 302.        ,  30.        ],
           [123.        ,  44.        ,  23.55555556],
           [260.        , 173.        ,  30.        ],
           [197.        ,  44.        ,  20.33333333]])

    Notes
    -----
    The radius of each blob is approximately `sigma`.
    Computation of Determinant of Hessians is independent of the standard
    deviation. Therefore detecting larger blobs won't take more time. In
    methods line :py:meth:`blob_dog` and :py:meth:`blob_log` the computation
    of Gaussians for larger `sigma` takes more time. The downside is that
    this method can't be used for detecting blobs of radius less than `3px`
    due to the box filters used in the approximation of Hessian Determinant.
    """
    check_nD(image, 2)

    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)

    image = integral_image(image)

    if log_scale:
        start, stop = math.log(min_sigma, 10), math.log(max_sigma, 10)
        sigma_list = cp.logspace(start, stop, num_sigma)
    else:
        sigma_list = cp.linspace(min_sigma, max_sigma, num_sigma)

    hessian_images = [_hessian_matrix_det(image, s) for s in sigma_list]
    image_cube = cp.dstack(hessian_images)

    local_maxima = peak_local_max(image_cube,
                                  threshold_abs=threshold,
                                  threshold_rel=threshold_rel,
                                  exclude_border=False,
                                  footprint=cp.ones((3,) * image_cube.ndim))

    # Catch no peaks
    if local_maxima.size == 0:
        return cp.empty((0, 3))
    # Convert local_maxima to float type of input image
    lm = local_maxima.astype(float_dtype)
    # Convert the last index to its corresponding scale value
    lm[:, -1] = sigma_list[local_maxima[:, -1]]
    return _prune_blobs(lm, overlap)
