import itertools

import cupy as cp
import numpy as np

from .._shared.utils import _supported_float_type, check_nD
from ._moments_analytical import moments_raw_to_central


def moments_coords(coords, order=3):
    """Calculate all raw image moments up to a certain order.

    The following properties can be calculated from raw image moments:
     * Area as: ``M[0, 0]``.
     * Centroid as: {``M[1, 0] / M[0, 0]``, ``M[0, 1] / M[0, 0]``}.

    Note that raw moments are neither translation, scale nor rotation
    invariant.

    Parameters
    ----------
    coords : (N, D) double or uint8 array
        Array of N points that describe an image of D dimensionality in
        Cartesian space.
    order : int, optional
        Maximum order of moments. Default is 3.

    Returns
    -------
    M : (``order + 1``, ``order + 1``, ...) array
        Raw image moments. (D dimensions)

    References
    ----------
    .. [1] Johannes Kilian. Simple Image Analysis By Moments. Durham
           University, version 0.2, Durham, 2001.

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage.measure import moments_coords
    >>> coords = cp.array([[row, col]
    ...                    for row in range(13, 17)
    ...                    for col in range(14, 18)], dtype=cp.float64)
    >>> M = moments_coords(coords)
    >>> centroid = (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])
    >>> centroid
    (array(14.5), array(15.5))
    """
    return moments_coords_central(coords, 0, order=order)


def moments_coords_central(coords, center=None, order=3):
    """Calculate all central image moments up to a certain order.

    The following properties can be calculated from raw image moments:
     * Area as: ``M[0, 0]``.
     * Centroid as: {``M[1, 0] / M[0, 0]``, ``M[0, 1] / M[0, 0]``}.

    Note that raw moments are neither translation, scale nor rotation
    invariant.

    Parameters
    ----------
    coords : (N, D) double or uint8 array
        Array of N points that describe an image of D dimensionality in
        Cartesian space. A tuple of coordinates as returned by
        ``cp.nonzero`` is also accepted as input.
    center : tuple of float, optional
        Coordinates of the image centroid. This will be computed if it
        is not provided.
    order : int, optional
        Maximum order of moments. Default is 3.

    Returns
    -------
    Mc : (``order + 1``, ``order + 1``, ...) array
        Central image moments. (D dimensions)

    References
    ----------
    .. [1] Johannes Kilian. Simple Image Analysis By Moments. Durham
           University, version 0.2, Durham, 2001.

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage.measure import moments_coords_central
    >>> coords = cp.array([[row, col]
    ...                    for row in range(13, 17)
    ...                    for col in range(14, 18)])
    >>> moments_coords_central(coords)
    array([[16.,  0., 20.,  0.],
           [ 0.,  0.,  0.,  0.],
           [20.,  0., 25.,  0.],
           [ 0.,  0.,  0.,  0.]])

    As seen above, for symmetric objects, odd-order moments (columns 1 and 3,
    rows 1 and 3) are zero when centered on the centroid, or center of mass,
    of the object (the default). If we break the symmetry by adding a new
    point, this no longer holds:

    >>> coords2 = cp.concatenate((coords, cp.array([[17, 17]])), axis=0)
    >>> cp.around(moments_coords_central(coords2),
    ...           decimals=2)  # doctest: +NORMALIZE_WHITESPACE
    array([[17.  ,  0.  , 22.12, -2.49],
           [ 0.  ,  3.53,  1.73,  7.4 ],
           [25.88,  6.02, 36.63,  8.83],
           [ 4.15, 19.17, 14.8 , 39.6 ]])

    Image moments and central image moments are equivalent (by definition)
    when the center is (0, 0):

    >>> cp.allclose(moments_coords(coords),
    ...             moments_coords_central(coords, (0, 0)))
    array(True)
    """
    if isinstance(coords, tuple):
        # This format corresponds to coordinate tuples as returned by
        # e.g. cp.nonzero: (row_coords, column_coords).
        # We represent them as an npoints x ndim array.
        coords = cp.stack(coords, axis=-1)
    check_nD(coords, 2)
    ndim = coords.shape[1]

    float_type = _supported_float_type(coords.dtype)
    if center is None:
        center = cp.mean(coords, axis=0, dtype=float)
        center = center.astype(float_type, copy=False)
    else:
        center = cp.asarray(center, dtype=float_type)

    # center the coordinates
    coords = coords.astype(float_type, copy=False)
    coords -= center

    # CuPy backend: for efficiency, sum over the last axis
    #               (which is memory contiguous)
    # generate all possible exponents for each axis in the given set of points
    # produces a matrix of shape (order + 1, D, N)
    coords = coords.T
    powers = cp.arange(order + 1, dtype=float_type)[:, np.newaxis, np.newaxis]
    coords = coords[cp.newaxis, ...] ** powers

    # add extra dimensions for proper broadcasting
    coords = coords.reshape((1,) * (ndim - 1) + coords.shape)

    calc = cp.moveaxis(coords[..., 0, :], -2, 0)

    for axis in range(1, ndim):
        # isolate each point's axis
        isolated_axis = coords[..., axis, :]

        # rotate orientation of matrix for proper broadcasting
        isolated_axis = cp.moveaxis(isolated_axis, -2, axis)

        # calculate the moments for each point, one axis at a time
        calc = calc * isolated_axis
    # sum all individual point moments to get our final answer
    Mc = cp.sum(calc, axis=-1)

    return Mc


def moments(image, order=3, *, spacing=None):
    """Calculate all raw image moments up to a certain order.

    The following properties can be calculated from raw image moments:
     * Area as: ``M[0, 0]``.
     * Centroid as: {``M[1, 0] / M[0, 0]``, ``M[0, 1] / M[0, 0]``}.

    Note that raw moments are neither translation, scale nor rotation
    invariant.

    Parameters
    ----------
    image : nD double or uint8 array
        Rasterized shape as image.
    order : int, optional
        Maximum order of moments. Default is 3.
    spacing: tuple of float, shape (ndim, )
        The pixel spacing along each axis of the image.

    Returns
    -------
    m : (``order + 1``, ``order + 1``) array
        Raw image moments.

    References
    ----------
    .. [1] Wilhelm Burger, Mark Burge. Principles of Digital Image Processing:
           Core Algorithms. Springer-Verlag, London, 2009.
    .. [2] B. Jähne. Digital Image Processing. Springer-Verlag,
           Berlin-Heidelberg, 6. edition, 2005.
    .. [3] T. H. Reiss. Recognizing Planar Objects Using Invariant Image
           Features, from Lecture notes in computer science, p. 676. Springer,
           Berlin, 1993.
    .. [4] https://en.wikipedia.org/wiki/Image_moment

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage.measure import moments
    >>> image = cp.zeros((20, 20), dtype=cp.float64)
    >>> image[13:17, 13:17] = 1
    >>> M = moments(image)
    >>> centroid = (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])
    >>> centroid
    (array(14.5), array(14.5))
    """
    float_dtype = _supported_float_type(image.dtype)
    calc = image.astype(float_dtype, copy=False)
    powers = cp.arange(order + 1, dtype=float_dtype)
    _delta = cp.arange(max(image.shape), dtype=float_dtype)[:, cp.newaxis]
    if spacing is None:
        # when spacing is not used can compute the powers outside the loop
        _powers_of_delta = _delta ** powers
    for dim, dim_length in enumerate(image.shape):
        if spacing is None:
            powers_of_delta = _powers_of_delta[:dim_length]
        else:
            delta = _delta[:dim_length] * spacing[dim]
            powers_of_delta = delta ** powers
        calc = cp.moveaxis(calc, source=dim, destination=-1)
        calc = cp.dot(calc, powers_of_delta)
        calc = cp.moveaxis(calc, source=-1, destination=dim)
    return calc


def moments_central(image, center=None, order=3, *, spacing=None, **kwargs):
    """Calculate all central image moments up to a certain order.

    The center coordinates (cr, cc) can be calculated from the raw moments as:
    {``M[1, 0] / M[0, 0]``, ``M[0, 1] / M[0, 0]``}.

    Note that central moments are translation invariant but not scale and
    rotation invariant.

    Parameters
    ----------
    image : nD double or uint8 array
        Rasterized shape as image.
    center : tuple of float, optional
        Coordinates of the image centroid. This will be computed if it
        is not provided.
    order : int, optional
        The maximum order of moments computed.
    spacing: tuple of float, shape (ndim, )
        The pixel spacing along each axis of the image.

    Returns
    -------
    mu : (``order + 1``, ``order + 1``) array
        Central image moments.

    References
    ----------
    .. [1] Wilhelm Burger, Mark Burge. Principles of Digital Image Processing:
           Core Algorithms. Springer-Verlag, London, 2009.
    .. [2] B. Jähne. Digital Image Processing. Springer-Verlag,
           Berlin-Heidelberg, 6. edition, 2005.
    .. [3] T. H. Reiss. Recognizing Planar Objects Using Invariant Image
           Features, from Lecture notes in computer science, p. 676. Springer,
           Berlin, 1993.
    .. [4] https://en.wikipedia.org/wiki/Image_moment

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage.measure import moments, moments_central
    >>> image = cp.zeros((20, 20), dtype=cp.float64)
    >>> image[13:17, 13:17] = 1
    >>> M = moments(image)
    >>> centroid = (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])
    >>> moments_central(image, centroid)
    array([[16.,  0., 20.,  0.],
           [ 0.,  0.,  0.,  0.],
           [20.,  0., 25.,  0.],
           [ 0.,  0.,  0.,  0.]])
    """
    if center is None:
        # Note: No need for an explicit call to centroid.
        #       The centroid will be obtained from the raw moments.
        moments_raw = moments(image, order=order, spacing=spacing)
        return moments_raw_to_central(moments_raw)
    if spacing is None:
        spacing = np.ones(image.ndim)
    float_dtype = _supported_float_type(image.dtype)
    calc = image.astype(float_dtype, copy=False)
    powers = cp.arange(order + 1, dtype=float_dtype)
    _delta = cp.arange(max(image.shape), dtype=float_dtype)[:, cp.newaxis]
    for dim, dim_length in enumerate(image.shape):
        delta = _delta[:dim_length] * spacing[dim] - center[dim]
        powers_of_delta = delta ** powers
        calc = cp.moveaxis(calc, source=dim, destination=-1)
        calc = cp.dot(calc, powers_of_delta)
        calc = cp.moveaxis(calc, source=-1, destination=dim)
    return calc


def _get_moments_norm_operation(ndim, order, unit_scale=True):
    """Full normalization computation kernel for 2D or 3D cases.

    Variants with or without scaling are provided.
    """
    operation = f"""
        double mu0 = static_cast<double>(mu[0]);
        double ndim = {ndim};
        int _i = i;
        int coord_i;
        int order_of_current_index = 0;
        int n_rows = order + 1;
        double denom;
    """

    if not unit_scale:
        operation += """
        double s_pow;"""

    operation += f"""
        for (int d=0; d<{ndim}; d++)"""
    operation += """
        {
            // This loop computes the coordinate index along each axis of the
            // matrix in turn and sums them up to get the order of the moment
            // at the current index in mu.
            coord_i = _i % n_rows;
            _i /= n_rows;
            order_of_current_index += coord_i;
        }
        if ((order_of_current_index > order) || (order_of_current_index < 2))
        {
            continue;
        }
    """
    if unit_scale:
        operation += """
        denom = pow(mu0, static_cast<double>(order_of_current_index) / ndim + 1);
        nu = mu[i] / denom;"""  # noqa
    else:
        operation += """
        s_pow = pow(scale, static_cast<double>(order_of_current_index));
        denom = pow(mu0, static_cast<double>(order_of_current_index) / ndim + 1);
        nu = (mu[i] / s_pow) / denom;"""  # noqa
    return operation


@cp.memoize()
def _get_normalize_kernel(ndim, order, unit_scale=True):
    return cp.ElementwiseKernel(
        'raw F mu, int32 order, float64 scale',
        'F nu',
        operation=_get_moments_norm_operation(ndim, order, unit_scale),
        name=f"moments_normmalize_2d_kernel"
    )


def moments_normalized(mu, order=3, spacing=None):
    """Calculate all normalized central image moments up to a certain order.

    Note that normalized central moments are translation and scale invariant
    but not rotation invariant.

    Parameters
    ----------
    mu : (M,[ ...,] M) array
        Central image moments, where M must be greater than or equal
        to ``order``.
    order : int, optional
        Maximum order of moments. Default is 3.

    Returns
    -------
    nu : (``order + 1``,[ ...,] ``order + 1``) array
        Normalized central image moments.

    Notes
    -----
    Differs from the scikit-image implementation in that any moments greater
    than the requested `order` will be set to ``nan``.

    References
    ----------
    .. [1] Wilhelm Burger, Mark Burge. Principles of Digital Image Processing:
           Core Algorithms. Springer-Verlag, London, 2009.
    .. [2] B. Jähne. Digital Image Processing. Springer-Verlag,
           Berlin-Heidelberg, 6. edition, 2005.
    .. [3] T. H. Reiss. Recognizing Planar Objects Using Invariant Image
           Features, from Lecture notes in computer science, p. 676. Springer,
           Berlin, 1993.
    .. [4] https://en.wikipedia.org/wiki/Image_moment

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage.measure import (moments, moments_central,
    ...                                      moments_normalized)
    >>> image = cp.zeros((20, 20), dtype=cp.float64)
    >>> image[13:17, 13:17] = 1
    >>> m = moments(image)
    >>> centroid = (m[0, 1] / m[0, 0], m[1, 0] / m[0, 0])
    >>> mu = moments_central(image, centroid)
    >>> moments_normalized(mu)
    array([[       nan,        nan, 0.078125  , 0.        ],
           [       nan, 0.        , 0.        , 0.        ],
           [0.078125  , 0.        , 0.00610352, 0.        ],
           [0.        , 0.        , 0.        , 0.        ]])
    """
    if any(s <= order for s in mu.shape):
        raise ValueError("Shape of image moments must be >= `order`")
    if spacing is None:
        scale = 1.0
    else:
        if isinstance(spacing, cp.ndarray):
            scale = spacing.min()
        else:
            scale = min(spacing)
    # compute using in a single kernel for the 2D or 3D cases
    unit_scale = scale == 1.0
    kernel = _get_normalize_kernel(mu.ndim, order, unit_scale)
    nu = cp.full(mu.shape, cp.nan, dtype=mu.dtype)
    kernel(mu, order, scale, nu)
    return nu


def moments_hu(nu):
    """Calculate Hu's set of image moments (2D-only).

    Note that this set of moments is proved to be translation, scale and
    rotation invariant.

    Parameters
    ----------
    nu : (M, M) array
        Normalized central image moments, where M must be >= 4.

    Returns
    -------
    nu : (7,) array
        Hu's set of image moments.

    Notes
    -----
    Due to the small array sizes, this function will be faster on the CPU.
    Consider transfering ``nu`` to the host and running
    ``skimage.measure.moments_hu`` if the moments are not needed on the
    device.

    References
    ----------
    .. [1] M. K. Hu, "Visual Pattern Recognition by Moment Invariants",
           IRE Trans. Info. Theory, vol. IT-8, pp. 179-187, 1962
    .. [2] Wilhelm Burger, Mark Burge. Principles of Digital Image Processing:
           Core Algorithms. Springer-Verlag, London, 2009.
    .. [3] B. Jähne. Digital Image Processing. Springer-Verlag,
           Berlin-Heidelberg, 6. edition, 2005.
    .. [4] T. H. Reiss. Recognizing Planar Objects Using Invariant Image
           Features, from Lecture notes in computer science, p. 676. Springer,
           Berlin, 1993.
    .. [5] https://en.wikipedia.org/wiki/Image_moment

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage.measure import (moments_central, moments_hu,
    ...                                      moments_normalized)
    >>> image = cp.zeros((20, 20), dtype=np.float64)
    >>> image[13:17, 13:17] = 0.5
    >>> image[10:12, 10:12] = 1
    >>> mu = moments_central(image)
    >>> nu = moments_normalized(mu)
    >>> moments_hu(nu)
    array([7.45370370e-01, 3.51165981e-01, 1.04049179e-01, 4.06442107e-02,
           2.64312299e-03, 2.40854582e-02, 6.50521303e-19])
    """
    try:
        from skimage.measure import moments_hu
    except ImportError:
        raise ImportError("moments_hu requires scikit-image.")

    # CuPy Backend: TODO: Due to small arrays involved, just transfer to/from
    #                     the CPU implementation.
    float_dtype = cp.float32 if nu.dtype == cp.float32 else cp.float64
    return cp.asarray(moments_hu(cp.asnumpy(nu)), dtype=float_dtype)


def centroid(image, *, spacing=None):
    """Return the (weighted) centroid of an image.

    Parameters
    ----------
    image : array
        The input image.
    spacing: tuple of float, shape (ndim, )
        The pixel spacing along each axis of the image.

    Returns
    -------
    center : tuple of float, length ``image.ndim``
        The centroid of the (nonzero) pixels in ``image``.

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage.measure import centroid
    >>> image = cp.zeros((20, 20), dtype=cp.float64)
    >>> image[13:17, 13:17] = 0.5
    >>> image[10:12, 10:12] = 1
    >>> centroid(image)
    array([13.16666667, 13.16666667])
    """
    mu = moments(image, order=1, spacing=spacing)
    ndim = image.ndim
    mu0 = mu[(0,) * ndim]
    center = mu[tuple((0,) * dim + (1,) + (0,) * (ndim - dim - 1)
                for dim in range(ndim))]
    center /= mu0
    return center


def _get_inertia_tensor_2x2_kernel():
    operation = """
    F mu0, mxx, mxy, myy;
    mu0 = mu[0];
    mxx = mu[6];
    myy = mu[2];
    mxy = mu[4];

    result[0] = myy / mu0;
    result[1] = result[2] = -mxy / mu0;
    result[3] = mxx / mu0;
    """
    return cp.ElementwiseKernel(
        in_params='raw F mu',
        out_params='raw F result',
        operation=operation,
        name='cucim_skimage_measure_inertia_tensor_2x2'
    )


def _get_inertia_tensor_3x3_kernel():
    operation = """
    F mu0, mxx, myy, mzz, mxy, mxz, myz;
    mu0 = mu[0];   // mu[0, 0, 0]
    mxx = mu[18];  // mu[2, 0, 0]
    myy = mu[6];   // mu[0, 2, 0]
    mzz = mu[2];   // mu[0, 0, 2]

    mxy = mu[12];  // mu[1, 1, 0]
    mxz = mu[10];  // mu[1, 0, 1]
    myz = mu[4];   // mu[0, 1, 1]

    result[0] = (myy + mzz) / mu0;
    result[4] = (mxx + mzz) / mu0;
    result[8] = (mxx + myy) / mu0;

    result[1] = result[3] = -mxy / mu0;
    result[2] = result[6] = -mxz / mu0;
    result[5] = result[7] = -myz / mu0;
    """
    return cp.ElementwiseKernel(
        in_params='raw F mu',
        out_params='raw F result',
        operation=operation,
        name='cucim_skimage_measure_inertia_tensor_3x3'
    )


def inertia_tensor(image, mu=None, *, spacing=None):
    """Compute the inertia tensor of the input image.

    Parameters
    ----------
    image : array
        The input image.
    mu : array, optional
        The pre-computed central moments of ``image``. The inertia tensor
        computation requires the central moments of the image. If an
        application requires both the central moments and the inertia tensor
        (for example, `skimage.measure.regionprops`), then it is more
        efficient to pre-compute them and pass them to the inertia tensor
        call.
    spacing : tuple of float, optional
        The pixel spacing along each axis of the image.

    Returns
    -------
    T : array, shape ``(image.ndim, image.ndim)``
        The inertia tensor of the input image. :math:`T_{i, j}` contains
        the covariance of image intensity along axes :math:`i` and :math:`j`.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
    .. [2] Bernd Jähne. Spatio-Temporal Image Processing: Theory and
           Scientific Applications. (Chapter 8: Tensor Methods) Springer, 1993.
    """
    if mu is None:
        # don't need higher-order moments
        mu = moments_central(image, order=2, spacing=spacing)
    else:
        if mu.shape[0] < 3:
            raise ValueError("mu must contain second order moments")
        if mu.shape[0] > 3:
            # if higher than 2nd order moments are present trim the array to
            # match the expectations of the _get_inertia_tensor* kernels.
            mu = mu[(slice(0, 3),) * mu.ndim]
        mu = cp.ascontiguousarray(mu)
    if image.ndim == 2:
        result = cp.empty((2, 2), dtype=mu.dtype)
        kern = _get_inertia_tensor_2x2_kernel()
        kern(mu, result, size=1)
    elif image.ndim == 3:
        result = cp.empty((3, 3), dtype=mu.dtype)
        kern = _get_inertia_tensor_3x3_kernel()
        kern(mu, result, size=1)
    else:
        # CuPy Backend: mu and result are tiny, so faster on the CPU
        mu = cp.asnumpy(mu)
        mu0 = mu[(0,) * image.ndim]
        # nD expression to get coordinates ([2, 0], [0, 2]) (2D),
        # ([2, 0, 0], [0, 2, 0], [0, 0, 2]) (3D), etc.
        corners2 = tuple(2 * np.eye(image.ndim, dtype=int))
        # See https://ocw.mit.edu/courses/aeronautics-and-astronautics/
        #          16-07-dynamics-fall-2009/lecture-notes/MIT16_07F09_Lec26.pdf
        # Iii is the sum of second-order moments of every axis *except* i, not
        # the second order moment of axis i.
        # See also https://github.com/scikit-image/scikit-image/issues/3229
        result = np.diag((np.sum(mu[corners2]) - mu[corners2]) / mu0)

        for dims in itertools.combinations(range(image.ndim), 2):
            mu_index = np.zeros(image.ndim, dtype=int)
            mu_index[list(dims)] = 1
            result[dims] = -mu[tuple(mu_index)] / mu0
            result.T[dims] = -mu[tuple(mu_index)] / mu0
        result = cp.asarray(result)
    return result


def inertia_tensor_eigvals(image, mu=None, T=None, *, spacing=None):
    """Compute the eigenvalues of the inertia tensor of the image.

    The inertia tensor measures covariance of the image intensity along
    the image axes. (See `inertia_tensor`.) The relative magnitude of the
    eigenvalues of the tensor is thus a measure of the elongation of a
    (bright) object in the image.

    Parameters
    ----------
    image : array
        The input image.
    mu : array, optional
        The pre-computed central moments of ``image``.
    T : array, shape ``(image.ndim, image.ndim)``
        The pre-computed inertia tensor. If ``T`` is given, ``mu`` and
        ``image`` are ignored.
    spacing : tuple of float, optional
        The pixel spacing along each axis of the image.

    Returns
    -------
    eigvals : list of float, length ``image.ndim``
        The eigenvalues of the inertia tensor of ``image``, in descending
        order.

    Notes
    -----
    Computing the eigenvalues requires the inertia tensor of the input image.
    This is much faster if the central moments (``mu``) are provided, or,
    alternatively, one can provide the inertia tensor (``T``) directly.
    """
    # avoid circular import
    from ..feature.corner import (_image_orthogonal_matrix22_eigvals,
                                  _image_orthogonal_matrix33_eigvals)

    if T is None:
        T = inertia_tensor(image, mu, spacing=spacing)
    if image.ndim == 2:
        eigvals = _image_orthogonal_matrix22_eigvals(
            T[0, 0], T[0, 1], T[1, 1], sort='descending', abs_sort=False
        )
        cp.maximum(eigvals, 0.0, out=eigvals)
    elif image.ndim == 3:
        # fmt: off
        eigvals = _image_orthogonal_matrix33_eigvals(
            T[0, 0], T[0, 1], T[0, 2], T[1, 1], T[1, 2], T[2, 2],
            sort='descending', abs_sort=False
        )
        # fmt: on
        cp.maximum(eigvals, 0.0, out=eigvals)
    else:
        # sort in descending order
        eigvals = cp.sort(cp.linalg.eigvalsh(T))[::-1]
        # call without out argument so copy will be made -> positive strides
        eigvals = cp.maximum(eigvals, 0.0)
    # Floating point precision problems could make a positive
    # semidefinite matrix have an eigenvalue that is very slightly
    # negative. This can cause problems down the line, so set values
    # very near zero to zero.
    return eigvals
