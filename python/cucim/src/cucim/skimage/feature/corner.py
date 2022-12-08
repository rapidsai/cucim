import functools
import math
from itertools import combinations_with_replacement

import cupy as cp
import numpy as np
from scipy import spatial  # TODO: use cuSpatial if cKDTree becomes available

import cucim.skimage._vendored.ndimage as ndi
from cucim.skimage.util import img_as_float

from .._shared._gradient import gradient
from .._shared.utils import _supported_float_type, warn
from ..transform import integral_image
from ._hessian_det_appx import _hessian_matrix_det
from .peak import peak_local_max
from .util import _prepare_grayscale_input_nD


def _compute_derivatives(image, mode="constant", cval=0):
    """Compute derivatives in axis directions using the Sobel operator.

    Parameters
    ----------
    image : ndarray
        Input image.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.

    Returns
    -------
    derivatives : list of ndarray
        Derivatives in each axis direction.

    """

    derivatives = [
        ndi.sobel(image, axis=i, mode=mode, cval=cval)
        for i in range(image.ndim)
    ]

    return derivatives


def structure_tensor(image, sigma=1, mode="constant", cval=0, order='rc'):
    """Compute structure tensor using sum of squared differences.

    The (2-dimensional) structure tensor A is defined as::

        A = [Arr Arc]
            [Arc Acc]

    which is approximated by the weighted sum of squared differences in a local
    window around each pixel in the image. This formula can be extended to a
    larger number of dimensions (see [1]_).

    Parameters
    ----------
    image : ndarray
        Input image.
    sigma : float or array-like of float, optional
        Standard deviation used for the Gaussian kernel, which is used as a
        weighting function for the local summation of squared differences.
        If sigma is an iterable, its length must be equal to `image.ndim` and
        each element is used for the Gaussian kernel applied along its
        respective axis.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    order : {'rc', 'xy'}, optional
        NOTE: 'xy' is only an option for 2D images, higher dimensions must
        always use 'rc' order. This parameter allows for the use of reverse or
        forward order of the image axes in gradient computation. 'rc' indicates
        the use of the first axis initially (Arr, Arc, Acc), whilst 'xy'
        indicates the usage of the last axis initially (Axx, Axy, Ayy).

    Returns
    -------
    A_elems : list of ndarray
        Upper-diagonal elements of the structure tensor for each pixel in the
        input image.

    See also
    --------
    structure_tensor_eigenvalues

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Structure_tensor

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage.feature import structure_tensor
    >>> square = cp.zeros((5, 5))
    >>> square[2, 2] = 1
    >>> Arr, Arc, Acc = structure_tensor(square, sigma=0.1, order="rc")
    >>> Acc
    array([[0., 0., 0., 0., 0.],
           [0., 1., 0., 1., 0.],
           [0., 4., 0., 4., 0.],
           [0., 1., 0., 1., 0.],
           [0., 0., 0., 0., 0.]])

    """
    from .._shared.filters import gaussian  # avoid circular import

    if order == "xy" and image.ndim > 2:
        raise ValueError('Only "rc" order is supported for dim > 2.')

    if order not in ["rc", "xy"]:
        raise ValueError(
            f'order {order} is invalid. Must be either "rc" or "xy"'
        )

    if not np.isscalar(sigma):
        sigma = tuple(sigma)
        if len(sigma) != image.ndim:
            raise ValueError("sigma must have as many elements as image "
                             "has axes")

    image = _prepare_grayscale_input_nD(image)

    derivatives = _compute_derivatives(image, mode=mode, cval=cval)

    if order == "xy":
        derivatives = reversed(derivatives)

    # Autodetection as done internally to Gaussian, but set it here to silence
    # a warning.
    channel_axis = -1 if (image.ndim == 3 and image.shape[-1] == 3) else None

    # structure tensor
    A_elems = [gaussian(der0 * der1, sigma, mode=mode, cval=cval,
                        channel_axis=channel_axis)
               for der0, der1 in combinations_with_replacement(derivatives, 2)]

    return A_elems


def _hessian_matrix_with_gaussian(image, sigma=1, mode='reflect', cval=0,
                                  order='rc'):
    """Compute the Hessian via convolutions with Gaussian derivatives.

    In 2D, the Hessian matrix is defined as:
        H = [Hrr Hrc]
            [Hrc Hcc]

    which is computed by convolving the image with the second derivatives
    of the Gaussian kernel in the respective r- and c-directions.

    The implementation here also supports n-dimensional data.

    Parameters
    ----------
    image : ndarray
        Input image.
    sigma : float or sequence of float, optional
        Standard deviation used for the Gaussian kernel, which sets the
        amount of smoothing in terms of pixel-distances. It is
        advised to not choose a sigma much less than 1.0, otherwise
        aliasing artifacts may occur.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    order : {'rc', 'xy'}, optional
        NOTE: 'xy' is only an option for 2D images, higher dimensions must
        always use 'rc' order. This parameter allows for the use of reverse or
        forward order of the image axes in gradient computation. 'rc' indicates
        the use of the first axis initially (Hrr, Hrc, Hcc), whilst 'xy'
        indicates the usage of the last axis initially (Hxx, Hxy, Hyy).

    Returns
    -------
    H_elems : list of ndarray
        Upper-diagonal elements of the hessian matrix for each pixel in the
        input image. In 2D, this will be a three element list containing [Hrr,
        Hrc, Hcc]. In nD, the list will contain ``(n**2 + n) / 2`` arrays.

    """
    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    if image.ndim > 2 and order == "xy":
        raise ValueError("Order 'xy' is only supported for 2D images.")
    if order not in ["rc", "xy"]:
        raise ValueError(f"unrecognized order: {order}")

    if np.isscalar(sigma):
        sigma = (sigma,) * image.ndim

    # This function uses `scipy.ndimage.gaussian_filter` with the order
    # argument to compute convolutions. For example, specifying
    # ``order=[1, 0]`` would apply convolution with a first-order derivative of
    # the Gaussian along the first axis and simple Gaussian smoothing along the
    # second.

    # For small sigma, the SciPy Gaussian filter suffers from aliasing and edge
    # artifacts, given that the filter will approximate a sinc or sinc
    # derivative which only goes to 0 very slowly (order 1/n**2). Thus, we use
    # a much larger truncate value to reduce any edge artifacts.
    truncate = 8 if all(s > 1 for s in sigma) else 100
    sq1_2 = 1 / math.sqrt(2)
    sigma_scaled = tuple(sq1_2 * s for s in sigma)
    common_kwargs = dict(sigma=sigma_scaled, mode=mode, cval=cval,
                         truncate=truncate)
    gaussian_ = functools.partial(ndi.gaussian_filter, **common_kwargs)

    # Apply two successive first order Gaussian derivative operations, as
    # detailed in:
    # https://dsp.stackexchange.com/questions/78280/are-scipy-second-order-gaussian-derivatives-correct  # noqa

    # 1.) First order along one axis while smoothing (order=0) along the other
    ndim = image.ndim

    # orders in 2D = ([1, 0], [0, 1])
    #        in 3D = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    #        etc.
    orders = tuple([0] * d + [1] + [0] * (ndim - d - 1) for d in range(ndim))
    gradients = [gaussian_(image, order=orders[d]) for d in range(ndim)]

    # 2.) apply the derivative along another axis as well
    axes = range(ndim)
    if order == 'xy':
        axes = reversed(axes)
    H_elems = [gaussian_(gradients[ax0], order=orders[ax1])
               for ax0, ax1 in combinations_with_replacement(axes, 2)]
    return H_elems


def hessian_matrix(image, sigma=1, mode='constant', cval=0, order='rc',
                   use_gaussian_derivatives=None):
    r"""Compute the Hessian matrix.

    In 2D, the Hessian matrix is defined as::

        H = [Hrr Hrc]
            [Hrc Hcc]

    which is computed by convolving the image with the second derivatives
    of the Gaussian kernel in the respective r- and c-directions.

    The implementation here also supports n-dimensional data.

    Parameters
    ----------
    image : ndarray
        Input image.
    sigma : float
        Standard deviation used for the Gaussian kernel, which is used as
        weighting function for the auto-correlation matrix.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    order : {'rc', 'xy'}, optional
        NOTE: 'xy' is only an option for 2D images, higher dimensions must
        always use 'rc' order. This parameter allows for the use of reverse or
        forward order of the image axes in gradient computation. 'rc' indicates
        the use of the first axis initially (Hrr, Hrc, Hcc), whilst 'xy'
        indicates the usage of the last axis initially (Hxx, Hxy, Hyy).
    use_gaussian_derivatives : boolean, optional
        Indicates whether the Hessian is computed by convolving with Gaussian
        derivatives, or by a simple finite-difference operation.

    Returns
    -------
    H_elems : list of ndarray
        Upper-diagonal elements of the hessian matrix for each pixel in the
        input image. In 2D, this will be a three element list containing [Hrr,
        Hrc, Hcc]. In nD, the list will contain ``(n**2 + n) / 2`` arrays.


    Notes
    -----
    The distributive property of derivatives and convolutions allows us to
    restate the derivative of an image, I, smoothed with a Gaussian kernel, G,
    as the convolution of the image with the derivative of G.

    .. math::

        \frac{\partial }{\partial x_i}(I * G) =
        I * \left( \frac{\partial }{\partial x_i} G \right)

    When ``use_gaussian_derivatives`` is ``True``, this property is used to
    compute the second order derivatives that make up the Hessian matrix.

    When ``use_gaussian_derivatives`` is ``False``, simple finite differences
    on a Gaussian-smoothed image are used instead.

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage.feature import hessian_matrix
    >>> square = cp.zeros((5, 5))
    >>> square[2, 2] = 4
    >>> Hrr, Hrc, Hcc = hessian_matrix(square, sigma=0.1, order='rc',
    ...                                use_gaussian_derivatives=False)
    >>> Hrc
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  0., -1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0., -1.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    """
    from ..filters import gaussian  # avoid circular import

    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    if image.ndim > 2 and order == "xy":
        raise ValueError("Order 'xy' is only supported for 2D images.")
    if order not in ["rc", "xy"]:
        raise ValueError(f"unrecognized order: {order}")

    if use_gaussian_derivatives is None:
        use_gaussian_derivatives = False
        warn("use_gaussian_derivatives currently defaults to False, but will "
             "change to True in a future version. Please specify this "
             "argument explicitly to maintain the current behavior",
             category=FutureWarning, stacklevel=2)

    if use_gaussian_derivatives:
        return _hessian_matrix_with_gaussian(image, sigma=sigma, mode=mode,
                                             cval=cval, order=order)

    # Autodetection as done internally to Gaussian, but set it here to silence
    # a warning.
    # TODO: eventually remove this as this behavior of gaussian is deprecated
    channel_axis = -1 if (image.ndim == 3 and image.shape[-1] == 3) else None

    gaussian_filtered = gaussian(image, sigma=sigma, mode=mode, cval=cval,
                                 channel_axis=channel_axis)

    gradients = gradient(gaussian_filtered)
    axes = range(image.ndim)

    if order == "xy":
        axes = reversed(axes)

    H_elems = [
        gradient(gradients[ax0], axis=ax1)
        for ax0, ax1 in combinations_with_replacement(axes, 2)
    ]

    return H_elems


@cp.memoize(for_each_device=True)
def _get_real_symmetric_2x2_det_kernel():
    return cp.ElementwiseKernel(
        in_params="F M00, F M01, F M11",
        out_params="F det",
        operation="det = M00 * M11 - M01 * M01;",
        name="cucim_skimage_symmetric_det22_kernel")


@cp.memoize(for_each_device=True)
def _get_real_symmetric_3x3_det_kernel():
    operation = """
    det = M00 * (M11 * M22 - M12 * M12);
    det -= M01 * (M01 * M22 - M12 * M02);
    det += M02 * (M01 * M12 - M11 * M02);
    """
    return cp.ElementwiseKernel(
        in_params="F M00, F M01, F M02, F M11, F M12, F M22",
        out_params="F det",
        operation=operation,
        name="cucim_skimage_symmetric_det33_kernel")


def hessian_matrix_det(image, sigma=1, approximate=True):
    """Compute the approximate Hessian Determinant over an image.

    The 2D approximate method uses box filters over integral images to
    compute the approximate Hessian Determinant.

    Parameters
    ----------
    image : ndarray
        The image over which to compute the Hessian Determinant.
    sigma : float, optional
        Standard deviation of the Gaussian kernel used for the Hessian
        matrix.
    approximate : bool, optional
        If ``True`` and the image is 2D, use a much faster approximate
        computation. This argument has no effect on 3D and higher images.

    Returns
    -------
    out : array
        The array of the Determinant of Hessians.

    References
    ----------
    .. [1] Herbert Bay, Andreas Ess, Tinne Tuytelaars, Luc Van Gool,
           "SURF: Speeded Up Robust Features"
           ftp://ftp.vision.ee.ethz.ch/publications/articles/eth_biwi_00517.pdf

    Notes
    -----
    For 2D images when ``approximate=True``, the running time of this method
    only depends on size of the image. It is independent of `sigma` as one
    would expect. The downside is that the result for `sigma` less than `3`
    is not accurate, i.e., not similar to the result obtained if someone
    computed the Hessian and took its determinant.
    """
    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    if image.ndim == 2 and approximate:
        integral = integral_image(image)
        # integral image will promote to float64 for accuracy, but we can
        # cast back to float_dtype here.
        integral = integral.astype(float_dtype, copy=False)
        return _hessian_matrix_det(integral, sigma)
    else:  # slower brute-force implementation for nD images
        H = hessian_matrix(image, sigma, use_gaussian_derivatives=False)
        if image.ndim in [2, 3]:
            det = cp.empty_like(image)
            if image.ndim == 2:
                kernel = _get_real_symmetric_2x2_det_kernel()
            else:
                kernel = _get_real_symmetric_3x3_det_kernel()
            kernel(*H, det)
        else:
            # general, n-dimensional case (warning: high memory usage)
            hessian_mat_array = _symmetric_image(H)
            det = cp.linalg.det(hessian_mat_array)
        return det


@cp.memoize(for_each_device=True)
def _get_real_symmetric_2x2_eigvals_kernel(sort='ascending', abs_sort=False):

    operation = """
    F tmp1, tmp2;
    double m00 = static_cast<double>(M00);
    double m01 = static_cast<double>(M01);
    double m11 = static_cast<double>(M11);
    tmp1 = m01 * m01;
    tmp1 *= 4;

    tmp2 = m00 - m11;
    tmp2 *= tmp2;
    tmp2 += tmp1;
    tmp2 = sqrt(tmp2);
    tmp2 /= 2;

    tmp1 = m00 + m11;
    tmp1 /= 2;
    """
    if sort == 'ascending':
        operation += """
        lam1 = tmp1 - tmp2;
        lam2 = tmp1 + tmp2;
        """
        if abs_sort:
            operation += """
            F stmp;
            if (abs(lam1) > abs(lam2)) {
                stmp = lam1;
                lam1 = lam2;
                lam2 = stmp;
            }
            """
    elif sort == 'descending':
        operation += """
        lam1 = tmp1 + tmp2;
        lam2 = tmp1 - tmp2;
        """
        if abs_sort:
            operation += """
            F stmp;
            if (abs(lam1) < abs(lam2)) {
                stmp = lam1;
                lam1 = lam2;
                lam2 = stmp;
            }
            """
    else:
        raise ValueError(f"unknown sort type: {sort}")
    return cp.ElementwiseKernel(
        in_params="F M00, F M01, F M11",
        out_params="F lam1, F lam2",
        operation=operation,
        name="cucim_skimage_symmetric_eig22_kernel")


def _image_orthogonal_matrix22_eigvals(
    M00, M01, M11, sort='descending', abs_sort=False
):
    r"""Analytical expressions of the eigenvalues of a symmetric 2 x 2 matrix.
    It corresponds to::

    l1 = (M00 + M11) / 2 + cp.sqrt(4 * M01 ** 2 + (M00 - M11) ** 2) / 2
    l2 = (M00 + M11) / 2 - cp.sqrt(4 * M01 ** 2 + (M00 - M11) ** 2) / 2

    Parameters
    ----------
    M00, M01, M11 : cp.ndarray
        Images corresponding to the individual components of the matrix. For
        example, ``M01 = M[0, 1]``.
    sort : {"ascending", "descending"}, optional
        Eigenvalues should be sorted in the specified order.
    abs_sort : boolean, optional
        If ``True``, sort based on the absolute values.

    References
    ----------
    .. [1] C. Deledalle, L. Denis, S. Tabti, F. Tupin. Closed-form expressions
        of the eigen decomposition of 2 x 2 and 3 x 3 Hermitian matrices.
        [Research Report] Université de Lyon. 2017.
        https://hal.archives-ouvertes.fr/hal-01501221/file/matrix_exp_and_log_formula.pdf
    """  # noqa
    if M00.dtype.kind != "f":
        raise ValueError("expected real-valued floating point matrices")
    kernel = _get_real_symmetric_2x2_eigvals_kernel(
        sort=sort, abs_sort=abs_sort
    )
    eigs = cp.empty((2,) + M00.shape, dtype=M00.dtype)
    kernel(M00, M01, M11, eigs[0], eigs[1])
    return eigs


@cp.memoize(for_each_device=True)
def _get_real_symmetric_3x3_eigvals_kernel(sort='ascending', abs_sort=False):

    operation = """
    double x1, x2, phi;
    double a = static_cast<double>(aa);
    double b = static_cast<double>(bb);
    double c = static_cast<double>(cc);
    double d = static_cast<double>(dd);
    double e = static_cast<double>(ee);
    double f = static_cast<double>(ff);
    double d_sq = d * d;
    double e_sq = e * e;
    double f_sq = f * f;
    double tmpa = (2*a - b - c);
    double tmpb = (2*b - a - c);
    double tmpc = (2*c - a - b);
    x2 = - tmpa * tmpb * tmpc;
    x2 += 9 * (tmpc*d_sq + tmpb*f_sq + tmpa*e_sq);
    x2 -= 54 * (d * e * f);
    x1 = a*a + b*b + c*c - a*b - a*c - b*c + 3 * (d_sq + e_sq + f_sq);

    // grlee77: added max() here for numerical stability
    // (avoid NaN values in ridge filter test cases)
    x1 = max(x1, 0.0);

    if (x2 == 0.0) {
        phi = M_PI / 2.0;
    } else {
        // grlee77: added max() here for numerical stability
        // (avoid NaN values in test_hessian_matrix_eigvals_3d)
        double arg = max(4*x1*x1*x1 - x2*x2, 0.0);
        phi = atan(sqrt(arg)/x2);
        if (x2 < 0) {
            phi += M_PI;
        }
    }
    double x1_term = (2.0 / 3.0) * sqrt(x1);
    double abc = (a + b + c) / 3.0;
    lam1 = abc - x1_term * cos(phi / 3.0);
    lam2 = abc + x1_term * cos((phi - M_PI) / 3.0);
    lam3 = abc + x1_term * cos((phi + M_PI) / 3.0);
    """
    sort_template = """
        F stmp;
        if ({prefix}{var1} > {prefix}{var2}) {{
            stmp = {var2};
            {var2} = {var1};
            {var1} = stmp;
        }} if ({prefix}{var1} > {prefix}{var3}) {{
            stmp = {var3};
            {var3} = {var1};
            {var1} = stmp;
        }} if ({prefix}{var2} > {prefix}{var3}) {{
            stmp = {var3};
            {var3} = {var2};
            {var2} = stmp;
        }}
    """
    if abs_sort:
        operation += """
        F abs_lam1 = abs(lam1);
        F abs_lam2 = abs(lam2);
        F abs_lam3 = abs(lam3);
        """
        prefix = "abs_"
    else:
        prefix = ""
    if sort == 'ascending':
        var1 = "lam1"
        var3 = "lam3"
    elif sort == 'descending':
        var1 = "lam3"
        var3 = "lam1"
    operation += sort_template.format(
        prefix=prefix, var1=var1, var2="lam2", var3=var3
    )
    return cp.ElementwiseKernel(
        in_params="F aa, F bb, F cc, F dd, F ee, F ff",
        out_params="F lam1, F lam2, F lam3",
        operation=operation,
        name="cucim_skimage_symmetric_eig33_kernel")


def _image_orthogonal_matrix33_eigvals(
    a, d, f, b, e, c, sort='descending', abs_sort=False
):
    r"""Analytical expressions of the eigenvalues of a symmetric 3 x 3 matrix.

    Follows the expressions given for hermitian symmetric 3 x 3 matrices in
    [1]_, but simplified to handle real-valued matrices only.

    We are computing moments at each voxel of the volume, so each of ``a``,
    ``d``, ``f``, ``b``, ``e``, and ``c`` will be equal in shape to the 3D
    volume.

    Invidual arguments correspond to the following moment matrix entries

    .. math::

    M = \begin{bmatrix}
    a & d & f\\
    d & b & e\\
    f & e & c
    \end{bmatrix}

    Parameters
    ----------
    a, d, f, b, e, c : cp.ndarray
        Images corresponding to the individual components of the matrix, `M`,
        shown above. For example, ``d = M[0, 1]``.
    sort : {"ascending", "descending"}, optional
        Eigenvalues should be sorted in the specified order.
    abs_sort : boolean, optional
        If ``True``, sort based on the absolute values.

    References
    ----------
    .. [1] C. Deledalle, L. Denis, S. Tabti, F. Tupin. Closed-form expressions
        of the eigen decomposition of 2 x 2 and 3 x 3 Hermitian matrices.
        [Research Report] Université de Lyon. 2017.
        https://hal.archives-ouvertes.fr/hal-01501221/file/matrix_exp_and_log_formula.pdf
    """  # noqa
    if a.dtype.kind != "f":
        raise ValueError("expected real-valued floating point matrices")
    kernel = _get_real_symmetric_3x3_eigvals_kernel(
        sort=sort, abs_sort=abs_sort
    )
    eigs = cp.empty((3,) + a.shape, dtype=a.dtype)
    kernel(a, b, c, d, e, f, eigs[0], eigs[1], eigs[2])
    return eigs


def _symmetric_compute_eigenvalues(S_elems, sort='descending', abs_sort=False):
    """Compute eigenvalues from the upper-diagonal entries of a symmetric
    matrix.

    Parameters
    ----------
    S_elems : list of ndarray
        The upper-diagonal elements of the matrix, as returned by
        `hessian_matrix` or `structure_tensor`.
    sort : {"ascending", "descending"}, optional
        Eigenvalues should be sorted in the specified order.
    abs_sort : boolean, optional
        If ``True``, sort based on the absolute values.

    Returns
    -------
    eigs : ndarray
        The eigenvalues of the matrix, sorted in the specified order. The
        eigenvalues are the leading dimension. That is, ``eigs[i, j, k]``
        contains the ith eigenvalue at position (j, k).

    Notes
    -----
    In 2D and 3D, analytical formulas as given in [1]_ are used. For the nD
    case, the implementation is memory-inefficient as a large intermediate
    matrix is formed and used with NumPy's general symmetric eigenvalue
    solver.

    References
    ----------
    .. [1] C. Deledalle, L. Denis, S. Tabti, F. Tupin. Closed-form expressions
        of the eigen decomposition of 2 x 2 and 3 x 3 Hermitian matrices.
        [Research Report] Université de Lyon. 2017.
        https://hal.archives-ouvertes.fr/hal-01501221/file/matrix_exp_and_log_formula.pdf
    """

    if len(S_elems) == 3:  # Use fast analytical kernel for 2D
        eigs = _image_orthogonal_matrix22_eigvals(
            *S_elems, sort=sort, abs_sort=abs_sort
        )
    elif len(S_elems) == 6:  # Use fast analytical kernel for 3D
        eigs = _image_orthogonal_matrix33_eigvals(
            *S_elems, sort=sort, abs_sort=abs_sort
        )
    else:
        # n-dimensional case. warning: extremely memory inefficient!
        matrices = _symmetric_image(S_elems)
        # eigvalsh returns eigenvalues in increasing order. We want decreasing
        eigs = cp.linalg.eigvalsh(matrices)
        leading_axes = tuple(range(eigs.ndim - 1))
        eigs = cp.transpose(eigs, (eigs.ndim - 1,) + leading_axes)
        if abs_sort:
            # (sort by magnitude)
            eigs = cp.take_along_axis(eigs, cp.abs(eigs).argsort(0), 0)
        if sort == 'descending':
            eigs = eigs[::-1, ...]
    return eigs


def _symmetric_image(S_elems):
    """Convert the upper-diagonal elements of a matrix to the full
    symmetric matrix.

    Parameters
    ----------
    S_elems : list of array
        The upper-diagonal elements of the matrix, as returned by
        `hessian_matrix` or `structure_tensor`.

    Returns
    -------
    image : array
        An array of shape ``(M, N[, ...], image.ndim, image.ndim)``,
        containing the matrix corresponding to each coordinate.
    """
    image = S_elems[0]
    symmetric_image = cp.zeros(image.shape + (image.ndim, image.ndim),
                               dtype=image.dtype)
    for idx, (row, col) in enumerate(
        combinations_with_replacement(range(image.ndim), 2)
    ):
        symmetric_image[..., row, col] = S_elems[idx]
        symmetric_image[..., col, row] = S_elems[idx]
    return symmetric_image


def structure_tensor_eigenvalues(A_elems):
    """Compute eigenvalues of structure tensor.

    Parameters
    ----------
    A_elems : list of ndarray
        The upper-diagonal elements of the structure tensor, as returned
        by `structure_tensor`.

    Returns
    -------
    ndarray
        The eigenvalues of the structure tensor, in decreasing order. The
        eigenvalues are the leading dimension. That is, the coordinate
        [i, j, k] corresponds to the ith-largest eigenvalue at position (j, k).

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage.feature import structure_tensor
    >>> from cucim.skimage.feature import structure_tensor_eigenvalues
    >>> square = cp.zeros((5, 5))
    >>> square[2, 2] = 1
    >>> A_elems = structure_tensor(square, sigma=0.1, order='rc')
    >>> structure_tensor_eigenvalues(A_elems)[0]
    array([[0., 0., 0., 0., 0.],
           [0., 2., 4., 2., 0.],
           [0., 4., 0., 4., 0.],
           [0., 2., 4., 2., 0.],
           [0., 0., 0., 0., 0.]])

    See also
    --------
    structure_tensor
    """
    return _symmetric_compute_eigenvalues(A_elems)


def hessian_matrix_eigvals(H_elems):
    """Compute eigenvalues of Hessian matrix.

    Parameters
    ----------
    H_elems : list of ndarray
        The upper-diagonal elements of the Hessian matrix, as returned
        by `hessian_matrix`.

    Returns
    -------
    eigs : ndarray
        The eigenvalues of the Hessian matrix, in decreasing order. The
        eigenvalues are the leading dimension. That is, ``eigs[i, j, k]``
        contains the ith-largest eigenvalue at position (j, k).

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage.feature import (hessian_matrix,
    ...                                    hessian_matrix_eigvals)
    >>> square = cp.zeros((5, 5))
    >>> square[2, 2] = 4
    >>> H_elems = hessian_matrix(square, sigma=0.1, order='rc',
    ...                          use_gaussian_derivatives=False)
    >>> hessian_matrix_eigvals(H_elems)[0]
    array([[ 0.,  0.,  2.,  0.,  0.],
           [ 0.,  1.,  0.,  1.,  0.],
           [ 2.,  0., -2.,  0.,  2.],
           [ 0.,  1.,  0.,  1.,  0.],
           [ 0.,  0.,  2.,  0.,  0.]])
    """
    return _symmetric_compute_eigenvalues(H_elems)


def shape_index(image, sigma=1, mode="constant", cval=0):
    """Compute the shape index.

    The shape index, as defined by Koenderink & van Doorn [1]_, is a
    single valued measure of local curvature, assuming the image as a 3D plane
    with intensities representing heights.

    It is derived from the eigenvalues of the Hessian, and its
    value ranges from -1 to 1 (and is undefined (=NaN) in *flat* regions),
    with following ranges representing following shapes:

    .. table:: Ranges of the shape index and corresponding shapes.

      ===================  =============
      Interval (s in ...)  Shape
      ===================  =============
      [  -1, -7/8)         Spherical cup
      [-7/8, -5/8)         Through
      [-5/8, -3/8)         Rut
      [-3/8, -1/8)         Saddle rut
      [-1/8, +1/8)         Saddle
      [+1/8, +3/8)         Saddle ridge
      [+3/8, +5/8)         Ridge
      [+5/8, +7/8)         Dome
      [+7/8,   +1]         Spherical cap
      ===================  =============

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    sigma : float, optional
        Standard deviation used for the Gaussian kernel, which is used for
        smoothing the input data before Hessian eigen value calculation.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.

    Returns
    -------
    s : ndarray
        Shape index

    References
    ----------
    .. [1] Koenderink, J. J. & van Doorn, A. J.,
           "Surface shape and curvature scales",
           Image and Vision Computing, 1992, 10, 557-564.
           :DOI:`10.1016/0262-8856(92)90076-F`

    Examples
    --------
    >>> from cucim.skimage.feature import shape_index
    >>> square = cp.zeros((5, 5))
    >>> square[2, 2] = 4
    >>> s = shape_index(square, sigma=0.1)
    >>> s
    array([[ nan,  nan, -0.5,  nan,  nan],
           [ nan, -0. ,  nan, -0. ,  nan],
           [-0.5,  nan, -1. ,  nan, -0.5],
           [ nan, -0. ,  nan, -0. ,  nan],
           [ nan,  nan, -0.5,  nan,  nan]])
    """

    H = hessian_matrix(image, sigma=sigma, mode=mode, cval=cval, order="rc",
                       use_gaussian_derivatives=False)
    l1, l2 = hessian_matrix_eigvals(H)

    # don't warn on divide by 0 as occurs in the docstring example
    with np.errstate(divide='ignore', invalid='ignore'):
        return (2.0 / np.pi) * np.arctan((l2 + l1) / (l2 - l1))


@cp.memoize(for_each_device=True)
def _get_kitchen_rosenfeld_kernel():
    return cp.ElementwiseKernel(
        in_params='F imx, F imy, F imxx, F imxy, F imyy',
        out_params='F response',
        operation="""
        F numerator, denominator, imx_sq, imy_sq;
        imx_sq = imx * imx;
        imy_sq = imy * imy;
        denominator = imx_sq;
        denominator += imy_sq;
        if (denominator == 0) {
            response = 0.0;
        } else {
            numerator = imxx * imy_sq + imyy * imx_sq - 2 * imxy * imx * imy;
            response = numerator / denominator;
        }
        """,  # noqa
        name='cucim_feature_kitchen_rosenfeld'
    )


def corner_kitchen_rosenfeld(image, mode="constant", cval=0):
    """Compute Kitchen and Rosenfeld corner measure response image.

    The corner measure is calculated as follows::

        (imxx * imy**2 + imyy * imx**2 - 2 * imxy * imx * imy)
            / (imx**2 + imy**2)

    Where imx and imy are the first and imxx, imxy, imyy the second
    derivatives.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.

    Returns
    -------
    response : ndarray
        Kitchen and Rosenfeld response image.

    References
    ----------
    .. [1] Kitchen, L., & Rosenfeld, A. (1982). Gray-level corner detection.
           Pattern recognition letters, 1(2), 95-102.
           :DOI:`10.1016/0167-8655(82)90020-4`
    """

    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)

    imy, imx = _compute_derivatives(image, mode=mode, cval=cval)
    imxy, imxx = _compute_derivatives(imx, mode=mode, cval=cval)
    imyy, _ = _compute_derivatives(imy, mode=mode, cval=cval)

    kernel = _get_kitchen_rosenfeld_kernel()
    response = cp.empty_like(image, order='C')
    return kernel(imx, imy, imxx, imxy, imyy, response)


@cp.memoize(for_each_device=True)
def _get_corner_harris_k_kernel():
    return cp.ElementwiseKernel(
        in_params='F Arr, F Acc, F Arc, float64 k',
        out_params='F response',
        operation="""
        F detA, traceA;
        // determinant
        detA = Arr * Acc - Arc * Arc;
        // trace
        traceA = Arr + Acc;
        response = detA - k * traceA * traceA;
        """,
        name='cucim_skimage_feature_corner_harris_k'
    )


@cp.memoize(for_each_device=True)
def _get_corner_harris_kernel():
    return cp.ElementwiseKernel(
        in_params='F Arr, F Acc, F Arc, float64 eps',
        out_params='F response',
        operation="""
        F detA, traceA;
        // determinant
        detA = Arr * Acc - Arc * Arc;
        // trace
        traceA = Arr + Acc;
        response = 2 * detA / (traceA + eps);
        """,
        name='cucim_skimage_feature_corner_harris_k'
    )


def corner_harris(image, method="k", k=0.05, eps=1e-6, sigma=1):
    """Compute Harris corner measure response image.

    This corner detector uses information from the auto-correlation matrix A::

        A = [(imx**2)   (imx*imy)] = [Axx Axy]
            [(imx*imy)   (imy**2)]   [Axy Ayy]

    Where imx and imy are first derivatives, averaged with a gaussian filter.
    The corner measure is then defined as::

        det(A) - k * trace(A)**2

    or::

        2 * det(A) / (trace(A) + eps)

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    method : {'k', 'eps'}, optional
        Method to compute the response image from the auto-correlation matrix.
    k : float, optional
        Sensitivity factor to separate corners from edges, typically in range
        `[0, 0.2]`. Small values of k result in detection of sharp corners.
    eps : float, optional
        Normalisation factor (Noble's corner measure).
    sigma : float, optional
        Standard deviation used for the Gaussian kernel, which is used as
        weighting function for the auto-correlation matrix.

    Returns
    -------
    response : ndarray
        Harris response image.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Corner_detection

    Examples
    --------
    >>> from cucim.skimage.feature import corner_harris, corner_peaks
    >>> square = cp.zeros([10, 10])
    >>> square[2:8, 2:8] = 1
    >>> square.astype(int)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> corner_peaks(corner_harris(square), min_distance=1)  # doctest: +SKIP
    array([[2, 2],
           [2, 7],
           [7, 2],
           [7, 7]])

    """
    Arr, Arc, Acc = structure_tensor(image, sigma, order="rc")
    response = cp.zeros_like(Arr)
    if method == "k":
        kernel = _get_corner_harris_k_kernel()
        kernel(Arr, Acc, Arc, k, response)
    else:
        kernel = _get_corner_harris_kernel()
        kernel(Arr, Acc, Arc, eps, response)
    return response


@cp.memoize(for_each_device=True)
def _get_shi_tomasi_kernel():
    return cp.ElementwiseKernel(
        in_params='F Arr, F Acc, F Arc',
        out_params='F response',
        operation="""
        F tmp;
        tmp = (Arr - Acc);
        tmp *= tmp;
        response = (Arr + Acc - sqrt(tmp + 4 * Arc * Arc)) / 2.0;
        """,
        name='cucim_skimage_feature_shi_tomasi'
    )


def corner_shi_tomasi(image, sigma=1):
    """Compute Shi-Tomasi (Kanade-Tomasi) corner measure response image.

    This corner detector uses information from the auto-correlation matrix A::

        A = [(imx**2)   (imx*imy)] = [Axx Axy]
            [(imx*imy)   (imy**2)]   [Axy Ayy]

    Where imx and imy are first derivatives, averaged with a gaussian filter.
    The corner measure is then defined as the smaller eigenvalue of A::

        ((Axx + Ayy) - sqrt((Axx - Ayy)**2 + 4 * Axy**2)) / 2

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    sigma : float, optional
        Standard deviation used for the Gaussian kernel, which is used as
        weighting function for the auto-correlation matrix.

    Returns
    -------
    response : ndarray
        Shi-Tomasi response image.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Corner_detection

    Examples
    --------
    >>> from cucim.skimage.feature import corner_shi_tomasi, corner_peaks
    >>> square = cp.zeros([10, 10])
    >>> square[2:8, 2:8] = 1
    >>> square.astype(int)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> corner_peaks(corner_shi_tomasi(square),
    ...              min_distance=1)  # doctest: +SKIP
    array([[2, 2],
           [2, 7],
           [7, 2],
           [7, 7]])

    """
    Arr, Arc, Acc = structure_tensor(image, sigma, order="rc")
    # minimum eigenvalue of A
    response = cp.zeros_like(Arr)
    kernel = _get_shi_tomasi_kernel()
    return kernel(Arr, Acc, Arc, response)


@cp.memoize(for_each_device=True)
def _get_foerstner_kernel():
    return cp.ElementwiseKernel(
        in_params='F Arr, F Acc, F Arc',
        out_params='F w, F q',
        operation="""
        F detA, traceA;

        // determinant
        detA = Arr * Acc - Arc * Arc;
        // trace
        traceA = Arr + Acc;
        if (traceA == 0) {
            w = 0;
            q = 0;
        } else {
            w = detA / traceA;
            q = 4 * detA / (traceA * traceA);
        }
        """,
        name='cucim_skimage_feature_forstner'
    )


def corner_foerstner(image, sigma=1):
    """Compute Foerstner corner measure response image.

    This corner detector uses information from the auto-correlation matrix A::

        A = [(imx**2)   (imx*imy)] = [Axx Axy]
            [(imx*imy)   (imy**2)]   [Axy Ayy]

    Where imx and imy are first derivatives, averaged with a gaussian filter.
    The corner measure is then defined as::

        w = det(A) / trace(A)           (size of error ellipse)
        q = 4 * det(A) / trace(A)**2    (roundness of error ellipse)

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    sigma : float, optional
        Standard deviation used for the Gaussian kernel, which is used as
        weighting function for the auto-correlation matrix.

    Returns
    -------
    w : ndarray
        Error ellipse sizes.
    q : ndarray
        Roundness of error ellipse.

    References
    ----------
    .. [1] Förstner, W., & Gülch, E. (1987, June). A fast operator for
           detection and precise location of distinct points, corners and
           centres of circular features. In Proc. ISPRS intercommission
           conference on fast processing of photogrammetric data (pp. 281-305).
    .. [2] https://en.wikipedia.org/wiki/Corner_detection

    Examples
    --------
    >>> from cucim.skimage.feature import corner_foerstner, corner_peaks
    >>> square = cp.zeros([10, 10])
    >>> square[2:8, 2:8] = 1
    >>> square.astype(int)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> w, q = corner_foerstner(square)
    >>> accuracy_thresh = 0.5
    >>> roundness_thresh = 0.3
    >>> foerstner = (q > roundness_thresh) * (w > accuracy_thresh) * w
    >>> corner_peaks(foerstner, min_distance=1)  # doctest: +SKIP
    array([[2, 2],
           [2, 7],
           [7, 2],
           [7, 7]])

    """
    Arr, Arc, Acc = structure_tensor(image, sigma, order="rc")
    w = cp.empty_like(Arr)
    q = cp.empty_like(Arr)
    kernel = _get_foerstner_kernel()
    return kernel(Arr, Acc, Arc, w, q)


def corner_peaks(
    image,
    min_distance=1,
    threshold_abs=None,
    threshold_rel=None,
    exclude_border=True,
    indices=True,
    num_peaks=np.inf,
    footprint=None,
    labels=None,
    *,
    num_peaks_per_label=np.inf,
    p_norm=np.inf,
):
    """Find peaks in corner measure response image.

    This differs from `skimage.feature.peak_local_max` in that it suppresses
    multiple connected peaks with the same accumulator value.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    min_distance : int, optional
        The minimal allowed distance separating peaks.
    * : *
        See :py:meth:`skimage.feature.peak_local_max`.
    p_norm : float
        Which Minkowski p-norm to use. Should be in the range [1, inf].
        A finite large p may cause a ValueError if overflow can occur.
        ``inf`` corresponds to the Chebyshev distance and 2 to the
        Euclidean distance.

    Returns
    -------
    output : ndarray or ndarray of bools

        * If `indices = True`  : (row, column, ...) coordinates of peaks.
        * If `indices = False` : Boolean array shaped like `image`, with peaks
          represented by True values.

    See also
    --------
    skimage.feature.peak_local_max

    Notes
    -----
    .. versionchanged:: 0.18
        The default value of `threshold_rel` has changed to None, which
        corresponds to letting `skimage.feature.peak_local_max` decide on the
        default. This is equivalent to `threshold_rel=0`.

    The `num_peaks` limit is applied before suppression of connected peaks.
    To limit the number of peaks after suppression, set `num_peaks=np.inf` and
    post-process the output of this function.

    Examples
    --------
    >>> from cucim.skimage.feature import peak_local_max
    >>> response = cp.zeros((5, 5))
    >>> response[2:4, 2:4] = 1
    >>> response
    array([[0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 1., 1., 0.],
           [0., 0., 1., 1., 0.],
           [0., 0., 0., 0., 0.]])
    >>> peak_local_max(response)
    array([[2, 2],
           [2, 3],
           [3, 2],
           [3, 3]])
    >>> corner_peaks(response)
    array([[2, 2]])

    """
    if cp.isinf(num_peaks):
        num_peaks = None

    # Get the coordinates of the detected peaks
    coords = peak_local_max(
        image,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        threshold_rel=threshold_rel,
        exclude_border=exclude_border,
        num_peaks=np.inf,
        footprint=footprint,
        labels=labels,
        num_peaks_per_label=num_peaks_per_label,
    )

    if len(coords):
        # TODO: modify to do KDTree on the GPU (cuSpatial?)
        coords = cp.asnumpy(coords)

        # Use KDtree to find the peaks that are too close to each other
        tree = spatial.cKDTree(coords)

        rejected_peaks_indices = set()
        for idx, point in enumerate(coords):
            if idx not in rejected_peaks_indices:
                candidates = tree.query_ball_point(
                    point, r=min_distance, p=p_norm
                )
                candidates.remove(idx)
                rejected_peaks_indices.update(candidates)

        # Remove the peaks that are too close to each other
        coords = np.delete(coords, tuple(rejected_peaks_indices), axis=0)[
            :num_peaks
        ]
        coords = cp.asarray(coords)

    if indices:
        return coords

    peaks = cp.zeros_like(image, dtype=bool)
    peaks[tuple(coords.T)] = True

    return peaks
