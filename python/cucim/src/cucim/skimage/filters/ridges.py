"""
Ridge filters.

Ridge filters can be used to detect continuous edges, such as vessels,
neurites, wrinkles, rivers, and other tube-like structures. The present
class of ridge filters relies on the eigenvalues of the Hessian matrix of
image intensities to detect tube-like structures where the intensity changes
perpendicular but not along the structure.
"""

import math
from warnings import warn

import cupy as cp
import numpy as np

from .._shared.utils import _supported_float_type, check_nD
from ..feature.corner import (
    _symmetric_compute_eigenvalues,
    hessian_matrix,
    hessian_matrix_eigvals,
)


@cp.memoize(for_each_device=True)
def _get_circulant_init_kernel(ndim, alpha):
    operation = f"""
    for (int x=0; x < {ndim}; x++) {{
        for (int y=0; y < {ndim}; y++) {{
            if (x == y) {{
               out[y + x*{ndim}] = 1.0;
            }} else {{
               out[y + x*{ndim}] = {alpha};
            }}
        }}
    }}
    """
    return cp.ElementwiseKernel(
        "",
        "raw F out",
        operation=operation,
        name=f"cucim_circulant_init_{ndim}d_alpha{int(1000*alpha)}",
    )


def meijering(
    image,
    sigmas=range(1, 10, 2),
    alpha=None,
    black_ridges=True,
    mode="reflect",
    cval=0,
):
    """
    Filter an image with the Meijering neuriteness filter.

    This filter can be used to detect continuous ridges, e.g. neurites,
    wrinkles, rivers. It can be used to calculate the fraction of the
    whole image containing such objects.

    Calculates the eigenvalues of the Hessian to compute the similarity of
    an image region to neurites, according to the method described in [1]_.

    Parameters
    ----------
    image : (N, M[, ..., P]) ndarray
        Array with input image data.
    sigmas : iterable of floats, optional
        Sigmas used as scales of filter
    alpha : float, optional
        Shaping filter constant, that selects maximally flat elongated
        features.  The default, None, selects the optimal value -1/(ndim+1).
    black_ridges : boolean, optional
        When True (the default), the filter detects black ridges; when
        False, it detects white ridges.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.

    Returns
    -------
    out : (N, M[, ..., P]) ndarray
        Filtered image (maximum of pixels across all scales).

    See also
    --------
    sato
    frangi
    hessian

    References
    ----------
    .. [1] Meijering, E., Jacob, M., Sarria, J. C., Steiner, P., Hirling, H.,
        Unser, M. (2004). Design and validation of a tool for neurite tracing
        and analysis in fluorescence microscopy images. Cytometry Part A,
        58(2), 167-176.
        :DOI:`10.1002/cyto.a.20022`
    """

    image = image.astype(_supported_float_type(image.dtype), copy=False)
    if not black_ridges:  # Normalize to black ridges.
        image = -image

    if alpha is None:
        alpha = 1 / (image.ndim + 1)

    # tiny matrix, so just use a single threaded kernel to assign the values
    circulant_init_kernel = _get_circulant_init_kernel(image.ndim, alpha)
    mtx = cp.empty((image.ndim, image.ndim), dtype=image.dtype)
    circulant_init_kernel(mtx, size=1)

    # Generate empty array for storing maximum value
    # from different (sigma) scales
    filtered_max = cp.zeros_like(image)
    for sigma in sigmas:  # Filter for all sigmas.
        H = hessian_matrix(
            image, sigma, mode=mode, cval=cval, use_gaussian_derivatives=True
        )
        eigvals = hessian_matrix_eigvals(H)

        # cucim's hessian_matrix differs numerically from the one in skimage.
        # Sometimes where skimage returns 0, it returns very small values
        # (1e-15-1e-14). Here we set values < 1e-12 to 0 to better replicate
        # the same behavior.
        eigvals[abs(eigvals) < 1e-12] = 0.0

        # Compute normalized eigenvalues l_i = e_i + sum_{j!=i} alpha * e_j.
        vals = cp.tensordot(mtx, eigvals, 1)
        # Get largest normalized eigenvalue (by magnitude) at each pixel.
        vals = cp.take_along_axis(vals, abs(vals).argmax(0)[None], 0).squeeze(0)
        # Remove negative values.
        vals = cp.maximum(vals, 0)
        # Normalize to max = 1 (unless everything is already zero).
        max_val = vals.max()
        if max_val > 0:
            vals /= max_val
        filtered_max = cp.maximum(filtered_max, vals)

    return filtered_max  # Return pixel-wise max over all sigmas.


def sato(
    image, sigmas=range(1, 10, 2), black_ridges=True, mode="reflect", cval=0
):
    """
    Filter an image with the Sato tubeness filter.

    This filter can be used to detect continuous ridges, e.g. tubes,
    wrinkles, rivers. It can be used to calculate the fraction of the
    whole image containing such objects.

    Defined only for 2-D and 3-D images. Calculates the eigenvalues of the
    Hessian to compute the similarity of an image region to tubes, according to
    the method described in [1]_.

    Parameters
    ----------
    image : (N, M[, P]) ndarray
        Array with input image data.
    sigmas : iterable of floats, optional
        Sigmas used as scales of filter.
    black_ridges : boolean, optional
        When True (the default), the filter detects black ridges; when
        False, it detects white ridges.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.

    Returns
    -------
    out : (N, M[, P]) ndarray
        Filtered image (maximum of pixels across all scales).

    See also
    --------
    meijering
    frangi
    hessian

    References
    ----------
    .. [1] Sato, Y., Nakajima, S., Shiraga, N., Atsumi, H., Yoshida, S.,
        Koller, T., ..., Kikinis, R. (1998). Three-dimensional multi-scale line
        filter for segmentation and visualization of curvilinear structures in
        medical images. Medical image analysis, 2(2), 143-168.
        :DOI:`10.1016/S1361-8415(98)80009-1`
    """

    check_nD(image, [2, 3])  # Check image dimensions.
    image = image.astype(_supported_float_type(image.dtype), copy=False)
    if not black_ridges:  # Normalize to black ridges.
        image = -image

    # Generate empty array for storing maximum value
    # from different (sigma) scales
    filtered_max = cp.zeros_like(image)
    for sigma in sigmas:  # Filter for all sigmas.
        H = hessian_matrix(
            image, sigma, mode=mode, cval=cval, use_gaussian_derivatives=True
        )
        eigvals = hessian_matrix_eigvals(H)

        # cucim's hessian_matrix differs numerically from the one in skimage.
        # Sometimes where skimage returns 0, it returns very small values
        # (1e-15-1e-14). Here we set values < 1e-12 to 0 to better replicate
        # the same behavior.
        eigvals[abs(eigvals) < 1e-12] = 0.0

        # Compute normalized tubeness (eqs. (9) and (22), ref. [1]_) as the
        # geometric mean of eigvals other than the lowest one
        # (hessian_matrix_eigvals returns eigvals in decreasing order), clipped
        # to 0, multiplied by sigma^2.
        eigvals = eigvals[:-1]
        vals = cp.prod(cp.maximum(eigvals, 0), 0) ** (1 / len(eigvals))
        vals *= sigma**2
        filtered_max = cp.maximum(filtered_max, vals)
    return filtered_max  # Return pixel-wise max over all sigmas.


@cp.memoize(for_each_device=True)
def _get_frangi2d_sum_kernel():
    return cp.ElementwiseKernel(
        in_params="F lambda1, F lambda2",  # noqa
        out_params="F r_g",
        operation="""
        // Compute sensitivity to areas of high variance/texture/structure,
        // see equation (12)in reference [1]_
        r_g = lambda1 * lambda1;
        r_g += lambda2 * lambda2;
        """,
        name="cucim_skimage_filters_frangi3d_inner",
    )


@cp.memoize(for_each_device=True)
def _get_frangi2d_inner_kernel():
    return cp.ElementwiseKernel(
        in_params="F lambda1, F lambda2, F r_g, float64 beta_sq, float64 gamma_sq",  # noqa
        out_params="F result",
        operation="""
        F r_b;

        // Compute sensitivity to deviation from a blob-like structure,
        // see equations (10) and (15) in reference [1]_,
        // np.abs(lambda2) in 2D, np.sqrt(np.abs(lambda2 * lambda3)) in 3D
        // CuPy Backend: cp.multiply does not have a reduce method
        // filtered_raw = np.abs(np.multiply.reduce(lambdas))**(1/len(lambdas))
        r_b = abs(lambda1) / max(lambda2, static_cast<F>(1.0e-10));
        r_b *= r_b;

        // Filtered image, eq. (15).  Our implementation relies on the
        // blobness exponential factor underflowing to zero whenever the second
        // or third eigenvalues are negative (we clip them to 1e-10, to make
        // r_b very large).
        result = exp(-r_b / beta_sq);
        result *= 1.0 - exp(-r_g / gamma_sq);
        """,
        name="cucim_skimage_filters_frangi2d_inner",
    )


@cp.memoize(for_each_device=True)
def _get_frangi3d_sum_kernel():
    return cp.ElementwiseKernel(
        in_params="F lambda1, F lambda2, F lambda3",  # noqa
        out_params="F r_g",
        operation="""
        // Compute sensitivity to areas of high variance/texture/structure,
        // see equation (12)in reference [1]_
        r_g = lambda1 * lambda1;
        r_g += lambda2 * lambda2;
        r_g += lambda3 * lambda3;
        """,
        name="cucim_skimage_filters_frangi3d_inner",
    )


@cp.memoize(for_each_device=True)
def _get_frangi3d_inner_kernel():
    return cp.ElementwiseKernel(
        in_params="F lambda1, F lambda2, F lambda3, F r_g, float64 alpha_sq, float64 beta_sq, float64 gamma_sq",  # noqa
        out_params="F result",
        operation="""
        F r_a, r_b;

        F lam2 = max(lambda2, static_cast<F>(1.0e-10));
        F lam3 = max(lambda3, static_cast<F>(1.0e-10));

        // Compute sensitivity to deviation from a plate-like
        // structure (see equations (11) and (15) in reference [1]_).
        r_a = lam2 / lam3;
        r_a *= r_a;

        // Compute sensitivity to deviation from a blob-like structure,
        // see equations (10) and (15) in reference [1]_,
        // np.abs(lambda2) in 2D, np.sqrt(np.abs(lambda2 * lambda3)) in 3D
        // CuPy Backend: cp.multiply does not have a reduce method
        // filtered_raw = np.abs(np.multiply.reduce(lambdas))**(1/len(lambdas))
        r_b = (lambda1 * lambda1) / (lam2 * lam3);

        // Filtered image, eq. (13).  Our implementation relies on the
        // blobness exponential factor underflowing to zero whenever the second
        // or third eigenvalues are negative (we clip them to 1e-10, to make
        // r_b very large).
        result = 1.0 - exp(-r_a / alpha_sq);
        result *= exp(-r_b / beta_sq);
        result *= 1.0 - exp(-r_g / gamma_sq);

        """,
        name="cucim_skimage_filters_frangi3d_inner",
    )


def frangi(
    image,
    sigmas=range(1, 10, 2),
    scale_range=None,
    scale_step=None,
    alpha=0.5,
    beta=0.5,
    gamma=None,
    black_ridges=True,
    mode="reflect",
    cval=0,
):
    """
    Filter an image with the Frangi vesselness filter.

    This filter can be used to detect continuous ridges, e.g. vessels,
    wrinkles, rivers. It can be used to calculate the fraction of the
    whole image containing such objects.

    Defined only for 2-D and 3-D images. Calculates the eigenvalues of the
    Hessian to compute the similarity of an image region to vessels, according
    to the method described in [1]_.

    Parameters
    ----------
    image : (N, M[, P]) ndarray
        Array with input image data.
    sigmas : iterable of floats, optional
        Sigmas used as scales of filter, i.e.,
        np.arange(scale_range[0], scale_range[1], scale_step)
    scale_range : 2-tuple of floats, optional
        The range of sigmas used.
    scale_step : float, optional
        Step size between sigmas.
    alpha : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to deviation from a plate-like structure.
    beta : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to deviation from a blob-like structure.
    gamma : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to areas of high variance/texture/structure.
        The default, None, uses half of the maximum Hessian norm.
    black_ridges : boolean, optional
        When True (the default), the filter detects black ridges; when
        False, it detects white ridges.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.

    Returns
    -------
    out : (N, M[, P]) ndarray
        Filtered image (maximum of pixels across all scales).

    Notes
    -----
    Earlier versions of this filter were implemented by Marc Schrijver,
    (November 2001), D. J. Kroon, University of Twente (May 2009) [2]_, and
    D. G. Ellis (January 2017) [3]_.

    See also
    --------
    meijering
    sato
    hessian

    References
    ----------
    .. [1] Frangi, A. F., Niessen, W. J., Vincken, K. L., & Viergever, M. A.
        (1998,). Multiscale vessel enhancement filtering. In International
        Conference on Medical Image Computing and Computer-Assisted
        Intervention (pp. 130-137). Springer Berlin Heidelberg.
        :DOI:`10.1007/BFb0056195`
    .. [2] Kroon, D. J.: Hessian based Frangi vesselness filter.
    .. [3] Ellis, D. G.: https://github.com/ellisdg/frangi3d/tree/master/frangi
    """
    if scale_range is not None and scale_step is not None:
        warn(
            "Use keyword parameter `sigmas` instead of `scale_range` and "
            "`scale_range` which will be removed in version 0.17.",
            stacklevel=2,
        )
        sigmas = np.arange(scale_range[0], scale_range[1], scale_step)

    check_nD(image, [2, 3])  # Check image dimensions.
    image = image.astype(_supported_float_type(image.dtype), copy=False)
    if not black_ridges:  # Normalize to black ridges.
        image = -image

    alpha_sq = 2 * alpha * alpha
    beta_sq = 2 * beta * beta
    if gamma is not None:
        gamma_sq = 2 * gamma * gamma

    ndim = image.ndim
    if ndim == 2:
        inner_kernel = _get_frangi2d_inner_kernel()
    elif ndim == 3:
        inner_kernel = _get_frangi3d_inner_kernel()

    vals = cp.empty(image.shape, dtype=image.dtype)
    ev_sq_sum = cp.empty_like(vals)
    for i, sigma in enumerate(sigmas):  # Filter for all sigmas.
        H = hessian_matrix(
            image, sigma, mode=mode, cval=cval, use_gaussian_derivatives=True
        )

        # Use _symmetric_compute_eigenvalues rather than
        # hessian_matrix_eigvals so we can directly sort by ascending magnitude
        eigvals = _symmetric_compute_eigenvalues(
            H, sort="ascending", abs_sort=True
        )

        # compute squared sum of the eigenvalues
        if ndim == 2:
            ev_sq_sum_kernel = _get_frangi2d_sum_kernel()
            ev_sq_sum_kernel(eigvals[0], eigvals[1], ev_sq_sum)
        else:
            ev_sq_sum_kernel = _get_frangi3d_sum_kernel()
            ev_sq_sum_kernel(eigvals[0], eigvals[1], eigvals[2], ev_sq_sum)

        if gamma is None:
            s_max = float(ev_sq_sum.max())
            gamma = math.sqrt(s_max) / 2.0
            if s_max == 0:
                gamma_sq = 2.0  # If s == 0 everywhere, gamma doesn't matter.
            else:
                gamma_sq = max(2 * gamma * gamma, 1e-10)

        if ndim == 2:
            inner_kernel(
                eigvals[0], eigvals[1], ev_sq_sum, beta_sq, gamma_sq, vals
            )
        else:
            inner_kernel(
                eigvals[0],
                eigvals[1],
                eigvals[2],
                ev_sq_sum,
                alpha_sq,
                beta_sq,
                gamma_sq,
                vals,
            )

        # Store maximum value from different (sigma) scales
        if i == 0:
            filtered_max = vals.copy()
        else:
            filtered_max = cp.maximum(filtered_max, vals)
    return filtered_max  # Return pixel-wise max over all sigmas.


def hessian(
    image,
    sigmas=range(1, 10, 2),
    scale_range=None,
    scale_step=None,
    alpha=0.5,
    beta=0.5,
    gamma=15,
    black_ridges=True,
    mode="reflect",
    cval=0,
):
    """Filter an image with the Hybrid Hessian filter.

    This filter can be used to detect continuous edges, e.g. vessels,
    wrinkles, rivers. It can be used to calculate the fraction of the whole
    image containing such objects.

    Defined only for 2-D and 3-D images. Almost equal to Frangi filter, but
    uses alternative method of smoothing. Refer to [1]_ to find the differences
    between Frangi and Hessian filters.

    Parameters
    ----------
    image : (N, M[, P]) ndarray
        Array with input image data.
    sigmas : iterable of floats, optional
        Sigmas used as scales of filter, i.e.,
        np.arange(scale_range[0], scale_range[1], scale_step)
    scale_range : 2-tuple of floats, optional
        The range of sigmas used.
    scale_step : float, optional
        Step size between sigmas.
    beta : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to deviation from a blob-like structure.
    gamma : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to areas of high variance/texture/structure.
    black_ridges : boolean, optional
        When True (the default), the filter detects black ridges; when
        False, it detects white ridges.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.

    Returns
    -------
    out : (N, M[, P]) ndarray
        Filtered image (maximum of pixels across all scales).

    Notes
    -----
    Written by Marc Schrijver (November 2001)
    Re-Written by D. J. Kroon University of Twente (May 2009) [2]_

    See also
    --------
    meijering
    sato
    frangi

    References
    ----------
    .. [1] Ng, C. C., Yap, M. H., Costen, N., & Li, B. (2014,). Automatic
        wrinkle detection using hybrid Hessian filter. In Asian Conference on
        Computer Vision (pp. 609-622). Springer International Publishing.
        :DOI:`10.1007/978-3-319-16811-1_40`
    .. [2] Kroon, D. J.: Hessian based Frangi vesselness filter.
    """
    filtered = frangi(
        image,
        sigmas=sigmas,
        scale_range=scale_range,
        scale_step=scale_step,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        black_ridges=black_ridges,
        mode=mode,
        cval=cval,
    )

    filtered[filtered <= 0] = 1
    return filtered
