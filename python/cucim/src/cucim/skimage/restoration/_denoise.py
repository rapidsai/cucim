import functools

import cupy as cp

from cucim.skimage.util import img_as_float

from .._shared import utils
from .._shared.utils import _supported_float_type


def _denoise_tv_chambolle_nd(image, weight=0.1, eps=2.0e-4, max_num_iter=200):
    """Perform total-variation denoising on n-dimensional images.

    Parameters
    ----------
    image : ndarray
        n-D input data to be denoised.
    weight : float, optional
        Denoising weight. The greater `weight`, the more denoising (at
        the expense of fidelity to `input`).
    eps : float, optional
        Relative difference of the value of the cost function that determines
        the stop criterion. The algorithm stops when:

            (E_(n-1) - E_n) < eps * E_0

    max_num_iter : int, optional
        Maximal number of iterations used for the optimization.

    Returns
    -------
    out : ndarray
        Denoised array of floats.

    Notes
    -----
    Rudin, Osher and Fatemi algorithm.
    """

    ndim = image.ndim
    p = cp.zeros((image.ndim,) + image.shape, dtype=image.dtype)
    g = cp.zeros_like(p)
    d = cp.zeros_like(image)
    i = 0
    slices_g = [slice(None)] * (ndim + 1)
    slices_d = [slice(None)] * ndim
    slices_p = [slice(None)] * (ndim + 1)
    while i < max_num_iter:
        if i > 0:
            # d will be the (negative) divergence of p
            d = -p.sum(0)
            for ax in range(ndim):
                slices_d[ax] = slice(1, None)
                slices_p[ax + 1] = slice(0, -1)
                slices_p[0] = ax
                d[tuple(slices_d)] += p[tuple(slices_p)]
                slices_d[ax] = slice(None)
                slices_p[ax + 1] = slice(None)
            out = image + d
            E = (d * d).sum()
        else:
            out = image
            E = 0.0

        # g stores the gradients of out along each axis
        # e.g. g[0] is the first order finite difference along axis 0
        for ax in range(ndim):
            slices_g[ax + 1] = slice(0, -1)
            slices_g[0] = ax
            g[tuple(slices_g)] = cp.diff(out, axis=ax)
            slices_g[ax + 1] = slice(None)

        norm = (g * g).sum(axis=0, keepdims=True)
        cp.sqrt(norm, out=norm)
        E += weight * norm.sum()
        tau = 1.0 / (2.0 * ndim)
        norm *= tau / weight
        norm += 1.0
        p -= tau * g
        p /= norm
        E /= float(image.size)
        if i == 0:
            E_init = E
            E_previous = E
        else:
            if abs(E_previous - E) < eps * E_init:
                break
            else:
                E_previous = E
        i += 1
    return out


@utils.deprecate_kwarg({'n_iter_max': 'max_num_iter'},
                       removed_version="23.02.00",
                       deprecated_version="22.06.00")
@utils.deprecate_multichannel_kwarg(multichannel_position=4)
def denoise_tv_chambolle(image, weight=0.1, eps=2.0e-4, max_num_iter=200,
                         multichannel=False, *, channel_axis=None):
    """Perform total-variation denoising on n-dimensional images.

    Parameters
    ----------
    image : ndarray of ints, uints or floats
        Input data to be denoised. `image` can be of any numeric type,
        but it is cast into an ndarray of floats for the computation
        of the denoised image.
    weight : float, optional
        Denoising weight. The greater `weight`, the more denoising (at
        the expense of fidelity to `input`).
    eps : float, optional
        Relative difference of the value of the cost function that
        determines the stop criterion. The algorithm stops when:

            (E_(n-1) - E_n) < eps * E_0

    max_num_iter : int, optional
        Maximal number of iterations used for the optimization.
    multichannel : bool, optional
        Apply total-variation denoising separately for each channel. This
        option should be true for color images, otherwise the denoising is
        also applied in the channels dimension. This argument is deprecated:
        specify `channel_axis` instead.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

    Returns
    -------
    out : ndarray
        Denoised image.

    Notes
    -----
    Make sure to set the multichannel parameter appropriately for color images.

    The principle of total variation denoising is explained in
    https://en.wikipedia.org/wiki/Total_variation_denoising

    The principle of total variation denoising is to minimize the
    total variation of the image, which can be roughly described as
    the integral of the norm of the image gradient. Total variation
    denoising tends to produce "cartoon-like" images, that is,
    piecewise-constant images.

    This code is an implementation of the algorithm of Rudin, Fatemi and Osher
    that was proposed by Chambolle in [1]_.

    References
    ----------
    .. [1] A. Chambolle, An algorithm for total variation minimization and
           applications, Journal of Mathematical Imaging and Vision,
           Springer, 2004, 20, 89-97.

    Examples
    --------
    2D example on astronaut image:

    >>> import cupy as cp
    >>> from cucim.skimage import color
    >>> from skimage import data
    >>> img = color.rgb2gray(cp.array(data.astronaut()[:50, :50]))
    >>> img += 0.5 * img.std() * cp.random.randn(*img.shape)
    >>> denoised_img = denoise_tv_chambolle(img, weight=60)

    3D example on synthetic data:

    >>> x, y, z = cp.ogrid[0:20, 0:20, 0:20]
    >>> mask = (x - 22)**2 + (y - 20)**2 + (z - 17)**2 < 8**2
    >>> mask = mask.astype(float)
    >>> mask += 0.2*cp.random.randn(*mask.shape)
    >>> res = denoise_tv_chambolle(mask, weight=100)

    """

    im_type = image.dtype
    if not im_type.kind == 'f':
        image = img_as_float(image)

    # enforce float16->float32 and float128->float64
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)

    if channel_axis is not None:
        channel_axis = channel_axis % image.ndim
        _at = functools.partial(utils.slice_at_axis, axis=channel_axis)
        out = cp.zeros_like(image)
        for c in range(image.shape[channel_axis]):
            out[_at(c)] = _denoise_tv_chambolle_nd(image[_at(c)], weight, eps,
                                                   max_num_iter)
    else:
        out = _denoise_tv_chambolle_nd(image, weight, eps, max_num_iter)
    return out
