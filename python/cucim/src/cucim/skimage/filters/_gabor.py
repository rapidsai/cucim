import math

import cupy as cp
from cupyx.scipy import ndimage as ndi

from .._shared.utils import _supported_float_type, check_nD, warn

__all__ = ["gabor_kernel", "gabor"]


def _sigma_prefactor(bandwidth):
    b = bandwidth
    # See http://www.cs.rug.nl/~imaging/simplecell.html
    return 1.0 / math.pi * math.sqrt(math.log(2) / 2.0) * \
        (2.0 ** b + 1) / (2.0 ** b - 1)


def gabor_kernel(frequency, theta=0, bandwidth=1, sigma_x=None, sigma_y=None,
                 n_stds=3, offset=0, dtype=None, *, float_dtype=None):
    """Return complex 2D Gabor filter kernel.

    Gabor kernel is a Gaussian kernel modulated by a complex harmonic function.
    Harmonic function consists of an imaginary sine function and a real
    cosine function. Spatial frequency is inversely proportional to the
    wavelength of the harmonic and to the standard deviation of a Gaussian
    kernel. The bandwidth is also inversely proportional to the standard
    deviation.

    Parameters
    ----------
    frequency : float
        Spatial frequency of the harmonic function. Specified in pixels.
    theta : float, optional
        Orientation in radians. If 0, the harmonic is in the x-direction.
    bandwidth : float, optional
        The bandwidth captured by the filter. For fixed bandwidth, ``sigma_x``
        and ``sigma_y`` will decrease with increasing frequency. This value is
        ignored if ``sigma_x`` and ``sigma_y`` are set by the user.
    sigma_x, sigma_y : float, optional
        Standard deviation in x- and y-directions. These directions apply to
        the kernel *before* rotation. If `theta = pi/2`, then the kernel is
        rotated 90 degrees so that ``sigma_x`` controls the *vertical*
        direction.
    n_stds : scalar, optional
        The linear size of the kernel is n_stds (3 by default) standard
        deviations
    offset : float, optional
        Phase offset of harmonic function in radians.
    dtype : {np.complex64, np.complex128}
        Specifies if the filter is single or double precision complex.

    Returns
    -------
    g : complex array
        Complex filter kernel.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Gabor_filter
    .. [2] https://web.archive.org/web/20180127125930/http://mplab.ucsd.edu/tutorials/gabor.pdf

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage.filters import gabor_kernel
    >>> from skimage import io
    >>> from matplotlib import pyplot as plt  # doctest: +SKIP

    >>> gk = gabor_kernel(frequency=0.2)
    >>> plt.figure()                    # doctest: +SKIP
    >>> io.imshow(cp.asnumpy(gk.real))  # doctest: +SKIP
    >>> io.show()                       # doctest: +SKIP

    >>> # more ripples (equivalent to increasing the size of the
    >>> # Gaussian spread)
    >>> gk = gabor_kernel(frequency=0.2, bandwidth=0.1)
    >>> plt.figure()                    # doctest: +SKIP
    >>> io.imshow(cp.asnumpy(gk.real))  # doctest: +SKIP
    >>> io.show()                       # doctest: +SKIP
    """  # noqa
    if sigma_x is None:
        sigma_x = _sigma_prefactor(bandwidth) / frequency
    if sigma_y is None:
        sigma_y = _sigma_prefactor(bandwidth) / frequency

    if float_dtype is not None:
        if dtype is not None:
            raise ValueError(
                "Cannot set both dtype and float_dtype. Please use dtype "
                "instead (float_dtype is deprecated)."
            )
        warn("`float_dtype` is deprecated: use `dtype` instead.")
        cplx_dtype = cp.promote_types(float_dtype, cp.complex64)
        dtype = cplx_dtype
    elif dtype is None:
        dtype = cp.complex128

    if cp.dtype(dtype).kind != 'c':
        raise ValueError("dtype must be complex")

    ct = math.cos(theta)
    st = math.sin(theta)
    x0 = math.ceil(
        max(abs(n_stds * sigma_x * ct), abs(n_stds * sigma_y * st), 1)
    )
    y0 = math.ceil(
        max(abs(n_stds * sigma_y * ct), abs(n_stds * sigma_x * st), 1)
    )
    y, x = cp.meshgrid(cp.arange(-y0, y0 + 1),
                       cp.arange(-x0, x0 + 1),
                       indexing='ij',
                       sparse=True)
    rotx = x * ct + y * st
    roty = -x * st + y * ct

    g = cp.empty(roty.shape, dtype=dtype)
    cp.exp(
        -0.5 * ((rotx * rotx) / sigma_x ** 2 + (roty * roty) / sigma_y ** 2),
        out=g,
    )
    g /= 2 * math.pi * sigma_x * sigma_y
    g *= cp.exp(1j * (2 * math.pi * frequency * rotx + offset))

    return g


def gabor(image, frequency, theta=0, bandwidth=1, sigma_x=None,
          sigma_y=None, n_stds=3, offset=0, mode='reflect', cval=0):
    """Return real and imaginary responses to Gabor filter.

    The real and imaginary parts of the Gabor filter kernel are applied to the
    image and the response is returned as a pair of arrays.

    Gabor filter is a linear filter with a Gaussian kernel which is modulated
    by a sinusoidal plane wave. Frequency and orientation representations of
    the Gabor filter are similar to those of the human visual system.
    Gabor filter banks are commonly used in computer vision and image
    processing. They are especially suitable for edge detection and texture
    classification.

    Parameters
    ----------
    image : 2-D array
        Input image.
    frequency : float
        Spatial frequency of the harmonic function. Specified in pixels.
    theta : float, optional
        Orientation in radians. If 0, the harmonic is in the x-direction.
    bandwidth : float, optional
        The bandwidth captured by the filter. For fixed bandwidth, ``sigma_x``
        and ``sigma_y`` will decrease with increasing frequency. This value is
        ignored if ``sigma_x`` and ``sigma_y`` are set by the user.
    sigma_x, sigma_y : float, optional
        Standard deviation in x- and y-directions. These directions apply to
        the kernel *before* rotation. If `theta = pi/2`, then the kernel is
        rotated 90 degrees so that ``sigma_x`` controls the *vertical*
        direction.
    n_stds : scalar, optional
        The linear size of the kernel is n_stds (3 by default) standard
        deviations.
    offset : float, optional
        Phase offset of harmonic function in radians.
    mode : {'constant', 'nearest', 'reflect', 'mirror', 'wrap'}, optional
        Mode used to convolve image with a kernel, passed to `ndi.convolve`
    cval : scalar, optional
        Value to fill past edges of input if ``mode`` of convolution is
        'constant'. The parameter is passed to `ndi.convolve`.

    Returns
    -------
    real, imag : arrays
        Filtered images using the real and imaginary parts of the Gabor filter
        kernel. Images are of the same dimensions as the input one.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Gabor_filter
    .. [2] https://web.archive.org/web/20180127125930/http://mplab.ucsd.edu/tutorials/gabor.pdf

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage.filters import gabor
    >>> from skimage import data, io
    >>> from matplotlib import pyplot as plt  # doctest: +SKIP

    >>> image = cp.array(data.coins())
    >>> # detecting edges in a coin image
    >>> filt_real, filt_imag = gabor(image, frequency=0.6)
    >>> plt.figure()                        # doctest: +SKIP
    >>> io.imshow(cp.asnumpy(filt_real))    # doctest: +SKIP
    >>> io.show()                           # doctest: +SKIP

    >>> # less sensitivity to finer details with the lower frequency kernel
    >>> filt_real, filt_imag = gabor(image, frequency=0.1)
    >>> plt.figure()                       # doctest: +SKIP
    >>> io.imshow(cp.asnumpy(filt_real)    # doctest: +SKIP
    >>> io.show()                          # doctest: +SKIP
    """  # noqa
    check_nD(image, 2)
    # do not cast integer types to float!
    if image.dtype.kind == 'f':
        float_dtype = _supported_float_type(image.dtype)
        image = image.astype(float_dtype, copy=False)
        kernel_dtype = cp.promote_types(image.dtype, cp.complex64)
    else:
        kernel_dtype = cp.complex128

    g = gabor_kernel(frequency, theta, bandwidth, sigma_x, sigma_y, n_stds,
                     offset, dtype=kernel_dtype)

    # separate real and imaginary calls in order to preserve integer dtypes
    filtered_real = ndi.convolve(image, g.real, mode=mode, cval=cval)
    filtered_imag = ndi.convolve(image, g.imag, mode=mode, cval=cval)

    return filtered_real, filtered_imag
