"""A vendored subset of cupyx.scipy.signal.signaltools

Note:

The version of ``choose_conv_method`` here differs from the one in CuPy and
does not restrict the choice of fftconvolve to only 1D arrays.
"""

import timeit
import warnings

import cupy
import numpy as np
from cupyx.scipy.ndimage import rank_filter, uniform_filter

from cucim import _misc
from cucim.skimage._vendored import _signaltools_core as _st_core
from cucim.skimage._vendored._ndimage_util import _fix_sequence_arg

_prod = _misc.prod


def convolve(in1, in2, mode='full', method='auto'):
    """Convolve two N-dimensional arrays.

    Convolve ``in1`` and ``in2``, with the output size determined by the
    ``mode`` argument.

    Args:
        in1 (cupy.ndarray): First input.
        in2 (cupy.ndarray): Second input. Should have the same number of
            dimensions as `in1`.
        mode (str): Indicates the size of the output:

            - ``'full'``: output is the full discrete linear convolution \
                (default)
            - ``'valid'``: output consists only of those elements that do \
                not rely on the zero-padding. Either ``in1`` or ``in2`` must \
                be at least as large as the other in every dimension.
            - ``'same'``: - output is the same size as ``in1``, centered with \
                respect to the ``'full'`` output

        method (str): Indicates which method to use for the computations:

            - ``'direct'``: The convolution is determined directly from sums, \
                the definition of convolution
            - ``'fft'``: The Fourier Transform is used to perform the \
                convolution by calling ``fftconvolve``.
            - ``'auto'``: Automatically choose direct of FFT based on an \
                estimate of which is faster for the arguments (default).

    Returns:
        cupy.ndarray: the result of convolution.

    .. seealso:: :func:`cupyx.scipy.signal.choose_conv_method`
    .. seealso:: :func:`cupyx.scipy.signal.correlation`
    .. seealso:: :func:`cupyx.scipy.signal.fftconvolve`
    .. seealso:: :func:`cupyx.scipy.signal.oaconvolve`
    .. seealso:: :func:`cupyx.scipy.ndimage.convolve`
    .. seealso:: :func:`scipy.signal.convolve`
    .. note::
        By default, ``convolve`` and ``correlate`` use ``method='auto'``, which
        calls ``choose_conv_method`` to choose the fastest method using
        pre-computed values. CuPy may not choose the same method to compute
        the convolution as SciPy does given the same inputs.
    """
    return _correlate(in1, in2, mode, method, True)


def correlate(in1, in2, mode='full', method='auto'):
    """Cross-correlate two N-dimensional arrays.

    Cross-correlate ``in1`` and ``in2``, with the output size determined by the
    ``mode`` argument.

    Args:
        in1 (cupy.ndarray): First input.
        in2 (cupy.ndarray): Second input. Should have the same number of
            dimensions as ``in1``.
        mode (str): Indicates the size of the output:

            - ``'full'``: output is the full discrete linear convolution \
                (default)
            - ``'valid'``: output consists only of those elements that do \
                not rely on the zero-padding. Either ``in1`` or ``in2`` must \
                be at least as large as the other in every dimension.
            - ``'same'``: - output is the same size as ``in1``, centered with \
                respect to the ``'full'`` output

        method (str): Indicates which method to use for the computations:

            - ``'direct'``: The convolution is determined directly from sums, \
                the definition of convolution
            - ``'fft'``: The Fourier Transform is used to perform the \
                convolution by calling ``fftconvolve``.
            - ``'auto'``: Automatically choose direct of FFT based on an \
                estimate of which is faster for the arguments (default).

    Returns:
        cupy.ndarray: the result of correlation.

    .. seealso:: :func:`cupyx.scipy.signal.choose_conv_method`
    .. seealso:: :func:`cupyx.scipy.signal.convolve`
    .. seealso:: :func:`cupyx.scipy.signal.fftconvolve`
    .. seealso:: :func:`cupyx.scipy.signal.oaconvolve`
    .. seealso:: :func:`cupyx.scipy.ndimage.correlation`
    .. seealso:: :func:`scipy.signal.correlation`
    .. note::
        By default, ``convolve`` and ``correlate`` use ``method='auto'``, which
        calls ``choose_conv_method`` to choose the fastest method using
        pre-computed values. CuPy may not choose the same method to compute
        the convolution as SciPy does given the same inputs.
    """
    return _correlate(in1, in2, mode, method, False)


def _correlate(in1, in2, mode='full', method='auto', convolution=False):
    quick_out = _st_core._check_conv_inputs(in1, in2, mode, convolution)
    if quick_out is not None:
        return quick_out
    if method not in ('auto', 'direct', 'fft'):
        raise ValueError('acceptable methods are "auto", "direct", or "fft"')

    if method == 'auto':
        method = choose_conv_method(in1, in2, mode=mode)

    if method == 'direct':
        return _st_core._direct_correlate(in1, in2, mode, in1.dtype,
                                          convolution)

    # if method == 'fft':
    inputs_swapped = _st_core._inputs_swap_needed(mode, in1.shape, in2.shape)
    if inputs_swapped:
        in1, in2 = in2, in1
    if not convolution:
        in2 = _st_core._reverse_and_conj(in2)
    out = fftconvolve(in1, in2, mode)
    result_type = cupy.result_type(in1, in2)
    if result_type.kind in 'ui':
        out = out.round()
    out = out.astype(result_type, copy=False)
    if not convolution and inputs_swapped:
        out = cupy.ascontiguousarray(_st_core._reverse_and_conj(out))
    return out


def fftconvolve(in1, in2, mode='full', axes=None):
    """Convolve two N-dimensional arrays using FFT.

    Convolve ``in1`` and ``in2`` using the fast Fourier transform method, with
    the output size determined by the ``mode`` argument.

    This is generally much faster than the ``'direct'`` method of ``convolve``
    for large arrays, but can be slower when only a few output values are
    needed, and can only output float arrays (int or object array inputs will
    be cast to float).

    Args:
        in1 (cupy.ndarray): First input.
        in2 (cupy.ndarray): Second input. Should have the same number of
            dimensions as ``in1``.
        mode (str): Indicates the size of the output:
            ``'full'``: output is the full discrete linear cross-correlation
                        (default)
            ``'valid'``: output consists only of those elements that do not
                         rely on the zero-padding. Either ``in1`` or ``in2``
                         must be at least as large as the other in every
                         dimension.
            ``'same'``: output is the same size as ``in1``, centered
                        with respect to the 'full' output
        axes (scalar or tuple of scalar or None): Axes over which to compute
            the convolution. The default is over all axes.

    Returns:
        cupy.ndarray: the result of convolution

    .. seealso:: :func:`cupyx.scipy.signal.choose_conv_method`
    .. seealso:: :func:`cupyx.scipy.signal.correlation`
    .. seealso:: :func:`cupyx.scipy.signal.convolve`
    .. seealso:: :func:`cupyx.scipy.signal.oaconvolve`
    .. seealso:: :func:`cupyx.scipy.ndimage.convolve`
    .. seealso:: :func:`scipy.signal.correlation`
    """
    out = _st_core._check_conv_inputs(in1, in2, mode)
    if out is not None:
        return out
    in1, in2, axes = _st_core._init_freq_conv_axes(in1, in2, mode, axes, False)
    shape = [max(x1, x2) if a not in axes else x1 + x2 - 1
             for a, (x1, x2) in enumerate(zip(in1.shape, in2.shape))]
    out = _st_core._freq_domain_conv(in1, in2, axes, shape, calc_fast_len=True)
    return _st_core._apply_conv_mode(out, in1.shape, in2.shape, mode, axes)


def _conv_ops(x_shape, h_shape, mode):
    """
    Find the number of operations required for direct/fft methods of
    convolution. The direct operations were recorded by making a dummy class to
    record the number of operations by overriding ``__mul__`` and ``__add__``.
    The FFT operations rely on the (well-known) computational complexity of the
    FFT (and the implementation of ``_freq_domain_conv``).

    """
    if mode == "full":
        out_shape = [n + k - 1 for n, k in zip(x_shape, h_shape)]
    elif mode == "valid":
        out_shape = [abs(n - k) + 1 for n, k in zip(x_shape, h_shape)]
    elif mode == "same":
        out_shape = x_shape
    else:
        raise ValueError(
            "Acceptable mode flags are 'valid',"
            " 'same', or 'full', not mode={}".format(mode)
        )

    s1, s2 = x_shape, h_shape
    if len(x_shape) == 1:
        s1, s2 = s1[0], s2[0]
        if mode == "full":
            direct_ops = s1 * s2
        elif mode == "valid":
            direct_ops = (s2 - s1 + 1) * s1 if s2 >= s1 else (s1 - s2 + 1) * s2
        elif mode == "same":
            direct_ops = (
                s1 * s2 if s1 < s2 else s1 * s2 - (s2 // 2) * ((s2 + 1) // 2)
            )
    else:
        if mode == "full":
            direct_ops = min(_prod(s1), _prod(s2)) * _prod(out_shape)
        elif mode == "valid":
            direct_ops = min(_prod(s1), _prod(s2)) * _prod(out_shape)
        elif mode == "same":
            direct_ops = _prod(s1) * _prod(s2)

    full_out_shape = [n + k - 1 for n, k in zip(x_shape, h_shape)]
    N = _prod(full_out_shape)
    fft_ops = 3 * N * np.log(N)  # 3 separate FFTs of size full_out_shape
    return fft_ops, direct_ops


def _fftconv_faster(x, h, mode):
    """
    See if using fftconvolve or convolve is faster.

    Parameters
    ----------
    x : cupy.ndarray
        Signal
    h : cupy.ndarray
        Kernel
    mode : str
        Mode passed to convolve

    Returns
    -------
    fft_faster : bool

    Notes
    -----
    See docstring of `choose_conv_method` for details on tuning hardware.

    See pull request 11031 for more detail:
    https://github.com/scipy/scipy/pull/11031.

    """
    fft_ops, direct_ops = _conv_ops(x.shape, h.shape, mode)
    offset = -1e-3 if x.ndim == 1 else -1e-4
    constants = (
        {
            "valid": (1.89095737e-9, 2.1364985e-10, offset),
            "full": (1.7649070e-9, 2.1414831e-10, offset),
            "same": (3.2646654e-9, 2.8478277e-10, offset)
            if h.size <= x.size
            else (3.21635404e-9, 1.1773253e-8, -1e-5),
        }
        if x.ndim == 1
        else {
            "valid": (1.85927e-9, 2.11242e-8, offset),
            "full": (1.99817e-9, 1.66174e-8, offset),
            "same": (2.04735e-9, 1.55367e-8, offset),
        }
    )
    O_fft, O_direct, O_offset = constants[mode]
    return O_fft * fft_ops < O_direct * direct_ops + O_offset


def _numeric_arrays(arrays, kinds="buifc"):
    """
    See if a list of arrays are all numeric.

    Parameters
    ----------
    ndarrays : array or list of arrays
        arrays to check if numeric.
    numeric_kinds : string-like
        The dtypes of the arrays to be checked. If the dtype.kind of
        the ndarrays are not in this string the function returns False and
        otherwise returns True.
    """
    if type(arrays) == cupy.ndarray:
        return arrays.dtype.kind in kinds
    for array_ in arrays:
        if array_.dtype.kind not in kinds:
            return False
    return True


def _timeit_fast(stmt="pass", setup="pass", repeat=3):
    """
    Returns the time the statement/function took, in seconds.

    Faster, less precise version of IPython's timeit. `stmt` can be a statement
    written as a string or a callable.

    Will do only 1 loop (like IPython's timeit) with no repetitions
    (unlike IPython) for very slow functions.  For fast functions, only does
    enough loops to take 5 ms, which seems to produce similar results (on
    Windows at least), and avoids doing an extraneous cycle that isn't
    measured.

    """
    timer = timeit.Timer(stmt, setup)

    # determine number of calls per rep so total time for 1 rep >= 5 ms
    x = 0
    for p in range(0, 10):
        number = 10 ** p
        x = timer.timeit(number)  # seconds
        if x >= 5e-3 / 10:  # 5 ms for final test, 1/10th that for this one
            break
    if x > 1:  # second
        # If it's macroscopic, don't bother with repetitions
        best = x
    else:
        number *= 10
        r = timer.repeat(repeat, number)
        best = min(r)

    sec = best / number
    return sec


# TODO: grlee77: tune this for CUDA when measure=False rather than falling
#                back to the choices made by SciPy

def choose_conv_method(in1, in2, mode="full", measure=False):
    """
    Find the fastest convolution/correlation method.

    This primarily exists to be called during the ``method='auto'`` option in
    `convolve` and `correlate`. It can also be used to determine the value of
    ``method`` for many different convolutions of the same dtype/shape.
    In addition, it supports timing the convolution to adapt the value of
    ``method`` to a particular set of inputs and/or hardware.

    Parameters
    ----------
    in1 : array_like
        The first argument passed into the convolution function.
    in2 : array_like
        The second argument passed into the convolution function.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    measure : bool, optional
        If True, run and time the convolution of `in1` and `in2` with both
        methods and return the fastest. If False (default), predict the fastest
        method using precomputed values.

    Returns
    -------
    method : str
        A string indicating which convolution method is fastest, either
        'direct' or 'fft'
    times : dict, optional
        A dictionary containing the times (in seconds) needed for each method.
        This value is only returned if ``measure=True``.

    See Also
    --------
    convolve
    correlate

    Notes
    -----
    Generally, this method is 99% accurate for 2D signals and 85% accurate
    for 1D signals for randomly chosen input sizes. For precision, use
    ``measure=True`` to find the fastest method by timing the convolution.
    This can be used to avoid the minimal overhead of finding the fastest
    ``method`` later, or to adapt the value of ``method`` to a particular set
    of inputs.

    Experiments were run on an Amazon EC2 r5a.2xlarge machine to test this
    function. These experiments measured the ratio between the time required
    when using ``method='auto'`` and the time required for the fastest method
    (i.e., ``ratio = time_auto / min(time_fft, time_direct)``). In these
    experiments, we found:

    * There is a 95% chance of this ratio being less than 1.5 for 1D signals
      and a 99% chance of being less than 2.5 for 2D signals.
    * The ratio was always less than 2.5/5 for 1D/2D signals respectively.
    * This function is most inaccurate for 1D convolutions that take between 1
      and 10 milliseconds with ``method='direct'``. A good proxy for this
      (at least in our experiments) is ``1e6 <= in1.size * in2.size <= 1e7``.

    The 2D results almost certainly generalize to 3D/4D/etc because the
    implementation is the same (the 1D implementation is different).

    All the numbers above are specific to the EC2 machine. However, we did find
    that this function generalizes fairly decently across hardware. The speed
    tests were of similar quality (and even slightly better) than the same
    tests performed on the machine to tune this function's numbers (a mid-2014
    15-inch MacBook Pro with 16GB RAM and a 2.5GHz Intel i7 processor).

    There are cases when `fftconvolve` supports the inputs but this function
    returns `direct` (e.g., to protect against floating point integer
    precision).

    .. versionadded:: 0.19

    Examples
    --------
    Estimate the fastest method for a given input:

    >>> from cucim.skimage import _vendored as signal
    >>> img = cupy.random.rand(32, 32)
    >>> filter = cupy.random.rand(8, 8)
    >>> method = signal.choose_conv_method(img, filter, mode='same')
    >>> method
    'fft'

    This can then be applied to other arrays of the same dtype and shape:

    >>> img2 = cupy.random.rand(32, 32)
    >>> filter2 = cupy.random.rand(8, 8)
    >>> corr2 = signal.correlate(img2, filter2, mode='same', method=method)
    >>> conv2 = signal.convolve(img2, filter2, mode='same', method=method)

    The output of this function (``method``) works with `correlate` and
    `convolve`.

    """
    volume = cupy.asarray(in1)
    kernel = cupy.asarray(in2)

    if measure:
        times = {}
        for method in ["fft", "direct"]:
            times[method] = _timeit_fast(
                lambda: convolve(volume, kernel, mode=mode, method=method)
            )

        chosen_method = "fft" if times["fft"] < times["direct"] else "direct"
        return chosen_method, times

    # for integer input,
    # catch when more precision required than float provides (representing an
    # integer as float can lose precision in fftconvolve if larger than 2**52)
    if any([_numeric_arrays([x], kinds="ui") for x in [volume, kernel]]):
        max_value = int(cupy.abs(volume).max()) * int(cupy.abs(kernel).max())
        max_value *= int(min(volume.size, kernel.size))
        if max_value > 2 ** np.finfo("float").nmant - 1:
            return "direct"

    if _numeric_arrays([volume, kernel], kinds="b"):
        return "direct"

    if _numeric_arrays([volume, kernel]):
        if _fftconv_faster(volume, kernel, mode):
            return "fft"

    return "direct"


def convolve2d(in1, in2, mode='full', boundary='fill', fillvalue=0):
    """Convolve two 2-dimensional arrays.

    Convolve ``in1`` and ``in2`` with output size determined by ``mode``, and
    boundary conditions determined by ``boundary`` and ``fillvalue``.

    Args:
        in1 (cupy.ndarray): First input.
        in2 (cupy.ndarray): Second input. Should have the same number of
            dimensions as ``in1``.
        mode (str): Indicates the size of the output:

            - ``'full'``: output is the full discrete linear convolution \
                (default)
            - ``'valid'``: output consists only of those elements that do \
                not rely on the zero-padding. Either ``in1`` or ``in2`` must \
                be at least as large as the other in every dimension.
            - ``'same'``: - output is the same size as ``in1``, centered with \
                respect to the ``'full'`` output

        boundary (str): Indicates how to handle boundaries:

            - ``fill``: pad input arrays with fillvalue (default)
            - ``wrap``: circular boundary conditions
            - ``symm``: symmetrical boundary conditions

        fillvalue (scalar): Value to fill pad input arrays with. Default is 0.

    Returns:
        cupy.ndarray: A 2-dimensional array containing a subset of the discrete
            linear convolution of ``in1`` with ``in2``.

    .. seealso:: :func:`cupyx.scipy.signal.convolve`
    .. seealso:: :func:`cupyx.scipy.signal.fftconvolve`
    .. seealso:: :func:`cupyx.scipy.signal.oaconvolve`
    .. seealso:: :func:`cupyx.scipy.signal.correlate2d`
    .. seealso:: :func:`cupyx.scipy.ndimage.convolve`
    .. seealso:: :func:`scipy.signal.convolve2d`
    """
    return _correlate2d(in1, in2, mode, boundary, fillvalue, True)


def correlate2d(in1, in2, mode='full', boundary='fill', fillvalue=0):
    """Cross-correlate two 2-dimensional arrays.

    Cross correlate ``in1`` and ``in2`` with output size determined by
    ``mode``, and boundary conditions determined by ``boundary`` and
    ``fillvalue``.

    Args:
        in1 (cupy.ndarray): First input.
        in2 (cupy.ndarray): Second input. Should have the same number of
            dimensions as ``in1``.
        mode (str): Indicates the size of the output:

            - ``'full'``: output is the full discrete linear convolution \
                (default)
            - ``'valid'``: output consists only of those elements that do \
                not rely on the zero-padding. Either ``in1`` or ``in2`` must \
                be at least as large as the other in every dimension.
            - ``'same'``: - output is the same size as ``in1``, centered with \
                respect to the ``'full'`` output

        boundary (str): Indicates how to handle boundaries:

            - ``fill``: pad input arrays with fillvalue (default)
            - ``wrap``: circular boundary conditions
            - ``symm``: symmetrical boundary conditions

        fillvalue (scalar): Value to fill pad input arrays with. Default is 0.

    Returns:
        cupy.ndarray: A 2-dimensional array containing a subset of the discrete
            linear cross-correlation of ``in1`` with ``in2``.

    Note:
        When using ``"same"`` mode with even-length inputs, the outputs of
        ``correlate`` and ``correlate2d`` differ: There is a 1-index offset
        between them.

    .. seealso:: :func:`cupyx.scipy.signal.correlate`
    .. seealso:: :func:`cupyx.scipy.signal.convolve2d`
    .. seealso:: :func:`cupyx.scipy.ndimage.correlate`
    .. seealso:: :func:`scipy.signal.correlate2d`
    """
    return _correlate2d(in1, in2, mode, boundary, fillvalue, False)


def _correlate2d(in1, in2, mode, boundary, fillvalue, convolution=False):
    if not (in1.ndim == in2.ndim == 2):
        raise ValueError('{} inputs must both be 2-D arrays'.format(
            'convolve2d' if convolution else 'correlate2d'))
    _boundaries = {
        'fill': 'constant', 'pad': 'constant',
        'wrap': 'wrap', 'circular': 'wrap',
        'symm': 'reflect', 'symmetric': 'reflect',
    }
    boundary = _boundaries.get(boundary)
    if boundary is None:
        raise ValueError('Acceptable boundary flags are "fill" (or "pad"), '
                         '"circular" (or "wrap"), and '
                         '"symmetric" (or "symm").')
    quick_out = _st_core._check_conv_inputs(in1, in2, mode, convolution)
    if quick_out is not None:
        return quick_out
    return _st_core._direct_correlate(in1, in2, mode, in1.dtype, convolution,
                                      boundary, fillvalue, not convolution)


def wiener(im, mysize=None, noise=None):
    """Perform a Wiener filter on an N-dimensional array.

    Apply a Wiener filter to the N-dimensional array `im`.

    Args:
        im (cupy.ndarray): An N-dimensional array.
        mysize (int or cupy.ndarray, optional): A scalar or an N-length list
            giving the size of the Wiener filter window in each dimension.
            Elements of mysize should be odd. If mysize is a scalar, then this
            scalar is used as the size in each dimension.
        noise (float, optional): The noise-power to use. If None, then noise is
            estimated as the average of the local variance of the input.

    Returns:
        cupy.ndarray: Wiener filtered result with the same shape as `im`.

    .. seealso:: :func:`scipy.signal.wiener`
    """
    if im.dtype.kind == 'c':
        # TODO: adding support for complex types requires ndimage filters
        # to support complex types (which they could easily if not for the
        # scipy compatibility requirement of forbidding complex and using
        # float64 intermediates)
        raise TypeError("complex types not currently supported")
    if mysize is None:
        mysize = 3
    mysize = _fix_sequence_arg(mysize, im.ndim, 'mysize', int)
    im = im.astype(float, copy=False)

    # Estimate the local mean
    local_mean = uniform_filter(im, mysize, mode='constant')

    # Estimate the local variance
    local_var = uniform_filter(im * im, mysize, mode='constant')
    local_var -= local_mean * local_mean

    # Estimate the noise power if needed.
    if noise is None:
        noise = local_var.mean()

    # Perform the filtering
    res = im - local_mean
    res *= 1 - noise / local_var
    res += local_mean
    return cupy.where(local_var < noise, local_mean, res)


def order_filter(a, domain, rank):
    """Perform an order filter on an N-D array.

    Perform an order filter on the array in. The domain argument acts as a mask
    centered over each pixel. The non-zero elements of domain are used to
    select elements surrounding each input pixel which are placed in a list.
    The list is sorted, and the output for that pixel is the element
    corresponding to rank in the sorted list.

    Args:
        a (cupy.ndarray): The N-dimensional input array.
        domain (cupy.ndarray): A mask array with the same number of dimensions
            as `a`. Each dimension should have an odd number of elements.
        rank (int): A non-negative integer which selects the element from the
            sorted list (0 corresponds to the smallest element).

    Returns:
        cupy.ndarray: The results of the order filter in an array with the same
            shape as `a`.

    .. seealso:: :func:`cupyx.scipy.ndimage.rank_filter`
    .. seealso:: :func:`scipy.signal.order_filter`
    """
    if a.dtype.kind in 'bc' or a.dtype == cupy.float16:
        # scipy doesn't support these types
        raise ValueError("data type not supported")
    if any(x % 2 != 1 for x in domain.shape):
        raise ValueError("Each dimension of domain argument "
                         " should have an odd number of elements.")
    return rank_filter(a, rank, footprint=domain, mode='constant')


def medfilt(volume, kernel_size=None):
    """Perform a median filter on an N-dimensional array.

    Apply a median filter to the input array using a local window-size
    given by `kernel_size`. The array will automatically be zero-padded.

    Args:
        volume (cupy.ndarray): An N-dimensional input array.
        kernel_size (int or list of ints): Gives the size of the median filter
            window in each dimension. Elements of `kernel_size` should be odd.
            If `kernel_size` is a scalar, then this scalar is used as the size
            in each dimension. Default size is 3 for each dimension.

    Returns:
        cupy.ndarray: An array the same size as input containing the median
        filtered result.

    .. seealso:: :func:`cupyx.scipy.ndimage.median_filter`
    .. seealso:: :func:`scipy.signal.medfilt`
    """
    if volume.dtype.kind == 'c':
        # scipy doesn't support complex
        # (and rank_filter raise TypeError)
        raise ValueError("complex types not supported")
    # output is forced to float64 to match scipy
    kernel_size = _get_kernel_size(kernel_size, volume.ndim)
    if any(k > s for k, s in zip(kernel_size, volume.shape)):
        warnings.warn('kernel_size exceeds volume extent: '
                      'volume will be zero-padded')

    size = np.prod(kernel_size)
    return rank_filter(volume, size // 2, size=kernel_size,
                       output=float, mode='constant')


def medfilt2d(input, kernel_size=3):
    """Median filter a 2-dimensional array.

    Apply a median filter to the `input` array using a local window-size given
    by `kernel_size` (must be odd). The array is zero-padded automatically.

    Args:
        input (cupy.ndarray): A 2-dimensional input array.
        kernel_size (int of list of ints of length 2): Gives the size of the
            median filter window in each dimension. Elements of `kernel_size`
            should be odd. If `kernel_size` is a scalar, then this scalar is
            used as the size in each dimension. Default is a kernel of size
            (3, 3).

    Returns:
        cupy.ndarray: An array the same size as input containing the median
            filtered result.
    See also
    --------
    .. seealso:: :func:`cupyx.scipy.ndimage.median_filter`
    .. seealso:: :func:`cupyx.scipy.signal.medfilt`
    .. seealso:: :func:`scipy.signal.medfilt2d`
    """
    if input.dtype not in (cupy.uint8, cupy.float32, cupy.float64):
        # Scipy's version only supports uint8, float32, and float64
        raise ValueError("only supports uint8, float32, and float64")
    if input.ndim != 2:
        raise ValueError('input must be 2d')
    kernel_size = _get_kernel_size(kernel_size, input.ndim)
    order = kernel_size[0] * kernel_size[1] // 2
    return rank_filter(input, order, size=kernel_size, mode='constant')


def _get_kernel_size(kernel_size, ndim):
    if kernel_size is None:
        kernel_size = (3,) * ndim
    kernel_size = _fix_sequence_arg(kernel_size, ndim, 'kernel_size', int)
    if any((k % 2) != 1 for k in kernel_size):
        raise ValueError("Each element of kernel_size should be odd")
    return kernel_size
