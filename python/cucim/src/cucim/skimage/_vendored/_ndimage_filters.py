"""A vendored subset of cupyx.scipy.ndimage._filters"""
import warnings

import cupy
import numpy

from cucim.skimage._vendored import _ndimage_filters_core as _filters_core
from cucim.skimage._vendored import _ndimage_util as _util
from cucim.skimage._vendored import _internal as internal


def correlate(input, weights, output=None, mode='reflect', cval=0.0, origin=0):
    """Multi-dimensional correlate.

    The array is correlated with the given kernel.

    Args:
        input (cupy.ndarray): The input array.
        weights (cupy.ndarray): Array of weights, same number of dimensions as
            input
        output (cupy.ndarray, dtype or None): The array in which to place the
            output.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``constant``. Default is ``0.0``.
        origin (scalar or tuple of scalar): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.

    Returns:
        cupy.ndarray: The result of correlate.

    .. seealso:: :func:`scipy.ndimage.correlate`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    return _correlate_or_convolve(input, weights, output, mode, cval, origin)


def convolve(input, weights, output=None, mode='reflect', cval=0.0, origin=0):
    """Multi-dimensional convolution.

    The array is convolved with the given kernel.

    Args:
        input (cupy.ndarray): The input array.
        weights (cupy.ndarray): Array of weights, same number of dimensions as
            input
        output (cupy.ndarray, dtype or None): The array in which to place the
            output.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``constant``. Default is ``0.0``.
        origin (scalar or tuple of scalar): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.

    Returns:
        cupy.ndarray: The result of convolution.

    .. seealso:: :func:`scipy.ndimage.convolve`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    return _correlate_or_convolve(input, weights, output, mode, cval, origin,
                                  True)


def correlate1d(input, weights, axis=-1, output=None, mode="reflect", cval=0.0,
                origin=0, *, algorithm=None):
    """One-dimensional correlate.

    The array is correlated with the given kernel.

    Args:
        input (cupy.ndarray): The input array.
        weights (cupy.ndarray): One-dimensional array of weights
        axis (int): The axis of input along which to calculate. Default is -1.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int): The origin parameter controls the placement of the
            filter, relative to the center of the current element of the
            input. Default is ``0``.

    Returns:
        cupy.ndarray: The result of the 1D correlation.

    .. seealso:: :func:`scipy.ndimage.correlate1d`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    return _correlate_or_convolve1d(
        input, weights, axis, output, mode, cval, origin, False, algorithm
    )


def convolve1d(input, weights, axis=-1, output=None, mode="reflect", cval=0.0,
               origin=0, *, algorithm=None):
    """One-dimensional convolution.

    The array is convolved with the given kernel.

    Args:
        input (cupy.ndarray): The input array.
        weights (cupy.ndarray): One-dimensional array of weights
        axis (int): The axis of input along which to calculate. Default is -1.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int): The origin parameter controls the placement of the
            filter, relative to the center of the current element of the
            input. Default is ``0``.
    Returns:
        cupy.ndarray: The result of the 1D convolution.

    .. seealso:: :func:`scipy.ndimage.convolve1d`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    return _correlate_or_convolve1d(
        input, weights, axis, output, mode, cval, origin, True, algorithm
    )


def _correlate_or_convolve(input, weights, output, mode, cval, origin,
                           convolution=False):
    origins, int_type = _filters_core._check_nd_args(input, weights,
                                                     mode, origin)
    if weights.size == 0:
        return cupy.zeros_like(input)

    _util._check_cval(mode, cval, _util._is_integer_output(output, input))

    if convolution:
        weights = weights[tuple([slice(None, None, -1)] * weights.ndim)]
        origins = list(origins)
        for i, wsize in enumerate(weights.shape):
            origins[i] = -origins[i]
            if wsize % 2 == 0:
                origins[i] -= 1
        origins = tuple(origins)
    elif weights.dtype.kind == "c":
        # numpy.correlate conjugates weights rather than input.
        weights = weights.conj()
    weights_dtype = _util._get_weights_dtype(input, weights)
    offsets = _filters_core._origins_to_offsets(origins, weights.shape)
    kernel = _get_correlate_kernel(mode, weights.shape, int_type,
                                   offsets, cval)
    output = _filters_core._call_kernel(kernel, input, weights, output,
                                        weights_dtype=weights_dtype)
    return output


def _correlate_or_convolve1d(input, weights, axis, output, mode, cval, origin,
                             convolution=False, algorithm=None):
    # Calls fast shared-memory convolution when possible, otherwise falls back
    # to the vendored elementwise _correlate_or_convolve
    if algorithm is None:
       if input.ndim == 2 and weights.size <= 256:
           algorithm = 'shared_memory'
       else:
           algorithm = 'elementwise'
    elif algorithm not in ['shared_memory', 'elementwise']:
        raise ValueError(
            "algorithm must be 'shared_memory', 'elementwise' or None"
        )
    if mode == 'wrap':
        mode = 'grid-wrap'
    if algorithm == 'shared_memory':
        from cucim.skimage.filters._separable_conv_shmem import (
            _shmem_convolve1d, ResourceLimitError
        )
        if input.ndim != 2:
            raise NotImplementedError(
                f"shared_memory not implemented for ndim={input.ndim}"
            )
        try:
            out = _shmem_convolve1d(input, weights, axis=axis, output=output,
                                    mode=mode, cval=cval, origin=origin,
                                    convolution=convolution)
            return out
        except ResourceLimitError:
            # fallback to elementwise if inadequate shared memory available
            warnings.warn(
                "Inadequate resources for algorithm='shared_memory: "
                "falling back to the elementwise implementation"
            )
            algorithm = 'elementwise'
    if algorithm == 'elementwise':
        weights, origins = _filters_core._convert_1d_args(
            input.ndim, weights, origin, axis
        )
        return _correlate_or_convolve(
            input, weights, output, mode, cval, origins, convolution
        )


@cupy.memoize(for_each_device=True)
def _get_correlate_kernel(mode, w_shape, int_type, offsets, cval):
    return _filters_core._generate_nd_kernel(
        'correlate',
        'W sum = (W)0;',
        'sum += cast<W>({value}) * wval;',
        'y = cast<Y>(sum);',
        mode, w_shape, int_type, offsets, cval, ctype='W')


def _run_1d_correlates(input, params, get_weights, output, mode, cval,
                       origin=0, **filter_kwargs):
    """
    Enhanced version of _run_1d_filters that uses correlate1d as the filter
    function. The params are a list of values to pass to the get_weights
    callable given. If duplicate param values are found, the weights are
    reused from the first invocation of get_weights. The get_weights callable
    must return a 1D array of weights to give to correlate1d.
    """
    wghts = {}
    for param in params:
        if param not in wghts:
            wghts[param] = get_weights(param)
    wghts = [wghts[param] for param in params]
    return _filters_core._run_1d_filters(
        [None if w is None else correlate1d for w in wghts],
        input, wghts, output, mode, cval, origin, **filter_kwargs)


def uniform_filter1d(input, size, axis=-1, output=None, mode="reflect",
                     cval=0.0, origin=0, *, algorithm=None):
    """One-dimensional uniform filter along the given axis.

    The lines of the array along the given axis are filtered with a uniform
    filter of the given size.

    Args:
        input (cupy.ndarray): The input array.
        size (int): Length of the uniform filter.
        axis (int): The axis of input along which to calculate. Default is -1.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int): The origin parameter controls the placement of the
            filter, relative to the center of the current element of the
            input. Default is ``0``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.uniform_filter1d`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    weights_dtype = cupy.promote_types(input.dtype, cupy.float32)
    weights = cupy.full(size, 1 / size, dtype=weights_dtype)
    return correlate1d(input, weights, axis, output, mode, cval,
                       origin, algorithm=algorithm)


def uniform_filter(input, size=3, output=None, mode="reflect", cval=0.0,
                   origin=0, *, algorithm=None):
    """Multi-dimensional uniform filter.

    Args:
        input (cupy.ndarray): The input array.
        size (int or sequence of int): Lengths of the uniform filter for each
            dimension. A single value applies to all axes.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int or sequence of int): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of ``0`` is equivalent to
            ``(0,)*input.ndim``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.uniform_filter`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    sizes = _util._fix_sequence_arg(size, input.ndim, 'size', int)
    weights_dtype = cupy.promote_types(input.dtype, cupy.float32)

    def get(size):
        return None if size <= 1 else cupy.full(size, 1 / size, dtype=weights_dtype)  # noqa

    return _run_1d_correlates(
        input, sizes, get, output, mode, cval, origin, algorithm=algorithm
    )


def gaussian_filter1d(input, sigma, axis=-1, order=0, output=None,
                      mode="reflect", cval=0.0, truncate=4.0, *,
                      algorithm=None):
    """One-dimensional Gaussian filter along the given axis.

    The lines of the array along the given axis are filtered with a Gaussian
    filter of the given standard deviation.

    Args:
        input (cupy.ndarray): The input array.
        sigma (scalar): Standard deviation for Gaussian kernel.
        axis (int): The axis of input along which to calculate. Default is -1.
        order (int): An order of ``0``, the default, corresponds to convolution
            with a Gaussian kernel. A positive order corresponds to convolution
            with that derivative of a Gaussian.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        truncate (float): Truncate the filter at this many standard deviations.
            Default is ``4.0``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.gaussian_filter1d`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    radius = int(float(truncate) * float(sigma) + 0.5)
    weights_dtype = cupy.promote_types(input.dtype, cupy.float32)
    weights = _gaussian_kernel1d(sigma, int(order), radius, weights_dtype)
    return correlate1d(
        input, weights, axis, output, mode, cval, algorithm=algorithm
    )


def gaussian_filter(input, sigma, order=0, output=None, mode="reflect",
                    cval=0.0, truncate=4.0, *, algorithm=None):
    """Multi-dimensional Gaussian filter.

    Args:
        input (cupy.ndarray): The input array.
        sigma (scalar or sequence of scalar): Standard deviations for each axis
            of Gaussian kernel. A single value applies to all axes.
        order (int or sequence of scalar): An order of ``0``, the default,
            corresponds to convolution with a Gaussian kernel. A positive order
            corresponds to convolution with that derivative of a Gaussian. A
            single value applies to all axes.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        truncate (float): Truncate the filter at this many standard deviations.
            Default is ``4.0``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.gaussian_filter`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    sigmas = _util._fix_sequence_arg(sigma, input.ndim, 'sigma', float)
    orders = _util._fix_sequence_arg(order, input.ndim, 'order', int)
    truncate = float(truncate)
    weights_dtype = cupy.promote_types(input, cupy.float32)

    def get(param, dtype=weights_dtype):
        sigma, order = param
        radius = int(truncate * float(sigma) + 0.5)
        if radius <= 0:
            return None
        return _gaussian_kernel1d(sigma, order, radius, dtype)

    return _run_1d_correlates(input, list(zip(sigmas, orders)), get, output,
                              mode, cval, 0, algorithm=algorithm)


def _gaussian_kernel1d(sigma, order, radius, dtype=cupy.float64):
    """
    Computes a 1-D Gaussian correlation kernel.
    """
    if order < 0:
        raise ValueError('order must be non-negative')
    sigma2 = sigma * sigma
    x = numpy.arange(-radius, radius+1)
    phi_x = numpy.exp(-0.5 / sigma2 * x ** 2)
    phi_x /= phi_x.sum()

    if order == 0:
        return cupy.asarray(phi_x)

    # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
    # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
    # p'(x) = -1 / sigma ** 2
    # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
    # coefficients of q(x)
    exponent_range = numpy.arange(order + 1)
    q = numpy.zeros(order + 1)
    q[0] = 1
    D = numpy.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
    P = numpy.diag(numpy.ones(order)/-sigma2, -1)  # P @ q(x) = q(x) * p'(x)
    Q_deriv = D + P
    for _ in range(order):
        q = Q_deriv.dot(q)
    q = (x[:, None] ** exponent_range).dot(q)
    return cupy.asarray((q * phi_x)[::-1], order='C', dtype=dtype)


def prewitt(input, axis=-1, output=None, mode="reflect", cval=0.0, *,
            algorithm=None):
    """Compute a Prewitt filter along the given axis.

    Args:
        input (cupy.ndarray): The input array.
        axis (int): The axis of input along which to calculate. Default is -1.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.prewitt`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    weights_dtype = cupy.promote_types(input.dtype, cupy.float32)
    smooth = cupy.ones(3, dtype=weights_dtype)
    return _prewitt_or_sobel(
        input, axis, output, mode, cval, smooth, algorithm
    )


def sobel(input, axis=-1, output=None, mode="reflect", cval=0.0, *,
          algorithm=None):
    """Compute a Sobel filter along the given axis.

    Args:
        input (cupy.ndarray): The input array.
        axis (int): The axis of input along which to calculate. Default is -1.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.sobel`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    weights_dtype = cupy.promote_types(input.dtype, cupy.float32)
    smooth = cupy.array([1, 2, 1], dtype=weights_dtype)
    return _prewitt_or_sobel(
        input, axis, output, mode, cval, smooth, algorithm
    )


def _prewitt_or_sobel(input, axis, output, mode, cval, weights, algorithm):
    axis = internal._normalize_axis_index(axis, input.ndim)

    weights_dtype = cupy.promote_types(input.dtype, cupy.float32)
    def get(is_diff, dtype=weights_dtype):
        return cupy.array([-1, 0, 1], dtype=dtype) if is_diff else weights  # noqa

    return _run_1d_correlates(input, [a == axis for a in range(input.ndim)],
                              get, output, mode, cval, algorithm=algorithm)


def generic_laplace(input, derivative2, output=None, mode="reflect",
                    cval=0.0, extra_arguments=(), extra_keywords=None):
    """Multi-dimensional Laplace filter using a provided second derivative
    function.

    Args:
        input (cupy.ndarray): The input array.
        derivative2 (callable): Function or other callable with the following
            signature that is called once per axis::

                derivative2(input, axis, output, mode, cval,
                            *extra_arguments, **extra_keywords)

            where ``input`` and ``output`` are ``cupy.ndarray``, ``axis`` is an
            ``int`` from ``0`` to the number of dimensions, and ``mode``,
            ``cval``, ``extra_arguments``, ``extra_keywords`` are the values
            given to this function.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        extra_arguments (sequence, optional):
            Sequence of extra positional arguments to pass to ``derivative2``.
        extra_keywords (dict, optional):
            dict of extra keyword arguments to pass ``derivative2``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.generic_laplace`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    if extra_keywords is None:
        extra_keywords = {}
    ndim = input.ndim
    modes = _util._fix_sequence_arg(mode, ndim, 'mode',
                                    _util._check_mode)
    output = _util._get_output(output, input)
    if ndim == 0:
        output[:] = input
        return output
    derivative2(input, 0, output, modes[0], cval,
                *extra_arguments, **extra_keywords)
    if ndim > 1:
        tmp = _util._get_output(output.dtype, input)
        for i in range(1, ndim):
            derivative2(input, i, tmp, modes[i], cval,
                        *extra_arguments, **extra_keywords)
            output += tmp
    return output


def laplace(input, output=None, mode="reflect", cval=0.0, *, algorithm=None):
    """Multi-dimensional Laplace filter based on approximate second
    derivatives.

    Args:
        input (cupy.ndarray): The input array.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.laplace`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    weights_dtype = cupy.promote_types(input.dtype, cupy.float32)
    weights = cupy.array([1, -2, 1], dtype=weights_dtype)

    def derivative2(input, axis, output, mode, cval):
        return correlate1d(
            input, weights, axis, output, mode, cval, algorithm=algorithm
        )

    return generic_laplace(input, derivative2, output, mode, cval)


def gaussian_laplace(input, sigma, output=None, mode="reflect",
                     cval=0.0, *, algorithm=None, **kwargs):
    """Multi-dimensional Laplace filter using Gaussian second derivatives.

    Args:
        input (cupy.ndarray): The input array.
        sigma (scalar or sequence of scalar): Standard deviations for each axis
            of Gaussian kernel. A single value applies to all axes.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        kwargs (dict, optional):
            dict of extra keyword arguments to pass ``gaussian_filter()``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.gaussian_laplace`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    def derivative2(input, axis, output, mode, cval):
        order = [0] * input.ndim
        order[axis] = 2
        return gaussian_filter(input, sigma, order, output, mode, cval,
                               algorithm=algorithm, **kwargs)
    return generic_laplace(input, derivative2, output, mode, cval)


def generic_gradient_magnitude(input, derivative, output=None,
                               mode="reflect", cval=0.0,
                               extra_arguments=(), extra_keywords=None):
    """Multi-dimensional gradient magnitude filter using a provided derivative
    function.

    Args:
        input (cupy.ndarray): The input array.
        derivative (callable): Function or other callable with the following
            signature that is called once per axis::

                derivative(input, axis, output, mode, cval,
                           *extra_arguments, **extra_keywords)

            where ``input`` and ``output`` are ``cupy.ndarray``, ``axis`` is an
            ``int`` from ``0`` to the number of dimensions, and ``mode``,
            ``cval``, ``extra_arguments``, ``extra_keywords`` are the values
            given to this function.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        extra_arguments (sequence, optional):
            Sequence of extra positional arguments to pass to ``derivative2``.
        extra_keywords (dict, optional):
            dict of extra keyword arguments to pass ``derivative2``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.generic_gradient_magnitude`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    if extra_keywords is None:
        extra_keywords = {}
    ndim = input.ndim
    modes = _util._fix_sequence_arg(mode, ndim, 'mode',
                                    _util._check_mode)
    output = _util._get_output(output, input)
    if ndim == 0:
        output[:] = input
        return output
    derivative(input, 0, output, modes[0], cval,
               *extra_arguments, **extra_keywords)
    output *= output
    if ndim > 1:
        tmp = _util._get_output(output.dtype, input)
        for i in range(1, ndim):
            derivative(input, i, tmp, modes[i], cval,
                       *extra_arguments, **extra_keywords)
            tmp *= tmp
            output += tmp
    return cupy.sqrt(output, output, casting='unsafe')


def gaussian_gradient_magnitude(input, sigma, output=None, mode="reflect",
                                cval=0.0, *, algorithm=None, **kwargs):
    """Multi-dimensional gradient magnitude using Gaussian derivatives.

    Args:
        input (cupy.ndarray): The input array.
        sigma (scalar or sequence of scalar): Standard deviations for each axis
            of Gaussian kernel. A single value applies to all axes.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        kwargs (dict, optional):
            dict of extra keyword arguments to pass ``gaussian_filter()``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.gaussian_gradient_magnitude`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    def derivative(input, axis, output, mode, cval):
        order = [0] * input.ndim
        order[axis] = 1
        return gaussian_filter(input, sigma, order, output, mode, cval,
                               algorithm=algorithm, **kwargs)
    return generic_gradient_magnitude(input, derivative, output, mode, cval)
