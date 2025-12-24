# SPDX-FileCopyrightText: Copyright (c) 2015 Preferred Infrastructure, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2015 Preferred Networks, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT

"""A vendored subset of cupyx.scipy.ndimage._filters"""

import math
import platform
import warnings

import cupy
import numpy

from cucim.skimage._vendored import (
    _internal as internal,
    _ndimage_filters_core as _filters_core,
    _ndimage_util as _util,
)
from cucim.skimage.filters._separable_filtering import (
    ResourceLimitError,
    _shmem_convolve1d,
)

try:
    from cupy.cuda.compiler import CompileException

    compile_errors = (ResourceLimitError, CompileException)
except ImportError:
    compile_errors = (ResourceLimitError,)

_is_not_windows = platform.system() != "Windows"


def correlate(
    input,
    weights,
    output=None,
    mode="reflect",
    cval=0.0,
    origin=0,
    *,
    axes=None,
):
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
        axes (tuple of int or None, optional):  If None, `input` is filtered
            along all axes. Otherwise, `input` is filtered along the specified
            axes. When `axes` is specified, any tuples used for `mode` or
            `origin` must match the length of `axes`. The ith entry in any of
            these tuples corresponds to the ith entry in `axes`.

    Returns:
        cupy.ndarray: The result of correlate.

    .. seealso:: :func:`scipy.ndimage.correlate`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    return _correlate_or_convolve(
        input, weights, output, mode, cval, origin, False, axes
    )


def convolve(
    input,
    weights,
    output=None,
    mode="reflect",
    cval=0.0,
    origin=0,
    *,
    axes=None,
):
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
        axes (tuple of int or None, optional):  If None, `input` is filtered
            along all axes. Otherwise, `input` is filtered along the specified
            axes. When `axes` is specified, any tuples used for `mode` or
            `origin` must match the length of `axes`. The ith entry in any of
            these tuples corresponds to the ith entry in `axes`.

    Returns:
        cupy.ndarray: The result of convolution.

    .. seealso:: :func:`scipy.ndimage.convolve`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    return _correlate_or_convolve(
        input, weights, output, mode, cval, origin, True, axes
    )


def correlate1d(
    input,
    weights,
    axis=-1,
    output=None,
    mode="reflect",
    cval=0.0,
    origin=0,
    *,
    algorithm=None,
):
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


def convolve1d(
    input,
    weights,
    axis=-1,
    output=None,
    mode="reflect",
    cval=0.0,
    origin=0,
    *,
    algorithm=None,
):
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


def _correlate_or_convolve(
    input, weights, output, mode, cval, origin, convolution=False, axes=None
):
    axes, weights, origins, modes, int_type = _filters_core._check_nd_args(
        input, weights, mode, origin, axes=axes
    )
    if weights.size == 0:
        return cupy.zeros_like(input)

    for mode in modes:
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
    weights_dtype = _util._get_weights_dtype(
        input, weights, use_cucim_casting=True
    )  # noqa
    offsets = _filters_core._origins_to_offsets(origins, weights.shape)
    kernel = _get_correlate_kernel(
        modes, weights.shape, int_type, offsets, cval
    )
    output = _filters_core._call_kernel(
        kernel, input, weights, output, weights_dtype=weights_dtype
    )
    return output


def _unsupported_shmem_cval(cval, dtype):
    """returns True if the cval + dtype combination is unsupported for the
    shared memory kernels."""
    return (
        # shared_memory kernels fail to compile for non-finite cval
        not numpy.isfinite(cval)
        # incorrect output for negative cval and unsigned integer input
        or (cval < 0 and dtype.kind == "u")
    )


def _correlate_or_convolve1d(
    input,
    weights,
    axis,
    output,
    mode,
    cval,
    origin,
    convolution=False,
    algorithm=None,
):
    # Calls fast shared-memory convolution when possible, otherwise falls back
    # to the vendored elementwise _correlate_or_convolve
    default_algorithm = False
    if algorithm is None:
        default_algorithm = True
        if (
            _is_not_windows
            and input.ndim == 2
            and weights.size <= 256
            and not (
                cval < 0 and mode == "constant" and input.dtype.kind == "u"
            )
        ):
            algorithm = "shared_memory"
        else:
            algorithm = "elementwise"
    elif algorithm == "shared_memory":
        if mode == "constant" and _unsupported_shmem_cval(cval, input.dtype):
            # shared_memory case can fail to compile on NaN cval and
            warnings.warn(
                f"{cval=} and {input.dtype=} is unsupported for algorithm "
                "'shared_memory', falling back to elementwise"
            )
            algorithm = "elementwise"
    elif algorithm != "elementwise":
        raise ValueError(
            "algorithm must be 'shared_memory', 'elementwise' or None"
        )
    if mode == "wrap":
        mode = "grid-wrap"
    if algorithm == "shared_memory":
        if input.ndim not in [2, 3]:
            raise NotImplementedError(
                f"shared_memory not implemented for ndim={input.ndim}"
            )
        try:
            out = _shmem_convolve1d(
                input,
                weights,
                axis=axis,
                output=output,
                mode=mode,
                cval=cval,
                origin=origin,
                convolution=convolution,
            )
            return out
        except compile_errors:
            # fallback to elementwise if inadequate shared memory available
            if not default_algorithm:
                # only warn if 'shared_memory' was explicitly requested
                warnings.warn(
                    "Inadequate resources for algorithm='shared_memory: "
                    "falling back to the elementwise implementation"
                )
            algorithm = "elementwise"
    if algorithm == "elementwise":
        weights, origins = _filters_core._convert_1d_args(
            input.ndim, weights, origin, axis
        )
        return _correlate_or_convolve(
            input, weights, output, mode, cval, origins, convolution
        )


@cupy.memoize(for_each_device=True)
def _get_correlate_kernel(modes, w_shape, int_type, offsets, cval):
    return _filters_core._generate_nd_kernel(
        "correlate",
        "W sum = (W)0;",
        "sum += cast<W>({value}) * wval;",
        "y = cast<Y>(sum);",
        modes,
        w_shape,
        int_type,
        offsets,
        cval,
        ctype="W",
    )


def _run_1d_correlates(
    input,
    axes,
    params,
    get_weights,
    output,
    modes,
    cval,
    origin=0,
    **filter_kwargs,
):
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
        input,
        axes,
        wghts,
        output,
        modes,
        cval,
        origin,
        **filter_kwargs,
    )


def uniform_filter1d(
    input,
    size,
    axis=-1,
    output=None,
    mode="reflect",
    cval=0.0,
    origin=0,
    *,
    algorithm=None,
):
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
    return correlate1d(
        input, weights, axis, output, mode, cval, origin, algorithm=algorithm
    )


def uniform_filter(
    input,
    size=3,
    output=None,
    mode="reflect",
    cval=0.0,
    origin=0,
    axes=None,
    *,
    algorithm=None,
):
    """Multi-dimensional uniform filter.

    Args:
        input (cupy.ndarray): The input array.
        size (int or sequence of int): Lengths of the uniform filter for each
            dimension. A single value applies to all axes.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str or sequence of str): The array borders are handled according
            to the given mode (``'reflect'``, ``'constant'``, ``'nearest'``,
            ``'mirror'``, ``'wrap'``). Default is ``'reflect'``. By passing a
            sequence of modes with length equal to the number of ``axes`` along
            which the input array is being filtered, different modes can be
            specified along each axis. For more details on the supported modes,
            see :func:`scipy.ndimage.uniform_filter`.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int or sequence of int): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of ``0`` is equivalent to
            ``(0,)*input.ndim``.
        axes (tuple of int or None): If None, ``input`` is filtered along all
            axes. Otherwise, ``input`` is filtered along the specified axes.
            When ``axes`` is specified, any tuples used for ``size``,
            ``mode`` and/or ``origin`` must match the length of ``axes``. The
            ith entry in any of these tuples corresponds to the ith entry in
            ``axes``. Default is ``None``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.uniform_filter`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    axes = _util._check_axes(axes, input.ndim)
    num_axes = len(axes)
    sizes = _util._fix_sequence_arg(size, num_axes, "size", int)
    origins = _util._fix_sequence_arg(origin, num_axes, "origin", int)
    modes = _util._fix_sequence_arg(mode, num_axes, "mode", str)

    weights_dtype = cupy.promote_types(input.dtype, cupy.float32)

    def get(size):
        return (
            None
            if size <= 1
            else cupy.full(size, 1 / size, dtype=weights_dtype)
        )  # noqa

    return _run_1d_correlates(
        input,
        axes,
        sizes,
        get,
        output,
        modes,
        cval,
        origins,
        algorithm=algorithm,
    )


def gaussian_filter1d(
    input,
    sigma,
    axis=-1,
    order=0,
    output=None,
    mode="reflect",
    cval=0.0,
    truncate=4.0,
    *,
    algorithm=None,
):
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
        The Gaussian kernel will have size ``2*radius + 1`` along each axis. If
        `radius` is None, a default ``radius = round(truncate * sigma)`` will
        be used.

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


def gaussian_filter(
    input,
    sigma,
    order=0,
    output=None,
    mode="reflect",
    cval=0.0,
    truncate=4.0,
    radius=None,
    axes=None,
    *,
    algorithm=None,
):
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
        mode (str or sequence of str): The array borders are handled according
            to the given mode (``'reflect'``, ``'constant'``, ``'nearest'``,
            ``'mirror'``, ``'wrap'``). Default is ``'reflect'``. By passing a
            sequence of modes with length equal to the number of ``axes`` along
            which the input array is being filtered, different modes can be
            specified along each axis. For more details on the supported modes,
            see :func:`scipy.ndimage.gaussian_filter`.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        truncate (float): Truncate the filter at this many standard deviations.
            Default is ``4.0``.
        radius (int, sequence of int, or None): Radius of the Gaussian kernel.
            The radius are given for each axis as a sequence, or as a single
            number, in which case it is equal for all axes. If specified, the
            size of the kernel along each axis will be ``2*radius + 1``, and
            `truncate` is ignored. Default is ``None``.
        axes (tuple of int or None): If None, ``input`` is filtered along all
            axes. Otherwise, ``input`` is filtered along the specified axes.
            When ``axes`` is specified, any tuples used for ``sigma``,
            ``order``, ``mode`` and/or ``radius`` must match the length of
            ``axes``. The ith entry in any of these tuples corresponds to the
            ith entry in ``axes``. Default is ``None``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.gaussian_filter`

    .. note::
        The Gaussian kernel will have size ``2*radius + 1`` along each axis. If
        `radius` is None, a default ``radius = round(truncate * sigma)`` will
        be used.

        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    axes = _util._check_axes(axes, input.ndim)
    num_axes = len(axes)

    sigmas = _util._fix_sequence_arg(sigma, num_axes, "sigma", float)
    sigma_threshold = 1e-15
    if num_axes == 0 or all(s < sigma_threshold for s in sigmas):
        if output is None:
            return input.copy()
        else:
            output = _util._get_output(output, input)
            output[:] = input
            return output
    orders = _util._fix_sequence_arg(order, num_axes, "order", int)
    modes = _util._fix_sequence_arg(mode, num_axes, "mode", str)
    radiuses = _util._fix_sequence_arg(radius, num_axes, "radius")
    truncate = float(truncate)
    weights_dtype = cupy.promote_types(input, cupy.float32)

    # omit any axes with sigma ~= 0.0
    params = [
        (axes[ii], sigmas[ii], orders[ii], modes[ii], radiuses[ii])
        for ii in range(num_axes)
        if sigmas[ii] > sigma_threshold
    ]
    axes, sigmas, orders, modes, radiuses = zip(*params)

    def get(param):
        _, sigma, order, _, radius = param
        if radius is None:
            radius = int(truncate * float(sigma) + 0.5)
        if radius <= 0:
            return None
        return _gaussian_kernel1d(sigma, order, radius, dtype=weights_dtype)

    return _run_1d_correlates(
        input,
        axes,
        params,
        get,
        output,
        modes,
        cval,
        0,
        algorithm=algorithm,
    )


def _gaussian_kernel1d(sigma, order, radius, dtype=cupy.float64):
    """
    Computes a 1-D Gaussian correlation kernel.
    """
    if order < 0:
        raise ValueError("order must be non-negative")
    sigma2 = sigma * sigma
    x = numpy.arange(-radius, radius + 1)
    phi_x = numpy.exp(-0.5 / sigma2 * x**2)
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
    P = numpy.diag(numpy.ones(order) / -sigma2, -1)  # P @ q(x) = q(x) * p'(x)
    Q_deriv = D + P
    for _ in range(order):
        q = Q_deriv.dot(q)
    q = (x[:, None] ** exponent_range).dot(q)
    return cupy.asarray((q * phi_x)[::-1], order="C", dtype=dtype)


def prewitt(
    input, axis=-1, output=None, mode="reflect", cval=0.0, *, algorithm=None
):
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
    return _prewitt_or_sobel(input, axis, output, mode, cval, smooth, algorithm)


def sobel(
    input, axis=-1, output=None, mode="reflect", cval=0.0, *, algorithm=None
):
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
    return _prewitt_or_sobel(input, axis, output, mode, cval, smooth, algorithm)


def _prewitt_or_sobel(input, axis, output, mode, cval, weights, algorithm):
    axis = internal._normalize_axis_index(axis, input.ndim)

    weights_dtype = cupy.promote_types(input.dtype, cupy.float32)

    def get(is_diff):
        return (
            cupy.array([-1, 0, 1], dtype=weights_dtype) if is_diff else weights
        )  # noqa

    axes = tuple(range(input.ndim))
    modes = (mode,) * input.ndim
    return _run_1d_correlates(
        input,
        axes,
        [a == axis for a in range(input.ndim)],
        get,
        output,
        modes,
        cval,
        algorithm=algorithm,
    )


def _dilate_mask(mask, structure, axes=None):
    """Dilate a binary mask using the given structure.

    Args:
        mask (cupy.ndarray): Binary mask to dilate.
        structure (cp.ndarray or int or tuple of int): Structure for dilation.
            If an integer, the same size is used for all axes.
            If a tuple, it should match the length of axes.
        axes (tuple of int or None): Axes along which to dilate.
            If None, dilates along all axes.

    Returns:
        cupy.ndarray: Dilated mask.
    """
    from cucim.skimage._vendored._ndimage_morphology import binary_dilation

    return binary_dilation(mask, structure=structure, axes=axes)


def generic_laplace(
    input,
    derivative2,
    output=None,
    mode="reflect",
    cval=0.0,
    extra_arguments=(),
    extra_keywords=None,
    *,
    axes=None,
):
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
        axes (tuple of int or None): The axes over which to apply the filter.
            If a `mode` tuple is provided, its length must match the number of
            axes.

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
    axes = _util._check_axes(axes, input.ndim)
    num_axes = len(axes)
    output = _util._get_output(output, input)
    if num_axes > 0:
        modes = _util._fix_sequence_arg(
            mode, num_axes, "mode", _util._check_mode
        )
        derivative2(
            input,
            axes[0],
            output,
            modes[0],
            cval,
            *extra_arguments,
            **extra_keywords,
        )
        if num_axes > 1:
            tmp = _util._get_output(output.dtype, input)
            for i in range(1, num_axes):
                derivative2(
                    input,
                    axes[i],
                    tmp,
                    modes[i],
                    cval,
                    *extra_arguments,
                    **extra_keywords,
                )
                output += tmp
    else:
        _core.elementwise_copy(input, output)
    return output


def laplace(
    input, output=None, mode="reflect", cval=0.0, *, axes=None, algorithm=None
):
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
        axes (tuple of int or None): The axes over which to apply the filter.
            If a `mode` tuple is provided, its length must match the number of
            axes.

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

    return generic_laplace(input, derivative2, output, mode, cval, axes=axes)


def gaussian_laplace(
    input,
    sigma,
    output=None,
    mode="reflect",
    cval=0.0,
    *,
    axes=None,
    algorithm=None,
    **kwargs,
):
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
        axes (tuple of int or None): The axes over which to apply the filter.
            If a `sigma` or `mode` tuples are provided, their length must match
            the number of axes.
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

    def derivative2(input, axis, output, mode, cval, sigma, **kwargs):
        order = [0] * input.ndim
        order[axis] = 2
        return gaussian_filter(
            input,
            sigma,
            order,
            output,
            mode,
            cval,
            algorithm=algorithm,
            **kwargs,
        )

    axes = _util._check_axes(axes, input.ndim)
    num_axes = len(axes)
    sigma = _util._fix_sequence_arg(sigma, num_axes, "sigma", float)
    if num_axes < input.ndim:
        # set sigma = 0 for any axes not being filtered
        sigma_temp = [
            0,
        ] * input.ndim
        for s, ax in zip(sigma, axes):
            sigma_temp[ax] = s
        sigma = sigma_temp

    return generic_laplace(
        input,
        derivative2,
        output,
        mode,
        cval,
        extra_arguments=(sigma,),
        extra_keywords=kwargs,
        axes=axes,
    )


def generic_gradient_magnitude(
    input,
    derivative,
    output=None,
    mode="reflect",
    cval=0.0,
    extra_arguments=(),
    extra_keywords=None,
    *,
    axes=None,
):
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
        axes (tuple of int or None): The axes over which to apply the filter.
            If a `mode` tuple is provided, its length must match the number of
            axes.

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
    axes = _util._check_axes(axes, ndim)
    num_axes = len(axes)
    modes = _util._fix_sequence_arg(mode, num_axes, "mode", _util._check_mode)
    output = _util._get_output(output, input)
    if ndim == 0:
        output[:] = input
        return output
    derivative(
        input,
        axes[0],
        output,
        modes[0],
        cval,
        *extra_arguments,
        **extra_keywords,
    )
    output *= output
    if ndim > 1:
        tmp = _util._get_output(output.dtype, input)
        for i in range(1, num_axes):
            derivative(
                input,
                axes[i],
                tmp,
                modes[i],
                cval,
                *extra_arguments,
                **extra_keywords,
            )
            tmp *= tmp
            output += tmp
    return cupy.sqrt(output, output, casting="unsafe")


def gaussian_gradient_magnitude(
    input,
    sigma,
    output=None,
    mode="reflect",
    cval=0.0,
    *,
    axes=None,
    algorithm=None,
    **kwargs,
):
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
        axes (tuple of int or None): The axes over which to apply the filter.
            If a `mode` tuple is provided, its length must match the number of
            axes.

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
        return gaussian_filter(
            input,
            sigma,
            order,
            output,
            mode,
            cval,
            algorithm=algorithm,
            **kwargs,
        )

    return generic_gradient_magnitude(
        input, derivative, output, mode, cval, axes=axes
    )


def minimum_filter(
    input,
    size=None,
    footprint=None,
    output=None,
    mode="reflect",
    cval=0.0,
    origin=0,
    axes=None,
):
    """Multi-dimensional minimum filter.

    Args:
        input (cupy.ndarray): The input array.
        size (int or sequence of int): One of ``size`` or ``footprint`` must be
            provided. If ``footprint`` is given, ``size`` is ignored. Otherwise
            ``footprint = cupy.ones(size)`` with ``size`` automatically made to
            match the number of dimensions in ``input``.
        footprint (cupy.ndarray): a boolean array which specifies which of the
            elements within this shape will get passed to the filter function.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str or sequence of str): The array borders are handled according
            to the given mode (``'reflect'``, ``'constant'``, ``'nearest'``,
            ``'mirror'``, ``'wrap'``). Default is ``'reflect'``. By passing a
            sequence of modes with length equal to the number of ``axes`` along
            which the input array is being filtered, different modes can be
            specified along each axis. For more details on the supported modes,
            see :func:`scipy.ndimage.minimum_filter`.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int or sequence of int): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
        axes (tuple of int or None): If None, ``input`` is filtered along all
            axes. Otherwise, ``input`` is filtered along the specified axes.
            When ``axes`` is specified, any tuples used for ``size``,
            ``mode`` and/or ``origin`` must match the length of ``axes``. The
            ith entry in any of these tuples corresponds to the ith entry in
            ``axes``. Default is ``None``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.minimum_filter`
    """
    return _min_or_max_filter(
        input,
        size,
        footprint,
        None,
        output,
        mode,
        cval,
        origin,
        "min",
        axes,
    )


def maximum_filter(
    input,
    size=None,
    footprint=None,
    output=None,
    mode="reflect",
    cval=0.0,
    origin=0,
    axes=None,
):
    """Multi-dimensional maximum filter.

    Args:
        input (cupy.ndarray): The input array.
        size (int or sequence of int): One of ``size`` or ``footprint`` must be
            provided. If ``footprint`` is given, ``size`` is ignored. Otherwise
            ``footprint = cupy.ones(size)`` with ``size`` automatically made to
            match the number of dimensions in ``input``.
        footprint (cupy.ndarray): a boolean array which specifies which of the
            elements within this shape will get passed to the filter function.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str or sequence of str): The array borders are handled according
            to the given mode (``'reflect'``, ``'constant'``, ``'nearest'``,
            ``'mirror'``, ``'wrap'``). Default is ``'reflect'``. By passing a
            sequence of modes with length equal to the number of ``axes`` along
            which the input array is being filtered, different modes can be
            specified along each axis. For more details on the supported modes,
            see :func:`scipy.ndimage.minimum_filter`.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int or sequence of int): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
        axes (tuple of int or None): If None, ``input`` is filtered along all
            axes. Otherwise, ``input`` is filtered along the specified axes.
            When ``axes`` is specified, any tuples used for ``size``,
            ``mode`` and/or ``origin`` must match the length of ``axes``. The
            ith entry in any of these tuples corresponds to the ith entry in
            ``axes``. Default is ``None``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.maximum_filter`
    """
    return _min_or_max_filter(
        input,
        size,
        footprint,
        None,
        output,
        mode,
        cval,
        origin,
        "max",
        axes,
    )


def _min_or_max_filter(
    input,
    size,
    ftprnt,
    structure,
    output,
    mode,
    cval,
    origin,
    func,
    axes,
    *,
    mask=None,
):
    # structure is used by morphology.grey_erosion() and grey_dilation()
    # and not by the regular min/max filters

    if isinstance(ftprnt, tuple) and size is None:
        size = ftprnt
        ftprnt = None

    axes, ftprnt, origins, modes, int_type = _filters_core._check_nd_args(
        input,
        ftprnt,
        mode,
        origin,
        "footprint",
        sizes=size,
        axes=axes,
        raise_on_zero_size_weight=True,
    )
    num_axes = len(axes)

    sizes, ftprnt, structure = _filters_core._check_size_footprint_structure(
        num_axes, size, ftprnt, structure
    )
    if cval is cupy.nan:
        raise NotImplementedError("NaN cval is unsupported")

    # Dilate mask if provided to ensure correct computation
    # (unless caller has already done this via skip_mask_dilation_and_restoration)
    dilated_mask = None
    original_mask = None
    if mask is not None:
        original_mask = cupy.asarray(mask, dtype=bool)
        # Determine structure for dilation based on filter size
        if sizes is not None:
            # Separable filter: use sizes directly
            dilation_structure = sizes
        else:
            # Non-separable filter: use footprint shape
            dilation_structure = ftprnt.shape
        dilated_mask = _dilate_mask(
            original_mask, dilation_structure, axes=axes
        )
    if sizes is not None:
        # Separable filter, run as a series of 1D filters
        fltr = minimum_filter1d if func == "min" else maximum_filter1d
        output = _filters_core._run_1d_filters(
            [fltr if size > 1 else None for size in sizes],
            input,
            axes,
            sizes,
            output,
            modes,
            cval,
            origins,
            mask=dilated_mask,
        )
        # Restore original values in unmasked regions
        if original_mask is not None:
            output[~original_mask] = input[~original_mask]
        return output

    if ftprnt.size == 0:
        return cupy.zeros_like(input)

    if num_axes < input.ndim:
        # expand origins ,footprint and structure if num_axes < input.ndim
        ftprnt = _util._expand_footprint(input.ndim, axes, ftprnt)
        origins = _util._expand_origin(input.ndim, axes, origin)
        modes = tuple(_util._expand_mode(input.ndim, axes, modes))

    if structure is not None:
        structure = _util._expand_footprint(
            input.ndim, axes, structure, footprint_name="structure"
        )

    offsets = _filters_core._origins_to_offsets(origins, ftprnt.shape)
    has_mask = dilated_mask is not None
    kernel = _get_min_or_max_kernel(
        modes,
        ftprnt.shape,
        func,
        offsets,
        float(cval),
        int_type,
        has_structure=structure is not None,
        has_central_value=bool(ftprnt[offsets]),
        has_mask=has_mask,
    )
    kwargs = dict(weights_dtype=bool)
    if has_mask:
        kwargs["mask"] = dilated_mask
    output = _filters_core._call_kernel(
        kernel, input, ftprnt, output, structure, **kwargs
    )
    # Restore original values in unmasked regions
    if original_mask is not None:
        output[~original_mask] = input[~original_mask]
    return output


def minimum_filter1d(
    input, size, axis=-1, output=None, mode="reflect", cval=0.0, origin=0
):
    """Compute the minimum filter along a single axis.

    Args:
        input (cupy.ndarray): The input array.
        size (int): Length of the minimum filter.
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

    .. seealso:: :func:`scipy.ndimage.minimum_filter1d`
    """
    return _min_or_max_1d(input, size, axis, output, mode, cval, origin, "min")


def maximum_filter1d(
    input, size, axis=-1, output=None, mode="reflect", cval=0.0, origin=0
):
    """Compute the maximum filter along a single axis.

    Args:
        input (cupy.ndarray): The input array.
        size (int): Length of the maximum filter.
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

    .. seealso:: :func:`scipy.ndimage.maximum_filter1d`
    """
    return _min_or_max_1d(input, size, axis, output, mode, cval, origin, "max")


def _min_or_max_1d(
    input,
    size,
    axis=-1,
    output=None,
    mode="reflect",
    cval=0.0,
    origin=0,
    func="min",
    *,
    mask=None,
):
    ftprnt = cupy.ones(size, dtype=bool)
    ftprnt, origin = _filters_core._convert_1d_args(
        input.ndim, ftprnt, origin, axis
    )
    axes, ftprnt, origins, modes, int_type = _filters_core._check_nd_args(
        input, ftprnt, mode, origin, "footprint", axes=None
    )
    offsets = _filters_core._origins_to_offsets(origins, ftprnt.shape)
    has_mask = mask is not None
    kernel = _get_min_or_max_kernel(
        modes,
        ftprnt.shape,
        func,
        offsets,
        float(cval),
        int_type,
        has_weights=False,
        has_mask=has_mask,
    )
    kwargs = dict(weights_dtype=bool)
    if has_mask:
        kwargs["mask"] = mask
    return _filters_core._call_kernel(kernel, input, None, output, **kwargs)


@cupy._util.memoize(for_each_device=True)
def _get_min_or_max_kernel(
    modes,
    w_shape,
    func,
    offsets,
    cval,
    int_type,
    has_weights=True,
    has_structure=False,
    has_central_value=True,
    has_mask=False,
):
    # When there are no 'weights' (the footprint, for the 1D variants) then
    # we need to make sure intermediate results are stored as doubles for
    # consistent results with scipy.
    ctype = "X" if has_weights else "double"
    value = "{value}"
    if not has_weights:
        value = f"cast<double>({value})"

    # Having a non-flat structure biases the values
    if has_structure:
        value += ("-" if func == "min" else "+") + "cast<X>(sval)"

    pre = ""
    if has_mask:
        pre += """
            // keep existing value if not within the mask
            bool mv = (bool)mask[i];
            if (!mv) {
                y = cast<Y>(x[i]);
                return;
            }\n"""

    if has_central_value:
        pre += f"{ctype} value = x[i];"
        found = f"value = {func}({value}, value);"
    else:
        # If the central pixel is not included in the footprint we cannot
        # assume `x[i]` is not below the min or above the max and thus cannot
        # seed with that value. Instead we keep track of having set `value`.
        pre += f"{ctype} value; bool set = false;"
        found = f"value = set ? {func}({value}, value) : {value}; set=true;"

    mask_str = "_masked" if has_mask else ""
    name = f"{func}{mask_str}"
    return _filters_core._generate_nd_kernel(
        name,
        pre,
        found,
        "y = cast<Y>(value);",
        modes,
        w_shape,
        int_type,
        offsets,
        cval,
        ctype=ctype,
        has_mask=has_mask,
        has_weights=has_weights,
        has_structure=has_structure,
    )


def rank_filter(
    input,
    rank,
    size=None,
    footprint=None,
    output=None,
    mode="reflect",
    cval=0.0,
    origin=0,
    axes=None,
):
    """Multi-dimensional rank filter.

    Args:
        input (cupy.ndarray): The input array.
        rank (int): The rank of the element to get. Can be negative to count
            from the largest value, e.g. ``-1`` indicates the largest value.
        size (int or sequence of int): One of ``size`` or ``footprint`` must be
            provided. If ``footprint`` is given, ``size`` is ignored. Otherwise
            ``footprint = cupy.ones(size)`` with ``size`` automatically made to
            match the number of dimensions in ``input``.
        footprint (cupy.ndarray): a boolean array which specifies which of the
            elements within this shape will get passed to the filter function.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int or sequence of int): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
        axes (tuple of int or None): If None, ``input`` is filtered along all
            axes. Otherwise, ``input`` is filtered along the specified axes.
            When ``axes`` is specified, any tuples used for ``size`` and/or
            ``origin`` must match the length of ``axes``. The ith entry in any
            of these tuples corresponds to the ith entry in ``axes``. Default
            is ``None``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.rank_filter`
    """
    rank = int(rank)
    return _rank_filter(
        input,
        lambda fs: rank + fs if rank < 0 else rank,
        size,
        footprint,
        output,
        mode,
        cval,
        origin,
        axes,
    )


def median_filter(
    input,
    size=None,
    footprint=None,
    output=None,
    mode="reflect",
    cval=0.0,
    origin=0,
    axes=None,
):
    """Multi-dimensional median filter.

    Args:
        input (cupy.ndarray): The input array.
        size (int or sequence of int): One of ``size`` or ``footprint`` must be
            provided. If ``footprint`` is given, ``size`` is ignored. Otherwise
            ``footprint = cupy.ones(size)`` with ``size`` automatically made to
            match the number of dimensions in ``input``.
        footprint (cupy.ndarray): a boolean array which specifies which of the
            elements within this shape will get passed to the filter function.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int or sequence of int): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
        axes (tuple of int or None): If None, ``input`` is filtered along all
            axes. Otherwise, ``input`` is filtered along the specified axes.
            When ``axes`` is specified, any tuples used for ``size`` and/or
            ``origin`` must match the length of ``axes``. The ith entry in any
            of these tuples corresponds to the ith entry in ``axes``. Default
            is ``None``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.median_filter`
    """
    return _rank_filter(
        input,
        lambda fs: fs // 2,
        size,
        footprint,
        output,
        mode,
        cval,
        origin,
        axes,
    )


def percentile_filter(
    input,
    percentile,
    size=None,
    footprint=None,
    output=None,
    mode="reflect",
    cval=0.0,
    origin=0,
    axes=None,
):
    """Multi-dimensional percentile filter.

    Args:
        input (cupy.ndarray): The input array.
        percentile (scalar): The percentile of the element to get (from ``0``
            to ``100``). Can be negative, thus ``-20`` equals ``80``.
        size (int or sequence of int): One of ``size`` or ``footprint`` must be
            provided. If ``footprint`` is given, ``size`` is ignored. Otherwise
            ``footprint = cupy.ones(size)`` with ``size`` automatically made to
            match the number of dimensions in ``input``.
        footprint (cupy.ndarray): a boolean array which specifies which of the
            elements within this shape will get passed to the filter function.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int or sequence of int): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
        axes (tuple of int or None): If None, ``input`` is filtered along all
            axes. Otherwise, ``input`` is filtered along the specified axes.
            When ``axes`` is specified, any tuples used for ``size`` and/or
            ``origin`` must match the length of ``axes``. The ith entry in any
            of these tuples corresponds to the ith entry in ``axes``. Default
            is ``None``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.percentile_filter`
    """
    percentile = float(percentile)
    if percentile < 0.0:
        percentile += 100.0
    if percentile < 0.0 or percentile > 100.0:
        raise RuntimeError("invalid percentile")
    if percentile == 100.0:

        def get_rank(fs):
            return fs - 1

    else:

        def get_rank(fs):
            return int(float(fs) * percentile / 100.0)

    return _rank_filter(
        input, get_rank, size, footprint, output, mode, cval, origin, axes
    )


def _rank_filter(
    input,
    get_rank,
    size=None,
    footprint=None,
    output=None,
    mode="reflect",
    cval=0.0,
    origin=0,
    axes=None,
):
    ndim = input.ndim
    axes = _util._check_axes(axes, ndim)
    num_axes = len(axes)
    default_footprint = footprint is None
    sizes, footprint, _ = _filters_core._check_size_footprint_structure(
        num_axes, size, footprint, None, force_footprint=False
    )
    if cval is cupy.nan:
        raise NotImplementedError("NaN cval is unsupported")

    has_weights = True
    if sizes is not None:
        has_weights = False
        filter_size = math.prod(sizes)
        if filter_size == 0:
            return cupy.zeros_like(input)
        footprint_shape = tuple(sizes)
        (
            axes,
            footprint,
            origins,
            modes,
            int_type,
        ) = _filters_core._check_nd_args(
            input,
            None,
            mode,
            origin,
            "footprint",
            axes=axes,
            sizes=footprint_shape,
        )
    else:
        if footprint.size == 0:
            return cupy.zeros_like(input)

        (
            axes,
            footprint,
            origins,
            modes,
            int_type,
        ) = _filters_core._check_nd_args(
            input, footprint, mode, origin, "footprint", axes=axes
        )

        if default_footprint:
            filter_size = footprint.size
        else:
            footprint_shape = footprint.shape
            filter_size = int(footprint.sum())
            if filter_size == footprint.size:
                # can omit passing the footprint if it is all ones
                sizes = footprint.shape
                has_weights = False

    if not has_weights:
        footprint = None

    rank = get_rank(filter_size)
    if rank < 0 or rank >= filter_size:
        raise RuntimeError("rank not within filter footprint size")
    if rank == 0:
        min_max_op = "min"
    elif rank == filter_size - 1:
        min_max_op = "max"
    else:
        min_max_op = None
    if min_max_op is not None:
        if sizes is not None:
            return _min_or_max_filter(
                input,
                sizes,  #  [0],
                None,
                None,
                output,
                modes,
                cval,
                origins,
                min_max_op,
                axes,
            )
        else:
            return _min_or_max_filter(
                input,
                None,
                footprint,
                None,
                output,
                modes,
                cval,
                origins,
                min_max_op,
                axes,
            )
    offsets = _filters_core._origins_to_offsets(origins, footprint_shape)
    if num_axes < ndim and not has_weights:
        offsets = tuple(_util._expand_origin(ndim, axes, offsets))
        modes = tuple(_util._expand_mode(ndim, axes, modes))
        footprint_shape_temp = [1] * ndim
        for s, ax in zip(footprint_shape, axes):
            footprint_shape_temp[ax] = s
        footprint_shape = tuple(footprint_shape_temp)
    kernel = _get_rank_kernel(
        filter_size,
        rank,
        modes,
        footprint_shape,
        offsets,
        float(cval),
        int_type,
        has_weights=has_weights,
    )
    return _filters_core._call_kernel(
        kernel, input, footprint, output, weights_dtype=bool
    )


__SHELL_SORT = """
__device__ void sort(X *array, int size) {{
    int gap = {gap};
    while (gap > 1) {{
        gap /= 3;
        for (int i = gap; i < size; ++i) {{
            X value = array[i];
            int j = i - gap;
            while (j >= 0 && value < array[j]) {{
                array[j + gap] = array[j];
                j -= gap;
            }}
            array[j + gap] = value;
        }}
    }}
}}"""


@cupy._util.memoize()
def _get_shell_gap(filter_size):
    gap = 1
    while gap < filter_size:
        gap = 3 * gap + 1
    return gap


@cupy._util.memoize(for_each_device=True)
def _get_rank_kernel(
    filter_size, rank, modes, w_shape, offsets, cval, int_type, has_weights
):
    s_rank = min(rank, filter_size - rank - 1)
    # The threshold was set based on the measurements on a V100
    # TODO(leofang, anaruse): Use Optuna to automatically tune the threshold,
    # as it may vary depending on the GPU in use, compiler version, dtype,
    # filter size, etc.
    if s_rank <= 80:
        # When s_rank is small and register usage is low, this partial
        # selection sort approach is faster than general sorting approach
        # using shell sort.
        if s_rank == rank:
            comp_op = "<"
        else:
            comp_op = ">"
        array_size = s_rank + 2
        found_post = f"""
            if (iv > {s_rank} + 1) {{{{
                int target_iv = 0;
                X target_val = values[0];
                for (int jv = 1; jv <= {s_rank} + 1; jv++) {{{{
                    if (target_val {comp_op} values[jv]) {{{{
                        target_val = values[jv];
                        target_iv = jv;
                    }}}}
                }}}}
                if (target_iv <= {s_rank}) {{{{
                    values[target_iv] = values[{s_rank} + 1];
                }}}}
                iv = {s_rank} + 1;
            }}}}"""
        post = f"""
            X target_val = values[0];
            for (int jv = 1; jv <= {s_rank}; jv++) {{
                if (target_val {comp_op} values[jv]) {{
                    target_val = values[jv];
                }}
            }}
            y=cast<Y>(target_val);"""
        sorter = ""
    else:
        array_size = filter_size
        found_post = ""
        post = f"sort(values,{filter_size});\ny=cast<Y>(values[{rank}]);"
        sorter = __SHELL_SORT.format(gap=_get_shell_gap(filter_size))

    return _filters_core._generate_nd_kernel(
        f"rank_{filter_size}_{rank}",
        f"int iv = 0;\nX values[{array_size}];",
        "values[iv++] = {value};" + found_post,
        post,
        modes,
        w_shape,
        int_type,
        offsets,
        cval,
        has_weights=has_weights,
        preamble=sorter,
    )
