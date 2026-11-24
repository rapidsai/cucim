# SPDX-FileCopyrightText: Copyright (c) 2015 Preferred Infrastructure, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2015 Preferred Networks, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT

"""A vendored subset of cupyx.scipy.ndimage._util"""

import operator
import threading
import warnings
import weakref
from collections import deque
from collections.abc import Iterable
from functools import lru_cache

try:
    from numpy.exceptions import AxisError
except ImportError:
    from numpy import AxisError

import cupy

_cache_array_streams = {}
_cache_array_streams_lock = threading.Lock()
_MAX_STREAMS_PER_CACHE_ARRAY = 32
# ids of live cache arrays known to contain no zero entries
_nonzero_cache_array_ids = set()
_retired_cache_allocations = {}
_CACHE_RETIREMENT_POLL_LIMIT = 8
# A finalizer can reenter cleanup on a thread that already holds this lock.
_retired_cache_lock = threading.RLock()


def _release_completed_cache_allocations(device_id):
    """Reclaim evicted storage after all queued consumers have finished."""
    with _retired_cache_lock:
        allocations = _retired_cache_allocations.get(device_id)
        if not allocations:
            return
        # Give pending allocations another turn without letting one busy
        # stream block cleanup of its peers or scanning the entire queue.
        for _ in range(min(len(allocations), _CACHE_RETIREMENT_POLL_LIMIT)):
            if not allocations:  # A nested cleanup can drain the queue.
                break
            entry = allocations.popleft()
            complete = False
            try:
                complete = all(event.done for event in entry[1])
            finally:
                if not complete:
                    allocations.append(entry)


def _retire_cache_allocation(array_id, device_id, memory, streams):
    # The last ndarray reference is gone, so every use has now been enqueued.
    # Keep its allocation alive separately until those reads complete; returning
    # it immediately to the initialization stream's arena would allow early reuse.
    del _cache_array_streams[array_id]
    with cupy.cuda.Device(device_id):
        events = []
        for stream in streams.values():
            event = cupy.cuda.Event(disable_timing=True)
            event.record(stream)
            events.append(event)
        with _retired_cache_lock:
            _retired_cache_allocations.setdefault(device_id, deque()).append(
                (memory, events)
            )
        _release_completed_cache_allocations(device_id)


def _register_cache_array(array, all_nonzero=False):
    """Protect an initialized array's allocation from reuse after eviction.

    If ``all_nonzero`` is True, the array (which must contain no zero entries)
    is also recorded for :func:`_is_nonzero_cache_array`.
    """
    device_id = array.device.id
    _release_completed_cache_allocations(device_id)
    streams = _cache_array_streams[id(array)] = {}
    # The memory pointer owns the allocation, without retaining the ndarray.
    finalizer = weakref.finalize(
        array,
        _retire_cache_allocation,
        id(array),
        device_id,
        array.data,
        streams,
    )
    finalizer.atexit = False
    if all_nonzero:
        _nonzero_cache_array_ids.add(id(array))
        weakref.finalize(
            array, _nonzero_cache_array_ids.discard, id(array)
        ).atexit = False
    return array


def _is_nonzero_cache_array(array):
    """Whether ``array`` is a cache array registered as having no zeros.

    Filter kernels skip zero weights so that a non-finite input value under a
    zero weight does not propagate (as in SciPy). For cached coefficients the
    absence of zeros is checked once, when the cache entry is created, which
    lets the kernels omit the per-tap test without a host synchronization.
    """
    return id(array) in _nonzero_cache_array_ids


def _coefficient_cache_stream():
    """Return a trackable stream or None for a per-thread default stream."""
    stream = cupy.cuda.get_current_stream()
    runtime = cupy.cuda.runtime
    if not runtime.is_hip and stream.ptr not in (0, runtime.streamLegacy):
        if stream.is_capturing():
            # Retirement only covers queued work, not future graph replays.
            raise RuntimeError(
                "Direct CUDA Graph capture is unsupported for operations "
                "using the coefficient cache"
            )
        if stream.ptr == runtime.streamPerThread:
            # The PTDS handle resolves relative to the issuing host thread.
            # A finalizer on another thread cannot record a completion event
            # for this consumer. Allocate in the caller's stream instead.
            return None
    return stream


def _record_cache_array_use(array, stream):
    """Return the shared array, or None if the caller needs private storage."""
    streams = _cache_array_streams[id(array)]
    ptr = stream.ptr
    if ptr not in streams:
        with _cache_array_streams_lock:
            if ptr not in streams:
                if len(streams) >= _MAX_STREAMS_PER_CACHE_ARRAY:
                    return None
                streams[ptr] = stream
    return array


@cupy.memoize(for_each_device=True)
def _get_cache_init_stream():
    """Share a private stream for small cache-initialization copies per device.

    Synchronize this stream before publishing an entry so other streams can
    reuse it immediately, without waiting for work queued by the caller.
    """
    return cupy.cuda.Stream(non_blocking=True)


def _cached_filter_weights(weights, dtype, divisor=1):
    """Reuse small built-in coefficient tuples; callers must not modify them.

    Optional normalization runs once in the working dtype on the device, as
    in the uncached edge filters. Keep it in the key so magnitude and single-
    axis filters never share coefficients with different normalization.
    """
    caller = _coefficient_cache_stream()
    if caller is None:
        return _make_filter_weights(weights, dtype, divisor)
    device_id = cupy.cuda.runtime.getDevice()
    array = _cached_filter_weights_for_device(
        device_id, weights, cupy.dtype(dtype).str, divisor
    )
    if _record_cache_array_use(array, caller) is None:
        return _make_filter_weights(weights, dtype, divisor)
    return array


def _make_filter_weights(weights, dtype, divisor):
    array = cupy.asarray(weights, dtype=dtype)
    if divisor != 1:
        array /= divisor
    return array


@lru_cache(maxsize=64)
def _cached_filter_weights_for_device(device_id, weights, dtype, divisor):
    """Publish initialized coefficients using the shared cache stream policy."""
    with cupy.cuda.Device(device_id):
        stream = _get_cache_init_stream()
        with stream:
            array = _make_filter_weights(weights, dtype, divisor)
            # checked once per cache entry (synchronizes the cache stream)
            all_nonzero = int(cupy.count_nonzero(array)) == array.size
        stream.synchronize()
        return _register_cache_array(array, all_nonzero=all_nonzero)


def _is_integer_output(output, input):
    if output is None:
        return input.dtype.kind in "iu"
    elif isinstance(output, cupy.ndarray):
        return output.dtype.kind in "iu"
    return cupy.dtype(output).kind in "iu"


def _check_cval(mode, cval, integer_output):
    if mode == "constant" and integer_output and not cupy.isfinite(cval):
        raise NotImplementedError(
            "Non-finite cval is not supported for outputs with integer dtype."
        )


def _get_weights_dtype(input, weights, use_cucim_casting=False):
    if weights.dtype.kind == "c" or input.dtype.kind == "c":
        return cupy.promote_types(input.real.dtype, cupy.complex64)
    elif weights.dtype.kind in "iub":
        if use_cucim_casting:
            from cucim.skimage._shared.utils import _supported_float_type

            return _supported_float_type(weights.dtype)
        else:
            # convert integer dtype weights to double as in SciPy
            return cupy.float64
    return cupy.promote_types(input.real.dtype, cupy.float32)


def _get_output(output, input, shape=None, complex_output=False):
    shape = input.shape if shape is None else shape
    if output is None:
        if complex_output:
            _dtype = cupy.promote_types(input.dtype, cupy.complex64)
        else:
            _dtype = input.dtype
        output = cupy.empty(shape, dtype=_dtype)
    elif isinstance(output, (type, cupy.dtype)):
        if complex_output and cupy.dtype(output).kind != "c":
            warnings.warn("promoting specified output dtype to complex")
            output = cupy.promote_types(output, cupy.complex64)
        output = cupy.empty(shape, dtype=output)
    elif isinstance(output, str):
        output = cupy.dtype(output)
        if complex_output and output.kind != "c":
            raise RuntimeError("output must have complex dtype")
        output = cupy.empty(shape, dtype=output)
    elif output.shape != shape:
        raise RuntimeError("output shape not correct")
    elif complex_output and output.dtype.kind != "c":
        raise RuntimeError("output must have complex dtype")
    return output


def _fix_sequence_arg(arg, ndim, name, conv=lambda x: x):
    if isinstance(arg, str):
        return [conv(arg)] * ndim
    try:
        arg = iter(arg)
    except TypeError:
        return [conv(arg)] * ndim
    lst = [conv(x) for x in arg]
    if len(lst) != ndim:
        msg = f"{name} must have length equal to input rank"
        raise RuntimeError(msg)
    return lst


def _check_origin(origin, width):
    origin = int(origin)
    if (width // 2 + origin < 0) or (width // 2 + origin >= width):
        raise ValueError("invalid origin")
    return origin


def _check_mode(mode):
    if mode not in (
        "reflect",
        "constant",
        "nearest",
        "mirror",
        "wrap",
        "grid-mirror",
        "grid-wrap",
        "grid-reflect",
    ):
        msg = f"boundary mode not supported (actual: {mode})"
        raise RuntimeError(msg)
    return mode


def _check_axes(axes, ndim):
    if axes is None:
        return tuple(range(ndim))
    elif cupy.isscalar(axes):
        axes = (operator.index(axes),)
    elif isinstance(axes, Iterable):
        axes = tuple(operator.index(ax) for ax in axes)
        for ax in axes:
            if ax < -ndim or ax > ndim - 1:
                raise AxisError(f"specified axis: {ax} is out of range")
        axes = tuple(ax % ndim if ax < 0 else ax for ax in axes)
    else:
        message = "axes must be an integer, iterable of integers, or None"
        raise ValueError(message)
    if len(tuple(set(axes))) != len(axes):
        raise ValueError("axes must be unique")
    return axes


def _get_inttype(input):
    # The integer type to use for indices in the input array
    # The indices actually use byte positions and we can't just use
    # input.nbytes since that won't tell us the number of bytes between the
    # first and last elements when the array is non-contiguous
    nbytes = (
        sum(
            (x - 1) * abs(stride)
            for x, stride in zip(input.shape, input.strides)
        )
        + input.dtype.itemsize
    )
    return "int" if nbytes < (1 << 31) else "ptrdiff_t"


def _expand_origin(ndim_image, axes, origin):
    num_axes = len(axes)
    origins = _fix_sequence_arg(origin, num_axes, "origin", int)
    if num_axes < ndim_image:
        # set origin = 0 for any axes not being filtered
        origins_temp = [
            0,
        ] * ndim_image
        for o, ax in zip(origins, axes):
            origins_temp[ax] = o
        origins = origins_temp
    return origins


def _expand_footprint(ndim_image, axes, footprint, footprint_name="footprint"):
    num_axes = len(axes)
    if num_axes < ndim_image:
        if footprint.ndim != num_axes:
            raise RuntimeError(
                f"{footprint_name}.ndim ({footprint.ndim}) "
                f"must match len(axes) ({num_axes})"
            )

        footprint = cupy.expand_dims(
            footprint, tuple(ax for ax in range(ndim_image) if ax not in axes)
        )
    return footprint


def _expand_mode(ndim_image, axes, mode):
    num_axes = len(axes)
    if not isinstance(mode, str) and isinstance(mode, Iterable):
        # set mode = 'constant' for any axes not being filtered
        modes = _fix_sequence_arg(mode, num_axes, "mode", str)
        modes_temp = ["constant"] * ndim_image
        for m, ax in zip(modes, axes):
            modes_temp[ax] = m
        mode = modes_temp
    return mode


def _generate_boundary_condition_ops(
    mode, ix, xsize, int_t="int", float_ix=False, separate=False
):
    """Generate boundary conditions

    If separate = True, a pair of conditions for the (lower, upper) boundary
    are provided instead of a single expression.
    """
    min_func = "fmin" if float_ix else "min"
    max_func = "fmax" if float_ix else "max"
    if mode in ["reflect", "grid-mirror"]:
        if separate:
            ops_upper = f"""
            {ix} %= {xsize} * 2;
            {ix} = {min_func}({ix}, 2 * {xsize} - 1 - {ix});
            """
            ops_lower = (
                f"""
            if ({ix} < 0) {{
                {ix} = - 1 -{ix};
            }}
            """
                + ops_upper
            )
            ops = (ops_lower, ops_upper)
        else:
            ops = f"""
            if ({ix} < 0) {{
                {ix} = - 1 -{ix};
            }}
            {ix} %= {xsize} * 2;
            {ix} = {min_func}({ix}, 2 * {xsize} - 1 - {ix});"""
    elif mode == "mirror":
        if separate:
            temp1 = f"""
            if ({xsize} == 1) {{
                {ix} = 0;
            }} else {{
            """
            temp2 = f"""
                if ({ix} < 0) {{
                    {ix} = -{ix};
                }}
            """
            temp3 = f"""
                {ix} = 1 + ({ix} - 1) % (({xsize} - 1) * 2);
                {ix} = {min_func}({ix}, 2 * {xsize} - 2 - {ix});
            }}"""
            ops_lower = temp1 + temp2 + temp3
            ops_upper = temp1 + temp3
            ops = (ops_lower, ops_upper)
        else:
            ops = f"""
            if ({xsize} == 1) {{
                {ix} = 0;
            }} else {{
                if ({ix} < 0) {{
                    {ix} = -{ix};
                }}
                {ix} = 1 + ({ix} - 1) % (({xsize} - 1) * 2);
                {ix} = {min_func}({ix}, 2 * {xsize} - 2 - {ix});
            }}"""
    elif mode == "nearest":
        T = "int" if int_t == "int" else "long long"
        if separate:
            ops_lower = f"""{ix} = {max_func}(({T}){ix}, ({T})0);"""
            ops_upper = f"""{ix} = {min_func}(({T}){ix}, ({T})({xsize} - 1));"""  # noqa
            ops = (ops_lower, ops_upper)
        else:
            ops = f"""{ix} = {min_func}({max_func}(({T}){ix}, ({T})0), ({T})({xsize} - 1));"""  # noqa
    elif mode == "grid-wrap":
        if separate:
            ops_upper = f"""
            {ix} %= {xsize};
            """
            ops_lower = (
                ops_upper
                + f"""
            while ({ix} < 0) {{
                {ix} += {xsize};
            }}"""
            )
            ops = (ops_lower, ops_upper)
        else:
            ops = f"""
            {ix} %= {xsize};
            if ({ix} < 0) {{
                {ix} += {xsize};
            }}"""

    elif mode == "wrap":
        if separate:
            ops_lower = f"""{ix} += ({xsize} - 1) * (({int_t})(-{ix} / ({xsize} - 1)) + 1);"""  # noqa
            ops_upper = (
                f"""{ix} -= ({xsize} - 1) * ({int_t})({ix} / ({xsize} - 1));"""  # noqa
            )
            ops = (ops_lower, ops_upper)
        else:
            ops = f"""
            if ({ix} < 0) {{
                {ix} += ({xsize} - 1) * (({int_t})(-{ix} / ({xsize} - 1)) + 1);
            }} else if ({ix} > ({xsize} - 1)) {{
                {ix} -= ({xsize} - 1) * ({int_t})({ix} / ({xsize} - 1));
            }};"""
    elif mode in ["constant", "grid-constant"]:
        if separate:
            ops_lower = f"""
            if ({ix} < 0) {{
                {ix} = -1;
            }}"""
            ops_upper = f"""
            if ({ix} >= {xsize}) {{
                {ix} = -1;
            }}"""
            ops = (ops_lower, ops_upper)
        else:
            ops = f"""
            if (({ix} < 0) || {ix} >= {xsize}) {{
                {ix} = -1;
            }}"""
        if separate:
            ops = (ops, ops)
    return ops


def _generate_indices_ops(ndim, int_type, offsets):
    code = "{type} ind_{j} = _i % ysize_{j} - {offset}; _i /= ysize_{j};"
    body = [
        code.format(type=int_type, j=j, offset=offsets[j])
        for j in range(ndim - 1, 0, -1)
    ]
    return "{type} _i = i;\n{body}\n{type} ind_0 = _i - {offset};".format(
        type=int_type, body="\n".join(body), offset=offsets[0]
    )
