# SPDX-FileCopyrightText: Copyright (c) 2015 Preferred Infrastructure, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2015 Preferred Networks, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT

"""
Spline poles and boundary handling implemented as in SciPy

https://github.com/scipy/scipy/blob/master/scipy/ndimage/src/ni_splines.c
"""

from __future__ import annotations

import functools
import math
import operator
import textwrap

import cupy


def get_poles(order):
    if order == 2:
        # sqrt(8.0) - 3.0
        return (-0.171572875253809902396622551580603843,)
    elif order == 3:
        # sqrt(3.0) - 2.0
        return (-0.267949192431122706472553658494127633,)
    elif order == 4:
        # sqrt(664.0 - sqrt(438976.0)) + sqrt(304.0) - 19.0
        # sqrt(664.0 + sqrt(438976.0)) - sqrt(304.0) - 19.0
        return (
            -0.361341225900220177092212841325675255,
            -0.013725429297339121360331226939128204,
        )
    elif order == 5:
        # sqrt(67.5 - sqrt(4436.25)) + sqrt(26.25) - 6.5
        # sqrt(67.5 + sqrt(4436.25)) - sqrt(26.25) - 6.5
        return (
            -0.430575347099973791851434783493520110,
            -0.043096288203264653822712376822550182,
        )
    else:
        raise ValueError("only order 2-5 supported")


def get_gain(poles):
    return functools.reduce(
        operator.mul, [(1.0 - z) * (1.0 - 1.0 / z) for z in poles]
    )


def _causal_init_code(mode):
    """Code for causal initialization step of IIR filtering.

    c is a 1d array of length n and z is a filter pole
    """
    code = f"""
        // causal init for mode={mode}"""
    if mode == "mirror":
        code += """
        z_i = z;
        z_n_1 = pow(z, (P)(n - 1));

        c[0] = c[0] + z_n_1 * gain * c[(n - 1) * element_stride];
        for (i = 1; i < min(n - 1, static_cast<idx_t>({n_boundary})); ++i) {{
            c[0] += z_i * gain * (c[i * element_stride] +
                                   z_n_1 * c[(n - 1 - i) * element_stride]);
            z_i *= z;
        }}
        c[0] /= 1 - z_n_1 * z_n_1;"""
    elif mode == "grid-wrap":
        code += """
        z_i = z;

        for (i = 1; i < min(n, static_cast<idx_t>({n_boundary})); ++i) {{
            c[0] += z_i * gain * c[(n - i) * element_stride];
            z_i *= z;
        }}
        c[0] /= 1 - z_i; /* z_i = pow(z, n) */"""
    elif mode == "reflect":
        code += """
        z_i = z;
        z_n = pow(z, (P)n);
        c0 = c[0];

        c[0] = c[0] + z_n * gain * c[(n - 1) * element_stride];
        for (i = 1; i < min(n - 1, static_cast<idx_t>({n_boundary})); ++i) {{
            c[0] += z_i * gain * (c[i * element_stride] +
                                   z_n * c[(n - 1 - i) * element_stride]);
            z_i *= z;
        }}
        // The last reflected pair reads the already scaled and updated c[0].
        // Apply gain only to its other sample, matching an upfront gain pass.
        if (n <= static_cast<idx_t>({n_boundary})) {{
            c[0] += z_i * (gain * c[(n - 1) * element_stride] + z_n * c[0]);
        }}
        c[0] *= z / (1 - z_n * z_n);
        c[0] += c0;"""
    else:
        raise ValueError(f"invalid mode: {mode}")
    return code


def _anticausal_init_code(mode):
    """Code for the anti-causal initialization step of IIR filtering.

    c is a 1d array of length n and z is a filter pole
    """
    code = f"""
        // anti-causal init for mode={mode}"""
    if mode == "mirror":
        code += """
        c[(n - 1) * element_stride] = (
            z * c[(n - 2) * element_stride] +
            c[(n - 1) * element_stride]) * z / (z * z - 1);"""
    elif mode == "grid-wrap":
        code += """
        z_i = z;

        for (i = 0; i < min(n - 1, static_cast<idx_t>({n_boundary})); ++i) {{
            c[(n - 1) * element_stride] += z_i * c[i * element_stride];
            z_i *= z;
        }}
        c[(n - 1) * element_stride] *= z / (z_i - 1); /* z_i = pow(z, n) */"""
    elif mode == "reflect":
        code += """
        c[(n - 1) * element_stride] *= z / (z - 1);"""
    else:
        raise ValueError(f"invalid mode: {mode}")
    return code


def _get_spline_mode(mode):
    """spline boundary mode for interpolation with order >= 2."""
    if mode in ["mirror", "reflect", "grid-wrap"]:
        # exact analytic boundary conditions exist for these modes.
        return mode
    elif mode == "grid-mirror":
        # grid-mirror is a synonym for 'reflect'
        return "reflect"
    # No exact analytical spline boundary condition implemented. Reflect gives
    # lower error than using mirror or wrap for mode 'nearest'. Otherwise, a
    # mirror spline boundary condition is used.
    return "reflect" if mode == "nearest" else "mirror"


def _get_spline1d_code(mode, poles, n_boundary):
    """Generates the code required for IIR filtering of a single 1d signal.

    Prefiltering is done by causal filtering followed by anti-causal filtering.
    Multiple boundary conditions have been implemented.
    """
    code = [
        """
    __device__ void spline_prefilter1d(
        T* __restrict__ c, idx_t signal_length, idx_t element_stride)
    {{"""
    ]

    # variables common to all boundary modes
    code.append(
        """
        idx_t i, n = signal_length;
        P z, z_i;"""
    )
    gain = get_gain(poles)
    code.append(
        f"""
        // apply prefilter gain
        P gain = {gain};
        c[0] *= gain;"""
    )

    # retrieve the spline boundary extension mode to use
    mode = _get_spline_mode(mode)

    if mode == "mirror":
        # variables specific to mirror boundary mode
        code.append(
            """
        P z_n_1;"""
        )
    elif mode == "reflect":
        # variables specific to reflect boundary mode
        code.append(
            """
        P z_n;
        T c0;"""
        )

    for pole in poles:
        code.append(
            f"""
        // select the current pole
        z = {pole};"""
        )

        # initialize and apply the causal filter
        code.append(_causal_init_code(mode))
        code.append(
            """
        // apply the causal filter for the current pole
        for (i = 1; i < n; ++i) {{
            c[i * element_stride] = gain * c[i * element_stride] +
                                    z * c[(i - 1) * element_stride];
        }}"""
        )
        # initialize and apply the anti-causal filter
        code.append(_anticausal_init_code(mode))
        code.append(
            """
        // apply the anti-causal filter for the current pole
        for (i = n - 2; i >= 0; --i) {{
            c[i * element_stride] = z * (c[(i + 1) * element_stride] -
                                         c[i * element_stride]);
        }}
        gain = 1;"""
        )

    code += [
        """
    }}"""
    ]
    return textwrap.dedent("\n".join(code)).format(n_boundary=n_boundary)


_FILTER_GENERAL = """
#include "cupy/carray.cuh"
#include "cupy/complex.cuh"
#include <cupy/float16.cuh>  // TODO(seberg): Add this via type_headers?

typedef {data_type} T;
typedef {pole_type} P;
typedef {index_type} idx_t;
template <typename T>
__device__ T* row(
        T* ptr, idx_t i, idx_t axis, idx_t ndim, const idx_t* shape) {{
    idx_t index = 0, stride = 1;
    for (idx_t a = ndim - 1; a > 0; --a) {{
        if (a != axis) {{
            index += (i % shape[a]) * stride;
            i /= shape[a];
        }}
        stride *= shape[a];
    }}
    return ptr + index + stride * i;
}}
"""


_batch_spline1d_strided_template = """
extern "C" __global__
__launch_bounds__({block_size})
void {kernel_name}(T* __restrict__ y, idx_t n_signals, idx_t n_samples,
                   {shape_parameters}) {{
    const idx_t shape[] = {{{shape_values}}};
    idx_t y_elem_stride = 1;
    for (int a = {ndim} - 1; a > {axis}; --a) {{ y_elem_stride *= shape[a]; }}
    idx_t unraveled_idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx_t batch_idx = unraveled_idx;
    if (batch_idx < n_signals)
    {{
        T* __restrict__ y_i = row(y, batch_idx, {axis}, {ndim}, shape);
        spline_prefilter1d(y_i, n_samples, y_elem_stride);
    }}
}}
"""


def _get_n_boundary(poles, pole_type):
    """Number of samples used by the truncated boundary initializations."""
    # SciPy uses n_boundary = n_samples, but this is excessive.
    largest_pole = max(abs(p) for p in poles)
    # tol < 1e-7 fails test cases comparing to SciPy at atol = rtol = 1e-5
    tol = 1e-10 if pole_type == "float" else 1e-18
    return math.ceil(math.log(tol, largest_pole))


_SEGMENTED_SPLINE1D_HEADER = """
#include "cupy/carray.cuh"
#include "cupy/complex.cuh"
#include <cupy/float16.cuh>  // TODO(seberg): Add this via type_headers?

typedef {data_type} T;
typedef {pole_type} P;
typedef {index_type} idx_t;

extern "C" __global__
__launch_bounds__({block_size})
void {kernel_name}(T* __restrict__ y, idx_t n_lines, idx_t n, idx_t inner)
{{
    // A line is indexed by its (outer, inner) position. Element i of line
    // `line` is at (line / inner) * n * inner + i * inner + line % inner, so
    // the 32 lines handled by a warp are adjacent in memory when inner >= 32.
    constexpr int n_segments = {n_segments};
    __shared__ P seg_A[n_segments][32];
    __shared__ T seg_B[n_segments][32];
    const int lane = threadIdx.x;
    const int segment = threadIdx.y;
    const idx_t line = static_cast<idx_t>(blockIdx.x) * 32 + lane;
    const bool active = line < n_lines;
    const idx_t outer = active ? line / inner : 0;
    const idx_t element_stride = inner;
    T* __restrict__ c = y + (outer * n * inner + (line - outer * inner));

    // Variables shared with the boundary initialization code.
    idx_t i;
    P z, z_i;
    P z_n, z_n_1;
    T c0;
    (void)z_n; (void)z_n_1; (void)c0;

    // Each of the n_segments threads of a line filters a contiguous run of
    // samples. A run maps an incoming carry x to A * x + B; composing the
    // transforms of earlier runs gives each run its exact incoming carry.
    const idx_t seg_len = (n - 2 + n_segments) / n_segments;
    idx_t start, end;
    P A, A_run;
    T B, carry;

    P gain = {gain};
    if (active && segment == 0) {{
        c[0] *= gain;
    }}
"""

_SEGMENTED_SPLINE1D_POLE = """
    // ---- pole {pole_index} ----
    z = {pole};

    // causal initialization and filter over samples 1 .. n - 1
    if (active && segment == 0) {{
{causal_init}
    }}
    start = 1 + segment * seg_len;
    end = min(n, start + seg_len);
    A = 1;
    B = 0;
    if (active) {{
        for (i = start; i < end; ++i) {{
            B = z * B + gain * c[i * element_stride];
            A *= z;
        }}
    }}
    seg_A[segment][lane] = A;
    seg_B[segment][lane] = B;
    __syncthreads();
    if (active) {{
        carry = c[0];
        for (int s = 0; s < segment; ++s) {{
            A_run = seg_A[s][lane];
            carry = A_run * carry + seg_B[s][lane];
        }}
        for (i = start; i < end; ++i) {{
            carry = z * carry + gain * c[i * element_stride];
            c[i * element_stride] = carry;
        }}
    }}
    __syncthreads();

    // anti-causal initialization and filter over samples n - 2 .. 0
    if (active && segment == 0) {{
{anticausal_init}
    }}
    start = segment * seg_len;
    end = min(n - 1, start + seg_len);
    A = 1;
    B = 0;
    if (active) {{
        for (idx_t p = start; p < end; ++p) {{
            i = n - 2 - p;
            B = z * B - z * c[i * element_stride];
            A *= z;
        }}
    }}
    seg_A[segment][lane] = A;
    seg_B[segment][lane] = B;
    __syncthreads();
    if (active) {{
        carry = c[(n - 1) * element_stride];
        for (int s = 0; s < segment; ++s) {{
            A_run = seg_A[s][lane];
            carry = A_run * carry + seg_B[s][lane];
        }}
        for (idx_t p = start; p < end; ++p) {{
            i = n - 2 - p;
            carry = z * carry - z * c[i * element_stride];
            c[i * element_stride] = carry;
        }}
    }}
    __syncthreads();
    gain = 1;
"""


def _get_segmented_spline1d_code(
    kernel_name, mode, poles, n_boundary, n_segments, types
):
    """Generate a block-parallel spline prefilter for strided lines."""
    mode = _get_spline_mode(mode)
    code = _SEGMENTED_SPLINE1D_HEADER.format(
        kernel_name=kernel_name,
        n_segments=n_segments,
        block_size=32 * n_segments,
        gain=get_gain(poles),
        **types,
    )
    for pole_index, pole in enumerate(poles):
        causal = _causal_init_code(mode).format(n_boundary=n_boundary)
        anticausal = _anticausal_init_code(mode).format(n_boundary=n_boundary)
        code += _SEGMENTED_SPLINE1D_POLE.format(
            pole_index=pole_index,
            pole=pole,
            causal_init=textwrap.indent(textwrap.dedent(causal), " " * 8),
            anticausal_init=textwrap.indent(
                textwrap.dedent(anticausal), " " * 8
            ),
        )
    return code + "}\n"


@cupy.memoize(for_each_device=True)
def get_raw_spline1d_kernel(
    axis,
    ndim,
    mode,
    order,
    index_type="int",
    data_type="double",
    pole_type="double",
    block_size=128,
):
    """Generate a spline prefilter with runtime sizes passed as scalar arguments.

    Kernel specialization depends on rank/axis/types and spline settings,
    rather than exact image shape. No separate device metadata allocation or
    host-to-device copy is needed.
    """
    poles = get_poles(order)

    # determine number of samples for the boundary approximation
    # (SciPy uses n_boundary = n_samples but this is excessive)
    largest_pole = max([abs(p) for p in poles])
    # tol < 1e-7 fails test cases comparing to SciPy at atol = rtol = 1e-5
    tol = 1e-10 if pole_type == "float" else 1e-18
    n_boundary = math.ceil(math.log(tol, largest_pole))

    # headers and general utility function for extracting rows of data
    code = _FILTER_GENERAL.format(
        index_type=index_type, data_type=data_type, pole_type=pole_type
    )

    # generate source for a 1d function for a given boundary mode and poles
    code += _get_spline1d_code(mode, poles, n_boundary)

    # generate code handling batch operation of the 1d filter
    mode_str = mode.replace("-", "_")  # cannot have '-' in kernel name
    kernel_name = (
        f"cupyx_scipy_ndimage_spline_filter_{ndim}d_ord{order}_"
        f"axis{axis}_{mode_str}"
    )
    code += _batch_spline1d_strided_template.format(
        ndim=ndim,
        axis=axis,
        block_size=block_size,
        kernel_name=kernel_name,
        shape_parameters=", ".join(
            f"idx_t shape_{axis}" for axis in range(ndim)
        ),
        shape_values=", ".join(f"shape_{axis}" for axis in range(ndim)),
    )
    return cupy.RawKernel(code, kernel_name)


def supports_segmented_spline1d(axis, ndim, data_type):
    """Return True if the block-parallel strided prefilter can be used.

    Lines along any axis other than the last are filtered with several
    threads per line, while adjacent lines keep global memory accesses
    coalesced. The workspace must be C-contiguous.
    """
    return axis < ndim - 1 and data_type in ("float", "double")


def get_segmented_spline1d_segments(n_samples, data_type="float"):
    """Number of threads (runs) that cooperatively filter each line."""
    # Keep runs at least 8 samples long so the recurrence dominates the carry
    # composition. Measured on an RTX A6000, 32 runs (1024 threads per block)
    # were best for float and 16 runs for double.
    max_segments = 32 if data_type == "float" else 16
    n_segments = 1
    while n_segments < max_segments and 2 * n_segments * 8 <= n_samples - 1:
        n_segments *= 2
    return n_segments


@cupy.memoize(for_each_device=True)
def get_segmented_spline1d_kernel(
    mode,
    order,
    n_segments,
    index_type="int",
    data_type="float",
    pole_type="float",
):
    """Generate a block-parallel spline prefilter for strided lines."""
    if data_type not in ("float", "double"):
        raise ValueError(f"unsupported segmented spline dtype: {data_type}")
    poles = get_poles(order)
    n_boundary = _get_n_boundary(poles, pole_type)
    mode_str = _get_spline_mode(mode).replace("-", "_")
    kernel_name = (
        f"cupyx_scipy_ndimage_segmented_spline_ord{order}_{mode_str}_"
        f"seg{n_segments}"
    )
    code = _get_segmented_spline1d_code(
        kernel_name,
        mode,
        poles,
        n_boundary,
        n_segments,
        dict(data_type=data_type, pole_type=pole_type, index_type=index_type),
    )
    return cupy.RawKernel(code, kernel_name)
