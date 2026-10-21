# SPDX-FileCopyrightText: Copyright (c) 2015 Preferred Infrastructure, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2015 Preferred Networks, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT

"""Spline prefilter kernels and their memory-access patterns.

Kernel families
---------------
``spline_filter1d`` in ``_ndimage_interpolation`` selects among three kernel
families after preparing a C-contiguous workspace. All implement the causal
and anti-causal IIR recurrences for each spline pole (orders 2--5), share the
boundary-initialization formulas, and fuse the gain into the first causal
pass. Cooperative kernels summarize a run as an affine carry map
``incoming -> A * incoming + B``, compose those maps, and replay each run with
its correct incoming carry.

``get_raw_spline1d_kernel`` is the general fallback: one thread processes an
entire line sequentially in global memory. Threads can process independent
lines without cooperation or shared-memory storage. This keeps setup small
for short lines and supports complex workspaces and real cases outside the
optimized dispatch. Long recurrences limit parallelism when there are few
lines, and last-axis lines give neighboring threads widely separated global
addresses, motivating the two cooperative layouts below.

``get_segmented_spline1d_kernel`` handles non-last axes of real float32/float64
workspaces. A block has shape ``(32, n_segments)``: a warp spans 32 adjacent
lines, while threads at the same x coordinate split one line into runs. At a
fixed sample, adjacent lines access neighboring trailing elements, so global
accesses coalesce well when the trailing dimension spans a warp. Shared memory
holds only the runs' carry maps; block barriers make them available for carry
composition before replay. This exposes several threads per long line while
preserving the favorable global-access pattern of filtering a strided axis.

``get_warp_spline1d_kernel`` handles contiguous last-axis lines and the axis
before a trailing dimension of 2--4 elements, such as RGB/RGBA channels. Each
warp stages a complete line or interleaved ``(samples, channels)`` block with
coalesced global loads, filters it in shared memory, then stores it coalesced.
Lanes own runs and use shuffle-based prefix scans to compose their carry
maps, carrying each channel independently. Odd run lengths avoid systematic
shared-memory bank collisions for contiguous and odd-channel layouts. Staging
avoids the widely separated global accesses that lane-owned runs would
otherwise generate; scalar loads also allow element-aligned output views.

The staged warp path needs at least 32 samples and enough device shared
memory for the staged block. Launches use 4, 2, or 1 independent warps per
block, opting in to larger shared memory for long lines when available.
Float64 uses this path for lines of at least 512 samples, or at most 256
single-channel lines: serial per-channel boundary initialization otherwise
outweighs the parallel recurrence. Ineligible strided cases use the segmented
kernel; remaining cases use the general fallback. The dispatch and run-count
choices are empirical RTX A6000 tuning, rather than universal cutoffs.

Spline poles and boundary handling are implemented as in SciPy:
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


_WARP_SPLINE1D_HEADER = """
#include "cupy/carray.cuh"
#include "cupy/complex.cuh"
#include <cupy/float16.cuh>  // TODO(seberg): Add this via type_headers?

typedef {data_type} T;
typedef {pole_type} P;
typedef {index_type} idx_t;

// Number of interleaved lines: 1 for a contiguous axis, or the size of a
// small trailing axis (e.g. RGB/RGBA channels) when filtering the axis
// before it. Element i of channel ch is at lines[i * CH + ch].
constexpr int CH = {channels};

__device__ __forceinline__ void scan_transform(P &A, T (&B)[CH])
{{
    // Inclusive warp scan composing the runs' carry maps x -> A * x + B.
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {{
        P A_up = __shfl_up_sync(0xffffffffu, A, offset);
        T B_up[CH];
        #pragma unroll
        for (int ch = 0; ch < CH; ++ch) {{
            B_up[ch] = __shfl_up_sync(0xffffffffu, B[ch], offset);
        }}
        if ((threadIdx.x & 31) >= offset) {{
            #pragma unroll
            for (int ch = 0; ch < CH; ++ch) {{
                B[ch] = B[ch] + A * B_up[ch];
            }}
            A = A * A_up;
        }}
    }}
}}

__device__ __forceinline__ void run_carry_in(
    P A, const T (&B)[CH], const T* __restrict__ init, T (&carry)[CH])
{{
    // Carries entering this lane's run, from the exclusive scan.
    P A_prev = __shfl_up_sync(0xffffffffu, A, 1);
    #pragma unroll
    for (int ch = 0; ch < CH; ++ch) {{
        T B_prev = __shfl_up_sync(0xffffffffu, B[ch], 1);
        carry[ch] = (threadIdx.x & 31) == 0 ? init[ch]
                                             : A_prev * init[ch] + B_prev;
    }}
}}

extern "C" __global__
__launch_bounds__({block_size})
void {kernel_name}(T* __restrict__ y, idx_t n_signals, idx_t n)
{{
    // Each warp filters one contiguous (n, CH) block, staged in shared memory
    // so that global loads and stores are coalesced.
    extern __shared__ __align__(8) unsigned char smem_raw[];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const idx_t line = static_cast<idx_t>(blockIdx.x) * {warps_per_block}
                       + warp;
    if (line >= n_signals) {{
        return;  // uniform across the warp; only warp-level syncs are used
    }}
    const idx_t n_elements = n * CH;
    T* __restrict__ g = y + static_cast<ptrdiff_t>(line) * n_elements;
    T* __restrict__ lines = reinterpret_cast<T*>(smem_raw) + warp * n_elements;
    for (idx_t j = lane; j < n_elements; j += 32) {{
        lines[j] = g[j];
    }}
    __syncwarp();

    // Variables shared with the boundary initialization code.
    const idx_t element_stride = CH;
    idx_t i;
    P z, z_i;
    P z_n, z_n_1;
    T c0;
    (void)z_n; (void)z_n_1; (void)c0;

    // Lane runs have odd length, so for odd CH the 32 lanes access distinct
    // shared memory banks.
    const idx_t seg_len = ((n - 2 + 32) / 32) | 1;
    idx_t start, end;
    P A;
    T B[CH], carry[CH];

    P gain = {gain};
    if (lane == 0) {{
        #pragma unroll
        for (int ch = 0; ch < CH; ++ch) {{
            lines[ch] *= gain;
        }}
    }}
"""

_WARP_SPLINE1D_POLE = """
    // ---- pole {pole_index} ----
    z = {pole};

    // causal initialization and filter over samples 1 .. n - 1
    if (lane == 0) {{
        for (int ch = 0; ch < CH; ++ch) {{
            T* __restrict__ c = lines + ch;
{causal_init}
        }}
    }}
    __syncwarp();
    start = 1 + lane * seg_len;
    end = min(n, start + seg_len);
    A = 1;
    #pragma unroll
    for (int ch = 0; ch < CH; ++ch) {{
        B[ch] = 0;
    }}
    for (i = start; i < end; ++i) {{
        #pragma unroll
        for (int ch = 0; ch < CH; ++ch) {{
            B[ch] = z * B[ch] + gain * lines[i * CH + ch];
        }}
        A *= z;
    }}
    scan_transform(A, B);
    run_carry_in(A, B, lines, carry);
    for (i = start; i < end; ++i) {{
        #pragma unroll
        for (int ch = 0; ch < CH; ++ch) {{
            carry[ch] = z * carry[ch] + gain * lines[i * CH + ch];
            lines[i * CH + ch] = carry[ch];
        }}
    }}
    __syncwarp();

    // anti-causal initialization and filter over samples n - 2 .. 0
    if (lane == 0) {{
        for (int ch = 0; ch < CH; ++ch) {{
            T* __restrict__ c = lines + ch;
{anticausal_init}
        }}
    }}
    __syncwarp();
    start = lane * seg_len;
    end = min(n - 1, start + seg_len);
    A = 1;
    #pragma unroll
    for (int ch = 0; ch < CH; ++ch) {{
        B[ch] = 0;
    }}
    for (idx_t p = start; p < end; ++p) {{
        i = n - 2 - p;
        #pragma unroll
        for (int ch = 0; ch < CH; ++ch) {{
            B[ch] = z * B[ch] - z * lines[i * CH + ch];
        }}
        A *= z;
    }}
    scan_transform(A, B);
    run_carry_in(A, B, lines + (n - 1) * CH, carry);
    for (idx_t p = start; p < end; ++p) {{
        i = n - 2 - p;
        #pragma unroll
        for (int ch = 0; ch < CH; ++ch) {{
            carry[ch] = z * carry[ch] - z * lines[i * CH + ch];
            lines[i * CH + ch] = carry[ch];
        }}
    }}
    __syncwarp();
    gain = 1;
"""

_WARP_SPLINE1D_FOOTER = """
    for (idx_t j = lane; j < n_elements; j += 32) {
        g[j] = lines[j];
    }
}
"""


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


# Dynamic shared memory available to a block without opting in.
_DEFAULT_MAX_DYNAMIC_SHARED_MEMORY = 48 * 1024


@cupy.memoize(for_each_device=True)
def _max_dynamic_shared_memory():
    attributes = cupy.cuda.Device().attributes
    return attributes.get(
        "MaxSharedMemoryPerBlockOptin", _DEFAULT_MAX_DYNAMIC_SHARED_MEMORY
    )


def get_warp_spline1d_channels(axis, shape):
    """Number of interleaved lines a warp filters together, or 0.

    A contiguous last axis is one line. Filtering the axis before a small
    trailing axis (e.g. RGB/RGBA channels) filters the trailing elements as
    interleaved lines of the same staged block.
    """
    ndim = len(shape)
    if axis == ndim - 1:
        return 1
    if axis == ndim - 2 and 2 <= shape[-1] <= 4:
        return shape[-1]
    return 0


def get_warp_spline1d_launch(n_samples, data_type, channels=1):
    """Return (warps per block, dynamic shared memory bytes) or None.

    Each warp stages one (n_samples, channels) block in shared memory. Use up
    to 4 warps per block within the default 48 KiB limit; longer lines use
    one warp per block and opt in to the device's larger limit when it is
    sufficient.
    """
    line_bytes = n_samples * channels * (4 if data_type == "float" else 8)
    for warps_per_block in (4, 2, 1):
        smem = warps_per_block * line_bytes
        if smem <= _DEFAULT_MAX_DYNAMIC_SHARED_MEMORY:
            return warps_per_block, smem
    if line_bytes <= _max_dynamic_shared_memory():
        return 1, line_bytes
    return None


def supports_warp_spline1d(data_type, n_samples, n_signals, channels):
    """Return True if the staged warp prefilter can be used.

    ``n_signals`` counts the 1D lines (including interleaved channels) and
    ``channels`` comes from :func:`get_warp_spline1d_channels`.
    """
    if not (
        channels
        and data_type in ("float", "double")
        and n_samples >= 32
        and get_warp_spline1d_launch(n_samples, data_type, channels) is not None
    ):
        return False
    # Lane 0 runs the boundary initialization serially for each line (and
    # channel). With slow double precision this outweighs the parallel
    # recurrence for many short lines (measured crossover on an RTX A6000).
    if data_type == "float" or n_samples >= 512:
        return True
    return channels == 1 and n_signals <= 256


@cupy.memoize(for_each_device=True)
def get_warp_spline1d_kernel(
    mode,
    order,
    index_type="int",
    data_type="float",
    pole_type="float",
    warps_per_block=4,
    channels=1,
):
    """Generate a warp-parallel spline prefilter for contiguous 1D lines.

    Each warp loads one line into dynamic shared memory (``warps_per_block``
    lines of ``n`` elements per block), splits it into one run per lane, and
    composes the runs' carries with warp scans. Boundary initialization uses
    the same generated code as the general kernel. All spline orders are
    supported; orders 4 and 5 filter both poles in the same launch.

    With ``channels > 1``, each warp instead stages an ``(n, channels)``
    block and filters its ``channels`` interleaved lines together.
    """
    if data_type not in ("float", "double"):
        raise ValueError(f"unsupported warp spline dtype: {data_type}")
    mode = _get_spline_mode(mode)
    poles = get_poles(order)
    n_boundary = _get_n_boundary(poles, pole_type)
    mode_str = mode.replace("-", "_")
    kernel_name = (
        f"cupyx_scipy_ndimage_warp_spline_ord{order}_{mode_str}_"
        f"ch{channels}_w{warps_per_block}"
    )
    code = _WARP_SPLINE1D_HEADER.format(
        channels=channels,
        index_type=index_type,
        data_type=data_type,
        pole_type=pole_type,
        block_size=32 * warps_per_block,
        kernel_name=kernel_name,
        gain=get_gain(poles),
        warps_per_block=warps_per_block,
    )
    for pole_index, pole in enumerate(poles):
        causal = _causal_init_code(mode).format(n_boundary=n_boundary)
        anticausal = _anticausal_init_code(mode).format(n_boundary=n_boundary)
        code += _WARP_SPLINE1D_POLE.format(
            pole_index=pole_index,
            pole=pole,
            causal_init=textwrap.indent(textwrap.dedent(causal), " " * 12),
            anticausal_init=textwrap.indent(
                textwrap.dedent(anticausal), " " * 12
            ),
        )
    code += _WARP_SPLINE1D_FOOTER
    return cupy.RawKernel(code, kernel_name)
