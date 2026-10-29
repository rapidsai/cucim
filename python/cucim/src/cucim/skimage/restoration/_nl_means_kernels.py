# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA source generation for non-local means denoising.

Two kernel variants are generated from Python string fragments (in the style
of ``cucim.skimage.filters._separable_filtering``):

``direct``
    One thread per output pixel. Each thread loops over all shifts of the
    search window and computes every patch distance from global memory. It
    supports every configuration and is used as the fallback.

``tiled``
    One thread block per output tile, with the loop over shifts inside the
    kernel. For each shift the block stores the squared differences over the
    tile plus the patch halo in shared memory, reduces them with separable sums
    along axes ``ndim - 1, ..., 1``, and each thread then finishes the sum
    along axis 0 for its own output pixel. The sequential sum along axis 0 is
    where the early exit of scikit-image's slow mode is applied.

Both variants use the gather form of the algorithm: every output pixel sums
over the full search window, so no atomics are needed. This is equivalent to
scikit-image's symmetric half-window scatter. The weights reproduce
scikit-image's, including its ``_fast_exp`` approximation of the exponential.
"""

import math

import cupy as cp
import numpy as np

# patch distances above this value get zero weight (as in scikit-image)
DISTANCE_CUTOFF = 5.0

# Largest slow-mode distance for which ``_fast_exp`` is meaningful. Beyond it
# the approximation underflows (float) or returns NaN/garbage (double), so the
# weight is set to zero instead.
_MAX_DISTANCE = {"float": 87.0, "double": 708.0}

_preamble_template = """
typedef {c_type} T;
typedef {index_type} idx_t;

// Schraudolph's approximation of exp(y), as used by scikit-image
// (skimage/_shared/fast_exp.h). This version is bit-identical to
// scikit-image's _fast_exp for -708 <= y <= 0.
__device__ __forceinline__ double nlm_fast_exp(double y) {{
    const int hi = static_cast<int>(1512775.3951951856938 * y) + 1072632447;
    return __hiloint2double(hi, 0);
}}

// The same approximation built directly from float bits, avoiding FP64 math.
// It agrees with (float)_fast_exp((double)y) to within one 2**-20 step for
// -5 <= y <= 0 (all weights that pass the distance cutoff). For
// -87 <= y < -5 (slow mode only) the float32 product loses a few steps
// (relative error < 1e-5 on weights below exp(-5)).
__device__ __forceinline__ float nlm_fast_exp(float y) {{
    const int hi = __float2int_rz(1512775.3951951856938f * y) + 1072632447;
    return __int_as_float((hi - (896 << 20)) << 3);
}}
"""


def _patch_taps(patch_size, fast_mode):
    """Return the patch footprint along each axis.

    Returns ``(lo, taps)`` where ``lo`` is the offset of the first patch
    element relative to the patch center and ``taps`` holds the separable
    per-axis weights. ``patch_size`` must be odd.
    """
    r = patch_size // 2
    if fast_mode:
        # scikit-image's integral-image distance sums over [x - r + 1, x + r]
        return -r + 1, (1.0,) * (2 * r)
    if patch_size == 1:
        # Gaussian width is zero; the caller uses scale = 0 (constant weights)
        return 0, (1.0,)
    a = (patch_size - 1) / 4.0
    k = np.arange(-r, r + 1, dtype=np.float64)
    return -r, tuple(np.exp(-k * k / (2 * a * a)).tolist())


def _literal(value, c_type):
    if c_type == "float":
        return f"{float(np.float32(value))!r}f"
    return repr(float(value))


def _c_order_strides(extents):
    strides = [1] * len(extents)
    for a in range(len(extents) - 2, -1, -1):
        strides[a] = strides[a + 1] * extents[a + 1]
    return strides


def _kernel_name(
    variant, ndim, n_channels, patch_size, fast_mode, c_type, index_type, tile
):
    mode = "fast" if fast_mode else "slow"
    name = f"nlm_{variant}_{mode}_{ndim}d_c{n_channels}_s{patch_size}"
    name += "_f32" if c_type == "float" else "_f64"
    name += "_i64" if index_type == "long long" else "_i32"
    if tile is not None:
        name += "_t" + "x".join(str(t) for t in tile)
    return name


def _weight_code(fast_mode, c_type):
    """Code computing weight ``w`` from ``dist`` (and ``dead`` if slow)."""
    cutoff = _literal(DISTANCE_CUTOFF, c_type)
    if fast_mode:
        return f"""
            const T dist_n = (dist > (T)0 ? dist : (T)0) * scale;
            const T w = dist_n > {cutoff} ? (T)0 : nlm_fast_exp(-dist_n);"""
    max_dist = _literal(_MAX_DISTANCE[c_type], c_type)
    return f"""
            const T w = (dead || dist > {max_dist})
                        ? (T)0 : nlm_fast_exp(dist > (T)0 ? -dist : (T)0);"""


def _unravel_code(var, linear, extents, prefix, indent):
    """Unravel C-order index ``linear`` into ``{prefix}0 .. {prefix}{n-1}``."""
    ndim = len(extents)
    lines = [f"int {var} = {linear};"]
    for a in range(ndim - 1, 0, -1):
        lines.append(f"const int {prefix}{a} = {var} % {extents[a]};")
        lines.append(f"{var} /= {extents[a]};")
    lines.append(f"const int {prefix}0 = {var};")
    return "".join(f"\n{indent}{line}" for line in lines)


def _signature(name, ndim, extra=""):
    shape_args = "".join(f"const int n{a}, " for a in range(ndim))
    stride_args = "".join(f"const idx_t ps{a}, " for a in range(ndim))
    return f"""
extern "C" __global__ void {extra}{name}(
    const T* __restrict__ padded, T* __restrict__ out,
    {shape_args}
    {stride_args}
    const idx_t plane, const int pad, const int d, const int chunk_len,
    const T scale, const T var_term)"""


# The shifts along axis 0 can be split into gridDim.y chunks of chunk_len
# shifts (to fill the GPU for small images). Each chunk then writes partial
# sums that the host adds up.
_chunk_code = """
    const int t0_first = -d + (int)blockIdx.y * chunk_len;
    const int t0_last = min(d, t0_first + chunk_len - 1);"""


def _taps_code(taps, fast_mode, c_type):
    if fast_mode:
        return ""
    values = ", ".join(_literal(t, c_type) for t in taps)
    return f"""
    const T TAPS[F] = {{{values}}};
    T SCALED_TAPS[F];
    #pragma unroll
    for (int k = 0; k < F; ++k) {{
        SCALED_TAPS[k] = scale * TAPS[k];
    }}"""


def _store_code(ndim, indent):
    pix = "(idx_t)x0"
    for a in range(1, ndim):
        pix = f"({pix} * n{a} + x{a})"
    return f"""
{indent}const idx_t pix = {pix};
{indent}if (gridDim.y == 1) {{
{indent}    const T inv_wsum = (T)1 / wsum;
{indent}    #pragma unroll
{indent}    for (int ch = 0; ch < C; ++ch) {{
{indent}        out[pix * C + ch] = acc[ch] * inv_wsum;
{indent}    }}
{indent}}} else {{
{indent}    // partial sums over this block's chunk of shifts
{indent}    T* const part = out + ((idx_t)blockIdx.y * n_pixels + pix) * (C + 1);
{indent}    #pragma unroll
{indent}    for (int ch = 0; ch < C; ++ch) {{
{indent}        part[ch] = acc[ch];
{indent}    }}
{indent}    part[C] = wsum;
{indent}}}"""


def _direct_distance_code(ndim, fast_mode):
    """Patch distance between patches at padded offsets center and cand."""
    ind = "            "

    def level(a, depth):
        pad = ind + "    " * depth
        q_prev = f"q{a - 1} + " if a > 0 else ""
        code = f"""
{pad}#pragma unroll
{pad}for (int k{a} = 0; k{a} < F; ++k{a}) {{"""
        if a == 0 and not fast_mode:
            # scikit-image checks the partial sum before each patch row
            code += f"""
{pad}    if (k0 > 0 && dist > (T){DISTANCE_CUTOFF}) {{
{pad}        dead = true;
{pad}        break;
{pad}    }}"""
        code += f"""
{pad}    const idx_t q{a} = {q_prev}(idx_t)(LO + k{a}) * ps{a};"""
        if a < ndim - 1:
            if not fast_mode:
                code += f"""
{pad}    T s{a + 1} = 0;"""
            code += level(a + 1, depth + 1)
            if not fast_mode:
                if a == 0:
                    code += f"""
{pad}    dist += SCALED_TAPS[k0] * s1;"""
                else:
                    code += f"""
{pad}    s{a} += TAPS[k{a}] * s{a + 1};"""
        else:
            q = f"q{a}"
            code += f"""
{pad}    T sq = -var_term;
{pad}    #pragma unroll
{pad}    for (int ch = 0; ch < C; ++ch) {{
{pad}        const T diff = __ldg(padded + center + {q} + ch * plane)
{pad}                       - __ldg(padded + cand + {q} + ch * plane);
{pad}        sq += diff * diff;
{pad}    }}"""
            if fast_mode:
                code += f"""
{pad}    dist += sq;"""
            else:
                code += f"""
{pad}    s{a} += TAPS[k{a}] * sq;"""
        code += f"""
{pad}}}"""
        return code

    code = f"""
{ind}T dist = 0;"""
    if not fast_mode:
        code += f"""
{ind}bool dead = false;"""
    return code + level(0, 0)


def _get_direct_kernel_src(
    ndim, n_channels, patch_size, fast_mode, c_type, index_type
):
    lo, taps = _patch_taps(patch_size, fast_mode)
    name = _kernel_name(
        "direct",
        ndim,
        n_channels,
        patch_size,
        fast_mode,
        c_type,
        index_type,
        None,
    )
    code = _preamble_template.format(c_type=c_type, index_type=index_type)
    code += _signature(name, ndim)
    code += f"""
{{
    constexpr int C = {n_channels};
    constexpr int F = {len(taps)};
    constexpr int LO = {lo};
    const idx_t n_pixels = {" * ".join(f"(idx_t)n{a}" for a in range(ndim))};
"""
    code += _taps_code(taps, fast_mode, c_type)
    code += _chunk_code
    code += """
    for (idx_t i = (idx_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n_pixels; i += (idx_t)blockDim.x * gridDim.x) {
        idx_t rem = i;"""
    for a in range(ndim - 1, 0, -1):
        code += f"""
        const int x{a} = rem % n{a};
        rem /= n{a};"""
    code += """
        const int x0 = rem;"""
    center = " + ".join(f"(idx_t)(x{a} + pad) * ps{a}" for a in range(ndim))
    code += f"""
        const idx_t center = {center};
        T wsum = 0;
        T acc[C];
        #pragma unroll
        for (int ch = 0; ch < C; ++ch) {{
            acc[ch] = 0;
        }}"""
    indent = "        "
    prev = "center"
    for a in range(ndim):
        t_lo, t_hi = ("t0_first", "t0_last") if a == 0 else ("-d", "d")
        if not fast_mode:
            # slow mode: search window clipped to the image domain
            t_lo, t_hi = f"max({t_lo}, -x{a})", f"min({t_hi}, n{a} - 1 - x{a})"
        if a == ndim - 1:
            # two-level summation: partial sums over the innermost shift axis
            code += f"""
{indent}T wrow = 0;
{indent}T arow[C];
{indent}#pragma unroll
{indent}for (int ch = 0; ch < C; ++ch) {{
{indent}    arow[ch] = 0;
{indent}}}"""
        code += f"""
{indent}for (int t{a} = {t_lo}; t{a} <= {t_hi}; ++t{a}) {{
{indent}    const idx_t c{a} = {prev} + (idx_t)t{a} * ps{a};"""
        prev = f"c{a}"
        indent += "    "
    code += f"""
{indent}const idx_t cand = c{ndim - 1};"""
    code += _direct_distance_code(ndim, fast_mode).replace(
        "\n            ", "\n" + indent
    )
    code += _weight_code(fast_mode, c_type).replace(
        "\n            ", "\n" + indent
    )
    code += f"""
{indent}wrow += w;
{indent}#pragma unroll
{indent}for (int ch = 0; ch < C; ++ch) {{
{indent}    arow[ch] += w * __ldg(padded + cand + ch * plane);
{indent}}}"""
    for a in range(ndim - 1, -1, -1):
        indent = indent[:-4]
        code += f"""
{indent}}}"""
        if a == ndim - 1:
            code += f"""
{indent}wsum += wrow;
{indent}#pragma unroll
{indent}for (int ch = 0; ch < C; ++ch) {{
{indent}    acc[ch] += arow[ch];
{indent}}}"""
    code += _store_code(ndim, "        ")
    code += """
    }
}
"""
    return name, code


def _tiled_buffers(ndim, n_taps, tile):
    """Extents of the shared memory buffers of the tiled kernel.

    Buffer 0 holds the squared differences over the tile plus patch halo.
    Buffer ``i`` (``i >= 1``) holds the result of the separable sum along
    axis ``ndim - i``.
    """
    ext = [t + n_taps - 1 for t in tile]
    buffers = [ext]
    for a in range(ndim - 1, 0, -1):
        buffers.append(ext[:a] + list(tile[a:]))
    return buffers


def _tiled_smem_elements(ndim, n_taps, tile):
    return sum(math.prod(b) for b in _tiled_buffers(ndim, n_taps, tile))


def _get_tiled_kernel_src(
    ndim, n_channels, patch_size, fast_mode, c_type, index_type, tile
):
    lo, taps = _patch_taps(patch_size, fast_mode)
    n_taps = len(taps)
    if n_taps < 1:
        raise ValueError("tiled kernel requires a non-empty patch")
    name = _kernel_name(
        "tiled",
        ndim,
        n_channels,
        patch_size,
        fast_mode,
        c_type,
        index_type,
        tile,
    )
    n_threads = math.prod(tile)
    buffers = _tiled_buffers(ndim, n_taps, tile)
    sizes = [math.prod(b) for b in buffers]
    offsets = [sum(sizes[:i]) for i in range(len(sizes))]
    ext = buffers[0]
    sq_size = sizes[0]
    items_a = -(-sq_size // n_threads)

    code = _preamble_template.format(c_type=c_type, index_type=index_type)
    code += _signature(name, ndim, extra=f"__launch_bounds__({n_threads}) ")
    code += f"""
{{
    constexpr int C = {n_channels};
    constexpr int F = {n_taps};
    constexpr int LO = {lo};
    constexpr int NT = {n_threads};
    constexpr int SQ_SIZE = {sq_size};
    constexpr int ITEMS_A = {items_a};
"""
    code += _taps_code(taps, fast_mode, c_type)
    code += """
    extern __shared__ __align__(16) unsigned char nlm_smem[];"""
    for i, offset in enumerate(offsets):
        code += f"""
    T* const buf{i} = reinterpret_cast<T*>(nlm_smem) + {offset};"""
    code += _chunk_code
    code += f"""
    const idx_t n_pixels = {" * ".join(f"(idx_t)n{a}" for a in range(ndim))};
    const int tid = threadIdx.x;

    // origin of this block's output tile
    int bid = blockIdx.x;"""
    for a in range(ndim - 1, 0, -1):
        code += f"""
    const int ntiles{a} = (n{a} + {tile[a] - 1}) / {tile[a]};
    const int o{a} = (bid % ntiles{a}) * {tile[a]};
    bid /= ntiles{a};"""
    code += f"""
    const int o0 = bid * {tile[0]};

    // this thread's output pixel"""
    code += _unravel_code("rem_y", "tid", list(tile), "y", "    ")
    for a in range(ndim):
        code += f"""
    const int x{a} = o{a} + y{a};"""
    valid = " && ".join(f"(x{a} < n{a})" for a in range(ndim))
    center = " + ".join(f"(idx_t)(x{a} + pad) * ps{a}" for a in range(ndim))
    final_strides = _c_order_strides(buffers[-1])
    p_idx = " + ".join(f"y{a} * {final_strides[a]}" for a in range(ndim))
    code += f"""
    const bool valid = {valid};
    const idx_t center = {center};
    const T* const pfinal = buf{len(buffers) - 1} + {p_idx};

    // reference values for this thread's squared-difference elements
    idx_t ref_off[ITEMS_A];
    T ref_val[ITEMS_A][C];
    #pragma unroll
    for (int it = 0; it < ITEMS_A; ++it) {{
        const int e = tid + it * NT;
        ref_off[it] = 0;
        #pragma unroll
        for (int ch = 0; ch < C; ++ch) {{
            ref_val[it][ch] = 0;
        }}
        if (it < ITEMS_A - 1 || e < SQ_SIZE) {{"""
    code += _unravel_code("rem_z", "e", ext, "z", "            ")
    ref = " + ".join(
        f"(idx_t)(o{a} + z{a} + LO + pad) * ps{a}" for a in range(ndim)
    )
    code += f"""
            ref_off[it] = {ref};
            #pragma unroll
            for (int ch = 0; ch < C; ++ch) {{
                ref_val[it][ch] = __ldg(padded + ref_off[it] + ch * plane);
            }}
        }}
    }}"""
    # base offsets into the input buffer of each separable pass
    for i in range(1, len(buffers)):
        axis = ndim - i
        out_ext = buffers[i]
        in_strides = _c_order_strides(buffers[i - 1])
        n_items = -(-sizes[i] // n_threads)
        base = " + ".join(f"u{a} * {in_strides[a]}" for a in range(ndim))
        code += f"""
    // pass {i}: separable sum along axis {axis}
    int pbase{i}[{n_items}];
    #pragma unroll
    for (int it = 0; it < {n_items}; ++it) {{
        const int e = tid + it * NT;
        pbase{i}[it] = 0;
        if (it < {n_items - 1} || e < {sizes[i]}) {{"""
        code += _unravel_code("rem_u", "e", out_ext, "u", "            ")
        code += f"""
            pbase{i}[it] = {base};
        }}
    }}"""

    code += """

    T wsum = 0;
    T acc[C];
    #pragma unroll
    for (int ch = 0; ch < C; ++ch) {
        acc[ch] = 0;
    }"""
    indent = "    "
    prev = None
    for a in range(ndim):
        if a == ndim - 1:
            code += f"""
{indent}T wrow = 0;
{indent}T arow[C];
{indent}#pragma unroll
{indent}for (int ch = 0; ch < C; ++ch) {{
{indent}    arow[ch] = 0;
{indent}}}"""
        t_lo, t_hi = ("t0_first", "t0_last") if a == 0 else ("-d", "d")
        code += f"""
{indent}for (int t{a} = {t_lo}; t{a} <= {t_hi}; ++t{a}) {{"""
        indent += "    "
        if not fast_mode:
            # skip shifts no pixel of the tile can use (block-uniform)
            code += f"""
{indent}if (o{a} + t{a} >= n{a} || min(o{a} + {tile[a]}, n{a}) - 1 + t{a} < 0) {{
{indent}    continue;
{indent}}}
{indent}const bool in{a} = {f"in{a - 1} && " if a > 0 else ""}((unsigned)(x{a} + t{a}) < (unsigned)n{a});"""
        off = f"{prev} + " if prev is not None else ""
        code += f"""
{indent}const idx_t off{a} = {off}(idx_t)t{a} * ps{a};"""
        prev = f"off{a}"

    off_t = f"off{ndim - 1}"
    code += f"""

{indent}// phase A: squared differences over the tile plus patch halo
{indent}#pragma unroll
{indent}for (int it = 0; it < ITEMS_A; ++it) {{
{indent}    const int e = tid + it * NT;
{indent}    if (it < ITEMS_A - 1 || e < SQ_SIZE) {{
{indent}        const T* const src = padded + ref_off[it] + {off_t};
{indent}        T v = -var_term;
{indent}        #pragma unroll
{indent}        for (int ch = 0; ch < C; ++ch) {{
{indent}            const T diff = ref_val[it][ch] - __ldg(src + ch * plane);
{indent}            v += diff * diff;
{indent}        }}
{indent}        buf0[e] = v;
{indent}    }}
{indent}}}
{indent}__syncthreads();"""
    for i in range(1, len(buffers)):
        axis = ndim - i
        tap_stride = _c_order_strides(buffers[i - 1])[axis]
        n_items = -(-sizes[i] // n_threads)
        tap = "" if fast_mode else "TAPS[k] * "
        code += f"""

{indent}// phase B, pass {i}: separable sum along axis {axis}
{indent}#pragma unroll
{indent}for (int it = 0; it < {n_items}; ++it) {{
{indent}    const int e = tid + it * NT;
{indent}    if (it < {n_items - 1} || e < {sizes[i]}) {{
{indent}        const T* const src = buf{i - 1} + pbase{i}[it];
{indent}        T s = 0;
{indent}        #pragma unroll
{indent}        for (int k = 0; k < F; ++k) {{
{indent}            s += {tap}src[k * {tap_stride}];
{indent}        }}
{indent}        buf{i}[e] = s;
{indent}    }}
{indent}}}
{indent}__syncthreads();"""

    final_stride0 = final_strides[0]
    code += f"""

{indent}// phase C: sum along axis 0 for this thread's own pixel
{indent}T dist = 0;"""
    if fast_mode:
        code += f"""
{indent}#pragma unroll
{indent}for (int k = 0; k < F; ++k) {{
{indent}    dist += pfinal[k * {final_stride0}];
{indent}}}"""
        inside = "valid"
    else:
        code += f"""
{indent}bool dead = false;
{indent}#pragma unroll
{indent}for (int k = 0; k < F; ++k) {{
{indent}    // scikit-image checks the partial sum before each patch row
{indent}    if (k > 0 && dist > (T){DISTANCE_CUTOFF}) {{
{indent}        dead = true;
{indent}        break;
{indent}    }}
{indent}    dist += SCALED_TAPS[k] * pfinal[k * {final_stride0}];
{indent}}}"""
        inside = f"valid && in{ndim - 1}"
    code += _weight_code(fast_mode, c_type).replace(
        "\n            ", "\n" + indent
    )
    code += f"""
{indent}if ({inside}) {{
{indent}    wrow += w;
{indent}    #pragma unroll
{indent}    for (int ch = 0; ch < C; ++ch) {{
{indent}        arow[ch] += w * __ldg(padded + center + {off_t} + ch * plane);
{indent}    }}
{indent}}}"""
    # No trailing __syncthreads() is needed: every buffer is written only after
    # the barrier that follows the phase reading it in the previous shift.
    for a in range(ndim - 1, -1, -1):
        indent = indent[:-4]
        code += f"""
{indent}}}"""
        if a == ndim - 1:
            code += f"""
{indent}wsum += wrow;
{indent}#pragma unroll
{indent}for (int ch = 0; ch < C; ++ch) {{
{indent}    acc[ch] += arow[ch];
{indent}}}"""
    code += """
    if (valid) {"""
    code += _store_code(ndim, "        ")
    code += """
    }
}
"""
    return name, code


@cp.memoize(for_each_device=True)
def _get_direct_kernel(
    ndim, n_channels, patch_size, fast_mode, c_type, index_type
):
    name, code = _get_direct_kernel_src(
        ndim, n_channels, patch_size, fast_mode, c_type, index_type
    )
    return cp.RawModule(code=code).get_function(name)


@cp.memoize(for_each_device=True)
def _get_tiled_kernel(
    ndim, n_channels, patch_size, fast_mode, c_type, index_type, tile
):
    name, code = _get_tiled_kernel_src(
        ndim, n_channels, patch_size, fast_mode, c_type, index_type, tile
    )
    return cp.RawModule(code=code).get_function(name)


_reflect_code = """
// index into an axis of length n for NumPy's 'reflect' padding mode (also
// valid for pad widths larger than the axis)
__device__ __forceinline__ int nlm_reflect(int k, const int n) {
    if (n == 1) {
        return 0;
    }
    const int period = 2 * (n - 1);
    k = abs(k) % period;
    return k < n ? k : period - k;
}
"""


@cp.memoize(for_each_device=True)
def _get_pad_kernel(ndim):
    """Kernel reflect-padding a contiguous ``(*shape, C)`` array.

    The output is the planar ``(C, *padded_shape)`` array used by the
    denoising kernels. Padding and transposing in a single kernel is much
    cheaper for small images than ``cupy.pad``, which launches many kernels.
    The padding starts ``pad`` elements before the image on every axis.
    """
    params = ", ".join(f"int32 n{a}, int32 p{a}" for a in range(ndim))
    body = """
        ptrdiff_t rem = i;"""
    for a in range(ndim - 1, -1, -1):
        body += f"""
        const int j{a} = rem % p{a};
        rem /= p{a};"""
    body += """
        const int ch = rem;
        ptrdiff_t src_idx = 0;"""
    for a in range(ndim):
        body += f"""
        src_idx = src_idx * n{a} + nlm_reflect(j{a} - pad, n{a});"""
    body += """
        dst = src[src_idx * n_channels + ch];"""
    return cp.ElementwiseKernel(
        f"raw T src, {params}, int32 pad, int32 n_channels",
        "T dst",
        body,
        f"nlm_reflect_pad_{ndim}d",
        preamble=_reflect_code,
    )


@cp.memoize(for_each_device=True)
def _get_finalize_kernel():
    """Kernel adding up the partial sums of all shift chunks.

    ``part`` has shape ``(n_chunks, n_pixels, n_channels + 1)`` holding the
    weighted sums of each channel followed by the sum of weights. Chunks are
    always added in the same order, so the result is deterministic.
    """
    return cp.ElementwiseKernel(
        "raw T part, int32 n_chunks, int64 n_pixels, int32 n_channels",
        "T out",
        """
        const ptrdiff_t pix = i / n_channels;
        const int ch = i % n_channels;
        T acc = 0;
        T wsum = 0;
        for (int k = 0; k < n_chunks; ++k) {
            const ptrdiff_t base = (k * n_pixels + pix) * (n_channels + 1);
            acc += part[base + ch];
            wsum += part[base + n_channels];
        }
        out = acc / wsum;
        """,
        "nlm_finalize",
    )
