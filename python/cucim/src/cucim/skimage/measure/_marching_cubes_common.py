# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2012-2015, P. M. Neila
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import base64

import cupy as cp
import numpy as np

# Code shared by the Lorensen and Lewiner marching cubes implementations:
# input validation, step_size handling, the kernel code for indexing the vertex
# ids of crossed edges, the edge classification and vertex kernels (which call
# the method's own mc_crosses), the driver of the methods whose count kernel
# resolves each cell once (_run_cell_table_method), and the removal of
# degenerate faces.
#
# Implementation notes
# --------------------
# Both methods are CuPy RawKernel implementations. The public API validation
# and post-processing behavior follows scikit-image's
# _marching_cubes_lewiner.py. The general design (a per-node, three-axis
# vertex-id grid built by count, vertex and face passes, as in CuMCubes, and
# output sized from prefix scans rather than atomic appends, as in the legacy
# native CUDA marching cubes of NVIDIA Warp, versions 1.6.2 to 1.7.2, which are
# Apache-2.0 licensed) is a common approach for regular grids. The kernels here
# were written independently and do not copy code from either project.


def _decode_lut(shape, text):
    byts = base64.decodebytes(text.encode("utf-8"))
    return np.frombuffer(byts, dtype=np.int8).reshape(shape)


def _validate_marching_cubes_inputs(
    volume,
    level,
    spacing,
    gradient_direction,
    step_size,
    allow_degenerate,
    mask,
):
    if not isinstance(volume, cp.ndarray) or volume.ndim != 3:
        raise ValueError("Input volume should be a 3D CuPy array.")
    if volume.shape[0] < 2 or volume.shape[1] < 2 or volume.shape[2] < 2:
        raise ValueError("Input array must be at least 2x2x2.")

    volume = cp.ascontiguousarray(volume, dtype=cp.float32)
    vol_min = float(volume.min())
    vol_max = float(volume.max())
    if level is None:
        level = 0.5 * (vol_min + vol_max)
    else:
        level = float(level)
        if level < vol_min or level > vol_max:
            raise ValueError("Surface level must be within volume data range.")

    if len(spacing) != 3:
        raise ValueError("`spacing` must consist of three floats.")
    spacing = tuple(float(s) for s in spacing)

    step_size = int(step_size)
    if step_size < 1:
        raise ValueError("step_size must be at least one.")

    if gradient_direction not in ("descent", "ascent"):
        raise ValueError(
            f"Incorrect input {gradient_direction} in `gradient_direction`, "
            "see docstring."
        )

    if mask is not None:
        if not isinstance(mask, cp.ndarray) or mask.shape != volume.shape:
            raise ValueError("volume and mask must have the same shape.")
        mask = cp.ascontiguousarray(mask, dtype=cp.bool_)

    return volume, level, spacing, step_size, mask


_INT32_LIMIT = 2**31


def _check_int32_range(count, what):
    """Raise if ``count`` entries of ``what`` cannot be indexed by the kernels.

    The kernels index with 32-bit integers (up to three values per vertex,
    face or edge), so ``count`` must stay below ``2**31``.
    """
    if count >= _INT32_LIMIT:
        raise ValueError(
            f"The volume is too large for marching_cubes: {what} would "
            f"require more than 2**31 elements ({count}). Use a larger "
            "`step_size` or process the volume in blocks."
        )


def _apply_step_size(volume, spacing, step_size, mask):
    if step_size == 1:
        return volume, spacing, mask

    volume = cp.ascontiguousarray(volume[::step_size, ::step_size, ::step_size])
    if mask is not None:
        mask = cp.ascontiguousarray(mask[::step_size, ::step_size, ::step_size])
    if volume.shape[0] < 2 or volume.shape[1] < 2 or volume.shape[2] < 2:
        raise RuntimeError("No surface found at the given iso value.")
    spacing = tuple(s * step_size for s in spacing)
    return volume, spacing, mask


# common kernels shared across Lewiner and Lorensen implementations
_COMMON_KERNEL_CODE = r"""
extern "C" __device__ inline int node_index(int i, int j, int k, int ny, int nz) {
    return (i * ny + j) * nz + k;
}

extern "C" __device__ inline float vol_at(
    const float* volume, int i, int j, int k, int ny, int nz) {
    return volume[node_index(i, j, k, ny, nz)];
}

extern "C" __device__ inline bool mc_mask_cell_active(
    const bool* mask, int i, int j, int k, int nx, int ny, int nz) {
    return i >= 0 && i < nx - 1 &&
           j >= 0 && j < ny - 1 &&
           k >= 0 && k < nz - 1 &&
           mask[node_index(i + 1, j + 1, k + 1, ny, nz)];
}

extern "C" __device__ inline bool mc_edge_has_active_masked_cell(
    const bool* mask, int i, int j, int k, int side, int nx, int ny, int nz) {
    if (side == 0) {
        if (i >= nx - 1) {
            return false;
        }
        return mc_mask_cell_active(mask, i, j - 1, k - 1, nx, ny, nz) ||
               mc_mask_cell_active(mask, i, j, k - 1, nx, ny, nz) ||
               mc_mask_cell_active(mask, i, j - 1, k, nx, ny, nz) ||
               mc_mask_cell_active(mask, i, j, k, nx, ny, nz);
    }
    if (side == 1) {
        if (j >= ny - 1) {
            return false;
        }
        return mc_mask_cell_active(mask, i - 1, j, k - 1, nx, ny, nz) ||
               mc_mask_cell_active(mask, i, j, k - 1, nx, ny, nz) ||
               mc_mask_cell_active(mask, i - 1, j, k, nx, ny, nz) ||
               mc_mask_cell_active(mask, i, j, k, nx, ny, nz);
    }
    if (k >= nz - 1) {
        return false;
    }
    return mc_mask_cell_active(mask, i - 1, j - 1, k, nx, ny, nz) ||
           mc_mask_cell_active(mask, i, j - 1, k, nx, ny, nz) ||
           mc_mask_cell_active(mask, i - 1, j, k, nx, ny, nz) ||
           mc_mask_cell_active(mask, i, j, k, nx, ny, nz);
}

extern "C" __device__ inline float grad_at(
    const float* volume, int i, int j, int k, int axis, int nx, int ny, int nz) {
    if (axis == 0) {
        if (i == 0) {
            return vol_at(volume, 1, j, k, ny, nz) - vol_at(volume, 0, j, k, ny, nz);
        }
        if (i == nx - 1) {
            return vol_at(volume, i, j, k, ny, nz) - vol_at(volume, i - 1, j, k, ny, nz);
        }
        return 0.5f * (vol_at(volume, i + 1, j, k, ny, nz) - vol_at(volume, i - 1, j, k, ny, nz));
    }
    if (axis == 1) {
        if (j == 0) {
            return vol_at(volume, i, 1, k, ny, nz) - vol_at(volume, i, 0, k, ny, nz);
        }
        if (j == ny - 1) {
            return vol_at(volume, i, j, k, ny, nz) - vol_at(volume, i, j - 1, k, ny, nz);
        }
        return 0.5f * (vol_at(volume, i, j + 1, k, ny, nz) - vol_at(volume, i, j - 1, k, ny, nz));
    }
    if (k == 0) {
        return vol_at(volume, i, j, 1, ny, nz) - vol_at(volume, i, j, 0, ny, nz);
    }
    if (k == nz - 1) {
        return vol_at(volume, i, j, k, ny, nz) - vol_at(volume, i, j, k - 1, ny, nz);
    }
    return 0.5f * (vol_at(volume, i, j, k + 1, ny, nz) - vol_at(volume, i, j, k - 1, ny, nz));
}

// Crossed edges are numbered in the order of (node, side). edge_mask holds one
// bit per side for each node and node_scan the inclusive scan of the number of
// set bits, so the id of a crossed edge is the scan value minus the number of
// crossed edges of that node at this side or after it. Faces only reference
// crossed edges.
extern "C" __device__ inline int edge_vid(
    const int* node_scan, const unsigned char* edge_mask, int i, int j, int k,
    int side, int ny, int nz) {
    int node = node_index(i, j, k, ny, nz);
    return node_scan[node] - __popc((unsigned int)edge_mask[node] >> side);
}

// One thread per node: write the (flat) index node * 3 + side of each crossed
// edge at its vertex id, so the vertex kernel can run with one thread per
// vertex (surfaces cross only a small fraction of the edges).
extern "C" __global__ void mc_compact_edges(
    const unsigned char* edge_mask, const int* node_scan, int* edge_idx,
    int n_nodes) {
    int node = blockDim.x * blockIdx.x + threadIdx.x;
    if (node >= n_nodes) {
        return;
    }
    unsigned int bits = edge_mask[node];
    for (int side = 0; side < 3; side++) {
        if ((bits >> side) & 1u) {
            edge_idx[node_scan[node] - __popc(bits >> side)] = node * 3 + side;
        }
    }
}

extern "C" __device__ inline int local_edge_vid(
    const int* node_scan, const unsigned char* edge_mask, int edge, int i,
    int j, int k, int ny, int nz) {
    switch (edge) {
        case 0: return edge_vid(node_scan, edge_mask, i, j, k, 0, ny, nz);
        case 1: return edge_vid(node_scan, edge_mask, i + 1, j, k, 1, ny, nz);
        case 2: return edge_vid(node_scan, edge_mask, i, j + 1, k, 0, ny, nz);
        case 3: return edge_vid(node_scan, edge_mask, i, j, k, 1, ny, nz);
        case 4: return edge_vid(node_scan, edge_mask, i, j, k + 1, 0, ny, nz);
        case 5: return edge_vid(node_scan, edge_mask, i + 1, j, k + 1, 1, ny, nz);
        case 6: return edge_vid(node_scan, edge_mask, i, j + 1, k + 1, 0, ny, nz);
        case 7: return edge_vid(node_scan, edge_mask, i, j, k + 1, 1, ny, nz);
        case 8: return edge_vid(node_scan, edge_mask, i, j, k, 2, ny, nz);
        case 9: return edge_vid(node_scan, edge_mask, i + 1, j, k, 2, ny, nz);
        case 10: return edge_vid(node_scan, edge_mask, i + 1, j + 1, k, 2, ny, nz);
        default: return edge_vid(node_scan, edge_mask, i, j + 1, k, 2, ny, nz);
    }
}
extern "C" __global__ void mc_classify_edges(
    const float* volume, unsigned char* edge_mask, unsigned char* edge_count,
    int nx, int ny, int nz, float level) {
    int node = blockDim.x * blockIdx.x + threadIdx.x;
    if (node >= nx * ny * nz) {
        return;
    }

    int k = node % nz;
    int j = (node / nz) % ny;
    int i = node / (ny * nz);

    // bit `side` is set when the edge leaving this node along that axis crosses
    float a = vol_at(volume, i, j, k, ny, nz);
    int bits = 0;
    if (i + 1 < nx && mc_crosses(a, vol_at(volume, i + 1, j, k, ny, nz), level)) {
        bits |= 1;
    }
    if (j + 1 < ny && mc_crosses(a, vol_at(volume, i, j + 1, k, ny, nz), level)) {
        bits |= 2;
    }
    if (k + 1 < nz && mc_crosses(a, vol_at(volume, i, j, k + 1, ny, nz), level)) {
        bits |= 4;
    }
    edge_mask[node] = bits;
    edge_count[node] = __popc(bits);
}

extern "C" __global__ void mc_classify_edges_masked(
    const float* volume, const bool* mask, unsigned char* edge_mask,
    unsigned char* edge_count, int nx, int ny, int nz, float level) {
    int node = blockDim.x * blockIdx.x + threadIdx.x;
    if (node >= nx * ny * nz) {
        return;
    }

    int k = node % nz;
    int j = (node / nz) % ny;
    int i = node / (ny * nz);

    float a = vol_at(volume, i, j, k, ny, nz);
    int bits = 0;
    if (i + 1 < nx && mc_crosses(a, vol_at(volume, i + 1, j, k, ny, nz), level) &&
        mc_edge_has_active_masked_cell(mask, i, j, k, 0, nx, ny, nz)) {
        bits |= 1;
    }
    if (j + 1 < ny && mc_crosses(a, vol_at(volume, i, j + 1, k, ny, nz), level) &&
        mc_edge_has_active_masked_cell(mask, i, j, k, 1, nx, ny, nz)) {
        bits |= 2;
    }
    if (k + 1 < nz && mc_crosses(a, vol_at(volume, i, j, k + 1, ny, nz), level) &&
        mc_edge_has_active_masked_cell(mask, i, j, k, 2, nx, ny, nz)) {
        bits |= 4;
    }
    edge_mask[node] = bits;
    edge_count[node] = __popc(bits);
}

extern "C" __global__ void mc_generate_vertices(
    const float* volume, const int* edge_idx, int n_vertices,
    float* vertices, float* normals, float* values,
    int nx, int ny, int nz, float level,
    float sx, float sy, float sz) {
    int vid = blockDim.x * blockIdx.x + threadIdx.x;
    if (vid >= n_vertices) {
        return;
    }

    int idx = edge_idx[vid];
    int side = idx % 3;
    int node = idx / 3;
    int k = node % nz;
    int j = (node / nz) % ny;
    int i = node / (ny * nz);
    int i2 = i + (side == 0);
    int j2 = j + (side == 1);
    int k2 = k + (side == 2);

    float a = vol_at(volume, i, j, k, ny, nz);
    float b = vol_at(volume, i2, j2, k2, ny, nz);

    float t = (level - a) / (b - a);
    t = fminf(1.0f, fmaxf(0.0f, t));
    float x = (float)i;
    float y = (float)j;
    float z = (float)k;
    if (side == 0) {
        x += t;
    } else if (side == 1) {
        y += t;
    } else {
        z += t;
    }

    int out = 3 * vid;
    vertices[out] = x * sx;
    vertices[out + 1] = y * sy;
    vertices[out + 2] = z * sz;
    values[vid] = fmaxf(a, b);

    float gx0 = grad_at(volume, i, j, k, 0, nx, ny, nz);
    float gy0 = grad_at(volume, i, j, k, 1, nx, ny, nz);
    float gz0 = grad_at(volume, i, j, k, 2, nx, ny, nz);
    float gx1 = grad_at(volume, i2, j2, k2, 0, nx, ny, nz);
    float gy1 = grad_at(volume, i2, j2, k2, 1, nx, ny, nz);
    float gz1 = grad_at(volume, i2, j2, k2, 2, nx, ny, nz);
    float gx = (1.0f - t) * gx0 + t * gx1;
    float gy = (1.0f - t) * gy0 + t * gy1;
    float gz = (1.0f - t) * gz0 + t * gz1;
    float norm = sqrtf(gx * gx + gy * gy + gz * gz);
    if (norm > 0.0f) {
        gx /= norm;
        gy /= norm;
        gz /= norm;
    }
    normals[out] = gx;
    normals[out + 1] = gy;
    normals[out + 2] = gz;
}

"""


def _number_crossed_edges(get_kernel, volume, level, mask, threads=256):
    """Classify the edges, scan the counts and list the crossed edges.

    ``get_kernel`` returns a kernel of the calling method's module (which
    defines ``mc_crosses``). Returns ``(edge_mask, node_scan, edge_idx)``.
    """
    nx, ny, nz = volume.shape
    n_nodes = nx * ny * nz
    node_blocks = ((n_nodes + threads - 1) // threads,)
    edge_mask = cp.empty(n_nodes, dtype=cp.uint8)
    edge_count = cp.empty(n_nodes, dtype=cp.uint8)
    if mask is None:
        get_kernel("mc_classify_edges")(
            node_blocks,
            (threads,),
            (volume, edge_mask, edge_count, nx, ny, nz, np.float32(level)),
        )
    else:
        get_kernel("mc_classify_edges_masked")(
            node_blocks,
            (threads,),
            (
                volume,
                mask,
                edge_mask,
                edge_count,
                nx,
                ny,
                nz,
                np.float32(level),
            ),
        )
    # one scan per node (not per edge); see edge_vid for how ids are recovered
    node_scan = _inclusive_scan_uint8(edge_count)
    n_edge_vertices = int(node_scan[-1])
    if n_edge_vertices == 0:
        raise RuntimeError("No surface found at the given iso value.")
    _check_int32_range(3 * n_edge_vertices, "the vertex array")

    edge_idx = cp.empty(n_edge_vertices, dtype=cp.int32)
    get_kernel("mc_compact_edges")(
        node_blocks,
        (threads,),
        (edge_mask, node_scan, edge_idx, n_nodes),
    )
    return edge_mask, node_scan, edge_idx


def _generate_edge_vertices(
    get_kernel,
    volume,
    edge_idx,
    vertices,
    normals,
    values,
    level,
    spacing,
    threads=256,
):
    """Write the edge vertices (one thread each) to the start of the arrays."""
    nx, ny, nz = volume.shape
    n_edge_vertices = edge_idx.size
    get_kernel("mc_generate_vertices")(
        ((n_edge_vertices + threads - 1) // threads,),
        (threads,),
        (
            volume,
            edge_idx,
            n_edge_vertices,
            vertices,
            normals,
            values,
            nx,
            ny,
            nz,
            np.float32(level),
            np.float32(spacing[0]),
            np.float32(spacing[1]),
            np.float32(spacing[2]),
        ),
    )


# Inclusive scan of uint8 counts into int32, in three steps: the sums of blocks
# of 1024 counts, a scan of those sums (1/1024 of the size), and a scan inside
# each block that adds the offset of the block. Compared with cp.cumsum this
# reads the counts twice (one byte each) and writes the result once, and
# gives the same values: the sums are integers, so the order of the additions
# does not matter.
_SCAN_KERNEL_CODE = r"""
#define SCAN_TILE 1024

extern "C" __device__ inline void scan_load4(
    const unsigned char* in, int base, int n, int* v) {
    if (base + 3 < n) {
        uchar4 u = *reinterpret_cast<const uchar4*>(in + base);
        v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
    } else {
        for (int k = 0; k < 4; k++) {
            v[k] = (base + k < n) ? (int)in[base + k] : 0;
        }
    }
}

extern "C" __global__ void mc_scan_block_sums(
    const unsigned char* in, int* block_sums, int n) {
    __shared__ int warp_sums[SCAN_TILE / 4 / 32];
    int base = blockIdx.x * SCAN_TILE + threadIdx.x * 4;
    int v[4];
    scan_load4(in, base, n, v);
    int x = v[0] + v[1] + v[2] + v[3];
    for (int d = 16; d > 0; d >>= 1) {
        x += __shfl_down_sync(0xffffffffu, x, d);
    }
    if ((threadIdx.x & 31) == 0) {
        warp_sums[threadIdx.x >> 5] = x;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int total = 0;
        for (int w = 0; w < SCAN_TILE / 4 / 32; w++) {
            total += warp_sums[w];
        }
        block_sums[blockIdx.x] = total;
    }
}

// block_scan is the inclusive scan of block_sums
extern "C" __global__ void mc_scan_apply(
    const unsigned char* in, const int* block_scan, const int* block_sums,
    int* out, int n) {
    __shared__ int warp_sums[SCAN_TILE / 4 / 32];
    int base = blockIdx.x * SCAN_TILE + threadIdx.x * 4;
    int v[4];
    scan_load4(in, base, n, v);
    int local = v[0] + v[1] + v[2] + v[3];
    int lane = threadIdx.x & 31;
    int x = local;
    for (int d = 1; d < 32; d <<= 1) {
        int y = __shfl_up_sync(0xffffffffu, x, d);
        if (lane >= d) {
            x += y;
        }
    }
    if (lane == 31) {
        warp_sums[threadIdx.x >> 5] = x;
    }
    __syncthreads();
    int offset = block_scan[blockIdx.x] - block_sums[blockIdx.x] + x - local;
    for (int w = 0; w < (int)(threadIdx.x >> 5); w++) {
        offset += warp_sums[w];
    }
    if (base + 3 < n) {
        int4 r;
        r.x = offset + v[0];
        r.y = r.x + v[1];
        r.z = r.y + v[2];
        r.w = r.z + v[3];
        *reinterpret_cast<int4*>(out + base) = r;
    } else {
        for (int k = 0; k < 4; k++) {
            offset += v[k];
            if (base + k < n) {
                out[base + k] = offset;
            }
        }
    }
}
"""

_SCAN_TILE = 1024


@cp.memoize(for_each_device=True)
def _get_scan_kernel(name):
    return cp.RawKernel(_SCAN_KERNEL_CODE, name)


# below this size the three launches of the block scan cost more than they save
# (measured on an RTX A6000: the crossover is between 2 and 3 million entries)
_SCAN_MIN_SIZE = 5 * 2**19


def _inclusive_scan_uint8(counts, min_size=_SCAN_MIN_SIZE):
    """Inclusive scan of a contiguous 1D uint8 array, as int32.

    Returns the same array as ``cp.cumsum(counts, dtype=cp.int32)``, which is
    what is used for arrays with fewer than ``min_size`` entries.
    """
    if counts.size < min_size:
        return cp.cumsum(counts, dtype=cp.int32)
    n = counts.size
    out = cp.empty(n, dtype=cp.int32)
    n_blocks = (n + _SCAN_TILE - 1) // _SCAN_TILE
    block_sums = cp.empty(n_blocks, dtype=cp.int32)
    threads = _SCAN_TILE // 4
    _get_scan_kernel("mc_scan_block_sums")(
        (n_blocks,), (threads,), (counts, block_sums, np.int32(n))
    )
    block_scan = cp.cumsum(block_sums, dtype=cp.int32)
    _get_scan_kernel("mc_scan_apply")(
        (n_blocks,),
        (threads,),
        (counts, block_scan, block_sums, out, np.int32(n)),
    )
    return out


# code specific to allow_degenerate=False
_REMOVE_DEGENERATE_KERNEL_CODE = r"""

extern "C" __device__ inline int mc_find_root(int* parent, int x) {
    int p = parent[x];
    while (p != parent[p]) {
        p = parent[p];
    }
    return p;
}

extern "C" __device__ inline void mc_union_vertices(int* parent, int a, int b) {
    while (true) {
        int ra = mc_find_root(parent, a);
        int rb = mc_find_root(parent, b);
        if (ra == rb) {
            return;
        }
        int hi = ra > rb ? ra : rb;
        int lo = ra > rb ? rb : ra;
        int old = atomicMin(parent + hi, lo);
        if (old == hi || old <= lo) {
            return;
        }
    }
}

extern "C" __global__ void mc_mark_degenerate_faces_parallel(
    const float* vertices, const int* faces, int n_faces,
    unsigned char* faces_ok, int* degenerate_count) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_faces) {
        return;
    }

    int i0 = faces[3 * idx];
    int i1 = faces[3 * idx + 1];
    int i2 = faces[3 * idx + 2];
    const float* v0 = vertices + 3 * i0;
    const float* v1 = vertices + 3 * i1;
    const float* v2 = vertices + 3 * i2;
    bool eq01 = v0[0] == v1[0] && v0[1] == v1[1] && v0[2] == v1[2];
    bool eq02 = v0[0] == v2[0] && v0[1] == v2[1] && v0[2] == v2[2];
    bool eq12 = v1[0] == v2[0] && v1[1] == v2[1] && v1[2] == v2[2];

    bool ok = !(eq01 || eq02 || eq12);
    faces_ok[idx] = ok;
    if (!ok) {
        atomicAdd(degenerate_count, 1);
    }
}

extern "C" __global__ void mc_union_degenerate_faces(
    const float* vertices, const int* faces, const unsigned char* faces_ok,
    int n_faces, int* parent) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_faces || faces_ok[idx]) {
        return;
    }

    int i0 = faces[3 * idx];
    int i1 = faces[3 * idx + 1];
    int i2 = faces[3 * idx + 2];
    const float* v0 = vertices + 3 * i0;
    const float* v1 = vertices + 3 * i1;
    const float* v2 = vertices + 3 * i2;
    if (v0[0] == v1[0] && v0[1] == v1[1] && v0[2] == v1[2]) {
        mc_union_vertices(parent, i0, i1);
    }
    if (v0[0] == v2[0] && v0[1] == v2[1] && v0[2] == v2[2]) {
        mc_union_vertices(parent, i0, i2);
    }
    if (v1[0] == v2[0] && v1[1] == v2[1] && v1[2] == v2[2]) {
        mc_union_vertices(parent, i1, i2);
    }
}

extern "C" __global__ void mc_init_vertex_map(int* parent, int n_vertices) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_vertices) {
        return;
    }
    parent[idx] = idx;
}

extern "C" __global__ void mc_compress_vertex_roots(int* parent, int n_vertices) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_vertices) {
        return;
    }
    parent[idx] = mc_find_root(parent, idx);
}

extern "C" __global__ void mc_mark_vertex_roots(
    const int* parent, int* root_flags, int n_vertices) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_vertices) {
        return;
    }
    root_flags[idx] = parent[idx] == idx;
}

extern "C" __global__ void mc_emit_non_degenerate_faces(
    const int* faces, const unsigned char* faces_ok, const int* face_scan,
    const int* vertex_map, const int* vertex_scan, int n_faces,
    int* faces_out) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_faces || !faces_ok[idx]) {
        return;
    }
    int out_idx = face_scan[idx] - 1;
    int i0 = vertex_map[faces[3 * idx]];
    int i1 = vertex_map[faces[3 * idx + 1]];
    int i2 = vertex_map[faces[3 * idx + 2]];
    faces_out[3 * out_idx] = vertex_scan[i0] - 1;
    faces_out[3 * out_idx + 1] = vertex_scan[i1] - 1;
    faces_out[3 * out_idx + 2] = vertex_scan[i2] - 1;
}

extern "C" __global__ void mc_compact_root_vertices(
    const float* vertices, const float* normals, const float* values,
    const int* root_flags, const int* vertex_scan, int n_vertices,
    float* vertices_out, float* normals_out, float* values_out) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_vertices || !root_flags[idx]) {
        return;
    }
    int out_idx = vertex_scan[idx] - 1;
    vertices_out[3 * out_idx] = vertices[3 * idx];
    vertices_out[3 * out_idx + 1] = vertices[3 * idx + 1];
    vertices_out[3 * out_idx + 2] = vertices[3 * idx + 2];
    normals_out[3 * out_idx] = normals[3 * idx];
    normals_out[3 * out_idx + 1] = normals[3 * idx + 1];
    normals_out[3 * out_idx + 2] = normals[3 * idx + 2];
    values_out[out_idx] = values[idx];
}

"""


@cp.memoize(for_each_device=True)
def _get_remove_degen_kernel(name):
    return cp.RawKernel(_REMOVE_DEGENERATE_KERNEL_CODE, name)


def _remove_degenerate_faces_gpu(vertices, faces, normals, values):
    n_vertices = vertices.shape[0]
    n_faces = faces.shape[0]
    if n_faces == 0:
        return vertices, faces, normals, values

    faces_ok = cp.empty(n_faces, dtype=cp.uint8)
    degenerate_count = cp.zeros(1, dtype=cp.int32)
    threads = 256
    face_blocks = ((n_faces + threads - 1) // threads,)
    vertex_blocks = ((n_vertices + threads - 1) // threads,)
    _get_remove_degen_kernel("mc_mark_degenerate_faces_parallel")(
        face_blocks,
        (threads,),
        (vertices, faces, n_faces, faces_ok, degenerate_count),
    )
    if int(degenerate_count[0]) == 0:
        return vertices, faces, normals, values

    vertex_map = cp.empty(n_vertices, dtype=cp.int32)
    _get_remove_degen_kernel("mc_init_vertex_map")(
        vertex_blocks,
        (threads,),
        (vertex_map, n_vertices),
    )
    _get_remove_degen_kernel("mc_union_degenerate_faces")(
        face_blocks,
        (threads,),
        (vertices, faces, faces_ok, n_faces, vertex_map),
    )
    _get_remove_degen_kernel("mc_compress_vertex_roots")(
        vertex_blocks,
        (threads,),
        (vertex_map, n_vertices),
    )

    root_flags = cp.empty(n_vertices, dtype=cp.int32)
    _get_remove_degen_kernel("mc_mark_vertex_roots")(
        vertex_blocks,
        (threads,),
        (vertex_map, root_flags, n_vertices),
    )
    vertex_scan = cp.cumsum(root_flags, dtype=cp.int32)
    n_vertices2 = int(vertex_scan[-1])
    face_scan = _inclusive_scan_uint8(faces_ok)
    n_faces2 = int(face_scan[-1])

    faces2 = cp.empty((n_faces2, 3), dtype=faces.dtype)
    vertices2 = cp.empty((n_vertices2, 3), dtype=vertices.dtype)
    normals2 = cp.empty((n_vertices2, 3), dtype=normals.dtype)
    values2 = cp.empty((n_vertices2,), dtype=values.dtype)
    _get_remove_degen_kernel("mc_emit_non_degenerate_faces")(
        face_blocks,
        (threads,),
        (faces, faces_ok, face_scan, vertex_map, vertex_scan, n_faces, faces2),
    )
    _get_remove_degen_kernel("mc_compact_root_vertices")(
        vertex_blocks,
        (threads,),
        (
            vertices,
            normals,
            values,
            root_flags,
            vertex_scan,
            n_vertices,
            vertices2,
            normals2,
            values2,
        ),
    )
    return vertices2, faces2, normals2, values2


def _count_cells_gpu(get_kernel, kernels, volume, level, mask=None):
    nx, ny, nz = volume.shape
    n_cells = (nx - 1) * (ny - 1) * (nz - 1)
    threads = 256
    cell_blocks = ((n_cells + threads - 1) // threads,)
    tri_counts = cp.empty(n_cells, dtype=cp.uint8)
    center_flags = cp.empty(n_cells, dtype=cp.uint8)
    row_offsets = cp.empty(n_cells, dtype=cp.uint16)
    center_count = cp.zeros(1, dtype=cp.int32)
    if mask is None:
        get_kernel(kernels["count"])(
            cell_blocks,
            (threads,),
            (
                volume,
                tri_counts,
                center_flags,
                row_offsets,
                center_count,
                nx,
                ny,
                nz,
                np.float32(level),
            ),
        )
    else:
        get_kernel(kernels["count_masked"])(
            cell_blocks,
            (threads,),
            (
                volume,
                mask,
                tri_counts,
                center_flags,
                row_offsets,
                center_count,
                nx,
                ny,
                nz,
                np.float32(level),
            ),
        )
    return tri_counts, center_flags, row_offsets, int(center_count[0])


def _generate_center_vertices_gpu(
    get_kernel,
    kernels,
    volume,
    level,
    spacing,
    center_flags,
    vertex_offset=0,
    n_centers=None,
    out=None,
):
    """Write the center vertices after the first ``vertex_offset`` entries.

    ``out`` is an optional ``(vertices, normals, values)`` tuple of arrays that
    already have ``vertex_offset + n_centers`` entries.
    """
    nx, ny, nz = volume.shape
    n_cells = (nx - 1) * (ny - 1) * (nz - 1)
    center_scan = _inclusive_scan_uint8(center_flags)
    if n_centers is None:
        n_centers = int(center_scan[-1]) if n_cells else 0
    n_vertices = vertex_offset + n_centers
    center_vertex_ids = cp.full(n_cells, -1, dtype=cp.int32)
    if out is None:
        vertices = cp.empty((n_vertices, 3), dtype=cp.float32)
        normals = cp.empty((n_vertices, 3), dtype=cp.float32)
        values = cp.empty(n_vertices, dtype=cp.float32)
    else:
        vertices, normals, values = out
    if n_centers == 0:
        return vertices, normals, values, center_vertex_ids

    threads = 256
    cell_blocks = ((n_cells + threads - 1) // threads,)
    get_kernel(kernels["center"])(
        cell_blocks,
        (threads,),
        (
            volume,
            center_flags,
            center_scan,
            center_vertex_ids,
            vertices,
            normals,
            values,
            nx,
            ny,
            nz,
            np.float32(level),
            np.float32(spacing[0]),
            np.float32(spacing[1]),
            np.float32(spacing[2]),
            np.int32(vertex_offset),
        ),
    )
    return vertices, normals, values, center_vertex_ids


def _generate_cell_table_faces_gpu(
    get_kernel,
    kernels,
    volume,
    row_offsets,
    tri_counts,
    node_scan,
    edge_mask,
    center_vertex_ids,
    gradient_direction,
):
    nx, ny, nz = volume.shape
    n_cells = (nx - 1) * (ny - 1) * (nz - 1)
    # a 64-bit total, so that a too large mesh is detected before the 32-bit
    # scan can wrap around
    n_faces = int(tri_counts.sum(dtype=cp.int64)) if n_cells else 0
    _check_int32_range(3 * n_faces, "the face array")
    tri_scan = _inclusive_scan_uint8(tri_counts)
    faces = cp.empty(n_faces * 3, dtype=cp.int32)
    if n_faces == 0:
        return faces.reshape(-1, 3)

    threads = 128
    cell_blocks = ((n_cells + threads - 1) // threads,)
    get_kernel(kernels["faces"])(
        cell_blocks,
        (threads,),
        (
            row_offsets,
            tri_counts,
            tri_scan,
            node_scan,
            edge_mask,
            center_vertex_ids,
            faces,
            nx,
            ny,
            nz,
            np.int32(gradient_direction == "descent"),
        ),
    )
    return faces.reshape(-1, 3)


def _run_cell_table_method(
    get_kernel,
    kernels,
    volume,
    level,
    spacing,
    gradient_direction,
    allow_degenerate,
    mask,
):
    """Run a method whose count kernel resolves each cell once.

    ``kernels`` names the kernels of the method (``count``, ``count_masked``,
    ``center`` and ``faces``) that ``get_kernel`` returns. The count kernel
    writes the number of triangles, the center flag and the offset of the
    triangle row of each cell.
    """
    nx, ny, nz = volume.shape
    n_edges = nx * ny * nz * 3
    _check_int32_range(n_edges, "the edge array")
    threads = 256

    edge_mask, node_scan, edge_idx = _number_crossed_edges(
        get_kernel, volume, level, mask, threads
    )
    n_edge_vertices = edge_idx.size
    tri_counts, center_flags, row_offsets, n_center_vertices = _count_cells_gpu(
        get_kernel, kernels, volume, level, mask
    )

    # edge vertices come first, followed by the center vertices
    n_vertices = n_edge_vertices + n_center_vertices
    _check_int32_range(3 * n_vertices, "the vertex array")
    vertices = cp.empty((n_vertices, 3), dtype=cp.float32)
    normals = cp.empty((n_vertices, 3), dtype=cp.float32)
    values = cp.empty(n_vertices, dtype=cp.float32)
    _generate_edge_vertices(
        get_kernel,
        volume,
        edge_idx,
        vertices,
        normals,
        values,
        level,
        spacing,
        threads,
    )
    if n_center_vertices:
        center_vertex_ids = _generate_center_vertices_gpu(
            get_kernel,
            kernels,
            volume,
            level,
            spacing,
            center_flags,
            vertex_offset=n_edge_vertices,
            n_centers=n_center_vertices,
            out=(vertices, normals, values),
        )[3]
    else:
        center_vertex_ids = cp.empty(0, dtype=cp.int32)

    faces = _generate_cell_table_faces_gpu(
        get_kernel,
        kernels,
        volume,
        row_offsets,
        tri_counts,
        node_scan,
        edge_mask,
        center_vertex_ids,
        gradient_direction,
    )
    if faces.size == 0:
        raise RuntimeError("No surface found at the given iso value.")
    if not allow_degenerate:
        vertices, faces, normals, values = _remove_degenerate_faces_gpu(
            vertices, faces, normals, values
        )
    return vertices, faces, normals, values
