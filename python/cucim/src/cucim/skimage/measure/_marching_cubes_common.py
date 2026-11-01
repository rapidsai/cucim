# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2012-2015, P. M. Neila
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import base64

import cupy as cp
import numpy as np

# Code shared by the Lorensen and Lewiner marching cubes implementations:
# input validation, step_size handling, the kernel code for indexing the vertex
# ids of crossed edges, and the removal of degenerate faces.
#
# Implementation notes
# --------------------
# Both methods are CuPy RawKernel implementations. The public API validation
# and post-processing behavior follows scikit-image's
# _marching_cubes_lewiner.py. The general design (a per-node, three-axis
# vertex-id grid built by count, vertex and face passes, as in CuMCubes, and
# output sized from prefix scans rather than atomic appends, as in NVIDIA Warp)
# is a common approach for regular grids. The kernels here were written
# independently and do not copy code from either project.


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

"""


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
    face_scan = cp.cumsum(faces_ok, dtype=cp.int32)
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
