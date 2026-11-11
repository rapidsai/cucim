# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2019 David E. Vega M. and Javier E. Abache R.
# SPDX-FileCopyrightText: Copyright (c) 2023 Guangyan Cai
# SPDX-License-Identifier: Apache-2.0 AND MIT

import base64

import cupy as cp
import numpy as np

from ._marching_cubes_common import (
    _COMMON_KERNEL_CODE,
    _apply_step_size,
    _count_cells_gpu,
    _generate_cell_table_faces_gpu,
    _generate_center_vertices_gpu,
    _idx_code,
    _run_cell_table_method,
    _validate_marching_cubes_inputs,
)
from ._marching_cubes_vega_luts import VEGA_TABLE

# Implementation notes
# --------------------
# Marching cubes 33 with the corrected interior test of Vega, Abache and Coll,
# "A Fast and Memory-Saving Marching Cubes 33 Implementation with the Correct
# Interior Test" (Journal of Computer Graphics Techniques 8(3), 2019). The
# case dispatch (vega_find_case and the face and interior tests it calls) is a
# port of MC33_findCase of the MC33 C library of Vega and Abache (MIT License,
# https://github.com/dvega68/MC33_c_library), made with the help of the port to
# CUDA in the isoext project (MIT License, Copyright (c) 2023 Guangyan Cai,
# https://github.com/GuangyanCai/isoext), and uses the lookup table of that
# library (see _marching_cubes_vega_luts.py). The rest of the pipeline (edge
# numbering, vertices, scans, faces) is the one shared with the other methods
# (see _marching_cubes_common.py): as for the Lewiner method, the counting
# kernel resolves each cell once and stores the offset of its triangles in the
# table, and the face kernel only reads the table.
#
# Differences from the Lewiner method: the corrected interior test (a diagonal
# of the cell is tested on the plane through the saddle point of the
# trilinear interpolant, which makes the result independent of the
# orientation of the cell), three results of the interior test for the case
# 13.5, one more triangulation for the case 13.5.2, and a center vertex at the
# center of the cell. All tests are made in float32, as in the library.


def _decode_vega_table():
    shape, text = VEGA_TABLE
    raw = base64.decodebytes(text.encode("utf-8"))
    table = np.frombuffer(raw, dtype="<u2").reshape(shape)
    return table


# The corners of the Vega corner order as offsets (i, j, k) of the cell, and
# the corners of the edges of the cell (in the Vega corner order)
_VEGA_CORNERS = (
    (0, 0, 0),
    (0, 1, 0),
    (0, 1, 1),
    (0, 0, 1),
    (1, 0, 0),
    (1, 1, 0),
    (1, 1, 1),
    (1, 0, 1),
)
_VEGA_EDGES = (
    (0, 1),
    (1, 2),
    (3, 2),
    (0, 3),
    (4, 5),
    (5, 6),
    (7, 6),
    (4, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)
# the corners (offsets (i, j, k)) and edges of the classic numbering that
# local_edge_vid uses
_CLASSIC_CORNERS = (
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
)
_CLASSIC_EDGES = _VEGA_EDGES  # the same corner pairs, in the classic corners


def _vega_edge_to_local():
    """The edge of local_edge_vid for each edge of the Vega table.

    The center (12) stays 12.
    """
    classic = {
        frozenset((_CLASSIC_CORNERS[a], _CLASSIC_CORNERS[b])): n
        for n, (a, b) in enumerate(_CLASSIC_EDGES)
    }
    mapping = []
    for a, b in _VEGA_EDGES:
        mapping.append(classic[frozenset((_VEGA_CORNERS[a], _VEGA_CORNERS[b]))])
    return tuple(mapping) + (12,)


def _generate_vega_lut_constants_code():
    table = _decode_vega_table()
    values = ", ".join(str(int(v)) for v in table)
    edges = ", ".join(str(v) for v in _vega_edge_to_local())
    return (
        f"__constant__ unsigned short lut_vega[{table.size}] = {{{values}}};\n"
        f"__constant__ signed char vega_edge_to_local[13] = {{{edges}}};"
    )


# crossing test of this method (the shared edge classification kernels in
# _COMMON_KERNEL_CODE call it, so it must precede that code): a sample is
# above the level when it is greater than the level, as for the Lewiner method
_CROSSING_TEST_CODE = r"""
extern "C" __device__ inline bool mc_crosses(float a, float b, float level) {
    return (a <= level && b > level) || (a > level && b <= level);
}
"""

_VEGA_KERNEL_CODE = (
    _generate_vega_lut_constants_code()
    + _CROSSING_TEST_CODE
    + _COMMON_KERNEL_CODE
    + r"""

// level minus the value of each corner, in the Vega corner order
extern "C" __device__ inline void vega_load_values(
    const float* volume, int i, int j, int k, int ny, int nz, float level,
    float* v) {
    v[0] = level - vol_at(volume, i, j, k, ny, nz);
    v[1] = level - vol_at(volume, i, j + 1, k, ny, nz);
    v[2] = level - vol_at(volume, i, j + 1, k + 1, ny, nz);
    v[3] = level - vol_at(volume, i, j, k + 1, ny, nz);
    v[4] = level - vol_at(volume, i + 1, j, k, ny, nz);
    v[5] = level - vol_at(volume, i + 1, j + 1, k, ny, nz);
    v[6] = level - vol_at(volume, i + 1, j + 1, k + 1, ny, nz);
    v[7] = level - vol_at(volume, i + 1, j, k + 1, ny, nz);
    // a value equal to the level is +0 (not -0) whatever the sign of the level
    for (int n = 0; n < 8; n++) {
        v[n] += 0.0f;
    }
}

// Sign of the bilinear interpolant at the saddle point of each face whose
// four corner signs alternate: 1 joins the positive vertices, -1 the negative
// ones, 0 means the face is not ambiguous. Returns the sum over the six faces.
extern "C" __device__ inline int vega_face_tests(
    const float* v, int* face, int ind) {
    if (ind & 0x80) {
        face[0] = ((ind & 0xCC) == 0x84 ? (v[0] * v[5] < v[1] * v[4] ? -1 : 1) : 0);
        face[3] = ((ind & 0x99) == 0x81 ? (v[0] * v[7] < v[3] * v[4] ? -1 : 1) : 0);
        face[4] = ((ind & 0xF0) == 0xA0 ? (v[0] * v[2] < v[1] * v[3] ? -1 : 1) : 0);
    } else {
        face[0] = ((ind & 0xCC) == 0x48 ? (v[0] * v[5] < v[1] * v[4] ? 1 : -1) : 0);
        face[3] = ((ind & 0x99) == 0x18 ? (v[0] * v[7] < v[3] * v[4] ? 1 : -1) : 0);
        face[4] = ((ind & 0xF0) == 0x50 ? (v[0] * v[2] < v[1] * v[3] ? 1 : -1) : 0);
    }
    if (ind & 0x02) {
        face[1] = ((ind & 0x66) == 0x42 ? (v[1] * v[6] < v[2] * v[5] ? -1 : 1) : 0);
        face[2] = ((ind & 0x33) == 0x12 ? (v[3] * v[6] < v[2] * v[7] ? -1 : 1) : 0);
        face[5] = ((ind & 0x0F) == 0x0A ? (v[4] * v[6] < v[5] * v[7] ? -1 : 1) : 0);
    } else {
        face[1] = ((ind & 0x66) == 0x24 ? (v[1] * v[6] < v[2] * v[5] ? 1 : -1) : 0);
        face[2] = ((ind & 0x33) == 0x21 ? (v[3] * v[6] < v[2] * v[7] ? 1 : -1) : 0);
        face[5] = ((ind & 0x0F) == 0x05 ? (v[4] * v[6] < v[5] * v[7] ? 1 : -1) : 0);
    }
    return face[0] + face[1] + face[2] + face[3] + face[4] + face[5];
}

// The same test for one face; returns the mask of the joined vertices (cases
// 3 and 6).
extern "C" __device__ inline int vega_face_test1(const float* v, int face) {
    switch (face) {
        case 0: return v[0] * v[5] < v[1] * v[4] ? 0x48 : 0x84;
        case 1: return v[1] * v[6] < v[2] * v[5] ? 0x24 : 0x42;
        case 2: return v[3] * v[6] < v[2] * v[7] ? 0x21 : 0x12;
        case 3: return v[0] * v[7] < v[3] * v[4] ? 0x18 : 0x81;
        case 4: return v[0] * v[2] < v[1] * v[3] ? 0x50 : 0xA0;
        default: return v[4] * v[6] < v[5] * v[7] ? 0x05 : 0x0A;
    }
}

// The corrected interior test: nonzero if the diagonally opposite vertices i
// and (i + 6) % 8 are joined through the interior of the cell. The trilinear
// interpolant is analyzed on the plane through its saddle point. With flag13
// set (case 13 only) it returns 2 if one of the vertices 0 to 3 joins the
// center of the cell and 1 if one of the vertices 4 to 7 does.
extern "C" __device__ inline int vega_interior_test(
    const float* v, int i, int flag13) {
    float at = v[4] - v[0], bt = v[5] - v[1];
    float ct = v[6] - v[2], dt = v[7] - v[3];
    float t = at * ct - bt * dt;
    if (signbit(t)) {
        if (i & 0x01) {
            return 0;
        }
    } else {
        if (!(i & 0x01) || t == 0.0f) {
            return 0;
        }
    }
    t = 0.5f * (v[3] * bt - v[2] * at + v[1] * dt - v[0] * ct) / t;
    if (t > 0.0f && t < 1.0f) {
        at = v[0] + at * t;
        bt = v[1] + bt * t;
        ct = v[2] + ct * t;
        dt = v[3] + dt * t;
        ct *= at;
        dt *= bt;
        if (i & 0x01) {
            if (ct < dt && !signbit(dt)) {
                return (signbit(bt) == signbit(v[i])) + flag13;
            }
        } else {
            if (ct > dt && !signbit(ct)) {
                return (signbit(at) == signbit(v[i])) + flag13;
            }
        }
    }
    return 0;
}

// Offset in lut_vega of the triangle pattern of a cell with the case index
// ind (bit 7 is the corner 0, bit 0 the corner 7, a bit is set when the corner
// is above the level), and whether its triangles are reversed. The first
// triangle is the entry after the offset.
extern "C" __device__ inline int vega_find_case(
    const float* v, unsigned int ind, int* reversed) {
    int f[6];
    unsigned int c, m;
    if (ind & 0x80) {
        c = lut_vega[ind ^ 0xFF];
        m = (c & 0x800) == 0;
    } else {
        c = lut_vega[ind];
        m = (c & 0x800) != 0;
    }
    *reversed = (int)m;
    unsigned int mi = m ? ind : (ind ^ 0xFF);
    int k = c & 0x7FF;
    int offset = 0;
    switch (c >> 12) {
        case 0:  // cases 1, 2, 5, 8, 9, 11 and 14
            offset = k;
            break;
        case 1:  // case 3
            offset = (mi & vega_face_test1(v, k >> 2)) ? 183 + 2 * k : 159 + k;
            break;
        case 2:  // case 4
            offset = vega_interior_test(v, k, 0) ? 239 + 6 * k : 231 + 2 * k;
            break;
        case 3:  // case 6
            if (mi & vega_face_test1(v, k % 6)) {
                offset = 575 + 5 * k;  // 6.2
            } else {
                offset = vega_interior_test(v, k / 6, 0) ? 407 + 7 * k
                                                          : 335 + 3 * k;  // 6.1
            }
            break;
        case 4:  // case 7
            switch (vega_face_tests(v, f, mi)) {
                case -3:
                    offset = 695 + 3 * k;  // 7.1
                    break;
                case -1:  // 7.2
                    offset = (f[4] + f[5] < 0 ? (f[0] + f[2] < 0 ? 759 : 799)
                                               : 719) + 5 * k;
                    break;
                case 1:  // 7.3
                    offset = (f[4] + f[5] < 0 ? 983
                                               : (f[0] + f[2] < 0 ? 839 : 911))
                             + 9 * k;
                    break;
                default:  // 7.4
                    offset = vega_interior_test(v, k >> 1, 0) ? 1095 + 9 * k
                                                               : 1055 + 5 * k;
            }
            break;
        case 5:  // case 10
            switch (vega_face_tests(v, f, mi)) {
                case -2:
                    if (k == 2 ? vega_interior_test(v, 0, 0)
                               : (vega_interior_test(v, 0, 0) ||
                                  vega_interior_test(v, k ? 1 : 3, 0))) {
                        offset = 1213 + 8 * k;  // 10.1.2
                    } else {
                        offset = 1189 + 4 * k;  // 10.1.1
                    }
                    break;
                case 0:  // 10.2
                    offset = (f[2 + k] < 0 ? 1261 : 1285) + 8 * k;
                    break;
                default:
                    if (k == 2 ? vega_interior_test(v, 1, 0)
                               : (vega_interior_test(v, 2, 0) ||
                                  vega_interior_test(v, k ? 3 : 1, 0))) {
                        offset = 1237 + 8 * k;  // 10.1.2
                    } else {
                        offset = 1201 + 4 * k;  // 10.1.1
                    }
            }
            break;
        case 6:  // case 12
            switch (vega_face_tests(v, f, mi)) {
                case -2:  // 12.1
                    offset = vega_interior_test(v, (0xDA010C >> (k << 1)) & 3, 0)
                                 ? 1453 + 8 * k : 1357 + 4 * k;
                    break;
                case 0:  // 12.2
                    offset = (f[k >> 1] < 0 ? 1645 : 1741) + 8 * k;
                    break;
                default:  // 12.1
                    offset = vega_interior_test(v, (0xA7B7E5 >> (k << 1)) & 3, 0)
                                 ? 1549 + 8 * k : 1405 + 4 * k;
            }
            break;
        default:  // case 13
            switch (abs(vega_face_tests(v, f, 165))) {
                case 0: {
                    k = ((f[1] < 0) << 1) | (f[5] < 0);
                    if (f[0] * f[1] == f[5]) {  // 13.4
                        offset = 2157 + 12 * k;
                    } else {  // 13.5.1 if joined == 0, else 13.5.2
                        int joined = vega_interior_test(v, k, 1);
                        offset = 2285 + (joined ? 10 * k - 40 * joined : 6 * k);
                    }
                    break;
                }
                case 2:  // 13.3
                    offset = 1917 + 10 * ((f[0] < 0 ? (f[2] > 0) : 12 + (f[2] < 0)) +
                                          (f[1] < 0 ? (f[3] < 0) : 6 + (f[3] > 0)));
                    if (f[4] > 0) {
                        offset += 30;
                    }
                    break;
                case 4:  // 13.2
                    k = 21 + 11 * f[0] + 4 * f[1] + 3 * f[2] + 2 * f[3] + f[4];
                    if (k >> 4) {
                        k -= (k & 32) ? 20 : 10;
                    }
                    offset = 1845 + 3 * k;
                    break;
                default:  // 13.1
                    offset = 1839 + 2 * f[0];
            }
    }
    return offset;
}

// Number of triangles of the pattern at offset, and whether it uses the
// center vertex (12).
extern "C" __device__ inline int vega_pattern(
    int offset, int* uses_center) {
    int count = 0;
    unsigned int e;
    *uses_center = 0;
    do {
        e = lut_vega[++offset];
        count++;
        if ((e & 0xF) == 12 || ((e >> 4) & 0xF) == 12 ||
            ((e >> 8) & 0xF) == 12) {
            *uses_center = 1;
        }
    } while (e >> 12);
    return count;
}

// the row of a cell: the offset of its pattern, bit 15 set when reversed
extern "C" __device__ inline void vega_resolve_cell(
    const float* volume, int i, int j, int k, int ny, int nz, float level,
    int* tri_count, int* center_flag, int* row) {
    float v[8];
    vega_load_values(volume, i, j, k, ny, nz, level, v);
    unsigned int index = 0;
    for (int n = 0; n < 8; n++) {
        if (signbit(v[n])) {
            index |= 0x80 >> n;
        }
    }
    *tri_count = 0;
    *center_flag = 0;
    *row = 0;
    if (index != 0 && index != 0xFF) {
        int reversed;
        int offset = vega_find_case(v, index, &reversed);
        *tri_count = vega_pattern(offset, center_flag);
        *row = offset | (reversed << 15);
    }
}

extern "C" __global__ void mc_vega_count_cells(
    const float* volume, unsigned char* tri_counts, unsigned char* center_flags,
    unsigned short* row_offsets, int* center_count, int nx, int ny, int nz,
    float level) {
    mc_idx idx = (mc_idx)blockDim.x * blockIdx.x + threadIdx.x;
    int cnx = nx - 1;
    int cny = ny - 1;
    int cnz = nz - 1;
    if (idx >= (mc_idx)cnx * cny * cnz) {
        return;
    }
    int k = (int)(idx % cnz);
    int j = (int)((idx / cnz) % cny);
    int i = (int)(idx / ((mc_idx)cny * cnz));
    int tri_count, center_flag, row;
    vega_resolve_cell(volume, i, j, k, ny, nz, level, &tri_count, &center_flag, &row);
    tri_counts[idx] = (unsigned char)tri_count;
    center_flags[idx] = (unsigned char)center_flag;
    row_offsets[idx] = (unsigned short)row;
    if (center_flag) {
        atomicAdd(center_count, 1);
    }
}

extern "C" __global__ void mc_vega_count_cells_masked(
    const float* volume, const bool* mask, unsigned char* tri_counts,
    unsigned char* center_flags, unsigned short* row_offsets,
    int* center_count, int nx, int ny, int nz, float level) {
    mc_idx idx = (mc_idx)blockDim.x * blockIdx.x + threadIdx.x;
    int cnx = nx - 1;
    int cny = ny - 1;
    int cnz = nz - 1;
    if (idx >= (mc_idx)cnx * cny * cnz) {
        return;
    }
    int k = (int)(idx % cnz);
    int j = (int)((idx / cnz) % cny);
    int i = (int)(idx / ((mc_idx)cny * cnz));
    if (!mask[node_index(i + 1, j + 1, k + 1, ny, nz)]) {
        tri_counts[idx] = 0;
        center_flags[idx] = 0;
        row_offsets[idx] = 0;
        return;
    }
    int tri_count, center_flag, row;
    vega_resolve_cell(volume, i, j, k, ny, nz, level, &tri_count, &center_flag, &row);
    tri_counts[idx] = (unsigned char)tri_count;
    center_flags[idx] = (unsigned char)center_flag;
    row_offsets[idx] = (unsigned short)row;
    if (center_flag) {
        atomicAdd(center_count, 1);
    }
}

// The center vertex is at the center of the cell. Its normal is the mean of
// the gradients at the corners (the trilinear interpolation at the center)
// and its value the range of the corner values (as for the edge vertices, the
// largest range of the cells that contain the vertex).
extern "C" __global__ void mc_vega_generate_center_vertices(
    const float* volume, const unsigned char* center_flags,
    const int* center_scan, int* center_vertex_ids, float* vertices,
    float* normals, float* values_out, int nx, int ny, int nz, float level,
    float sx, float sy, float sz, int vertex_offset) {
    mc_idx idx = (mc_idx)blockDim.x * blockIdx.x + threadIdx.x;
    int cnx = nx - 1;
    int cny = ny - 1;
    int cnz = nz - 1;
    if (idx >= (mc_idx)cnx * cny * cnz) {
        return;
    }
    center_vertex_ids[idx] = -1;
    if (center_flags[idx] == 0) {
        return;
    }
    int k = (int)(idx % cnz);
    int j = (int)((idx / cnz) % cny);
    int i = (int)(idx / ((mc_idx)cny * cnz));

    float gx = 0.0f, gy = 0.0f, gz = 0.0f;
    float vmax = vol_at(volume, i, j, k, ny, nz);
    float vmin = vmax;
    for (int n = 0; n < 8; n++) {
        int ci = i + (n & 1);
        int cj = j + ((n >> 1) & 1);
        int ck = k + ((n >> 2) & 1);
        gx += grad_at(volume, ci, cj, ck, 0, nx, ny, nz);
        gy += grad_at(volume, ci, cj, ck, 1, nx, ny, nz);
        gz += grad_at(volume, ci, cj, ck, 2, nx, ny, nz);
        float s = vol_at(volume, ci, cj, ck, ny, nz);
        vmax = fmaxf(vmax, s);
        vmin = fminf(vmin, s);
    }
    // the sign of the normals of scikit-image (towards lower values)
    gx = -gx;
    gy = -gy;
    gz = -gz;
    gx /= sx;
    gy /= sy;
    gz /= sz;
    float norm = sqrtf(gx * gx + gy * gy + gz * gz);
    if (norm > 0.0f) {
        gx /= norm;
        gy /= norm;
        gz /= norm;
    }
    int vid = vertex_offset + center_scan[idx] - 1;
    center_vertex_ids[idx] = vid;
    mc_idx out = (mc_idx)3 * vid;
    vertices[out] = ((float)i + 0.5f) * sx;
    vertices[out + 1] = ((float)j + 0.5f) * sy;
    vertices[out + 2] = ((float)k + 0.5f) * sz;
    normals[out] = gx;
    normals[out + 1] = gy;
    normals[out + 2] = gz;
    values_out[vid] = vmax - vmin;
}

extern "C" __device__ inline int vega_vertex_id(
    int nibble, const int* node_scan, const unsigned char* edge_mask,
    const int* center_vertex_ids, mc_idx idx, int i, int j, int k, int ny,
    int nz) {
    if (nibble == 12) {
        return center_vertex_ids[idx];
    }
    return local_edge_vid(
        node_scan, edge_mask, (int)vega_edge_to_local[nibble], i, j, k, ny, nz);
}

extern "C" __global__ void mc_vega_generate_faces(
    const unsigned short* row_offsets, const unsigned char* tri_counts,
    const int* tri_scan, const int* node_scan, const unsigned char* edge_mask,
    const int* center_vertex_ids, int* faces, int nx, int ny, int nz,
    int flip_winding) {
    mc_idx idx = (mc_idx)blockDim.x * blockIdx.x + threadIdx.x;
    int cnx = nx - 1;
    int cny = ny - 1;
    int cnz = nz - 1;
    if (idx >= (mc_idx)cnx * cny * cnz || tri_counts[idx] == 0) {
        return;
    }
    int k = (int)(idx % cnz);
    int j = (int)((idx / cnz) % cny);
    int i = (int)(idx / ((mc_idx)cny * cnz));

    int row = row_offsets[idx];
    int reversed = (row >> 15) & 1;
    int offset = row & 0x7FFF;
    int n_tri = tri_counts[idx];
    int face_start = tri_scan[idx] - n_tri;
    for (int t = 0; t < n_tri; t++) {
        unsigned int e = lut_vega[offset + 1 + t];
        int v0 = vega_vertex_id((e >> 8) & 0xF, node_scan, edge_mask, center_vertex_ids, idx, i, j, k, ny, nz);
        int v1 = vega_vertex_id((e >> 4) & 0xF, node_scan, edge_mask, center_vertex_ids, idx, i, j, k, ny, nz);
        int v2 = vega_vertex_id(e & 0xF, node_scan, edge_mask, center_vertex_ids, idx, i, j, k, ny, nz);
        if (reversed != VEGA_REVERSE_SENSE) {
            int tmp = v0;
            v0 = v1;
            v1 = tmp;
        }
        mc_idx out = (mc_idx)3 * (face_start + t);
        if (flip_winding) {
            faces[out] = v2;
            faces[out + 1] = v1;
            faces[out + 2] = v0;
        } else {
            faces[out] = v0;
            faces[out + 1] = v1;
            faces[out + 2] = v2;
        }
    }
}
"""
)

# the sense of the reversed flag of the table that gives the winding of the
# other methods (checked by the tests)
_VEGA_REVERSE_SENSE = 1
_VEGA_KERNEL_CODE = (
    f"#define VEGA_REVERSE_SENSE {_VEGA_REVERSE_SENSE}\n" + _VEGA_KERNEL_CODE
)

_VEGA_KERNELS = {
    "count": "mc_vega_count_cells",
    "count_masked": "mc_vega_count_cells_masked",
    "center": "mc_vega_generate_center_vertices",
    "faces": "mc_vega_generate_faces",
}


@cp.memoize(for_each_device=True)
def _get_vega_kernel(name, wide=False):
    return cp.RawKernel(_idx_code(wide) + _VEGA_KERNEL_CODE, name)


def _vega_count_cells_gpu(volume, level, mask=None):
    return _count_cells_gpu(
        _get_vega_kernel, _VEGA_KERNELS, volume, level, mask
    )


def _vega_generate_center_vertices_gpu(
    volume,
    level,
    spacing,
    center_flags,
    vertex_offset=0,
    n_centers=None,
    out=None,
):
    return _generate_center_vertices_gpu(
        _get_vega_kernel,
        _VEGA_KERNELS,
        volume,
        level,
        spacing,
        center_flags,
        vertex_offset,
        n_centers,
        out,
    )


def _vega_generate_faces_gpu(
    volume,
    row_offsets,
    tri_counts,
    node_scan,
    edge_mask,
    center_vertex_ids,
    gradient_direction,
):
    return _generate_cell_table_faces_gpu(
        _get_vega_kernel,
        _VEGA_KERNELS,
        volume,
        row_offsets,
        tri_counts,
        node_scan,
        edge_mask,
        center_vertex_ids,
        gradient_direction,
    )


def _marching_cubes_vega(
    volume,
    level,
    spacing,
    gradient_direction,
    step_size,
    allow_degenerate,
    mask,
):
    volume, level, spacing, step_size, mask = _validate_marching_cubes_inputs(
        volume,
        level,
        spacing,
        gradient_direction,
        step_size,
        allow_degenerate,
        mask,
    )
    volume, spacing, mask = _apply_step_size(volume, spacing, step_size, mask)
    return _run_cell_table_method(
        _get_vega_kernel,
        _VEGA_KERNELS,
        volume,
        level,
        spacing,
        gradient_direction,
        allow_degenerate,
        mask,
    )
