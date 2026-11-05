# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2012-2015, P. M. Neila
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

# See implementation notes and attributions in _marching_cubes_common.py
#
# The same general approach was adapted to Lewiner marching cubes by following
# the scikit-image implementation.

import cupy as cp
import numpy as np

from . import _marching_cubes_lewiner_luts as _mcluts
from ._marching_cubes_common import (
    _COMMON_KERNEL_CODE,
    _apply_step_size,
    _count_cells_gpu,
    _decode_lut,
    _generate_cell_table_faces_gpu,
    _generate_center_vertices_gpu,
    _idx_code,
    _run_cell_table_method,
    _validate_marching_cubes_inputs,
)

# Implementation notes
# --------------------
# This is a CuPy RawKernel implementation of Lewiner marching cubes.
# It shares the edge classification, vertex generation and degenerate face
# removal with the Lorensen implementation (see _marching_cubes_common.py) and
# follows the scikit-image implementation to ensure equivalent behavior.


def _decode_lut_for_cuda_constants(name):
    shape, text = getattr(_mcluts, name)
    return _decode_lut(shape, text)


# Edge numbering used by the Lewiner tables -> local cell edge numbering used
# by ``local_edge_vid``. The center vertex (12) is kept as is.
_LEWINER_EDGE_TO_LOCAL = (8, 7, 11, 3, 9, 5, 10, 1, 0, 4, 6, 2)

# The triangle tables (everything named TILING*) in the order of their ids.
_LEWINER_TILING_NAMES = tuple(
    name for name in _mcluts._LEWINER_LUT_NAMES if name.startswith("TILING")
)


def _build_lewiner_resolved_tables():
    """Concatenate all triangle tables, with edges in local numbering.

    Returns ``(flat, base, width, nsub, center)``. The edges of triangle table ``t``
    for ``(config, subconfig)`` start at
    ``flat[base[t] + (config * nsub[t] + subconfig) * width[t]]`` and the
    table has ``width[t] // 3`` triangles. Resolving which table (and
    subconfiguration) applies to a cell is the expensive part (it needs
    double precision tests), so it is done once, by the counting kernel.
    """
    to_local = np.asarray(_LEWINER_EDGE_TO_LOCAL + (12,), dtype=np.int8)
    chunks, base, width, nsub, center = [], [], [], [], []
    offset = 0
    for name in _LEWINER_TILING_NAMES:
        table = _decode_lut_for_cuda_constants(name)
        if table.ndim == 2:
            table = table[:, np.newaxis, :]
        n_sub, w = table.shape[1], table.shape[2]
        # entries past the last triangle (if any) are never read
        mapped = np.where(table >= 0, to_local[np.clip(table, 0, 12)], table)
        chunks.append(mapped.astype(np.int8).ravel())
        has_center = (table == 12).any(axis=2)
        # a table either always or never uses the center vertex
        assert has_center.all() or not has_center.any()
        center.append(int(has_center.any()))
        base.append(offset)
        width.append(w)
        nsub.append(n_sub)
        offset += mapped.size
    return np.concatenate(chunks), base, width, nsub, center


def _generate_lewiner_lut_constants_code():
    lines = []
    # The triangle tables are read through the resolved table below, except
    # that the case 13 internal test reads the first edge of TILING13_5_1 in
    # the original edge numbering.
    for name in _mcluts._LEWINER_LUT_NAMES:
        if name.startswith("TILING") and name != "TILING13_5_1":
            continue
        values = _decode_lut_for_cuda_constants(name).ravel()
        identifier = f"lut_{name.lower()}"
        values_text = ", ".join(str(int(value)) for value in values)
        lines.append(
            f"__constant__ signed char {identifier}[{values.size}] = "
            f"{{{values_text}}};"
        )
    flat, base, width, nsub, center = _build_lewiner_resolved_tables()
    assert flat.size < 2**16  # row offsets are stored as uint16
    for i, name in enumerate(_LEWINER_TILING_NAMES):
        lines.append(f"#define T_{name} {i}")
    lines.append(
        f"__constant__ signed char lut_resolved[{flat.size}] = "
        f"{{{', '.join(str(int(v)) for v in flat)}}};"
    )
    for ident, values in (
        ("tbl_base", base),
        ("tbl_width", width),
        ("tbl_nsub", nsub),
        ("tbl_center", center),
    ):
        lines.append(
            f"__constant__ int {ident}[{len(values)}] = "
            f"{{{', '.join(str(v) for v in values)}}};"
        )
    return "\n".join(lines)


# crossing test of this method (the shared edge classification kernels in
# _COMMON_KERNEL_CODE call it, so it must precede that code)
_CROSSING_TEST_CODE = r"""
extern "C" __device__ inline bool mc_crosses(float a, float b, float level) {
    return (a <= level && b > level) || (a > level && b <= level);
}
"""

_LEWINER_KERNEL_CODE = (
    _generate_lewiner_lut_constants_code()
    + _CROSSING_TEST_CODE
    + _COMMON_KERNEL_CODE
    + r"""

extern "C" __device__ inline bool lewiner_test_face(const double* v, int face) {
    int abs_face = face < 0 ? -face : face;
    double a = 0.0, b = 0.0, c = 0.0, d = 0.0;
    if (abs_face == 1) {
        a = v[0]; b = v[4]; c = v[5]; d = v[1];
    } else if (abs_face == 2) {
        a = v[1]; b = v[5]; c = v[6]; d = v[2];
    } else if (abs_face == 3) {
        a = v[2]; b = v[6]; c = v[7]; d = v[3];
    } else if (abs_face == 4) {
        a = v[3]; b = v[7]; c = v[4]; d = v[0];
    } else if (abs_face == 5) {
        a = v[0]; b = v[3]; c = v[2]; d = v[1];
    } else if (abs_face == 6) {
        a = v[4]; b = v[7]; c = v[6]; d = v[5];
    } else {
        return false;
    }

    double ac_bd = a * c - b * d;
    const double eps = 2.2204460492503131e-16;
    if (ac_bd > -eps && ac_bd < eps) {
        return face >= 0;
    }
    return ((double)face) * a * ac_bd >= 0.0;
}

extern "C" __device__ inline bool lewiner_test_internal(
    const double* v, int case_id, int config, int subconfig, int s) {
    const double eps = 2.2204460492503131e-16;
    double at = 0.0, bt = 0.0, ct = 0.0, dt = 0.0;

    if (case_id == 4 || case_id == 10) {
        double a = (v[4] - v[0]) * (v[6] - v[2]) -
                   (v[7] - v[3]) * (v[5] - v[1]);
        double b = v[2] * (v[4] - v[0]) + v[0] * (v[6] - v[2]) -
                   v[1] * (v[7] - v[3]) - v[3] * (v[5] - v[1]);
        double t = -b / (2.0 * a + eps);
        if (t < 0.0 || t > 1.0) {
            return s > 0;
        }
        at = v[0] + (v[4] - v[0]) * t;
        bt = v[3] + (v[7] - v[3]) * t;
        ct = v[2] + (v[6] - v[2]) * t;
        dt = v[1] + (v[5] - v[1]) * t;
    } else if (case_id == 6 || case_id == 7 || case_id == 12 || case_id == 13) {
        int edge = -1;
        if (case_id == 6) {
            edge = (int)lut_test6[config * 3 + 2];
        } else if (case_id == 7) {
            edge = (int)lut_test7[config * 5 + 4];
        } else if (case_id == 12) {
            edge = (int)lut_test12[config * 4 + 3];
        } else {
            edge = (int)lut_tiling13_5_1[config * 4 * 18 + subconfig * 18];
        }

        double t = 0.0;
        if (edge == 0) {
            t = v[0] / (v[0] - v[1] + eps);
            bt = v[3] + (v[2] - v[3]) * t;
            ct = v[7] + (v[6] - v[7]) * t;
            dt = v[4] + (v[5] - v[4]) * t;
        } else if (edge == 1) {
            t = v[1] / (v[1] - v[2] + eps);
            bt = v[0] + (v[3] - v[0]) * t;
            ct = v[4] + (v[7] - v[4]) * t;
            dt = v[5] + (v[6] - v[5]) * t;
        } else if (edge == 2) {
            t = v[2] / (v[2] - v[3] + eps);
            bt = v[1] + (v[0] - v[1]) * t;
            ct = v[5] + (v[4] - v[5]) * t;
            dt = v[6] + (v[7] - v[6]) * t;
        } else if (edge == 3) {
            t = v[3] / (v[3] - v[0] + eps);
            bt = v[2] + (v[1] - v[2]) * t;
            ct = v[6] + (v[5] - v[6]) * t;
            dt = v[7] + (v[4] - v[7]) * t;
        } else if (edge == 4) {
            t = v[4] / (v[4] - v[5] + eps);
            bt = v[7] + (v[6] - v[7]) * t;
            ct = v[3] + (v[2] - v[3]) * t;
            dt = v[0] + (v[1] - v[0]) * t;
        } else if (edge == 5) {
            t = v[5] / (v[5] - v[6] + eps);
            bt = v[4] + (v[7] - v[4]) * t;
            ct = v[0] + (v[3] - v[0]) * t;
            dt = v[1] + (v[2] - v[1]) * t;
        } else if (edge == 6) {
            t = v[6] / (v[6] - v[7] + eps);
            bt = v[5] + (v[4] - v[5]) * t;
            ct = v[1] + (v[0] - v[1]) * t;
            dt = v[2] + (v[3] - v[2]) * t;
        } else if (edge == 7) {
            t = v[7] / (v[7] - v[4] + eps);
            bt = v[6] + (v[5] - v[6]) * t;
            ct = v[2] + (v[1] - v[2]) * t;
            dt = v[3] + (v[0] - v[3]) * t;
        } else if (edge == 8) {
            t = v[0] / (v[0] - v[4] + eps);
            bt = v[3] + (v[7] - v[3]) * t;
            ct = v[2] + (v[6] - v[2]) * t;
            dt = v[1] + (v[5] - v[1]) * t;
        } else if (edge == 9) {
            t = v[1] / (v[1] - v[5] + eps);
            bt = v[0] + (v[4] - v[0]) * t;
            ct = v[3] + (v[7] - v[3]) * t;
            dt = v[2] + (v[6] - v[2]) * t;
        } else if (edge == 10) {
            t = v[2] / (v[2] - v[6] + eps);
            bt = v[1] + (v[5] - v[1]) * t;
            ct = v[0] + (v[4] - v[0]) * t;
            dt = v[3] + (v[7] - v[3]) * t;
        } else if (edge == 11) {
            t = v[3] / (v[3] - v[7] + eps);
            bt = v[2] + (v[6] - v[2]) * t;
            ct = v[1] + (v[5] - v[1]) * t;
            dt = v[0] + (v[4] - v[0]) * t;
        } else {
            return s < 0;
        }
    } else {
        return s < 0;
    }

    int test = 0;
    if (at >= 0.0) test += 1;
    if (bt >= 0.0) test += 2;
    if (ct >= 0.0) test += 4;
    if (dt >= 0.0) test += 8;

    if (test == 0 || test == 1 || test == 2 || test == 3 ||
        test == 4 || test == 6 || test == 8 || test == 9 || test == 12) {
        return s > 0;
    }
    // scikit-image's cdef function has no return statement when the extra
    // condition of test 5 or 10 fails, which returns 0 (false) rather than s < 0.
    if (test == 5) {
        return (at * ct - bt * dt < eps) && s > 0;
    }
    if (test == 10) {
        return (at * ct - bt * dt >= eps) && s > 0;
    }
    return s < 0;
}

extern "C" __device__ inline int lewiner_row(int table, int config, int sub) {
    return tbl_base[table] + (config * tbl_nsub[table] + sub) * tbl_width[table];
}

// Select the triangle table (and subconfiguration) that applies to a cell and
// return the offset of its edges in lut_resolved. This is the only place that
// needs the (double precision) ambiguity tests.
extern "C" __device__ inline int lewiner_resolve(
    const double* v, int case_id, int config, int* table_out) {
    int subconfig = 0;
    int table = -1;
    int sub = 0;

    if (case_id == 1) {
        table = T_TILING1;
    } else if (case_id == 2) {
        table = T_TILING2;
    } else if (case_id == 3) {
        table = lewiner_test_face(v, (int)lut_test3[config]) ?
            T_TILING3_2 : T_TILING3_1;
    } else if (case_id == 4) {
        table = lewiner_test_internal(
            v, case_id, config, subconfig, (int)lut_test4[config]) ?
            T_TILING4_1 : T_TILING4_2;
    } else if (case_id == 5) {
        table = T_TILING5;
    } else if (case_id == 6) {
        if (lewiner_test_face(v, (int)lut_test6[config * 3])) {
            table = T_TILING6_2;
        } else if (lewiner_test_internal(
                       v, case_id, config, subconfig,
                       (int)lut_test6[config * 3 + 1])) {
            table = T_TILING6_1_1;
        } else {
            table = T_TILING6_1_2;
        }
    } else if (case_id == 7) {
        if (lewiner_test_face(v, (int)lut_test7[config * 5])) subconfig += 1;
        if (lewiner_test_face(v, (int)lut_test7[config * 5 + 1])) subconfig += 2;
        if (lewiner_test_face(v, (int)lut_test7[config * 5 + 2])) subconfig += 4;
        if (subconfig == 0) {
            table = T_TILING7_1;
        } else if (subconfig == 1 || subconfig == 2 || subconfig == 4) {
            table = T_TILING7_2;
            sub = subconfig == 1 ? 0 : (subconfig == 2 ? 1 : 2);
        } else if (subconfig == 3 || subconfig == 5 || subconfig == 6) {
            table = T_TILING7_3;
            sub = subconfig == 3 ? 0 : (subconfig == 5 ? 1 : 2);
        } else if (lewiner_test_internal(
                       v, case_id, config, subconfig,
                       (int)lut_test7[config * 5 + 3])) {
            table = T_TILING7_4_2;
        } else {
            table = T_TILING7_4_1;
        }
    } else if (case_id == 8) {
        table = T_TILING8;
    } else if (case_id == 9) {
        table = T_TILING9;
    } else if (case_id == 10) {
        if (lewiner_test_face(v, (int)lut_test10[config * 3])) {
            table = lewiner_test_face(v, (int)lut_test10[config * 3 + 1]) ?
                T_TILING10_1_1_ : T_TILING10_2;
        } else if (lewiner_test_face(v, (int)lut_test10[config * 3 + 1])) {
            table = T_TILING10_2_;
        } else if (lewiner_test_internal(
                       v, case_id, config, subconfig,
                       (int)lut_test10[config * 3 + 2])) {
            table = T_TILING10_1_1;
        } else {
            table = T_TILING10_1_2;
        }
    } else if (case_id == 11) {
        table = T_TILING11;
    } else if (case_id == 12) {
        if (lewiner_test_face(v, (int)lut_test12[config * 4])) {
            table = lewiner_test_face(v, (int)lut_test12[config * 4 + 1]) ?
                T_TILING12_1_1_ : T_TILING12_2;
        } else if (lewiner_test_face(v, (int)lut_test12[config * 4 + 1])) {
            table = T_TILING12_2_;
        } else if (lewiner_test_internal(
                       v, case_id, config, subconfig,
                       (int)lut_test12[config * 4 + 2])) {
            table = T_TILING12_1_1;
        } else {
            table = T_TILING12_1_2;
        }
    } else if (case_id == 13) {
        for (int n = 0; n < 6; n++) {
            if (lewiner_test_face(v, (int)lut_test13[config * 7 + n])) {
                subconfig += 1 << n;
            }
        }
        subconfig = (int)lut_subconfig13[subconfig];
        if (subconfig == 0) {
            table = T_TILING13_1;
        } else if (subconfig >= 1 && subconfig <= 6) {
            table = T_TILING13_2;
            sub = subconfig - 1;
        } else if (subconfig >= 7 && subconfig <= 18) {
            table = T_TILING13_3;
            sub = subconfig - 7;
        } else if (subconfig >= 19 && subconfig <= 22) {
            table = T_TILING13_4;
            sub = subconfig - 19;
        } else if (subconfig >= 23 && subconfig <= 26) {
            sub = subconfig - 23;
            table = lewiner_test_internal(
                v, case_id, config, sub, (int)lut_test13[config * 7 + 6]) ?
                T_TILING13_5_1 : T_TILING13_5_2;
        } else if (subconfig >= 27 && subconfig <= 38) {
            table = T_TILING13_3_;
            sub = subconfig - 27;
        } else if (subconfig >= 39 && subconfig <= 44) {
            table = T_TILING13_2_;
            sub = subconfig - 39;
        } else if (subconfig == 45) {
            table = T_TILING13_1_;
        }
    } else if (case_id == 14) {
        table = T_TILING14;
    }

    *table_out = table;
    return table < 0 ? 0 : lewiner_row(table, config, sub);
}

extern "C" __device__ inline void load_lewiner_values(
    const float* volume, int i, int j, int k, int ny, int nz, float level,
    double* v);

extern "C" __device__ inline void load_lewiner_values_float(
    const float* volume, int i, int j, int k, int ny, int nz, float level,
    float* v);

extern "C" __device__ inline bool lewiner_count_case_needs_values(
    int case_id) {
    return case_id == 3 || case_id == 4 || case_id == 6 ||
           case_id == 7 || case_id == 10 || case_id == 12 ||
           case_id == 13;
}

extern "C" __global__ void mc_lewiner_count_cells(
    const float* volume, unsigned char* tri_counts, unsigned char* center_flags,
    unsigned short* row_offsets, int* center_count, int nx, int ny, int nz,
    float level) {
    mc_idx idx = (mc_idx)blockDim.x * blockIdx.x + threadIdx.x;
    int cnx = nx - 1;
    int cny = ny - 1;
    int cnz = nz - 1;
    mc_idx total = (mc_idx)cnx * cny * cnz;
    if (idx >= total) {
        return;
    }

    int k = (int)(idx % cnz);
    int j = (int)((idx / cnz) % cny);
    int i = (int)(idx / ((mc_idx)cny * cnz));
    float vf[8];
    load_lewiner_values_float(volume, i, j, k, ny, nz, level, vf);

    int code = 0;
    if (vf[0] > 0.0f) code |= 1;
    if (vf[1] > 0.0f) code |= 2;
    if (vf[2] > 0.0f) code |= 4;
    if (vf[3] > 0.0f) code |= 8;
    if (vf[4] > 0.0f) code |= 16;
    if (vf[5] > 0.0f) code |= 32;
    if (vf[6] > 0.0f) code |= 64;
    if (vf[7] > 0.0f) code |= 128;

    int case_id = (int)lut_cases[code * 2];
    int config = (int)lut_cases[code * 2 + 1];
    int tri_count = 0;
    int center_flag = 0;
    int row = 0;
    if (case_id != 0) {
        // Keep upstream-compatible double ambiguity tests, but only promote
        // the cases that need them.
        double v[8];
        if (lewiner_count_case_needs_values(case_id)) {
            for (int n = 0; n < 8; n++) {
                v[n] = (double)vf[n];
            }
        }
        int table;
        row = lewiner_resolve(v, case_id, config, &table);
        tri_count = tbl_width[table] / 3;
        center_flag = tbl_center[table];
    }
    tri_counts[idx] = (unsigned char)tri_count;
    center_flags[idx] = (unsigned char)center_flag;
    row_offsets[idx] = (unsigned short)row;
    if (center_flag) {
        atomicAdd(center_count, 1);
    }
}

extern "C" __global__ void mc_lewiner_count_cells_masked(
    const float* volume, const bool* mask, unsigned char* tri_counts,
    unsigned char* center_flags,
    unsigned short* row_offsets, int* center_count, int nx, int ny, int nz,
    float level) {
    mc_idx idx = (mc_idx)blockDim.x * blockIdx.x + threadIdx.x;
    int cnx = nx - 1;
    int cny = ny - 1;
    int cnz = nz - 1;
    mc_idx total = (mc_idx)cnx * cny * cnz;
    if (idx >= total) {
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

    float vf[8];
    load_lewiner_values_float(volume, i, j, k, ny, nz, level, vf);

    int code = 0;
    if (vf[0] > 0.0f) code |= 1;
    if (vf[1] > 0.0f) code |= 2;
    if (vf[2] > 0.0f) code |= 4;
    if (vf[3] > 0.0f) code |= 8;
    if (vf[4] > 0.0f) code |= 16;
    if (vf[5] > 0.0f) code |= 32;
    if (vf[6] > 0.0f) code |= 64;
    if (vf[7] > 0.0f) code |= 128;

    int case_id = (int)lut_cases[code * 2];
    int config = (int)lut_cases[code * 2 + 1];
    int tri_count = 0;
    int center_flag = 0;
    int row = 0;
    if (case_id != 0) {
        // Keep upstream-compatible double ambiguity tests, but only promote
        // the cases that need them.
        double v[8];
        if (lewiner_count_case_needs_values(case_id)) {
            for (int n = 0; n < 8; n++) {
                v[n] = (double)vf[n];
            }
        }
        int table;
        row = lewiner_resolve(v, case_id, config, &table);
        tri_count = tbl_width[table] / 3;
        center_flag = tbl_center[table];
    }
    tri_counts[idx] = (unsigned char)tri_count;
    center_flags[idx] = (unsigned char)center_flag;
    row_offsets[idx] = (unsigned short)row;
    if (center_flag) {
        atomicAdd(center_count, 1);
    }
}

extern "C" __global__ void mc_lewiner_generate_center_vertices(
    const float* volume, const unsigned char* center_flags,
    const int* center_scan,
    int* center_vertex_ids, float* vertices, float* normals, float* values_out,
    int nx, int ny, int nz, float level, float sx, float sy, float sz,
    int vertex_offset) {
    mc_idx idx = (mc_idx)blockDim.x * blockIdx.x + threadIdx.x;
    int cnx = nx - 1;
    int cny = ny - 1;
    int cnz = nz - 1;
    mc_idx total = (mc_idx)cnx * cny * cnz;
    if (idx >= total) {
        return;
    }

    center_vertex_ids[idx] = -1;
    if (center_flags[idx] == 0) {
        return;
    }

    int k = (int)(idx % cnz);
    int j = (int)((idx / cnz) % cny);
    int i = (int)(idx / ((mc_idx)cny * cnz));
    double v[8];
    v[0] = (double)vol_at(volume, i, j, k, ny, nz) - (double)level;
    v[1] = (double)vol_at(volume, i + 1, j, k, ny, nz) - (double)level;
    v[2] = (double)vol_at(volume, i + 1, j + 1, k, ny, nz) - (double)level;
    v[3] = (double)vol_at(volume, i, j + 1, k, ny, nz) - (double)level;
    v[4] = (double)vol_at(volume, i, j, k + 1, ny, nz) - (double)level;
    v[5] = (double)vol_at(volume, i + 1, j, k + 1, ny, nz) - (double)level;
    v[6] = (double)vol_at(volume, i + 1, j + 1, k + 1, ny, nz) - (double)level;
    v[7] = (double)vol_at(volume, i, j + 1, k + 1, ny, nz) - (double)level;

    const double eps = 2.2204460492503131e-16;
    double w[8];
    double total_w = 0.0;
    for (int n = 0; n < 8; n++) {
        w[n] = 1.0 / (eps + fabs(v[n]));
        total_w += w[n];
    }

    double px = (w[1] + w[2] + w[5] + w[6]) / total_w;
    double py = (w[2] + w[3] + w[6] + w[7]) / total_w;
    double pz = (w[4] + w[5] + w[6] + w[7]) / total_w;

    double gx[8], gy[8], gz[8];
    gx[0] = v[0] - v[1]; gy[0] = v[0] - v[3]; gz[0] = v[0] - v[4];
    gx[1] = v[0] - v[1]; gy[1] = v[1] - v[2]; gz[1] = v[1] - v[5];
    gx[2] = v[3] - v[2]; gy[2] = v[1] - v[2]; gz[2] = v[2] - v[6];
    gx[3] = v[3] - v[2]; gy[3] = v[0] - v[3]; gz[3] = v[3] - v[7];
    gx[4] = v[4] - v[5]; gy[4] = v[4] - v[7]; gz[4] = v[0] - v[4];
    gx[5] = v[4] - v[5]; gy[5] = v[5] - v[6]; gz[5] = v[1] - v[5];
    gx[6] = v[7] - v[6]; gy[6] = v[5] - v[6]; gz[6] = v[2] - v[6];
    gx[7] = v[7] - v[6]; gy[7] = v[4] - v[7]; gz[7] = v[3] - v[7];

    double nxg = 0.0, nyg = 0.0, nzg = 0.0;
    for (int n = 0; n < 8; n++) {
        nxg += w[n] * gx[n];
        nyg += w[n] * gy[n];
        nzg += w[n] * gz[n];
    }
    // the gradient in physical units
    nxg /= (double)sx;
    nyg /= (double)sy;
    nzg /= (double)sz;
    double norm = sqrt(nxg * nxg + nyg * nyg + nzg * nzg);
    if (norm > 0.0) {
        nxg /= norm;
        nyg /= norm;
        nzg /= norm;
    }

    int vid = vertex_offset + center_scan[idx] - 1;
    center_vertex_ids[idx] = vid;
    mc_idx out = (mc_idx)3 * vid;
    vertices[out] = ((float)i + (float)px) * sx;
    vertices[out + 1] = ((float)j + (float)py) * sy;
    vertices[out + 2] = ((float)k + (float)pz) * sz;
    normals[out] = (float)nxg;
    normals[out + 1] = (float)nyg;
    normals[out + 2] = (float)nzg;

    double vmin = v[0];
    double vmax = v[0];
    for (int n = 1; n < 8; n++) {
        vmin = fmin(vmin, v[n]);
        vmax = fmax(vmax, v[n]);
    }
    values_out[vid] = (float)(vmax - vmin);
}

extern "C" __device__ inline void load_lewiner_values_float(
    const float* volume, int i, int j, int k, int ny, int nz, float level,
    float* v) {
    float local[8];
    local[0] = vol_at(volume, i, j, k, ny, nz) - level;
    local[1] = vol_at(volume, i + 1, j, k, ny, nz) - level;
    local[2] = vol_at(volume, i + 1, j + 1, k, ny, nz) - level;
    local[3] = vol_at(volume, i, j + 1, k, ny, nz) - level;
    local[4] = vol_at(volume, i, j, k + 1, ny, nz) - level;
    local[5] = vol_at(volume, i + 1, j, k + 1, ny, nz) - level;
    local[6] = vol_at(volume, i + 1, j + 1, k + 1, ny, nz) - level;
    local[7] = vol_at(volume, i, j + 1, k + 1, ny, nz) - level;

    v[0] = local[0];
    v[1] = local[4];
    v[2] = local[7];
    v[3] = local[3];
    v[4] = local[1];
    v[5] = local[5];
    v[6] = local[6];
    v[7] = local[2];
}

extern "C" __device__ inline void load_lewiner_values(
    const float* volume, int i, int j, int k, int ny, int nz, float level,
    double* v) {
    float vf[8];
    load_lewiner_values_float(volume, i, j, k, ny, nz, level, vf);
    for (int n = 0; n < 8; n++) {
        v[n] = (double)vf[n];
    }
}

extern "C" __global__ void mc_lewiner_generate_faces_direct(
    const unsigned short* row_offsets,
    const unsigned char* tri_counts, const int* tri_scan,
    const int* node_scan, const unsigned char* edge_mask,
    const int* center_vertex_ids, int* faces,
    int nx, int ny, int nz, int flip_winding) {
    mc_idx idx = (mc_idx)blockDim.x * blockIdx.x + threadIdx.x;
    int cnx = nx - 1;
    int cny = ny - 1;
    int cnz = nz - 1;
    mc_idx total = (mc_idx)cnx * cny * cnz;
    if (idx >= total || tri_counts[idx] == 0) {
        return;
    }

    int k = (int)(idx % cnz);
    int j = (int)((idx / cnz) % cny);
    int i = (int)(idx / ((mc_idx)cny * cnz));

    const signed char* edges = lut_resolved + row_offsets[idx];
    int n_tri = tri_counts[idx];
    int face_start = tri_scan[idx] - n_tri;
    for (int t = 0; t < n_tri; t++) {
        int e0 = (int)edges[3 * t];
        int e1 = (int)edges[3 * t + 1];
        int e2 = (int)edges[3 * t + 2];
        int v0 = e0 == 12 ? center_vertex_ids[idx] :
            local_edge_vid(node_scan, edge_mask, e0, i, j, k, ny, nz);
        int v1 = e1 == 12 ? center_vertex_ids[idx] :
            local_edge_vid(node_scan, edge_mask, e1, i, j, k, ny, nz);
        int v2 = e2 == 12 ? center_vertex_ids[idx] :
            local_edge_vid(node_scan, edge_mask, e2, i, j, k, ny, nz);

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


@cp.memoize(for_each_device=True)
def _get_lewiner_kernel(name, wide=False):
    return cp.RawKernel(_idx_code(wide) + _LEWINER_KERNEL_CODE, name)


_LEWINER_KERNELS = {
    "count": "mc_lewiner_count_cells",
    "count_masked": "mc_lewiner_count_cells_masked",
    "center": "mc_lewiner_generate_center_vertices",
    "faces": "mc_lewiner_generate_faces_direct",
}


def _lewiner_count_cells_gpu(volume, level, mask=None):
    return _count_cells_gpu(
        _get_lewiner_kernel, _LEWINER_KERNELS, volume, level, mask
    )


def _lewiner_generate_center_vertices_gpu(
    volume,
    level,
    spacing,
    center_flags,
    vertex_offset=0,
    n_centers=None,
    out=None,
):
    return _generate_center_vertices_gpu(
        _get_lewiner_kernel,
        _LEWINER_KERNELS,
        volume,
        level,
        spacing,
        center_flags,
        vertex_offset,
        n_centers,
        out,
    )


def _lewiner_generate_faces_direct_gpu(
    volume,
    row_offsets,
    tri_counts,
    node_scan,
    edge_mask,
    center_vertex_ids,
    gradient_direction,
):
    return _generate_cell_table_faces_gpu(
        _get_lewiner_kernel,
        _LEWINER_KERNELS,
        volume,
        row_offsets,
        tri_counts,
        node_scan,
        edge_mask,
        center_vertex_ids,
        gradient_direction,
    )


def _marching_cubes_lewiner(
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
    return _run_lewiner(
        volume, level, spacing, gradient_direction, allow_degenerate, mask
    )


def _run_lewiner(
    volume, level, spacing, gradient_direction, allow_degenerate, mask
):
    return _run_cell_table_method(
        _get_lewiner_kernel,
        _LEWINER_KERNELS,
        volume,
        level,
        spacing,
        gradient_direction,
        allow_degenerate,
        mask,
    )
