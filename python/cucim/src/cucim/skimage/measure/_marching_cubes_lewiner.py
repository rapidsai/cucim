# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2012-2015, P. M. Neila
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64

import cupy as cp
import numpy as np

from . import _marching_cubes_lewiner_luts as _mcluts
from ._marching_cubes_lorensen import (
    _COMMON_KERNEL_CODE,
    _apply_step_size,
    _decode_lut,
    _remove_degenerate_faces_gpu,
    _validate_marching_cubes_inputs,
)

# Implementation notes
# --------------------
# This is a CuPy RawKernel implementation of Lewiner marching cubes.
# It is implemented by updating the Lorensen kerenels based on referencing
# the scikit-image implementation to ensure equivalent behavior.

_LEWINER_LUT_NAMES = (
    "CASES",
    "TILING1",
    "TILING2",
    "TILING3_1",
    "TILING3_2",
    "TILING4_1",
    "TILING4_2",
    "TILING5",
    "TILING6_1_1",
    "TILING6_1_2",
    "TILING6_2",
    "TILING7_1",
    "TILING7_2",
    "TILING7_3",
    "TILING7_4_1",
    "TILING7_4_2",
    "TILING8",
    "TILING9",
    "TILING10_1_1",
    "TILING10_1_1_",
    "TILING10_1_2",
    "TILING10_2",
    "TILING10_2_",
    "TILING11",
    "TILING12_1_1",
    "TILING12_1_1_",
    "TILING12_1_2",
    "TILING12_2",
    "TILING12_2_",
    "TILING13_1",
    "TILING13_1_",
    "TILING13_2",
    "TILING13_2_",
    "TILING13_3",
    "TILING13_3_",
    "TILING13_4",
    "TILING13_5_1",
    "TILING13_5_2",
    "TILING14",
    "TEST3",
    "TEST4",
    "TEST6",
    "TEST7",
    "TEST10",
    "TEST12",
    "TEST13",
    "SUBCONFIG13",
)

_FLT_EPSILON = float(np.spacing(1.0))


def _decode_lut_for_cuda_constants(name):
    shape, text = getattr(_mcluts, name)
    byts = base64.decodebytes(text.encode("utf-8"))
    return np.frombuffer(byts, dtype=np.int8).reshape(shape)


def _generate_lewiner_lut_constants_code():
    lines = []
    for name in _LEWINER_LUT_NAMES:
        values = _decode_lut_for_cuda_constants(name).ravel()
        identifier = f"lut_{name.lower()}"
        values_text = ", ".join(str(int(value)) for value in values)
        lines.append(
            f"__constant__ signed char {identifier}[{values.size}] = "
            f"{{{values_text}}};"
        )
    return "\n".join(lines)


_LEWINER_KERNEL_CODE = (
    _generate_lewiner_lut_constants_code()
    + _COMMON_KERNEL_CODE
    + r"""

extern "C" __device__ inline bool crosses_lewiner(float a, float b, float level) {
    return (a <= level && b > level) || (a > level && b <= level);
}

extern "C" __global__ void mc_lewiner_classify_edges(
    const float* volume, unsigned char* edge_flags, int nx, int ny, int nz,
    float level) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = nx * ny * nz * 3;
    if (idx >= total) {
        return;
    }

    int side = idx % 3;
    int node = idx / 3;
    int k = node % nz;
    int j = (node / nz) % ny;
    int i = node / (ny * nz);
    int i2 = i + (side == 0);
    int j2 = j + (side == 1);
    int k2 = k + (side == 2);

    int flag = 0;
    if (i2 < nx && j2 < ny && k2 < nz) {
        float a = vol_at(volume, i, j, k, ny, nz);
        float b = vol_at(volume, i2, j2, k2, ny, nz);
        flag = crosses_lewiner(a, b, level) ? 1 : 0;
    }
    edge_flags[idx] = flag;
}

extern "C" __global__ void mc_lewiner_classify_edges_masked(
    const float* volume, const bool* mask, unsigned char* edge_flags,
    int nx, int ny, int nz, float level) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = nx * ny * nz * 3;
    if (idx >= total) {
        return;
    }

    int side = idx % 3;
    int node = idx / 3;
    int k = node % nz;
    int j = (node / nz) % ny;
    int i = node / (ny * nz);
    int i2 = i + (side == 0);
    int j2 = j + (side == 1);
    int k2 = k + (side == 2);

    int flag = 0;
    if (i2 < nx && j2 < ny && k2 < nz) {
        float a = vol_at(volume, i, j, k, ny, nz);
        float b = vol_at(volume, i2, j2, k2, ny, nz);
        if (crosses_lewiner(a, b, level) &&
            mc_edge_has_active_masked_cell(mask, i, j, k, side, nx, ny, nz)) {
            flag = 1;
        }
    }
    edge_flags[idx] = flag;
}

extern "C" __global__ void mc_lewiner_generate_vertices(
    const float* volume, const int* edge_scan, int* edge_vertex_ids,
    float* vertices, float* normals, float* values,
    int nx, int ny, int nz, float level,
    float sx, float sy, float sz) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = nx * ny * nz * 3;
    if (idx >= total) {
        return;
    }

    int side = idx % 3;
    int node = idx / 3;
    int k = node % nz;
    int j = (node / nz) % ny;
    int i = node / (ny * nz);
    int i2 = i + (side == 0);
    int j2 = j + (side == 1);
    int k2 = k + (side == 2);

    edge_vertex_ids[idx] = -1;
    if (i2 >= nx || j2 >= ny || k2 >= nz) {
        return;
    }

    float a = vol_at(volume, i, j, k, ny, nz);
    float b = vol_at(volume, i2, j2, k2, ny, nz);
    if (!crosses_lewiner(a, b, level)) {
        return;
    }

    int prev = idx == 0 ? 0 : edge_scan[idx - 1];
    if (edge_scan[idx] == prev) {
        return;
    }
    int vid = edge_scan[idx] - 1;
    edge_vertex_ids[idx] = vid;

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
    const double* v, const signed char* test6, const signed char* test7,
    const signed char* test12, const signed char* tiling13_5_1,
    int case_id, int config, int subconfig, int s) {
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
            edge = (int)test6[config * 3 + 2];
        } else if (case_id == 7) {
            edge = (int)test7[config * 5 + 4];
        } else if (case_id == 12) {
            edge = (int)test12[config * 4 + 3];
        } else {
            edge = (int)tiling13_5_1[config * 4 * 18 + subconfig * 18];
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
    if (test == 5) {
        if (at * ct - bt * dt < eps) return s > 0;
    } else if (test == 10) {
        if (at * ct - bt * dt >= eps) return s > 0;
    }
    return s < 0;
}

extern "C" __device__ inline void lewiner_select_count_center(
    const double* v, int case_id, int config, const signed char* test3,
    const signed char* test4, const signed char* test6, const signed char* test7,
    const signed char* test10, const signed char* test12,
    const signed char* test13, const signed char* subconfig13,
    const signed char* tiling13_5_1, int* tri_count, int* center_flag) {
    int subconfig = 0;
    *tri_count = 0;
    *center_flag = 0;

    if (case_id == 1) {
        *tri_count = 1;
    } else if (case_id == 2) {
        *tri_count = 2;
    } else if (case_id == 3) {
        *tri_count = lewiner_test_face(v, (int)test3[config]) ? 4 : 2;
    } else if (case_id == 4) {
        *tri_count = lewiner_test_internal(v, test6, test7, test12, tiling13_5_1,
                                           case_id, config, subconfig, (int)test4[config]) ? 2 : 6;
    } else if (case_id == 5) {
        *tri_count = 3;
    } else if (case_id == 6) {
        if (lewiner_test_face(v, (int)test6[config * 3])) {
            *tri_count = 5;
        } else if (lewiner_test_internal(v, test6, test7, test12, tiling13_5_1,
                                         case_id, config, subconfig, (int)test6[config * 3 + 1])) {
            *tri_count = 3;
        } else {
            *tri_count = 9;
            *center_flag = 1;
        }
    } else if (case_id == 7) {
        if (lewiner_test_face(v, (int)test7[config * 5])) subconfig += 1;
        if (lewiner_test_face(v, (int)test7[config * 5 + 1])) subconfig += 2;
        if (lewiner_test_face(v, (int)test7[config * 5 + 2])) subconfig += 4;
        if (subconfig == 0) {
            *tri_count = 3;
        } else if (subconfig == 1 || subconfig == 2 || subconfig == 4) {
            *tri_count = 5;
        } else if (subconfig == 3 || subconfig == 5 || subconfig == 6) {
            *tri_count = 9;
            *center_flag = 1;
        } else if (lewiner_test_internal(v, test6, test7, test12, tiling13_5_1,
                                         case_id, config, subconfig, (int)test7[config * 5 + 3])) {
            *tri_count = 9;
        } else {
            *tri_count = 5;
        }
    } else if (case_id == 8) {
        *tri_count = 2;
    } else if (case_id == 9) {
        *tri_count = 4;
    } else if (case_id == 10) {
        if (lewiner_test_face(v, (int)test10[config * 3])) {
            if (lewiner_test_face(v, (int)test10[config * 3 + 1])) {
                *tri_count = 4;
            } else {
                *tri_count = 8;
                *center_flag = 1;
            }
        } else if (lewiner_test_face(v, (int)test10[config * 3 + 1])) {
            *tri_count = 8;
            *center_flag = 1;
        } else if (lewiner_test_internal(v, test6, test7, test12, tiling13_5_1,
                                         case_id, config, subconfig, (int)test10[config * 3 + 2])) {
            *tri_count = 4;
        } else {
            *tri_count = 8;
        }
    } else if (case_id == 11) {
        *tri_count = 4;
    } else if (case_id == 12) {
        if (lewiner_test_face(v, (int)test12[config * 4])) {
            if (lewiner_test_face(v, (int)test12[config * 4 + 1])) {
                *tri_count = 4;
            } else {
                *tri_count = 8;
                *center_flag = 1;
            }
        } else if (lewiner_test_face(v, (int)test12[config * 4 + 1])) {
            *tri_count = 8;
            *center_flag = 1;
        } else if (lewiner_test_internal(v, test6, test7, test12, tiling13_5_1,
                                         case_id, config, subconfig, (int)test12[config * 4 + 2])) {
            *tri_count = 4;
        } else {
            *tri_count = 8;
        }
    } else if (case_id == 13) {
        for (int n = 0; n < 6; n++) {
            if (lewiner_test_face(v, (int)test13[config * 7 + n])) {
                subconfig += 1 << n;
            }
        }
        subconfig = (int)subconfig13[subconfig];
        if (subconfig == 0 || subconfig == 45) {
            *tri_count = 4;
        } else if ((subconfig >= 1 && subconfig <= 6) ||
                   (subconfig >= 39 && subconfig <= 44)) {
            *tri_count = 6;
        } else if ((subconfig >= 7 && subconfig <= 18) ||
                   (subconfig >= 27 && subconfig <= 38)) {
            *tri_count = 10;
            *center_flag = 1;
        } else if (subconfig >= 19 && subconfig <= 22) {
            *tri_count = 12;
            *center_flag = 1;
        } else if (subconfig >= 23 && subconfig <= 26) {
            int internal_subconfig = subconfig - 23;
            if (lewiner_test_internal(v, test6, test7, test12, tiling13_5_1,
                                      case_id, config, internal_subconfig,
                                      (int)test13[config * 7 + 6])) {
                *tri_count = 6;
            } else {
                *tri_count = 10;
            }
        }
    } else if (case_id == 14) {
        *tri_count = 4;
    }
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

extern "C" __device__ inline int lewiner_simple_case_tri_count(int case_id) {
    if (case_id == 1) return 1;
    if (case_id == 2 || case_id == 8) return 2;
    if (case_id == 5) return 3;
    if (case_id == 9 || case_id == 11 || case_id == 14) return 4;
    return 0;
}

extern "C" __global__ void mc_lewiner_count_cells(
    const float* volume, int* tri_counts, int* center_flags,
    unsigned char* case_codes, int* center_count, int nx, int ny, int nz,
    float level) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int cnx = nx - 1;
    int cny = ny - 1;
    int cnz = nz - 1;
    int total = cnx * cny * cnz;
    if (idx >= total) {
        return;
    }

    int k = idx % cnz;
    int j = (idx / cnz) % cny;
    int i = idx / (cny * cnz);
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
    // Keep upstream-compatible double ambiguity tests, but avoid promoting
    // simple cases that only need a fixed triangle count.
    if (lewiner_count_case_needs_values(case_id)) {
        double v[8];
        for (int n = 0; n < 8; n++) {
            v[n] = (double)vf[n];
        }
        lewiner_select_count_center(
            v, case_id, config, lut_test3, lut_test4, lut_test6, lut_test7,
            lut_test10, lut_test12, lut_test13, lut_subconfig13,
            lut_tiling13_5_1,
            &tri_count, &center_flag);
    } else {
        tri_count = lewiner_simple_case_tri_count(case_id);
    }
    tri_counts[idx] = tri_count;
    center_flags[idx] = center_flag;
    case_codes[idx] = (unsigned char)code;
    if (center_flag) {
        atomicAdd(center_count, 1);
    }
}

extern "C" __global__ void mc_lewiner_count_cells_masked(
    const float* volume, const bool* mask, int* tri_counts, int* center_flags,
    unsigned char* case_codes, int* center_count, int nx, int ny, int nz,
    float level) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int cnx = nx - 1;
    int cny = ny - 1;
    int cnz = nz - 1;
    int total = cnx * cny * cnz;
    if (idx >= total) {
        return;
    }

    int k = idx % cnz;
    int j = (idx / cnz) % cny;
    int i = idx / (cny * cnz);
    if (!mask[node_index(i + 1, j + 1, k + 1, ny, nz)]) {
        tri_counts[idx] = 0;
        center_flags[idx] = 0;
        case_codes[idx] = 0;
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
    // Keep upstream-compatible double ambiguity tests, but avoid promoting
    // simple cases that only need a fixed triangle count.
    if (lewiner_count_case_needs_values(case_id)) {
        double v[8];
        for (int n = 0; n < 8; n++) {
            v[n] = (double)vf[n];
        }
        lewiner_select_count_center(
            v, case_id, config, lut_test3, lut_test4, lut_test6, lut_test7,
            lut_test10, lut_test12, lut_test13, lut_subconfig13,
            lut_tiling13_5_1,
            &tri_count, &center_flag);
    } else {
        tri_count = lewiner_simple_case_tri_count(case_id);
    }
    tri_counts[idx] = tri_count;
    center_flags[idx] = center_flag;
    case_codes[idx] = (unsigned char)code;
    if (center_flag) {
        atomicAdd(center_count, 1);
    }
}

extern "C" __global__ void mc_lewiner_generate_center_vertices(
    const float* volume, const int* center_flags, const int* center_scan,
    int* center_vertex_ids, float* vertices, float* normals, float* values_out,
    int nx, int ny, int nz, float level, float sx, float sy, float sz,
    int vertex_offset) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int cnx = nx - 1;
    int cny = ny - 1;
    int cnz = nz - 1;
    int total = cnx * cny * cnz;
    if (idx >= total) {
        return;
    }

    center_vertex_ids[idx] = -1;
    if (center_flags[idx] == 0) {
        return;
    }

    int k = idx % cnz;
    int j = (idx / cnz) % cny;
    int i = idx / (cny * cnz);
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
    double norm = sqrt(nxg * nxg + nyg * nyg + nzg * nzg);
    if (norm > 0.0) {
        nxg /= norm;
        nyg /= norm;
        nzg /= norm;
    }

    int vid = vertex_offset + center_scan[idx] - 1;
    center_vertex_ids[idx] = vid;
    int out = 3 * vid;
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

extern "C" __device__ inline int lewiner_edge_to_local(int edge) {
    switch (edge) {
        case 0: return 8;
        case 1: return 7;
        case 2: return 11;
        case 3: return 3;
        case 4: return 9;
        case 5: return 5;
        case 6: return 10;
        case 7: return 1;
        case 8: return 0;
        case 9: return 4;
        case 10: return 6;
        default: return 2;
    }
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

extern "C" __device__ inline void lewiner_write_lut2(
    signed char* out, int start, const signed char* lut, int config,
    int width, int n_triangles) {
    int n_values = n_triangles * 3;
    for (int n = 0; n < n_values; n++) {
        signed char edge = lut[config * width + n];
        out[start + n] = edge == 12 ? 12 : lewiner_edge_to_local((int)edge);
    }
}

extern "C" __device__ inline void lewiner_write_lut3(
    signed char* out, int start, const signed char* lut, int config,
    int subconfig, int subconfigs, int width, int n_triangles) {
    int base = config * subconfigs * width + subconfig * width;
    int n_values = n_triangles * 3;
    for (int n = 0; n < n_values; n++) {
        signed char edge = lut[base + n];
        out[start + n] = edge == 12 ? 12 : lewiner_edge_to_local((int)edge);
    }
}

extern "C" __device__ inline void lewiner_write_case13_edges(
    signed char* out, int start, const double* v, int config,
    const signed char* test6, const signed char* test7,
    const signed char* test12, const signed char* test13,
    const signed char* subconfig13, const signed char* tiling13_1,
    const signed char* tiling13_1_, const signed char* tiling13_2,
    const signed char* tiling13_2_, const signed char* tiling13_3,
    const signed char* tiling13_3_, const signed char* tiling13_4,
    const signed char* tiling13_5_1, const signed char* tiling13_5_2) {
    int subconfig = 0;
    for (int n = 0; n < 6; n++) {
        if (lewiner_test_face(v, (int)test13[config * 7 + n])) {
            subconfig += 1 << n;
        }
    }
    subconfig = (int)subconfig13[subconfig];
    if (subconfig == 0) {
        lewiner_write_lut2(out, start, tiling13_1, config, 12, 4);
    } else if (subconfig >= 1 && subconfig <= 6) {
        lewiner_write_lut3(out, start, tiling13_2, config, subconfig - 1, 6, 18, 6);
    } else if (subconfig >= 7 && subconfig <= 18) {
        lewiner_write_lut3(out, start, tiling13_3, config, subconfig - 7, 12, 30, 10);
    } else if (subconfig >= 19 && subconfig <= 22) {
        lewiner_write_lut3(out, start, tiling13_4, config, subconfig - 19, 4, 36, 12);
    } else if (subconfig >= 23 && subconfig <= 26) {
        int internal_subconfig = subconfig - 23;
        if (lewiner_test_internal(v, test6, test7, test12, tiling13_5_1,
                                  13, config, internal_subconfig,
                                  (int)test13[config * 7 + 6])) {
            lewiner_write_lut3(out, start, tiling13_5_1, config, internal_subconfig, 4, 18, 6);
        } else {
            lewiner_write_lut3(out, start, tiling13_5_2, config, internal_subconfig, 4, 30, 10);
        }
    } else if (subconfig >= 27 && subconfig <= 38) {
        lewiner_write_lut3(out, start, tiling13_3_, config, subconfig - 27, 12, 30, 10);
    } else if (subconfig >= 39 && subconfig <= 44) {
        lewiner_write_lut3(out, start, tiling13_2_, config, subconfig - 39, 6, 18, 6);
    } else if (subconfig == 45) {
        lewiner_write_lut2(out, start, tiling13_1_, config, 12, 4);
    }
}

extern "C" __device__ inline void lewiner_write_edges(
    signed char* out, int start, const double* v, int case_id, int config,
    const signed char* test3, const signed char* test4,
    const signed char* test6, const signed char* test7,
    const signed char* test10, const signed char* test12,
    const signed char* test13, const signed char* subconfig13,
    const signed char* tiling1, const signed char* tiling2,
    const signed char* tiling3_1, const signed char* tiling3_2,
    const signed char* tiling4_1, const signed char* tiling4_2,
    const signed char* tiling5, const signed char* tiling6_1_1,
    const signed char* tiling6_1_2, const signed char* tiling6_2,
    const signed char* tiling7_1, const signed char* tiling7_2,
    const signed char* tiling7_3, const signed char* tiling7_4_1,
    const signed char* tiling7_4_2, const signed char* tiling8,
    const signed char* tiling9, const signed char* tiling10_1_1,
    const signed char* tiling10_1_1_, const signed char* tiling10_1_2,
    const signed char* tiling10_2, const signed char* tiling10_2_,
    const signed char* tiling11, const signed char* tiling12_1_1,
    const signed char* tiling12_1_1_, const signed char* tiling12_1_2,
    const signed char* tiling12_2, const signed char* tiling12_2_,
    const signed char* tiling13_1, const signed char* tiling13_1_,
    const signed char* tiling13_2, const signed char* tiling13_2_,
    const signed char* tiling13_3, const signed char* tiling13_3_,
    const signed char* tiling13_4, const signed char* tiling13_5_1,
    const signed char* tiling13_5_2, const signed char* tiling14) {
    int subconfig = 0;
    if (case_id == 1) {
        lewiner_write_lut2(out, start, tiling1, config, 3, 1);
    } else if (case_id == 2) {
        lewiner_write_lut2(out, start, tiling2, config, 6, 2);
    } else if (case_id == 3) {
        if (lewiner_test_face(v, (int)test3[config])) {
            lewiner_write_lut2(out, start, tiling3_2, config, 12, 4);
        } else {
            lewiner_write_lut2(out, start, tiling3_1, config, 6, 2);
        }
    } else if (case_id == 4) {
        if (lewiner_test_internal(v, test6, test7, test12, tiling13_5_1,
                                  case_id, config, subconfig, (int)test4[config])) {
            lewiner_write_lut2(out, start, tiling4_1, config, 6, 2);
        } else {
            lewiner_write_lut2(out, start, tiling4_2, config, 18, 6);
        }
    } else if (case_id == 5) {
        lewiner_write_lut2(out, start, tiling5, config, 9, 3);
    } else if (case_id == 6) {
        if (lewiner_test_face(v, (int)test6[config * 3])) {
            lewiner_write_lut2(out, start, tiling6_2, config, 15, 5);
        } else if (lewiner_test_internal(v, test6, test7, test12, tiling13_5_1,
                                         case_id, config, subconfig, (int)test6[config * 3 + 1])) {
            lewiner_write_lut2(out, start, tiling6_1_1, config, 9, 3);
        } else {
            lewiner_write_lut2(out, start, tiling6_1_2, config, 27, 9);
        }
    } else if (case_id == 7) {
        if (lewiner_test_face(v, (int)test7[config * 5])) subconfig += 1;
        if (lewiner_test_face(v, (int)test7[config * 5 + 1])) subconfig += 2;
        if (lewiner_test_face(v, (int)test7[config * 5 + 2])) subconfig += 4;
        if (subconfig == 0) {
            lewiner_write_lut2(out, start, tiling7_1, config, 9, 3);
        } else if (subconfig == 1) {
            lewiner_write_lut3(out, start, tiling7_2, config, 0, 3, 15, 5);
        } else if (subconfig == 2) {
            lewiner_write_lut3(out, start, tiling7_2, config, 1, 3, 15, 5);
        } else if (subconfig == 3) {
            lewiner_write_lut3(out, start, tiling7_3, config, 0, 3, 27, 9);
        } else if (subconfig == 4) {
            lewiner_write_lut3(out, start, tiling7_2, config, 2, 3, 15, 5);
        } else if (subconfig == 5) {
            lewiner_write_lut3(out, start, tiling7_3, config, 1, 3, 27, 9);
        } else if (subconfig == 6) {
            lewiner_write_lut3(out, start, tiling7_3, config, 2, 3, 27, 9);
        } else if (lewiner_test_internal(v, test6, test7, test12, tiling13_5_1,
                                         case_id, config, subconfig, (int)test7[config * 5 + 3])) {
            lewiner_write_lut2(out, start, tiling7_4_2, config, 27, 9);
        } else {
            lewiner_write_lut2(out, start, tiling7_4_1, config, 15, 5);
        }
    } else if (case_id == 8) {
        lewiner_write_lut2(out, start, tiling8, config, 6, 2);
    } else if (case_id == 9) {
        lewiner_write_lut2(out, start, tiling9, config, 12, 4);
    } else if (case_id == 10) {
        if (lewiner_test_face(v, (int)test10[config * 3])) {
            if (lewiner_test_face(v, (int)test10[config * 3 + 1])) {
                lewiner_write_lut2(out, start, tiling10_1_1_, config, 12, 4);
            } else {
                lewiner_write_lut2(out, start, tiling10_2, config, 24, 8);
            }
        } else if (lewiner_test_face(v, (int)test10[config * 3 + 1])) {
            lewiner_write_lut2(out, start, tiling10_2_, config, 24, 8);
        } else if (lewiner_test_internal(v, test6, test7, test12, tiling13_5_1,
                                         case_id, config, subconfig, (int)test10[config * 3 + 2])) {
            lewiner_write_lut2(out, start, tiling10_1_1, config, 12, 4);
        } else {
            lewiner_write_lut2(out, start, tiling10_1_2, config, 24, 8);
        }
    } else if (case_id == 11) {
        lewiner_write_lut2(out, start, tiling11, config, 12, 4);
    } else if (case_id == 12) {
        if (lewiner_test_face(v, (int)test12[config * 4])) {
            if (lewiner_test_face(v, (int)test12[config * 4 + 1])) {
                lewiner_write_lut2(out, start, tiling12_1_1_, config, 12, 4);
            } else {
                lewiner_write_lut2(out, start, tiling12_2, config, 24, 8);
            }
        } else if (lewiner_test_face(v, (int)test12[config * 4 + 1])) {
            lewiner_write_lut2(out, start, tiling12_2_, config, 24, 8);
        } else if (lewiner_test_internal(v, test6, test7, test12, tiling13_5_1,
                                         case_id, config, subconfig, (int)test12[config * 4 + 2])) {
            lewiner_write_lut2(out, start, tiling12_1_1, config, 12, 4);
        } else {
            lewiner_write_lut2(out, start, tiling12_1_2, config, 24, 8);
        }
    } else if (case_id == 13) {
        lewiner_write_case13_edges(
            out, start, v, config, test6, test7, test12, test13, subconfig13,
            tiling13_1, tiling13_1_, tiling13_2, tiling13_2_, tiling13_3,
            tiling13_3_, tiling13_4, tiling13_5_1, tiling13_5_2);
    } else if (case_id == 14) {
        lewiner_write_lut2(out, start, tiling14, config, 12, 4);
    }
}

extern "C" __global__ void mc_lewiner_generate_edge_ids(
    const float* volume, const signed char* cases,
    const signed char* test3, const signed char* test4,
    const signed char* test6, const signed char* test7,
    const signed char* test10, const signed char* test12,
    const signed char* test13, const signed char* subconfig13,
    const signed char* tiling1, const signed char* tiling2,
    const signed char* tiling3_1, const signed char* tiling3_2,
    const signed char* tiling4_1, const signed char* tiling4_2,
    const signed char* tiling5, const signed char* tiling6_1_1,
    const signed char* tiling6_1_2, const signed char* tiling6_2,
    const signed char* tiling7_1, const signed char* tiling7_2,
    const signed char* tiling7_3, const signed char* tiling7_4_1,
    const signed char* tiling7_4_2, const signed char* tiling8,
    const signed char* tiling9, const signed char* tiling10_1_1,
    const signed char* tiling10_1_1_, const signed char* tiling10_1_2,
    const signed char* tiling10_2, const signed char* tiling10_2_,
    const signed char* tiling11, const signed char* tiling12_1_1,
    const signed char* tiling12_1_1_, const signed char* tiling12_1_2,
    const signed char* tiling12_2, const signed char* tiling12_2_,
    const signed char* tiling13_1, const signed char* tiling13_1_,
    const signed char* tiling13_2, const signed char* tiling13_2_,
    const signed char* tiling13_3, const signed char* tiling13_3_,
    const signed char* tiling13_4, const signed char* tiling13_5_1,
    const signed char* tiling13_5_2, const signed char* tiling14,
    const int* tri_counts, const int* tri_scan, signed char* edge_ids,
    int nx, int ny, int nz, float level) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int cnx = nx - 1;
    int cny = ny - 1;
    int cnz = nz - 1;
    int total = cnx * cny * cnz;
    if (idx >= total || tri_counts[idx] == 0) {
        return;
    }

    int k = idx % cnz;
    int j = (idx / cnz) % cny;
    int i = idx / (cny * cnz);
    double v[8];
    load_lewiner_values(volume, i, j, k, ny, nz, level, v);

    int code = 0;
    if (v[0] > 0.0) code |= 1;
    if (v[1] > 0.0) code |= 2;
    if (v[2] > 0.0) code |= 4;
    if (v[3] > 0.0) code |= 8;
    if (v[4] > 0.0) code |= 16;
    if (v[5] > 0.0) code |= 32;
    if (v[6] > 0.0) code |= 64;
    if (v[7] > 0.0) code |= 128;

    int case_id = (int)cases[code * 2];
    if (case_id == 0) {
        return;
    }
    int config = (int)cases[code * 2 + 1];
    int start = (tri_scan[idx] - tri_counts[idx]) * 3;
    lewiner_write_edges(
        edge_ids, start, v, case_id, config, test3, test4, test6, test7,
        test10, test12, test13, subconfig13, tiling1, tiling2, tiling3_1,
        tiling3_2, tiling4_1, tiling4_2, tiling5, tiling6_1_1,
        tiling6_1_2, tiling6_2, tiling7_1, tiling7_2, tiling7_3,
        tiling7_4_1, tiling7_4_2, tiling8, tiling9, tiling10_1_1,
        tiling10_1_1_, tiling10_1_2, tiling10_2, tiling10_2_,
        tiling11, tiling12_1_1, tiling12_1_1_, tiling12_1_2,
        tiling12_2, tiling12_2_, tiling13_1, tiling13_1_,
        tiling13_2, tiling13_2_, tiling13_3, tiling13_3_, tiling13_4,
        tiling13_5_1, tiling13_5_2, tiling14);
}

extern "C" __global__ void mc_lewiner_generate_faces_from_edge_ids(
    const signed char* edge_ids, const int* tri_counts, const int* tri_scan,
    const int* edge_vertex_ids, const int* center_vertex_ids, int* faces,
    int nx, int ny, int nz, int flip_winding) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int cnx = nx - 1;
    int cny = ny - 1;
    int cnz = nz - 1;
    int total = cnx * cny * cnz;
    if (idx >= total || tri_counts[idx] == 0) {
        return;
    }

    int k = idx % cnz;
    int j = (idx / cnz) % cny;
    int i = idx / (cny * cnz);
    int face_start = tri_scan[idx] - tri_counts[idx];
    int edge_start = 3 * face_start;

    for (int t = 0; t < tri_counts[idx]; t++) {
        int e0 = (int)edge_ids[edge_start + 3 * t];
        int e1 = (int)edge_ids[edge_start + 3 * t + 1];
        int e2 = (int)edge_ids[edge_start + 3 * t + 2];
        int v0 = e0 == 12 ? center_vertex_ids[idx] :
            local_edge_vid(edge_vertex_ids, e0, i, j, k, ny, nz);
        int v1 = e1 == 12 ? center_vertex_ids[idx] :
            local_edge_vid(edge_vertex_ids, e1, i, j, k, ny, nz);
        int v2 = e2 == 12 ? center_vertex_ids[idx] :
            local_edge_vid(edge_vertex_ids, e2, i, j, k, ny, nz);

        int out = 3 * (face_start + t);
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

extern "C" __device__ inline bool lewiner_case_needs_values(int case_id) {
    return case_id == 3 || case_id == 4 || case_id == 6 ||
           case_id == 7 || case_id == 10 || case_id == 12 ||
           case_id == 13;
}

extern "C" __global__ void mc_lewiner_generate_faces_direct(
    const float* volume, const unsigned char* case_codes,
    const int* tri_counts, const int* tri_scan,
    const int* edge_vertex_ids, const int* center_vertex_ids, int* faces,
    int nx, int ny, int nz, float level, int flip_winding) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int cnx = nx - 1;
    int cny = ny - 1;
    int cnz = nz - 1;
    int total = cnx * cny * cnz;
    if (idx >= total || tri_counts[idx] == 0) {
        return;
    }

    int k = idx % cnz;
    int j = (idx / cnz) % cny;
    int i = idx / (cny * cnz);

    int code = (int)case_codes[idx];
    int case_id = (int)lut_cases[code * 2];
    if (case_id == 0) {
        return;
    }
    int config = (int)lut_cases[code * 2 + 1];
    double v[8];
    if (lewiner_case_needs_values(case_id)) {
        load_lewiner_values(volume, i, j, k, ny, nz, level, v);
    }
    signed char local_edges[36];
    lewiner_write_edges(
        local_edges, 0, v, case_id, config, lut_test3, lut_test4,
        lut_test6, lut_test7, lut_test10, lut_test12, lut_test13,
        lut_subconfig13, lut_tiling1, lut_tiling2, lut_tiling3_1,
        lut_tiling3_2, lut_tiling4_1, lut_tiling4_2, lut_tiling5,
        lut_tiling6_1_1, lut_tiling6_1_2, lut_tiling6_2, lut_tiling7_1,
        lut_tiling7_2, lut_tiling7_3, lut_tiling7_4_1, lut_tiling7_4_2,
        lut_tiling8, lut_tiling9, lut_tiling10_1_1, lut_tiling10_1_1_,
        lut_tiling10_1_2, lut_tiling10_2, lut_tiling10_2_,
        lut_tiling11, lut_tiling12_1_1, lut_tiling12_1_1_,
        lut_tiling12_1_2, lut_tiling12_2, lut_tiling12_2_,
        lut_tiling13_1, lut_tiling13_1_, lut_tiling13_2,
        lut_tiling13_2_, lut_tiling13_3, lut_tiling13_3_,
        lut_tiling13_4, lut_tiling13_5_1, lut_tiling13_5_2,
        lut_tiling14);

    int face_start = tri_scan[idx] - tri_counts[idx];
    for (int t = 0; t < tri_counts[idx]; t++) {
        int e0 = (int)local_edges[3 * t];
        int e1 = (int)local_edges[3 * t + 1];
        int e2 = (int)local_edges[3 * t + 2];
        int v0 = e0 == 12 ? center_vertex_ids[idx] :
            local_edge_vid(edge_vertex_ids, e0, i, j, k, ny, nz);
        int v1 = e1 == 12 ? center_vertex_ids[idx] :
            local_edge_vid(edge_vertex_ids, e1, i, j, k, ny, nz);
        int v2 = e2 == 12 ? center_vertex_ids[idx] :
            local_edge_vid(edge_vertex_ids, e2, i, j, k, ny, nz);

        int out = 3 * (face_start + t);
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
def _get_lewiner_kernel(name):
    return cp.RawKernel(_LEWINER_KERNEL_CODE, name)


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
    nx, ny, nz = volume.shape
    n_edges = nx * ny * nz * 3
    threads = 256
    edge_blocks = ((n_edges + threads - 1) // threads,)

    edge_flags = cp.empty(n_edges, dtype=cp.uint8)
    if mask is None:
        _get_lewiner_kernel("mc_lewiner_classify_edges")(
            edge_blocks,
            (threads,),
            (volume, edge_flags, nx, ny, nz, np.float32(level)),
        )
    else:
        _get_lewiner_kernel("mc_lewiner_classify_edges_masked")(
            edge_blocks,
            (threads,),
            (volume, mask, edge_flags, nx, ny, nz, np.float32(level)),
        )
    edge_scan = cp.cumsum(edge_flags, dtype=cp.int32)
    n_edge_vertices = int(edge_scan[-1])
    if n_edge_vertices == 0:
        raise RuntimeError("No surface found at the given iso value.")

    edge_vertex_ids = cp.empty(n_edges, dtype=cp.int32)
    edge_vertices = cp.empty((n_edge_vertices, 3), dtype=cp.float32)
    edge_normals = cp.empty((n_edge_vertices, 3), dtype=cp.float32)
    edge_values = cp.empty(n_edge_vertices, dtype=cp.float32)
    _get_lewiner_kernel("mc_lewiner_generate_vertices")(
        edge_blocks,
        (threads,),
        (
            volume,
            edge_scan,
            edge_vertex_ids,
            edge_vertices,
            edge_normals,
            edge_values,
            nx,
            ny,
            nz,
            np.float32(level),
            np.float32(spacing[0]),
            np.float32(spacing[1]),
            np.float32(spacing[2]),
        ),
    )

    tri_counts, center_flags, case_codes, n_center_vertices = (
        _lewiner_count_cells_gpu(volume, level, mask)
    )
    if n_center_vertices:
        vertices, normals, values, center_vertex_ids = (
            _lewiner_generate_center_vertices_gpu(
                volume,
                level,
                spacing,
                center_flags,
                vertex_offset=n_edge_vertices,
                n_centers=n_center_vertices,
            )
        )
        vertices[:n_edge_vertices] = edge_vertices
        normals[:n_edge_vertices] = edge_normals
        values[:n_edge_vertices] = edge_values
    else:
        vertices = edge_vertices
        normals = edge_normals
        values = edge_values
        center_vertex_ids = cp.empty(0, dtype=cp.int32)

    faces = _lewiner_generate_faces_direct_gpu(
        volume,
        level,
        case_codes,
        tri_counts,
        edge_vertex_ids,
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


def _lewiner_count_cells_gpu(volume, level, mask=None):
    nx, ny, nz = volume.shape
    n_cells = (nx - 1) * (ny - 1) * (nz - 1)
    threads = 256
    cell_blocks = ((n_cells + threads - 1) // threads,)
    tri_counts = cp.empty(n_cells, dtype=cp.int32)
    center_flags = cp.empty(n_cells, dtype=cp.int32)
    case_codes = cp.empty(n_cells, dtype=cp.uint8)
    center_count = cp.zeros(1, dtype=cp.int32)
    if mask is None:
        _get_lewiner_kernel("mc_lewiner_count_cells")(
            cell_blocks,
            (threads,),
            (
                volume,
                tri_counts,
                center_flags,
                case_codes,
                center_count,
                nx,
                ny,
                nz,
                np.float32(level),
            ),
        )
    else:
        _get_lewiner_kernel("mc_lewiner_count_cells_masked")(
            cell_blocks,
            (threads,),
            (
                volume,
                mask,
                tri_counts,
                center_flags,
                case_codes,
                center_count,
                nx,
                ny,
                nz,
                np.float32(level),
            ),
        )
    return tri_counts, center_flags, case_codes, int(center_count[0])


def _lewiner_generate_center_vertices_gpu(
    volume, level, spacing, center_flags, vertex_offset=0, n_centers=None
):
    nx, ny, nz = volume.shape
    n_cells = (nx - 1) * (ny - 1) * (nz - 1)
    center_scan = cp.cumsum(center_flags, dtype=cp.int32)
    if n_centers is None:
        n_centers = int(center_scan[-1]) if n_cells else 0
    n_vertices = vertex_offset + n_centers
    center_vertex_ids = cp.full(n_cells, -1, dtype=cp.int32)
    vertices = cp.empty((n_vertices, 3), dtype=cp.float32)
    normals = cp.empty((n_vertices, 3), dtype=cp.float32)
    values = cp.empty(n_vertices, dtype=cp.float32)
    if n_centers == 0:
        return vertices, normals, values, center_vertex_ids

    threads = 256
    cell_blocks = ((n_cells + threads - 1) // threads,)
    _get_lewiner_kernel("mc_lewiner_generate_center_vertices")(
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


def _lewiner_generate_edge_ids_gpu(volume, level, tri_counts):
    nx, ny, nz = volume.shape
    n_cells = (nx - 1) * (ny - 1) * (nz - 1)
    tri_scan = cp.cumsum(tri_counts, dtype=cp.int32)
    n_faces = int(tri_scan[-1]) if n_cells else 0
    edge_ids = cp.empty(n_faces * 3, dtype=cp.int8)
    if n_faces == 0:
        return edge_ids

    threads = 256
    cell_blocks = ((n_cells + threads - 1) // threads,)
    luts = _get_lewiner_luts_device()
    _get_lewiner_kernel("mc_lewiner_generate_edge_ids")(
        cell_blocks,
        (threads,),
        (
            volume,
            luts["cases"],
            luts["test3"],
            luts["test4"],
            luts["test6"],
            luts["test7"],
            luts["test10"],
            luts["test12"],
            luts["test13"],
            luts["subconfig13"],
            luts["tiling1"],
            luts["tiling2"],
            luts["tiling3_1"],
            luts["tiling3_2"],
            luts["tiling4_1"],
            luts["tiling4_2"],
            luts["tiling5"],
            luts["tiling6_1_1"],
            luts["tiling6_1_2"],
            luts["tiling6_2"],
            luts["tiling7_1"],
            luts["tiling7_2"],
            luts["tiling7_3"],
            luts["tiling7_4_1"],
            luts["tiling7_4_2"],
            luts["tiling8"],
            luts["tiling9"],
            luts["tiling10_1_1"],
            luts["tiling10_1_1_"],
            luts["tiling10_1_2"],
            luts["tiling10_2"],
            luts["tiling10_2_"],
            luts["tiling11"],
            luts["tiling12_1_1"],
            luts["tiling12_1_1_"],
            luts["tiling12_1_2"],
            luts["tiling12_2"],
            luts["tiling12_2_"],
            luts["tiling13_1"],
            luts["tiling13_1_"],
            luts["tiling13_2"],
            luts["tiling13_2_"],
            luts["tiling13_3"],
            luts["tiling13_3_"],
            luts["tiling13_4"],
            luts["tiling13_5_1"],
            luts["tiling13_5_2"],
            luts["tiling14"],
            tri_counts,
            tri_scan,
            edge_ids,
            nx,
            ny,
            nz,
            np.float32(level),
        ),
    )
    return edge_ids


def _lewiner_generate_faces_from_edge_ids_gpu(
    edge_ids,
    tri_counts,
    edge_vertex_ids,
    center_vertex_ids,
    volume_shape,
    gradient_direction,
):
    nx, ny, nz = volume_shape
    n_cells = (nx - 1) * (ny - 1) * (nz - 1)
    tri_scan = cp.cumsum(tri_counts, dtype=cp.int32)
    n_faces = int(tri_scan[-1]) if n_cells else 0
    faces = cp.empty(n_faces * 3, dtype=cp.int32)
    if n_faces == 0:
        return faces.reshape(-1, 3)

    threads = 256
    cell_blocks = ((n_cells + threads - 1) // threads,)
    _get_lewiner_kernel("mc_lewiner_generate_faces_from_edge_ids")(
        cell_blocks,
        (threads,),
        (
            edge_ids,
            tri_counts,
            tri_scan,
            edge_vertex_ids,
            center_vertex_ids,
            faces,
            nx,
            ny,
            nz,
            np.int32(gradient_direction == "descent"),
        ),
    )
    return faces.reshape(-1, 3)


def _lewiner_generate_faces_direct_gpu(
    volume,
    level,
    case_codes,
    tri_counts,
    edge_vertex_ids,
    center_vertex_ids,
    gradient_direction,
):
    nx, ny, nz = volume.shape
    n_cells = (nx - 1) * (ny - 1) * (nz - 1)
    tri_scan = cp.cumsum(tri_counts, dtype=cp.int32)
    n_faces = int(tri_scan[-1]) if n_cells else 0
    faces = cp.empty(n_faces * 3, dtype=cp.int32)
    if n_faces == 0:
        return faces.reshape(-1, 3)

    threads = 128
    cell_blocks = ((n_cells + threads - 1) // threads,)
    _get_lewiner_kernel("mc_lewiner_generate_faces_direct")(
        cell_blocks,
        (threads,),
        (
            volume,
            case_codes,
            tri_counts,
            tri_scan,
            edge_vertex_ids,
            center_vertex_ids,
            faces,
            nx,
            ny,
            nz,
            np.float32(level),
            np.int32(gradient_direction == "descent"),
        ),
    )
    return faces.reshape(-1, 3)


def _decode_named_lut(name):
    shape, text = getattr(_mcluts, name)
    return _decode_lut(shape, text)


def _get_lewiner_luts_cpu():
    return {name: _decode_named_lut(name) for name in _LEWINER_LUT_NAMES}


def _append_triangles_from_lut(out, luts, name, config, *args, trace=None):
    if len(args) == 1:
        subconfig = None
        n_triangles = args[0]
    elif len(args) == 2:
        subconfig, n_triangles = args
    else:
        raise TypeError("expected n_triangles or subconfig, n_triangles")

    lut = luts[name]
    n_values = n_triangles * 3
    if subconfig is None:
        values = lut[config, :n_values]
    else:
        values = lut[config, subconfig, :n_values]
    out.extend(int(v) for v in values)
    if trace is not None:
        trace.append(
            (
                name,
                int(config),
                None if subconfig is None else int(subconfig),
                int(n_triangles),
            )
        )


def _lewiner_test_face(values, face):
    abs_face = abs(int(face))
    if abs_face == 1:
        a, b, c, d = values[0], values[4], values[5], values[1]
    elif abs_face == 2:
        a, b, c, d = values[1], values[5], values[6], values[2]
    elif abs_face == 3:
        a, b, c, d = values[2], values[6], values[7], values[3]
    elif abs_face == 4:
        a, b, c, d = values[3], values[7], values[4], values[0]
    elif abs_face == 5:
        a, b, c, d = values[0], values[3], values[2], values[1]
    elif abs_face == 6:
        a, b, c, d = values[4], values[7], values[6], values[5]
    else:
        return False

    ac_bd = a * c - b * d
    if -_FLT_EPSILON < ac_bd < _FLT_EPSILON:
        return face >= 0
    return face * a * ac_bd >= 0


def _lewiner_test_internal(values, luts, case, config, subconfig, s):
    v = values
    at = bt = ct = dt = 0.0

    if case in (4, 10):
        a = (v[4] - v[0]) * (v[6] - v[2]) - (v[7] - v[3]) * (v[5] - v[1])
        b = (
            v[2] * (v[4] - v[0])
            + v[0] * (v[6] - v[2])
            - v[1] * (v[7] - v[3])
            - v[3] * (v[5] - v[1])
        )
        t = -b / (2 * a + _FLT_EPSILON)
        if t < 0 or t > 1:
            return s > 0

        at = v[0] + (v[4] - v[0]) * t
        bt = v[3] + (v[7] - v[3]) * t
        ct = v[2] + (v[6] - v[2]) * t
        dt = v[1] + (v[5] - v[1]) * t
    elif case in (6, 7, 12, 13):
        if case == 6:
            edge = int(luts["TEST6"][config, 2])
        elif case == 7:
            edge = int(luts["TEST7"][config, 4])
        elif case == 12:
            edge = int(luts["TEST12"][config, 3])
        else:
            edge = int(luts["TILING13_5_1"][config, subconfig, 0])

        if edge == 0:
            t = v[0] / (v[0] - v[1] + _FLT_EPSILON)
            bt = v[3] + (v[2] - v[3]) * t
            ct = v[7] + (v[6] - v[7]) * t
            dt = v[4] + (v[5] - v[4]) * t
        elif edge == 1:
            t = v[1] / (v[1] - v[2] + _FLT_EPSILON)
            bt = v[0] + (v[3] - v[0]) * t
            ct = v[4] + (v[7] - v[4]) * t
            dt = v[5] + (v[6] - v[5]) * t
        elif edge == 2:
            t = v[2] / (v[2] - v[3] + _FLT_EPSILON)
            bt = v[1] + (v[0] - v[1]) * t
            ct = v[5] + (v[4] - v[5]) * t
            dt = v[6] + (v[7] - v[6]) * t
        elif edge == 3:
            t = v[3] / (v[3] - v[0] + _FLT_EPSILON)
            bt = v[2] + (v[1] - v[2]) * t
            ct = v[6] + (v[5] - v[6]) * t
            dt = v[7] + (v[4] - v[7]) * t
        elif edge == 4:
            t = v[4] / (v[4] - v[5] + _FLT_EPSILON)
            bt = v[7] + (v[6] - v[7]) * t
            ct = v[3] + (v[2] - v[3]) * t
            dt = v[0] + (v[1] - v[0]) * t
        elif edge == 5:
            t = v[5] / (v[5] - v[6] + _FLT_EPSILON)
            bt = v[4] + (v[7] - v[4]) * t
            ct = v[0] + (v[3] - v[0]) * t
            dt = v[1] + (v[2] - v[1]) * t
        elif edge == 6:
            t = v[6] / (v[6] - v[7] + _FLT_EPSILON)
            bt = v[5] + (v[4] - v[5]) * t
            ct = v[1] + (v[0] - v[1]) * t
            dt = v[2] + (v[3] - v[2]) * t
        elif edge == 7:
            t = v[7] / (v[7] - v[4] + _FLT_EPSILON)
            bt = v[6] + (v[5] - v[6]) * t
            ct = v[2] + (v[1] - v[2]) * t
            dt = v[3] + (v[0] - v[3]) * t
        elif edge == 8:
            t = v[0] / (v[0] - v[4] + _FLT_EPSILON)
            bt = v[3] + (v[7] - v[3]) * t
            ct = v[2] + (v[6] - v[2]) * t
            dt = v[1] + (v[5] - v[1]) * t
        elif edge == 9:
            t = v[1] / (v[1] - v[5] + _FLT_EPSILON)
            bt = v[0] + (v[4] - v[0]) * t
            ct = v[3] + (v[7] - v[3]) * t
            dt = v[2] + (v[6] - v[2]) * t
        elif edge == 10:
            t = v[2] / (v[2] - v[6] + _FLT_EPSILON)
            bt = v[1] + (v[5] - v[1]) * t
            ct = v[0] + (v[4] - v[0]) * t
            dt = v[3] + (v[7] - v[3]) * t
        elif edge == 11:
            t = v[3] / (v[3] - v[7] + _FLT_EPSILON)
            bt = v[2] + (v[6] - v[2]) * t
            ct = v[1] + (v[5] - v[1]) * t
            dt = v[0] + (v[4] - v[0]) * t
        else:
            return s < 0
    else:
        return s < 0

    test = 0
    if at >= 0:
        test += 1
    if bt >= 0:
        test += 2
    if ct >= 0:
        test += 4
    if dt >= 0:
        test += 8

    if test in (0, 1, 2, 3, 4, 6, 8, 9, 12):
        return s > 0
    if test == 5:
        if at * ct - bt * dt < _FLT_EPSILON:
            return s > 0
    elif test == 10:
        if at * ct - bt * dt >= _FLT_EPSILON:
            return s > 0
    return s < 0


def _lewiner_tri_edges_for_case_values(values, luts=None):
    edges, _ = _lewiner_tri_edges_and_trace_for_case_values(values, luts=luts)
    return edges


def _lewiner_tri_edges_and_trace_for_case_values(values, luts=None):
    values = tuple(float(v) for v in values)
    if len(values) != 8:
        raise ValueError("values must contain 8 cube corner values.")
    if luts is None:
        luts = _get_lewiner_luts_cpu()

    index = 0
    for i, value in enumerate(values):
        if value > 0.0:
            index += 1 << i

    case = int(luts["CASES"][index, 0])
    if case == 0:
        return (), ()
    config = int(luts["CASES"][index, 1])
    out = []
    trace = []
    _lewiner_big_switch(values, luts, case, config, out, trace=trace)
    return tuple(out), tuple(trace)


def _lewiner_case_needs_center_vertex(values, luts=None):
    return 12 in _lewiner_tri_edges_for_case_values(values, luts=luts)


def _lewiner_center_vertex(values):
    values = tuple(float(v) for v in values)
    if len(values) != 8:
        raise ValueError("values must contain 8 cube corner values.")

    weights = [1.0 / (_FLT_EPSILON + abs(v)) for v in values]
    total = sum(weights)
    position = np.asarray(
        [
            (weights[1] + weights[2] + weights[5] + weights[6]) / total,
            (weights[2] + weights[3] + weights[6] + weights[7]) / total,
            (weights[4] + weights[5] + weights[6] + weights[7]) / total,
        ],
        dtype=np.float64,
    )

    v = values
    gradients = np.asarray(
        [
            (v[0] - v[1], v[0] - v[3], v[0] - v[4]),
            (v[0] - v[1], v[1] - v[2], v[1] - v[5]),
            (v[3] - v[2], v[1] - v[2], v[2] - v[6]),
            (v[3] - v[2], v[0] - v[3], v[3] - v[7]),
            (v[4] - v[5], v[4] - v[7], v[0] - v[4]),
            (v[4] - v[5], v[5] - v[6], v[1] - v[5]),
            (v[7] - v[6], v[5] - v[6], v[2] - v[6]),
            (v[7] - v[6], v[4] - v[7], v[3] - v[7]),
        ],
        dtype=np.float64,
    )
    gradient = np.sum(
        np.asarray(weights, dtype=np.float64)[:, np.newaxis] * gradients,
        axis=0,
    )
    norm = np.linalg.norm(gradient)
    if norm > 0:
        gradient = gradient / norm
    return position, gradient


def _lewiner_big_switch(values, luts, case, config, out, trace=None):
    subconfig = 0

    def add(out, luts, name, config, *args):
        _append_triangles_from_lut(out, luts, name, config, *args, trace=trace)

    test_face = _lewiner_test_face
    test_internal = _lewiner_test_internal

    if case == 1:
        add(out, luts, "TILING1", config, 1)
    elif case == 2:
        add(out, luts, "TILING2", config, 2)
    elif case == 3:
        if test_face(values, luts["TEST3"][config]):
            add(out, luts, "TILING3_2", config, 4)
        else:
            add(out, luts, "TILING3_1", config, 2)
    elif case == 4:
        if test_internal(
            values, luts, case, config, subconfig, luts["TEST4"][config]
        ):
            add(out, luts, "TILING4_1", config, 2)
        else:
            add(out, luts, "TILING4_2", config, 6)
    elif case == 5:
        add(out, luts, "TILING5", config, 3)
    elif case == 6:
        if test_face(values, luts["TEST6"][config, 0]):
            add(out, luts, "TILING6_2", config, 5)
        elif test_internal(
            values, luts, case, config, subconfig, luts["TEST6"][config, 1]
        ):
            add(out, luts, "TILING6_1_1", config, 3)
        else:
            add(out, luts, "TILING6_1_2", config, 9)
    elif case == 7:
        if test_face(values, luts["TEST7"][config, 0]):
            subconfig += 1
        if test_face(values, luts["TEST7"][config, 1]):
            subconfig += 2
        if test_face(values, luts["TEST7"][config, 2]):
            subconfig += 4
        if subconfig == 0:
            add(out, luts, "TILING7_1", config, 3)
        elif subconfig in (1, 2, 4):
            add(
                out, luts, "TILING7_2", config, {1: 0, 2: 1, 4: 2}[subconfig], 5
            )
        elif subconfig in (3, 5, 6):
            add(
                out, luts, "TILING7_3", config, {3: 0, 5: 1, 6: 2}[subconfig], 9
            )
        elif test_internal(
            values, luts, case, config, subconfig, luts["TEST7"][config, 3]
        ):
            add(out, luts, "TILING7_4_2", config, 9)
        else:
            add(out, luts, "TILING7_4_1", config, 5)
    elif case == 8:
        add(out, luts, "TILING8", config, 2)
    elif case == 9:
        add(out, luts, "TILING9", config, 4)
    elif case == 10:
        if test_face(values, luts["TEST10"][config, 0]):
            if test_face(values, luts["TEST10"][config, 1]):
                add(out, luts, "TILING10_1_1_", config, 4)
            else:
                add(out, luts, "TILING10_2", config, 8)
        elif test_face(values, luts["TEST10"][config, 1]):
            add(out, luts, "TILING10_2_", config, 8)
        elif test_internal(
            values, luts, case, config, subconfig, luts["TEST10"][config, 2]
        ):
            add(out, luts, "TILING10_1_1", config, 4)
        else:
            add(out, luts, "TILING10_1_2", config, 8)
    elif case == 11:
        add(out, luts, "TILING11", config, 4)
    elif case == 12:
        if test_face(values, luts["TEST12"][config, 0]):
            if test_face(values, luts["TEST12"][config, 1]):
                add(out, luts, "TILING12_1_1_", config, 4)
            else:
                add(out, luts, "TILING12_2", config, 8)
        elif test_face(values, luts["TEST12"][config, 1]):
            add(out, luts, "TILING12_2_", config, 8)
        elif test_internal(
            values, luts, case, config, subconfig, luts["TEST12"][config, 2]
        ):
            add(out, luts, "TILING12_1_1", config, 4)
        else:
            add(out, luts, "TILING12_1_2", config, 8)
    elif case == 13:
        _lewiner_case_13(values, luts, config, out, trace=trace)
    elif case == 14:
        add(out, luts, "TILING14", config, 4)


def _lewiner_case_13(values, luts, config, out, trace=None):
    subconfig = 0
    for i in range(6):
        if _lewiner_test_face(values, luts["TEST13"][config, i]):
            subconfig += 1 << i
    subconfig = int(luts["SUBCONFIG13"][subconfig])

    def add(out, luts, name, config, *args):
        _append_triangles_from_lut(out, luts, name, config, *args, trace=trace)

    if subconfig == 0:
        add(out, luts, "TILING13_1", config, 4)
    elif 1 <= subconfig <= 6:
        add(out, luts, "TILING13_2", config, subconfig - 1, 6)
    elif 7 <= subconfig <= 18:
        add(out, luts, "TILING13_3", config, subconfig - 7, 10)
    elif 19 <= subconfig <= 22:
        add(out, luts, "TILING13_4", config, subconfig - 19, 12)
    elif 23 <= subconfig <= 26:
        internal_subconfig = subconfig - 23
        if _lewiner_test_internal(
            values,
            luts,
            13,
            config,
            internal_subconfig,
            luts["TEST13"][config, 6],
        ):
            add(out, luts, "TILING13_5_1", config, internal_subconfig, 6)
        else:
            add(out, luts, "TILING13_5_2", config, internal_subconfig, 10)
    elif 27 <= subconfig <= 38:
        add(out, luts, "TILING13_3_", config, subconfig - 27, 10)
    elif 39 <= subconfig <= 44:
        add(out, luts, "TILING13_2_", config, subconfig - 39, 6)
    elif subconfig == 45:
        add(out, luts, "TILING13_1_", config, 4)


@cp.memoize(for_each_device=True)
def _get_lewiner_luts_device():
    return {
        name.lower(): cp.asarray(_decode_named_lut(name))
        for name in _LEWINER_LUT_NAMES
    }
