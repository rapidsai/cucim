# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2012-2015, P. M. Neila
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import base64

import cupy as cp
import numpy as np

from . import _marching_cubes_lewiner_luts as _mcluts

# Implementation notes
# --------------------
# This is a CuPy RawKernel implementation of classic Lorensen marching cubes.
# The public API validation and post-processing behavior follows
# scikit-image's _marching_cubes_lewiner.py. The dense-grid edge/face
# organization follows the CuMCubes CUDA kernel structure, while the output
# allocation strategy follows Warp's scan-based implementation in
# https://github.com/NVIDIA/warp/blob/v1.13.0/warp/_src/marching_cubes.py
# and the older release-1.8 native CUDA implementation in
# https://github.com/NVIDIA/warp/blob/release-1.8/warp/native/marching.cu
# rather than atomic append.
#
# Classic Lorensen triangle table. This compact representation is vendored
# from scikit-image's generated _marching_cubes_lewiner_luts.py.
_CASES_CLASSIC = (
    (256, 16),
    """
/////////////////////wAIA/////////////////8AAQn/////////////////AQgDCQgB////
/////////wECCv////////////////8ACAMBAgr/////////////CQIKAAIJ/////////////wII
AwIKCAoJCP////////8DCwL/////////////////AAsCCAsA/////////////wEJAAIDC///////
//////8BCwIBCQsJCAv/////////AwoBCwoD/////////////wAKAQAICggLCv////////8DCQAD
CwkLCgn/////////CQgKCggL/////////////wQHCP////////////////8EAwAHAwT/////////
////AAEJCAQH/////////////wQBCQQHAQcDAf////////8BAgoIBAf/////////////AwQHAwAE
AQIK/////////wkCCgkAAggEB/////////8CCgkCCQcCBwMHCQT/////CAQHAwsC////////////
/wsEBwsCBAIABP////////8JAAEIBAcCAwv/////////BAcLCQQLCQsCCQIB/////wMKAQMLCgcI
BP////////8BCwoBBAsBAAQHCwT/////BAcICQALCQsKCwAD/////wQHCwQLCQkLCv////////8J
BQT/////////////////CQUEAAgD/////////////wAFBAEFAP////////////8IBQQIAwUDAQX/
////////AQIKCQUE/////////////wMACAECCgQJBf////////8FAgoFBAIEAAL/////////AgoF
AwIFAwUEAwQI/////wkFBAIDC/////////////8ACwIACAsECQX/////////AAUEAAEFAgML////
/////wIBBQIFCAIICwQIBf////8KAwsKAQMJBQT/////////BAkFAAgBCAoBCAsK/////wUEAAUA
CwULCgsAA/////8FBAgFCAoKCAv/////////CQcIBQcJ/////////////wkDAAkFAwUHA///////
//8ABwgAAQcBBQf/////////AQUDAwUH/////////////wkHCAkFBwoBAv////////8KAQIJBQAF
AwAFBwP/////CAACCAIFCAUHCgUC/////wIKBQIFAwMFB/////////8HCQUHCAkDCwL/////////
CQUHCQcCCQIAAgcL/////wIDCwABCAEHCAEFB/////8LAgELAQcHAQX/////////CQUICAUHCgED
CgML/////wUHAAUACQcLAAEACgsKAP8LCgALAAMKBQAIAAcFBwD/CwoFBwsF/////////////woG
Bf////////////////8ACAMFCgb/////////////CQABBQoG/////////////wEIAwEJCAUKBv//
//////8BBgUCBgH/////////////AQYFAQIGAwAI/////////wkGBQkABgACBv////////8FCQgF
CAIFAgYDAgj/////AgMLCgYF/////////////wsACAsCAAoGBf////////8AAQkCAwsFCgb/////
////BQoGAQkCCQsCCQgL/////wYDCwYFAwUBA/////////8ACAsACwUABQEFCwb/////AwsGAAMG
AAYFAAUJ/////wYFCQYJCwsJCP////////8FCgYEBwj/////////////BAMABAcDBgUK////////
/wEJAAUKBggEB/////////8KBgUBCQcBBwMHCQT/////BgECBgUBBAcI/////////wECBQUCBgMA
BAMEB/////8IBAcJAAUABgUAAgb/////BwMJBwkEAwIJBQkGAgYJ/wMLAgcIBAoGBf////////8F
CgYEBwIEAgACBwv/////AAEJBAcIAgMLBQoG/////wkCAQkLAgkECwcLBAUKBv8IBAcDCwUDBQEF
Cwb/////BQELBQsGAQALBwsEAAQL/wAFCQAGBQADBgsGAwgEB/8GBQkGCQsEBwkHCwn/////CgQJ
BgQK/////////////wQKBgQJCgAIA/////////8KAAEKBgAGBAD/////////CAMBCAEGCAYEBgEK
/////wEECQECBAIGBP////////8DAAgBAgkCBAkCBgT/////AAIEBAIG/////////////wgDAggC
BAQCBv////////8KBAkKBgQLAgP/////////AAgCAggLBAkKBAoG/////wMLAgABBgAGBAYBCv//
//8GBAEGAQoECAECAQsICwH/CQYECQMGCQEDCwYD/////wgLAQgBAAsGAQkBBAYEAf8DCwYDBgAA
BgT/////////BgQICwYI/////////////wcKBgcICggJCv////////8ABwMACgcACQoGBwr/////
CgYHAQoHAQcIAQgA/////woGBwoHAQEHA/////////8BAgYBBggBCAkIBgf/////AgYJAgkBBgcJ
AAkDBwMJ/wcIAAcABgYAAv////////8HAwIGBwL/////////////AgMLCgYICggJCAYH/////wIA
BwIHCwAJBwYHCgkKB/8BCAABBwgBCgcGBwoCAwv/CwIBCwEHCgYBBgcB/////wgJBggGBwkBBgsG
AwEDBv8ACQELBgf/////////////BwgABwAGAwsACwYA/////wcLBv////////////////8HBgv/
////////////////AwAICwcG/////////////wABCQsHBv////////////8IAQkIAwELBwb/////
////CgECBgsH/////////////wECCgMACAYLB/////////8CCQACCgkGCwf/////////BgsHAgoD
CggDCgkI/////wcCAwYCB/////////////8HAAgHBgAGAgD/////////AgcGAgMHAAEJ////////
/wEGAgEIBgEJCAgHBv////8KBwYKAQcBAwf/////////CgcGAQcKAQgHAQAI/////wADBwAHCgAK
CQYKB/////8HBgoHCggICgn/////////BggECwgG/////////////wMGCwMABgAEBv////////8I
BgsIBAYJAAH/////////CQQGCQYDCQMBCwMG/////wYIBAYLCAIKAf////////8BAgoDAAsABgsA
BAb/////BAsIBAYLAAIJAgoJ/////woJAwoDAgkEAwsDBgQGA/8IAgMIBAIEBgL/////////AAQC
BAYC/////////////wEJAAIDBAIEBgQDCP////8BCQQBBAICBAb/////////CAEDCAYBCAQGBgoB
/////woBAAoABgYABP////////8EBgMEAwgGCgMAAwkKCQP/CgkEBgoE/////////////wQJBQcG
C/////////////8ACAMECQULBwb/////////BQABBQQABwYL/////////wsHBggDBAMFBAMBBf//
//8JBQQKAQIHBgv/////////BgsHAQIKAAgDBAkF/////wcGCwUECgQCCgQAAv////8DBAgDBQQD
AgUKBQILBwb/BwIDBwYCBQQJ/////////wkFBAAIBgAGAgYIB/////8DBgIDBwYBBQAFBAD/////
BgIIBggHAgEIBAgFAQUI/wkFBAoBBgEHBgEDB/////8BBgoBBwYBAAcIBwAJBQT/BAAKBAoFAAMK
BgoHAwcK/wcGCgcKCAUECgQICv////8GCQUGCwkLCAn/////////AwYLAAYDAAUGAAkF/////wAL
CAAFCwABBQUGC/////8GCwMGAwUFAwH/////////AQIKCQULCQsICwUG/////wALAwAGCwAJBgUG
CQECCv8LCAULBQYIAAUKBQIAAgX/BgsDBgMFAgoDCgUD/////wUICQUCCAUGAgMIAv////8JBQYJ
BgAABgL/////////AQUIAQgABQYIAwgCBgII/wEFBgIBBv////////////8BAwYBBgoDCAYFBgkI
CQb/CgEACgAGCQUABQYA/////wADCAUGCv////////////8KBQb/////////////////CwUKBwUL
/////////////wsFCgsHBQgDAP////////8FCwcFCgsBCQD/////////CgcFCgsHCQgBCAMB////
/wsBAgsHAQcFAf////////8ACAMBAgcBBwUHAgv/////CQcFCQIHCQACAgsH/////wcFAgcCCwUJ
AgMCCAkIAv8CBQoCAwUDBwX/////////CAIACAUCCAcFCgIF/////wkAAQUKAwUDBwMKAv////8J
CAIJAgEIBwIKAgUHBQL/AQMFAwcF/////////////wAIBwAHAQEHBf////////8JAAMJAwUFAwf/
////////CQgHBQkH/////////////wUIBAUKCAoLCP////////8FAAQFCwAFCgsLAwD/////AAEJ
CAQKCAoLCgQF/////woLBAoEBQsDBAkEAQMBBP8CBQECCAUCCwgEBQj/////AAQLAAsDBAULAgsB
BQEL/wACBQAFCQILBQQFCAsIBf8JBAUCCwP/////////////AgUKAwUCAwQFAwgE/////wUKAgUC
BAQCAP////////8DCgIDBQoDCAUEBQgAAQn/BQoCBQIEAQkCCQQC/////wgEBQgFAwMFAf//////
//8ABAUBAAX/////////////CAQFCAUDCQAFAAMF/////wkEBf////////////////8ECwcECQsJ
Cgv/////////AAgDBAkHCQsHCQoL/////wEKCwELBAEEAAcEC/////8DAQQDBAgBCgQHBAsKCwT/
BAsHCQsECQILCQEC/////wkHBAkLBwkBCwILAQAIA/8LBwQLBAICBAD/////////CwcECwQCCAME
AwIE/////wIJCgIHCQIDBwcECf////8JCgcJBwQKAgcIBwACAAf/AwcKAwoCBwQKAQoABAAK/wEK
AggHBP////////////8ECQEEAQcHAQP/////////BAkBBAEHAAgBCAcB/////wQAAwcEA///////
//////8ECAf/////////////////CQoICgsI/////////////wMACQMJCwsJCv////////8AAQoA
CggICgv/////////AwEKCwMK/////////////wECCwELCQkLCP////////8DAAkDCQsBAgkCCwn/
////AAILCAAL/////////////wMCC/////////////////8CAwgCCAoKCAn/////////CQoCAAkC
/////////////wIDCAIICgABCAEKCP////8BCgL/////////////////AQMICQEI////////////
/wAJAf////////////////8AAwj//////////////////////////////////////w==
""",
)

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


_KERNEL_CODE = r"""
extern "C" __device__ inline int node_index(int i, int j, int k, int ny, int nz) {
    return (i * ny + j) * nz + k;
}

extern "C" __device__ inline float vol_at(
    const float* volume, int i, int j, int k, int ny, int nz) {
    return volume[node_index(i, j, k, ny, nz)];
}

extern "C" __device__ inline bool crosses(float a, float b, float level) {
    return (a < level && b >= level) || (a >= level && b < level);
}

extern "C" __device__ inline bool crosses_lewiner(float a, float b, float level) {
    return (a <= level && b > level) || (a > level && b <= level);
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

extern "C" __global__ void mc_classify_edges(
    const float* volume, int* edge_flags, int nx, int ny, int nz, float level) {
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
        flag = crosses(a, b, level) ? 1 : 0;
    }
    edge_flags[idx] = flag;
}

extern "C" __global__ void mc_lewiner_classify_edges(
    const float* volume, int* edge_flags, int nx, int ny, int nz, float level) {
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

extern "C" __global__ void mc_classify_edges_masked(
    const float* volume, const bool* mask, int* edge_flags,
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
        if (crosses(a, b, level) &&
            mc_edge_has_active_masked_cell(mask, i, j, k, side, nx, ny, nz)) {
            flag = 1;
        }
    }
    edge_flags[idx] = flag;
}

extern "C" __global__ void mc_lewiner_classify_edges_masked(
    const float* volume, const bool* mask, int* edge_flags,
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

extern "C" __global__ void mc_generate_vertices(
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
    if (!crosses(a, b, level)) {
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

extern "C" __global__ void mc_count_faces(
    const float* volume, const signed char* tri_table, int* tri_counts,
    int nx, int ny, int nz, float level) {
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

    int code = 0;
    code |= (vol_at(volume, i, j, k, ny, nz) >= level) ? 1 : 0;
    code |= (vol_at(volume, i + 1, j, k, ny, nz) >= level) ? 2 : 0;
    code |= (vol_at(volume, i + 1, j + 1, k, ny, nz) >= level) ? 4 : 0;
    code |= (vol_at(volume, i, j + 1, k, ny, nz) >= level) ? 8 : 0;
    code |= (vol_at(volume, i, j, k + 1, ny, nz) >= level) ? 16 : 0;
    code |= (vol_at(volume, i + 1, j, k + 1, ny, nz) >= level) ? 32 : 0;
    code |= (vol_at(volume, i + 1, j + 1, k + 1, ny, nz) >= level) ? 64 : 0;
    code |= (vol_at(volume, i, j + 1, k + 1, ny, nz) >= level) ? 128 : 0;

    int count = 0;
    const signed char* row = tri_table + code * 16;
    for (int n = 0; n < 15; n += 3) {
        if (row[n] < 0) {
            break;
        }
        count++;
    }
    tri_counts[idx] = count;
}

extern "C" __global__ void mc_count_faces_masked(
    const float* volume, const bool* mask, const signed char* tri_table,
    int* tri_counts, int nx, int ny, int nz, float level) {
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
        return;
    }

    int code = 0;
    code |= (vol_at(volume, i, j, k, ny, nz) >= level) ? 1 : 0;
    code |= (vol_at(volume, i + 1, j, k, ny, nz) >= level) ? 2 : 0;
    code |= (vol_at(volume, i + 1, j + 1, k, ny, nz) >= level) ? 4 : 0;
    code |= (vol_at(volume, i, j + 1, k, ny, nz) >= level) ? 8 : 0;
    code |= (vol_at(volume, i, j, k + 1, ny, nz) >= level) ? 16 : 0;
    code |= (vol_at(volume, i + 1, j, k + 1, ny, nz) >= level) ? 32 : 0;
    code |= (vol_at(volume, i + 1, j + 1, k + 1, ny, nz) >= level) ? 64 : 0;
    code |= (vol_at(volume, i, j + 1, k + 1, ny, nz) >= level) ? 128 : 0;

    int count = 0;
    const signed char* row = tri_table + code * 16;
    for (int n = 0; n < 15; n += 3) {
        if (row[n] < 0) {
            break;
        }
        count++;
    }
    tri_counts[idx] = count;
}

extern "C" __device__ inline int edge_vid(
    const int* edge_vertex_ids, int i, int j, int k, int side, int ny, int nz) {
    return edge_vertex_ids[(node_index(i, j, k, ny, nz) * 3) + side];
}

extern "C" __device__ inline int local_edge_vid(
    const int* edge_vertex_ids, int edge, int i, int j, int k, int ny, int nz) {
    switch (edge) {
        case 0: return edge_vid(edge_vertex_ids, i, j, k, 0, ny, nz);
        case 1: return edge_vid(edge_vertex_ids, i + 1, j, k, 1, ny, nz);
        case 2: return edge_vid(edge_vertex_ids, i, j + 1, k, 0, ny, nz);
        case 3: return edge_vid(edge_vertex_ids, i, j, k, 1, ny, nz);
        case 4: return edge_vid(edge_vertex_ids, i, j, k + 1, 0, ny, nz);
        case 5: return edge_vid(edge_vertex_ids, i + 1, j, k + 1, 1, ny, nz);
        case 6: return edge_vid(edge_vertex_ids, i, j + 1, k + 1, 0, ny, nz);
        case 7: return edge_vid(edge_vertex_ids, i, j, k + 1, 1, ny, nz);
        case 8: return edge_vid(edge_vertex_ids, i, j, k, 2, ny, nz);
        case 9: return edge_vid(edge_vertex_ids, i + 1, j, k, 2, ny, nz);
        case 10: return edge_vid(edge_vertex_ids, i + 1, j + 1, k, 2, ny, nz);
        default: return edge_vid(edge_vertex_ids, i, j + 1, k, 2, ny, nz);
    }
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

extern "C" __global__ void mc_lewiner_count_cells(
    const float* volume, const signed char* cases,
    const signed char* test3, const signed char* test4,
    const signed char* test6, const signed char* test7,
    const signed char* test10, const signed char* test12,
    const signed char* test13, const signed char* subconfig13,
    const signed char* tiling13_5_1,
    int* tri_counts, int* center_flags,
    int nx, int ny, int nz, float level) {
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
    int config = (int)cases[code * 2 + 1];
    int tri_count = 0;
    int center_flag = 0;
    if (case_id > 0) {
        lewiner_select_count_center(
            v, case_id, config, test3, test4, test6, test7, test10,
            test12, test13, subconfig13, tiling13_5_1,
            &tri_count, &center_flag);
    }
    tri_counts[idx] = tri_count;
    center_flags[idx] = center_flag;
}

extern "C" __global__ void mc_lewiner_count_cells_masked(
    const float* volume, const bool* mask, const signed char* cases,
    const signed char* test3, const signed char* test4,
    const signed char* test6, const signed char* test7,
    const signed char* test10, const signed char* test12,
    const signed char* test13, const signed char* subconfig13,
    const signed char* tiling13_5_1,
    int* tri_counts, int* center_flags,
    int nx, int ny, int nz, float level) {
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
        return;
    }

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
    int config = (int)cases[code * 2 + 1];
    int tri_count = 0;
    int center_flag = 0;
    if (case_id > 0) {
        lewiner_select_count_center(
            v, case_id, config, test3, test4, test6, test7, test10,
            test12, test13, subconfig13, tiling13_5_1,
            &tri_count, &center_flag);
    }
    tri_counts[idx] = tri_count;
    center_flags[idx] = center_flag;
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

extern "C" __device__ inline void load_lewiner_values(
    const float* volume, int i, int j, int k, int ny, int nz, float level,
    double* v) {
    double local[8];
    local[0] = (double)vol_at(volume, i, j, k, ny, nz) - (double)level;
    local[1] = (double)vol_at(volume, i + 1, j, k, ny, nz) - (double)level;
    local[2] = (double)vol_at(volume, i + 1, j + 1, k, ny, nz) - (double)level;
    local[3] = (double)vol_at(volume, i, j + 1, k, ny, nz) - (double)level;
    local[4] = (double)vol_at(volume, i, j, k + 1, ny, nz) - (double)level;
    local[5] = (double)vol_at(volume, i + 1, j, k + 1, ny, nz) - (double)level;
    local[6] = (double)vol_at(volume, i + 1, j + 1, k + 1, ny, nz) - (double)level;
    local[7] = (double)vol_at(volume, i, j + 1, k + 1, ny, nz) - (double)level;

    v[0] = local[0];
    v[1] = local[4];
    v[2] = local[7];
    v[3] = local[3];
    v[4] = local[1];
    v[5] = local[5];
    v[6] = local[6];
    v[7] = local[2];
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

extern "C" __global__ void mc_generate_faces(
    const float* volume, const signed char* tri_table, const int* tri_counts,
    const int* tri_scan, const int* edge_vertex_ids, int* faces,
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

    int code = 0;
    code |= (vol_at(volume, i, j, k, ny, nz) >= level) ? 1 : 0;
    code |= (vol_at(volume, i + 1, j, k, ny, nz) >= level) ? 2 : 0;
    code |= (vol_at(volume, i + 1, j + 1, k, ny, nz) >= level) ? 4 : 0;
    code |= (vol_at(volume, i, j + 1, k, ny, nz) >= level) ? 8 : 0;
    code |= (vol_at(volume, i, j, k + 1, ny, nz) >= level) ? 16 : 0;
    code |= (vol_at(volume, i + 1, j, k + 1, ny, nz) >= level) ? 32 : 0;
    code |= (vol_at(volume, i + 1, j + 1, k + 1, ny, nz) >= level) ? 64 : 0;
    code |= (vol_at(volume, i, j + 1, k + 1, ny, nz) >= level) ? 128 : 0;

    int face_start = tri_scan[idx] - tri_counts[idx];
    const signed char* row = tri_table + code * 16;
    for (int n = 0; n < 15; n += 3) {
        if (row[n] < 0) {
            break;
        }
        int v0 = local_edge_vid(edge_vertex_ids, row[n], i, j, k, ny, nz);
        int v1 = local_edge_vid(edge_vertex_ids, row[n + 1], i, j, k, ny, nz);
        int v2 = local_edge_vid(edge_vertex_ids, row[n + 2], i, j, k, ny, nz);
        int out = 3 * face_start;
        if (flip_winding) {
            faces[out] = v2;
            faces[out + 1] = v1;
            faces[out + 2] = v0;
        } else {
            faces[out] = v0;
            faces[out + 1] = v1;
            faces[out + 2] = v2;
        }
        face_start++;
    }
}

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
    int* parent, int* faces_ok) {
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

    faces_ok[idx] = !(eq01 || eq02 || eq12);
    if (eq01) {
        mc_union_vertices(parent, i0, i1);
    }
    if (eq02) {
        mc_union_vertices(parent, i0, i2);
    }
    if (eq12) {
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
    const int* faces, const int* faces_ok, const int* face_scan,
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


def marching_cubes(
    volume,
    level=None,
    *,
    spacing=(1.0, 1.0, 1.0),
    gradient_direction="descent",
    step_size=1,
    allow_degenerate=True,
    method="lewiner",
    mask=None,
):
    """Marching cubes algorithm to find surfaces in 3D volumetric data.

    The ``method='lewiner'`` default and ``method='lorensen'`` are supported
    for CuPy inputs.

    When ``allow_degenerate=False``, zero-area faces are removed and the mesh
    geometry is preserved, but the exact surviving duplicate vertex IDs can
    differ from scikit-image's ordered CPU post-processing.
    """
    if method not in ("lewiner", "lorensen"):
        raise ValueError("method should be either 'lewiner' or 'lorensen'")

    if method == "lewiner":
        return _marching_cubes_lewiner(
            volume,
            level,
            spacing,
            gradient_direction,
            step_size,
            allow_degenerate,
            mask,
        )

    return _marching_cubes_lorensen(
        volume,
        level,
        spacing,
        gradient_direction,
        step_size,
        allow_degenerate,
        mask,
    )


def _marching_cubes_lorensen(
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
    return _run_lorensen(
        volume, level, spacing, gradient_direction, allow_degenerate, mask
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


def _run_lorensen(
    volume, level, spacing, gradient_direction, allow_degenerate, mask
):
    nx, ny, nz = volume.shape
    n_edges = nx * ny * nz * 3
    threads = 256
    edge_blocks = ((n_edges + threads - 1) // threads,)

    edge_flags = cp.empty(n_edges, dtype=cp.int32)
    if mask is None:
        _get_kernel("mc_classify_edges")(
            edge_blocks,
            (threads,),
            (volume, edge_flags, nx, ny, nz, np.float32(level)),
        )
    else:
        _get_kernel("mc_classify_edges_masked")(
            edge_blocks,
            (threads,),
            (volume, mask, edge_flags, nx, ny, nz, np.float32(level)),
        )
    edge_scan = cp.cumsum(edge_flags, dtype=cp.int32)
    n_vertices = int(edge_scan[-1])
    if n_vertices == 0:
        raise RuntimeError("No surface found at the given iso value.")

    edge_vertex_ids = cp.empty(n_edges, dtype=cp.int32)
    vertices = cp.empty((n_vertices, 3), dtype=cp.float32)
    normals = cp.empty((n_vertices, 3), dtype=cp.float32)
    values = cp.empty(n_vertices, dtype=cp.float32)
    _get_kernel("mc_generate_vertices")(
        edge_blocks,
        (threads,),
        (
            volume,
            edge_scan,
            edge_vertex_ids,
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

    n_cells = (nx - 1) * (ny - 1) * (nz - 1)
    cell_blocks = ((n_cells + threads - 1) // threads,)
    tri_counts = cp.empty(n_cells, dtype=cp.int32)
    tri_table = _get_tri_table()
    if mask is None:
        _get_kernel("mc_count_faces")(
            cell_blocks,
            (threads,),
            (volume, tri_table, tri_counts, nx, ny, nz, np.float32(level)),
        )
    else:
        _get_kernel("mc_count_faces_masked")(
            cell_blocks,
            (threads,),
            (
                volume,
                mask,
                tri_table,
                tri_counts,
                nx,
                ny,
                nz,
                np.float32(level),
            ),
        )
    tri_scan = cp.cumsum(tri_counts, dtype=cp.int32)
    n_faces = int(tri_scan[-1])
    if n_faces == 0:
        raise RuntimeError("No surface found at the given iso value.")

    faces = cp.empty(n_faces * 3, dtype=cp.int32)
    _get_kernel("mc_generate_faces")(
        cell_blocks,
        (threads,),
        (
            volume,
            tri_table,
            tri_counts,
            tri_scan,
            edge_vertex_ids,
            faces,
            nx,
            ny,
            nz,
            np.float32(level),
            np.int32(gradient_direction == "descent"),
        ),
    )
    faces = faces.reshape(-1, 3)
    if not allow_degenerate:
        vertices, faces, normals, values = _remove_degenerate_faces_gpu(
            vertices, faces, normals, values
        )
    return vertices, faces, normals, values


def _run_lewiner(
    volume, level, spacing, gradient_direction, allow_degenerate, mask
):
    nx, ny, nz = volume.shape
    n_edges = nx * ny * nz * 3
    threads = 256
    edge_blocks = ((n_edges + threads - 1) // threads,)

    edge_flags = cp.empty(n_edges, dtype=cp.int32)
    if mask is None:
        _get_kernel("mc_lewiner_classify_edges")(
            edge_blocks,
            (threads,),
            (volume, edge_flags, nx, ny, nz, np.float32(level)),
        )
    else:
        _get_kernel("mc_lewiner_classify_edges_masked")(
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
    _get_kernel("mc_lewiner_generate_vertices")(
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

    tri_counts, center_flags = _lewiner_count_cells_gpu(volume, level, mask)
    edge_ids = _lewiner_generate_edge_ids_gpu(volume, level, tri_counts)
    if edge_ids.size == 0:
        raise RuntimeError("No surface found at the given iso value.")

    vertices, normals, values, center_vertex_ids = (
        _lewiner_generate_center_vertices_gpu(
            volume,
            level,
            spacing,
            center_flags,
            vertex_offset=n_edge_vertices,
        )
    )
    vertices[:n_edge_vertices] = edge_vertices
    normals[:n_edge_vertices] = edge_normals
    values[:n_edge_vertices] = edge_values

    faces = _lewiner_generate_faces_from_edge_ids_gpu(
        edge_ids,
        tri_counts,
        edge_vertex_ids,
        center_vertex_ids,
        volume.shape,
        gradient_direction,
    )
    if not allow_degenerate:
        vertices, faces, normals, values = _remove_degenerate_faces_gpu(
            vertices, faces, normals, values
        )
    return vertices, faces, normals, values


def _remove_degenerate_faces_gpu(vertices, faces, normals, values):
    n_vertices = vertices.shape[0]
    n_faces = faces.shape[0]
    if n_faces == 0:
        return vertices, faces, normals, values

    vertex_map = cp.empty(n_vertices, dtype=cp.int32)
    faces_ok = cp.empty(n_faces, dtype=cp.int32)
    threads = 256
    face_blocks = ((n_faces + threads - 1) // threads,)
    vertex_blocks = ((n_vertices + threads - 1) // threads,)
    _get_kernel("mc_init_vertex_map")(
        vertex_blocks,
        (threads,),
        (vertex_map, n_vertices),
    )
    _get_kernel("mc_mark_degenerate_faces_parallel")(
        face_blocks,
        (threads,),
        (vertices, faces, n_faces, vertex_map, faces_ok),
    )
    _get_kernel("mc_compress_vertex_roots")(
        vertex_blocks,
        (threads,),
        (vertex_map, n_vertices),
    )

    root_flags = cp.empty(n_vertices, dtype=cp.int32)
    _get_kernel("mc_mark_vertex_roots")(
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
    _get_kernel("mc_emit_non_degenerate_faces")(
        face_blocks,
        (threads,),
        (faces, faces_ok, face_scan, vertex_map, vertex_scan, n_faces, faces2),
    )
    _get_kernel("mc_compact_root_vertices")(
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


def _lewiner_count_cells_gpu(volume, level, mask=None):
    nx, ny, nz = volume.shape
    n_cells = (nx - 1) * (ny - 1) * (nz - 1)
    threads = 256
    cell_blocks = ((n_cells + threads - 1) // threads,)
    tri_counts = cp.empty(n_cells, dtype=cp.int32)
    center_flags = cp.empty(n_cells, dtype=cp.int32)
    luts = _get_lewiner_luts_device()
    if mask is None:
        _get_kernel("mc_lewiner_count_cells")(
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
                luts["tiling13_5_1"],
                tri_counts,
                center_flags,
                nx,
                ny,
                nz,
                np.float32(level),
            ),
        )
    else:
        _get_kernel("mc_lewiner_count_cells_masked")(
            cell_blocks,
            (threads,),
            (
                volume,
                mask,
                luts["cases"],
                luts["test3"],
                luts["test4"],
                luts["test6"],
                luts["test7"],
                luts["test10"],
                luts["test12"],
                luts["test13"],
                luts["subconfig13"],
                luts["tiling13_5_1"],
                tri_counts,
                center_flags,
                nx,
                ny,
                nz,
                np.float32(level),
            ),
        )
    return tri_counts, center_flags


def _lewiner_generate_center_vertices_gpu(
    volume, level, spacing, center_flags, vertex_offset=0
):
    nx, ny, nz = volume.shape
    n_cells = (nx - 1) * (ny - 1) * (nz - 1)
    center_scan = cp.cumsum(center_flags, dtype=cp.int32)
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
    _get_kernel("mc_lewiner_generate_center_vertices")(
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
    _get_kernel("mc_lewiner_generate_edge_ids")(
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
    _get_kernel("mc_lewiner_generate_faces_from_edge_ids")(
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


def _decode_cases_classic():
    shape, text = _CASES_CLASSIC
    return _decode_lut(shape, text)


def _decode_lut(shape, text):
    byts = base64.decodebytes(text.encode("utf-8"))
    return np.frombuffer(byts, dtype=np.int8).reshape(shape)


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
def _get_tri_table():
    return cp.asarray(_decode_cases_classic())


@cp.memoize(for_each_device=True)
def _get_lewiner_luts_device():
    return {
        name.lower(): cp.asarray(_decode_named_lut(name))
        for name in _LEWINER_LUT_NAMES
    }


@cp.memoize(for_each_device=True)
def _get_kernel(name):
    return cp.RawKernel(_KERNEL_CODE, name)
