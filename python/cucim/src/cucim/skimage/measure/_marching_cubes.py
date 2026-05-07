# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2012-2015, P. M. Neila
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import base64

import cupy as cp
import numpy as np

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

    This initial cuCIM implementation supports only ``method='lorensen'``.
    The scikit-image default ``method='lewiner'`` raises
    ``NotImplementedError`` until a GPU Lewiner implementation is available.
    """
    if method == "lewiner":
        raise NotImplementedError(
            "method='lewiner' is not implemented in cuCIM yet; "
            "use method='lorensen'."
        )
    if method != "lorensen":
        raise ValueError("method should be either 'lewiner' or 'lorensen'")

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
    if step_size != 1:
        raise NotImplementedError("step_size > 1 is not implemented yet.")

    if gradient_direction not in ("descent", "ascent"):
        raise ValueError(
            f"Incorrect input {gradient_direction} in `gradient_direction`, "
            "see docstring."
        )

    if not allow_degenerate:
        raise NotImplementedError(
            "allow_degenerate=False is not implemented yet."
        )

    if mask is not None:
        if not isinstance(mask, cp.ndarray) or mask.shape != volume.shape:
            raise ValueError("volume and mask must have the same shape.")
        raise NotImplementedError("mask is not implemented yet.")

    return _run_lorensen(volume, level, spacing, gradient_direction)


def _run_lorensen(volume, level, spacing, gradient_direction):
    nx, ny, nz = volume.shape
    n_edges = nx * ny * nz * 3
    threads = 256
    edge_blocks = ((n_edges + threads - 1) // threads,)

    edge_flags = cp.empty(n_edges, dtype=cp.int32)
    _get_kernel("mc_classify_edges")(
        edge_blocks,
        (threads,),
        (volume, edge_flags, nx, ny, nz, np.float32(level)),
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
    _get_kernel("mc_count_faces")(
        cell_blocks,
        (threads,),
        (volume, tri_table, tri_counts, nx, ny, nz, np.float32(level)),
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
    return vertices, faces.reshape(-1, 3), normals, values


def _decode_cases_classic():
    shape, text = _CASES_CLASSIC
    byts = base64.decodebytes(text.encode("utf-8"))
    return np.frombuffer(byts, dtype=np.int8).reshape(shape)


@cp.memoize(for_each_device=True)
def _get_tri_table():
    return cp.asarray(_decode_cases_classic())


@cp.memoize(for_each_device=True)
def _get_kernel(name):
    return cp.RawKernel(_KERNEL_CODE, name)
