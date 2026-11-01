# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2012-2015, P. M. Neila
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import base64

import cupy as cp
import numpy as np

from ._marching_cubes_common import (
    _COMMON_KERNEL_CODE,
    _apply_step_size,
    _check_int32_range,
    _decode_lut,
    _remove_degenerate_faces_gpu,
    _validate_marching_cubes_inputs,
)

# Implementation notes
# --------------------
# This is a CuPy RawKernel implementation of classic Lorensen marching cubes.
# See _marching_cubes_common.py for the code shared with the Lewiner
# implementation and for the notes on the design and its references.
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


def _decode_classic_lut_for_cuda_constants():
    shape, text = _CASES_CLASSIC
    byts = base64.decodebytes(text.encode("utf-8"))
    return np.frombuffer(byts, dtype=np.int8).reshape(shape)


def _decode_cases_classic():
    shape, text = _CASES_CLASSIC
    return _decode_lut(shape, text)


@cp.memoize(for_each_device=True)
def _get_tri_table():
    return cp.asarray(_decode_cases_classic())


def _generate_lut_constants_code():
    classic_values = _decode_classic_lut_for_cuda_constants().ravel()
    classic_values_text = ", ".join(str(int(value)) for value in classic_values)
    lines = [
        f"__constant__ signed char lut_cases_classic[{classic_values.size}] = "
        f"{{{classic_values_text}}};"
    ]
    return "\n".join(lines)


# code specific to method = 'Lorensen'
_LORENSEN_KERNEL_CODE = (
    _generate_lut_constants_code()
    + _COMMON_KERNEL_CODE
    + r"""

extern "C" __device__ inline bool crosses(float a, float b, float level) {
    return (a < level && b >= level) || (a >= level && b < level);
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
    if (i + 1 < nx && crosses(a, vol_at(volume, i + 1, j, k, ny, nz), level)) {
        bits |= 1;
    }
    if (j + 1 < ny && crosses(a, vol_at(volume, i, j + 1, k, ny, nz), level)) {
        bits |= 2;
    }
    if (k + 1 < nz && crosses(a, vol_at(volume, i, j, k + 1, ny, nz), level)) {
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
    if (i + 1 < nx && crosses(a, vol_at(volume, i + 1, j, k, ny, nz), level) &&
        mc_edge_has_active_masked_cell(mask, i, j, k, 0, nx, ny, nz)) {
        bits |= 1;
    }
    if (j + 1 < ny && crosses(a, vol_at(volume, i, j + 1, k, ny, nz), level) &&
        mc_edge_has_active_masked_cell(mask, i, j, k, 1, nx, ny, nz)) {
        bits |= 2;
    }
    if (k + 1 < nz && crosses(a, vol_at(volume, i, j, k + 1, ny, nz), level) &&
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

extern "C" __global__ void mc_count_faces(
    const float* volume, unsigned char* tri_counts, int nx, int ny, int nz, float level) {
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
    const signed char* row = lut_cases_classic + code * 16;
    for (int n = 0; n < 15; n += 3) {
        if (row[n] < 0) {
            break;
        }
        count++;
    }
    tri_counts[idx] = (unsigned char)count;
}

extern "C" __global__ void mc_count_faces_masked(
    const float* volume, const bool* mask, unsigned char* tri_counts,
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
    const signed char* row = lut_cases_classic + code * 16;
    for (int n = 0; n < 15; n += 3) {
        if (row[n] < 0) {
            break;
        }
        count++;
    }
    tri_counts[idx] = (unsigned char)count;
}

extern "C" __global__ void mc_generate_faces(
    const float* volume, const unsigned char* tri_counts, const int* tri_scan,
    const int* node_scan, const unsigned char* edge_mask, int* faces,
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
    const signed char* row = lut_cases_classic + code * 16;
    for (int n = 0; n < 15; n += 3) {
        if (row[n] < 0) {
            break;
        }
        int v0 = local_edge_vid(node_scan, edge_mask, row[n], i, j, k, ny, nz);
        int v1 = local_edge_vid(node_scan, edge_mask, row[n + 1], i, j, k, ny, nz);
        int v2 = local_edge_vid(node_scan, edge_mask, row[n + 2], i, j, k, ny, nz);
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
)


@cp.memoize(for_each_device=True)
def _get_lorensen_kernel(name):
    return cp.RawKernel(_LORENSEN_KERNEL_CODE, name)


def _run_lorensen(
    volume, level, spacing, gradient_direction, allow_degenerate, mask
):
    nx, ny, nz = volume.shape
    n_edges = nx * ny * nz * 3
    _check_int32_range(n_edges, "the edge array")
    threads = 256

    n_nodes = nx * ny * nz
    node_blocks = ((n_nodes + threads - 1) // threads,)
    edge_mask = cp.empty(n_nodes, dtype=cp.uint8)
    edge_count = cp.empty(n_nodes, dtype=cp.uint8)
    if mask is None:
        _get_lorensen_kernel("mc_classify_edges")(
            node_blocks,
            (threads,),
            (volume, edge_mask, edge_count, nx, ny, nz, np.float32(level)),
        )
    else:
        _get_lorensen_kernel("mc_classify_edges_masked")(
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
    node_scan = cp.cumsum(edge_count, dtype=cp.int32)
    n_vertices = int(node_scan[-1])
    if n_vertices == 0:
        raise RuntimeError("No surface found at the given iso value.")
    _check_int32_range(3 * n_vertices, "the vertex array")

    edge_idx = cp.empty(n_vertices, dtype=cp.int32)
    _get_lorensen_kernel("mc_compact_edges")(
        node_blocks,
        (threads,),
        (edge_mask, node_scan, edge_idx, n_nodes),
    )
    vertex_blocks = ((n_vertices + threads - 1) // threads,)
    vertices = cp.empty((n_vertices, 3), dtype=cp.float32)
    normals = cp.empty((n_vertices, 3), dtype=cp.float32)
    values = cp.empty(n_vertices, dtype=cp.float32)
    _get_lorensen_kernel("mc_generate_vertices")(
        vertex_blocks,
        (threads,),
        (
            volume,
            edge_idx,
            n_vertices,
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
    tri_counts = cp.empty(n_cells, dtype=cp.uint8)
    if mask is None:
        _get_lorensen_kernel("mc_count_faces")(
            cell_blocks,
            (threads,),
            (volume, tri_counts, nx, ny, nz, np.float32(level)),
        )
    else:
        _get_lorensen_kernel("mc_count_faces_masked")(
            cell_blocks,
            (threads,),
            (
                volume,
                mask,
                tri_counts,
                nx,
                ny,
                nz,
                np.float32(level),
            ),
        )
    # a 64-bit total, so that a too large mesh is detected before the 32-bit
    # scan can wrap around
    n_faces = int(tri_counts.sum(dtype=cp.int64))
    _check_int32_range(3 * n_faces, "the face array")
    tri_scan = cp.cumsum(tri_counts, dtype=cp.int32)
    if n_faces == 0:
        raise RuntimeError("No surface found at the given iso value.")

    faces = cp.empty(n_faces * 3, dtype=cp.int32)
    _get_lorensen_kernel("mc_generate_faces")(
        cell_blocks,
        (threads,),
        (
            volume,
            tri_counts,
            tri_scan,
            node_scan,
            edge_mask,
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
