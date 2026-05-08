# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2012-2015, P. M. Neila
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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


def _decode_classic_lut_for_cuda_constants():
    shape, text = _CASES_CLASSIC
    byts = base64.decodebytes(text.encode("utf-8"))
    return np.frombuffer(byts, dtype=np.int8).reshape(shape)


def _decode_lut(shape, text):
    byts = base64.decodebytes(text.encode("utf-8"))
    return np.frombuffer(byts, dtype=np.int8).reshape(shape)


def _decode_cases_classic():
    shape, text = _CASES_CLASSIC
    return _decode_lut(shape, text)


@cp.memoize(for_each_device=True)
def _get_tri_table():
    return cp.asarray(_decode_cases_classic())


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


def _generate_lut_constants_code():
    classic_values = _decode_classic_lut_for_cuda_constants().ravel()
    classic_values_text = ", ".join(str(int(value)) for value in classic_values)
    lines = [
        f"__constant__ signed char lut_cases_classic[{classic_values.size}] = "
        f"{{{classic_values_text}}};"
    ]
    return "\n".join(lines)


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

"""

# code specific to method = 'Lorensen'
_LORENSEN_KERNEL_CODE = (
    _generate_lut_constants_code()
    + _COMMON_KERNEL_CODE
    + r"""

extern "C" __device__ inline bool crosses(float a, float b, float level) {
    return (a < level && b >= level) || (a >= level && b < level);
}

extern "C" __global__ void mc_classify_edges(
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
        flag = crosses(a, b, level) ? 1 : 0;
    }
    edge_flags[idx] = flag;
}

extern "C" __global__ void mc_classify_edges_masked(
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
        if (crosses(a, b, level) &&
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

extern "C" __global__ void mc_count_faces(
    const float* volume, int* tri_counts, int nx, int ny, int nz, float level) {
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
    tri_counts[idx] = count;
}

extern "C" __global__ void mc_count_faces_masked(
    const float* volume, const bool* mask, int* tri_counts,
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
    tri_counts[idx] = count;
}

extern "C" __global__ void mc_generate_faces(
    const float* volume, const int* tri_counts, const int* tri_scan,
    const int* edge_vertex_ids, int* faces,
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
)


@cp.memoize(for_each_device=True)
def _get_lorensen_kernel(name):
    return cp.RawKernel(_LORENSEN_KERNEL_CODE, name)


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


def _run_lorensen(
    volume, level, spacing, gradient_direction, allow_degenerate, mask
):
    nx, ny, nz = volume.shape
    n_edges = nx * ny * nz * 3
    threads = 256
    edge_blocks = ((n_edges + threads - 1) // threads,)

    edge_flags = cp.empty(n_edges, dtype=cp.uint8)
    if mask is None:
        _get_lorensen_kernel("mc_classify_edges")(
            edge_blocks,
            (threads,),
            (volume, edge_flags, nx, ny, nz, np.float32(level)),
        )
    else:
        _get_lorensen_kernel("mc_classify_edges_masked")(
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
    _get_lorensen_kernel("mc_generate_vertices")(
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
    tri_scan = cp.cumsum(tri_counts, dtype=cp.int32)
    n_faces = int(tri_scan[-1])
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
