# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import cupy as cp
import numpy as np

from ._nl_means_kernels import (
    _get_direct_kernel,
    _get_finalize_kernel,
    _get_pad_kernel,
    _get_tiled_kernel,
    _patch_taps,
    _tiled_smem_elements,
)

# default output tile shapes (axis 0, ..., axis ndim - 1) of the tiled kernel
# (fastest in a sweep on an RTX A6000)
_default_tiles = {2: (16, 16), 3: (8, 8, 8), 4: (4, 4, 4, 4)}

# With only two taps per axis (fast mode, patch_size 3) the per-shift work is
# too small to amortize the tiled kernel's barriers; the direct kernel was
# 1.2-2x faster in 2D, 3D and 4D benchmarks.
_min_tiled_taps = 3

# shrink tiles no further than this many threads before using the direct
# kernel instead
_min_tile_threads = 64

# Launch at least this many waves of resident threads. For small images, the
# shifts along axis 0 are split into chunks handled by different blocks.
_min_waves = 2

_c_types = {np.dtype(np.float32): "float", np.dtype(np.float64): "double"}


@cp.memoize(for_each_device=True)
def _max_smem_per_block():
    """Maximum dynamic shared memory per block (with opt-in) in bytes."""
    device_id = cp.cuda.runtime.getDevice()
    props = cp.cuda.runtime.getDeviceProperties(device_id)
    return props.get("sharedMemPerBlockOptin") or props["sharedMemPerBlock"]


@cp.memoize(for_each_device=True)
def _max_resident_threads():
    """Maximum number of threads resident on the device at once."""
    device_id = cp.cuda.runtime.getDevice()
    props = cp.cuda.runtime.getDeviceProperties(device_id)
    return props["multiProcessorCount"] * props["maxThreadsPerMultiProcessor"]


def _n_shift_chunks(n_blocks, block_size, patch_distance):
    """Number of chunks to split the shifts along axis 0 into."""
    target_blocks = -(-_min_waves * _max_resident_threads() // block_size)
    if n_blocks >= target_blocks:
        return 1
    return min(2 * patch_distance + 1, -(-target_blocks // n_blocks), 65535)


def _select_tile(ndim, n_taps, itemsize, tile=None):
    """Return ``(tile, smem_bytes)`` for the tiled kernel, or ``(None, 0)``.

    Tiles whose shared memory needs exceed the device limit are shrunk by
    halving their largest dimension.
    """
    if tile is None:
        tile = _default_tiles.get(ndim)
        if tile is None:
            return None, 0
    tile = list(tile)
    max_bytes = _max_smem_per_block()
    while math.prod(tile) >= _min_tile_threads:
        nbytes = _tiled_smem_elements(ndim, n_taps, tile) * itemsize
        if nbytes <= max_bytes:
            return tuple(tile), nbytes
        tile[int(np.argmax(tile))] //= 2
    return None, 0


def _nl_means_denoising(
    image,
    patch_size,
    patch_distance,
    h,
    var,
    fast_mode,
    *,
    kernel="auto",
    tile=None,
    shift_chunks=None,
):
    """Non-local means denoising of an image with channels on the last axis.

    Parameters
    ----------
    image : cupy.ndarray
        Array of shape ``(*spatial_shape, n_channels)`` with dtype float32 or
        float64. The computation is performed in this dtype.
    patch_size, patch_distance, h, fast_mode
        As for `denoise_nl_means`.
    var : float
        Noise variance (``sigma**2``).
    kernel : {'auto', 'tiled', 'direct'}, optional
        Kernel variant to use. 'auto' uses the tiled shared memory kernel
        whenever it applies and fits in shared memory.
    tile : tuple of int, optional
        Output tile shape of the tiled kernel (for tuning).
    shift_chunks : int, optional
        Number of chunks the shifts along axis 0 are split into, each handled
        by separate thread blocks whose partial sums are then added. By default
        this is chosen so that small images still fill the GPU.

    Returns
    -------
    out : cupy.ndarray
        Denoised array with the same shape and dtype as `image`.
    """
    if kernel not in ("auto", "tiled", "direct"):
        raise ValueError(f"unknown kernel: {kernel}")
    dtype = image.dtype
    c_type = _c_types[dtype]
    spatial_shape = image.shape[:-1]
    ndim = len(spatial_shape)
    if tile is not None and len(tile) != ndim:
        raise ValueError(f"tile must have {ndim} elements")
    n_channels = image.shape[-1]

    # odd patch size for a symmetric patch (as in scikit-image)
    s = patch_size + 1 if patch_size % 2 == 0 else patch_size
    r = s // 2
    d = patch_distance
    _, taps = _patch_taps(s, fast_mode)
    n_taps = len(taps)
    if fast_mode:
        scale = 1.0 / (n_channels * h * h * s**ndim)
    elif s == 1:
        # Gaussian patch weights are undefined for s == 1. scikit-image ends up
        # with equal weights for all shifts: the mean over the search window.
        scale = 0.0
    else:
        scale = 1.0 / (n_channels * h * h * math.fsum(taps) ** ndim)
    var_term = n_channels * 2.0 * var

    out = cp.empty(image.shape, dtype=dtype)
    if out.size == 0:
        return out

    use_tiled = False
    smem_bytes = 0
    if (kernel == "tiled" and n_taps > 0) or (
        kernel == "auto" and n_taps >= _min_tiled_taps
    ):
        tile, smem_bytes = _select_tile(ndim, n_taps, dtype.itemsize, tile)
        use_tiled = tile is not None
    if kernel == "tiled" and not use_tiled:
        raise ValueError("the tiled kernel does not support this configuration")

    # Reflect-pad by the patch radius plus the search distance. For the tiled
    # kernel, the high side is also extended to a whole number of tiles so
    # that the kernel needs no bounds checks.
    pad = r + d
    if use_tiled:
        extra = [-(-n // t) * t - n for n, t in zip(spatial_shape, tile)]
    else:
        extra = [0] * ndim
    padded_shape = tuple(n + 2 * pad + e for n, e in zip(spatial_shape, extra))
    padded = cp.empty((n_channels,) + padded_shape, dtype=dtype)
    _get_pad_kernel(ndim)(
        cp.ascontiguousarray(image),
        *(
            np.int32(v)
            for n, p in zip(spatial_shape, padded_shape)
            for v in (n, p)
        ),
        np.int32(pad),
        np.int32(n_channels),
        padded,
    )

    if use_tiled:
        block = math.prod(tile)
        n_blocks = math.prod(-(-n // t) for n, t in zip(spatial_shape, tile))
    else:
        block = 128
        n_blocks = min(-(-math.prod(spatial_shape) // block), 2**31 - 1)

    # split the shifts along axis 0 across blocks if the image is too small to
    # fill the GPU (each chunk then writes partial sums, added up below)
    n_shifts0 = 2 * d + 1
    if shift_chunks is None:
        shift_chunks = _n_shift_chunks(n_blocks, block, d)
    chunk_len = -(-n_shifts0 // max(1, min(shift_chunks, n_shifts0)))
    n_chunks = -(-n_shifts0 // chunk_len)
    if n_chunks > 1:
        result = cp.empty(
            (n_chunks,) + spatial_shape + (n_channels + 1,), dtype=dtype
        )
    else:
        result = out

    large = max(padded.size, result.size) >= 2**31
    index_type = "long long" if large else "int"
    index_dtype = np.int64 if large else np.int32
    itemsize = dtype.itemsize
    strides = tuple(index_dtype(st // itemsize) for st in padded.strides[1:])
    plane = index_dtype(padded.strides[0] // itemsize)
    shape = tuple(np.int32(n) for n in spatial_shape)
    args = (
        (padded, result)
        + shape
        + strides
        + (
            plane,
            np.int32(pad),
            np.int32(d),
            np.int32(chunk_len),
            dtype.type(scale),
            dtype.type(var_term),
        )
    )
    if use_tiled:
        nlm_kernel = _get_tiled_kernel(
            ndim, n_channels, s, fast_mode, c_type, index_type, tile
        )
        if smem_bytes > nlm_kernel.max_dynamic_shared_size_bytes:
            nlm_kernel.max_dynamic_shared_size_bytes = smem_bytes
        nlm_kernel((n_blocks, n_chunks), (block,), args, shared_mem=smem_bytes)
    else:
        nlm_kernel = _get_direct_kernel(
            ndim, n_channels, s, fast_mode, c_type, index_type
        )
        nlm_kernel((n_blocks, n_chunks), (block,), args)

    if n_chunks > 1:
        _get_finalize_kernel()(
            result,
            np.int32(n_chunks),
            np.int64(math.prod(spatial_shape)),
            np.int32(n_channels),
            out,
        )
    return out
