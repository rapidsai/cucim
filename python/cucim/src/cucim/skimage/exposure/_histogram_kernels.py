# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integer histograms with caller-supplied bounds and no scalar readbacks."""

import cupy as cp
from cupy._core._scalar import get_typename


@cp.memoize(for_each_device=True)
def _get_bincount_rows_kernel(dtype, shared):
    """Privatize each block's histogram before merging into its output row."""
    dtype = cp.dtype(dtype)
    # CuPy names int8 `signed char`; plain `char` is unsigned on aarch64.
    ctype = get_typename(dtype)
    initialize = ""
    count = "atomicAdd(output + row * nbins + bin, 1ULL);"
    merge = ""
    if shared:
        initialize = """
        extern __shared__ unsigned int counts[];
        for (int bin = threadIdx.x; bin < nbins; bin += blockDim.x)
            counts[bin] = 0;
        __syncthreads();
        """
        count = "atomicAdd(counts + bin, 1u);"
        merge = """
        __syncthreads();
        for (int bin = threadIdx.x; bin < nbins; bin += blockDim.x)
            if (counts[bin])
                atomicAdd(output + row * nbins + bin,
                          (unsigned long long)counts[bin]);
        """
    code = f"""
    extern "C" __global__ void cucim_bincount_rows(
        const {ctype}* image, long long pixels, long long low,
        unsigned long long* output, int nbins, int blocks_per_row) {{
        long long row = blockIdx.x / blocks_per_row;
        int part = blockIdx.x % blocks_per_row;
        {initialize}
        for (long long i = (long long)part * blockDim.x + threadIdx.x;
             i < pixels; i += (long long)blocks_per_row * blockDim.x) {{
            // Unsigned subtraction avoids overflow at signed integer limits.
            unsigned long long bin = (unsigned long long)image[row * pixels + i]
                                   - (unsigned long long)low;
            if (bin < (unsigned long long)nbins) {{ {count} }}
        }}
        {merge}
    }}
    """
    return cp.RawKernel(code, "cucim_bincount_rows")


def _bincount_rows(image, nbins, low=0):
    """Count integer rows whose values are known to lie in [low, low + nbins).

    The caller supplies validated bounds, so no input extrema or negativity
    checks are needed. Up to 4096 bins use shared 32-bit block counts and
    64-bit global totals; wider histograms use global atomics directly.
    """
    image = cp.ascontiguousarray(image)
    rows, pixels = image.shape
    result = cp.zeros((rows, nbins), dtype=cp.int64)
    if not rows or not pixels:
        return result
    blocks = min((pixels + 255) // 256, max(1, 1024 // rows))
    shared = nbins <= 4096
    # Each private counter is bounded by this block's number of input values.
    shared &= (pixels + blocks - 1) // blocks <= 0xFFFFFFFF
    kernel = _get_bincount_rows_kernel(image.dtype.str, shared)
    kernel(
        (rows * blocks,),
        (256,),
        (image, pixels, low, result, nbins, blocks),
        shared_mem=4 * nbins if shared else 0,
    )
    return result
