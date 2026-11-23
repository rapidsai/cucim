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
    # Round to the full 256-thread iteration: an early block can process up
    # to one more iteration than the average pixels-per-block estimate.
    iterations = (pixels + blocks * 256 - 1) // (blocks * 256)
    shared &= iterations * 256 <= 0xFFFFFFFF
    kernel = _get_bincount_rows_kernel(image.dtype.str, shared)
    kernel(
        (rows * blocks,),
        (256,),
        (image, pixels, low, result, nbins, blocks),
        shared_mem=4 * nbins if shared else 0,
    )
    return result


_MAX_CLIP_BINS = 4096

_clip_rows_kernel = cp.RawKernel(
    r"""
    __device__ int sum_in_warp(int value) {
        for (int offset = 16; offset > 0; offset /= 2)
            value += __shfl_xor_sync(0xffffffff, value, offset);
        return value;
    }

    extern "C" __global__ void cucim_clip_histogram_rows(
        long long* hist, long long rows, int nbins, int limit) {
        const int warp = threadIdx.x / 32;
        const int lane = threadIdx.x % 32;
        const long long row = (long long)blockIdx.x * (blockDim.x / 32) + warp;
        if (row >= rows)
            return;
        hist += row * nbins;

        extern __shared__ int bins[];
        int* counts = bins + warp * nbins;
        int excess = 0;
        for (int i = lane; i < nbins; i += 32) {
            const int value = (int)hist[i];
            const int capped = min(value, limit);
            excess += value - capped;
            counts[i] = capped;
        }
        excess = sum_in_warp(excess);
        if (excess == 0)
            return;

        const int increment = excess / nbins;
        const int upper = limit - increment;
        int added_low = 0;
        int added_mid = 0;
        int available = 0;
        for (int i = lane; i < nbins; i += 32) {
            int value = counts[i];
            if (value < upper) {
                value += increment;
                added_low += increment;
            }
            // The reference checks the middle range after incrementing low bins.
            if (value >= upper && value < limit) {
                added_mid += limit - value;
                value = limit;
            }
            available += value < limit;
            counts[i] = value;
        }
        excess -= sum_in_warp(added_low);
        excess -= sum_in_warp(added_mid);
        available = sum_in_warp(available);

        while (excess > 0 && available > 0) {
            const int previous = excess;
            for (int offset = 0; offset < nbins && excess > 0 && available > 0; ++offset) {
                const int step = max(1, available / excess);
                int added = 0;
                int saturated = 0;
                for (int i = lane; i < nbins; i += 32) {
                    if (i >= offset && (step == 1 || (i - offset) % step == 0)
                            && counts[i] < limit) {
                        ++counts[i];
                        ++added;
                        saturated += counts[i] == limit;
                    }
                }
                // Update the whole slice before checking excess, as in NumPy.
                excess -= sum_in_warp(added);
                available -= sum_in_warp(saturated);
            }
            if (excess == previous)
                break;
        }

        for (int i = lane; i < nbins; i += 32)
            hist[i] = counts[i];
    }
    """,
    "cucim_clip_histogram_rows",
)


def _clip_histogram_rows(hist, limit):
    """Clip and redistribute contiguous int64 CLAHE histogram rows in place.

    One warp owns each row, and each lane owns its bins throughout the
    redistribution. The caller bounds the tile size so that the bin counts
    and reductions fit in signed 32-bit integers.
    """
    rows, nbins = hist.shape
    if rows == 0:
        return
    # Fit four rows into 32 KiB of shared memory for the typical bin counts.
    warps = min(4, 8192 // nbins)
    _clip_rows_kernel(
        ((rows + warps - 1) // warps,),
        (32 * warps,),
        (hist, rows, cp.int32(nbins), cp.int32(limit)),
        shared_mem=4 * warps * nbins,
    )
