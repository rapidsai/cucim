# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Combined range reductions for device-side warp output clipping.

Each reduction carries a NaN-safe minimum, maximum, and a flag recording
whether any input was NaN. Computing these together reads each image element
once instead of scanning separately for each statistic. The accumulator also
tracks whether it contains a non-NaN value, preserving all-NaN ranges without
numeric sentinels that would complicate support for integer and complex data.

CuPy's generic scalar ReductionKernel uses a single thread block. Small inputs
use it directly to avoid the extra dispatch and launch overhead of staging:
up to 512 KiB for C-contiguous arrays, or 4096 elements for other layouts.
Larger contiguous real inputs use a single launch: blocks publish partial
ranges and increment a completion counter after a device fence. The last
finishing block merges those ranges; other blocks exit without waiting. Each
call owns its scratch storage and counter, so streams can run independently.
Float16, complex, and strided inputs retain partial and merge launches. This
exposes more parallelism while avoiding another scan of the image; an
incomplete final chunk in the fallback is reduced separately before the merge.

The fallback uses fixed-size chunks for contiguous inputs. Strided
inputs use existing axes or split one axis into chunk views, avoiding a copy
of the full image merely to flatten it. The same accumulator and merge
operation are used for both image values and partial ranges, and all results
remain on the device.
"""

import cupy as cp
from cupy._core._scalar import get_typename

_RANGE_CHUNK_SIZE = 4096
# Keep the direct-reduction cutoff separate from the parallel chunk size.
# A conservative byte limit balances launch savings against the limited
# throughput of a single block.
# This empirical host-overhead tradeoff was tuned on an RTX A6000; GPUs with
# different SM counts or GPU-bound workloads may favor a different cutoff.
_RANGE_DIRECT_MAX_BYTES = 512 * 1024

_range_preamble = r"""
template<typename T>
__device__ bool warp_range_less(T a, T b) { return a < b; }

template<>
__device__ bool warp_range_less<float>(float a, float b) {
    // CuPy enables flush-to-zero. Ordered bits preserve subnormal extrema.
    unsigned int ka = __float_as_uint(a), kb = __float_as_uint(b);
    ka = (ka & 0x80000000u) ? ~ka : (ka | 0x80000000u);
    kb = (kb & 0x80000000u) ? ~kb : (kb | 0x80000000u);
    return ka < kb;
}

template<typename T>
struct WarpRange {
    T minimum, maximum;
    bool has_nan, valid;

    __device__ WarpRange()
        : minimum(0), maximum(0), has_nan(false), valid(false) {}

    __device__ WarpRange(T x)
        : minimum(x), maximum(x), has_nan(x != x), valid(x == x) {}

    __device__ WarpRange(T lo, T hi, bool nan)
        : minimum(lo), maximum(hi), has_nan(nan), valid(lo == lo) {}
};

template<typename T>
__device__ WarpRange<T> merge_warp_ranges(WarpRange<T> a, WarpRange<T> b) {
    // Keep an actual NaN when neither range contains a valid value, so an
    // all-NaN input produces NaN bounds without requiring a numeric sentinel.
    if (!a.valid && !b.valid) return a.has_nan ? a : b;
    if (!a.valid) {
        b.has_nan |= a.has_nan;
        return b;
    }
    if (!b.valid) {
        a.has_nan |= b.has_nan;
        return a;
    }
    a.minimum = warp_range_less(b.minimum, a.minimum) ? b.minimum : a.minimum;
    a.maximum = warp_range_less(a.maximum, b.maximum) ? b.maximum : a.maximum;
    a.has_nan |= b.has_nan;
    return a;
}
"""

_range_post_map = (
    "minimum = a.minimum, maximum = a.maximum, contains_nan = a.has_nan"
)

_warp_range_kernel = cp.ReductionKernel(
    "T x",
    "T minimum, T maximum, bool contains_nan",
    "WarpRange<T>(x)",
    "merge_warp_ranges(a, b)",
    _range_post_map,
    "WarpRange<T>()",
    name="cucim_warp_range",
    reduce_type="WarpRange<T>",
    preamble=_range_preamble,
)

_merge_warp_ranges_kernel = cp.ReductionKernel(
    "T lo, T hi, bool nan",
    "T minimum, T maximum, bool contains_nan",
    "WarpRange<T>(lo, hi, nan)",
    "merge_warp_ranges(a, b)",
    _range_post_map,
    "WarpRange<T>()",
    name="cucim_merge_warp_ranges",
    reduce_type="WarpRange<T>",
    preamble=_range_preamble,
)


def _empty_warp_range(shape, dtype):
    """Allocate device buffers for extrema and NaN flags at the given shape."""
    return (
        cp.empty(shape, dtype=dtype),
        cp.empty(shape, dtype=dtype),
        cp.empty(shape, dtype=cp.bool_),
    )


@cp.memoize(for_each_device=True)
def _get_last_block_range_kernel(dtype):
    """Merge parallel ranges in the last finishing block, without a spin wait.

    Each block publishes one range through volatile stores, then fences before
    incrementing a per-call completion counter. Only the last block reads the
    partials and writes the result; every other block exits. Keeping scratch
    and counter storage private to each invocation allows concurrent streams.
    """
    ctype = get_typename(cp.dtype(dtype))
    code = (
        _range_preamble
        + r"""
    typedef CTYPE T;

    __device__ WarpRange<T> block_range(WarpRange<T> value,
                                       WarpRange<T>* shared) {
        int t = threadIdx.x;
        shared[t] = value;
        __syncthreads();
        for (int offset = blockDim.x / 2; offset; offset /= 2) {
            if (t < offset)
                shared[t] = merge_warp_ranges(shared[t], shared[t + offset]);
            __syncthreads();
        }
        return shared[0];
    }

    extern "C" __global__ void cucim_warp_range_last_block(
        const T* image, long long size,
        volatile T* partial_min, volatile T* partial_max,
        volatile bool* partial_nan, unsigned int* counter,
        T* minimum, T* maximum, bool* contains_nan) {
        __shared__ __align__(16) unsigned char storage[256 * sizeof(WarpRange<T>)];
        WarpRange<T>* shared = reinterpret_cast<WarpRange<T>*>(storage);
        __shared__ bool last;
        WarpRange<T> value;
        for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
             i < size; i += (long long)blockDim.x * gridDim.x)
            value = merge_warp_ranges(value, WarpRange<T>(image[i]));
        value = block_range(value, shared);
        if (threadIdx.x == 0) {
            partial_min[blockIdx.x] = value.minimum;
            partial_max[blockIdx.x] = value.maximum;
            partial_nan[blockIdx.x] = value.has_nan;
            __threadfence();
            last = atomicAdd(counter, 1u) == gridDim.x - 1;
        }
        __syncthreads();
        if (!last) return;

        value = WarpRange<T>();
        for (unsigned int i = threadIdx.x; i < gridDim.x; i += blockDim.x)
            value = merge_warp_ranges(value,
                WarpRange<T>(partial_min[i], partial_max[i], partial_nan[i]));
        value = block_range(value, shared);
        if (threadIdx.x == 0) {
            *minimum = value.minimum;
            *maximum = value.maximum;
            *contains_nan = value.has_nan;
        }
    }
    """
    )
    return cp.RawKernel(
        code.replace("CTYPE", ctype), "cucim_warp_range_last_block"
    )


def _last_block_warp_range(image, result):
    """Launch parallel scan and final merge together for contiguous real data."""
    n_blocks = min(
        (image.size + _RANGE_CHUNK_SIZE - 1) // _RANGE_CHUNK_SIZE, 4096
    )
    partials = _empty_warp_range((n_blocks,), image.dtype)
    # An asynchronous memset initializes this invocation's counter. No host
    # synchronization or persistent, stream-shared counter state is needed.
    counter = cp.zeros((), dtype=cp.uint32)
    kernel = _get_last_block_range_kernel(image.dtype.str)
    kernel(
        (n_blocks,), (256,), (image, image.size, *partials, counter, *result)
    )


def _partial_warp_ranges(image, axis, chunk_size):
    """Reduce chunk views along one axis, including any incomplete tail."""
    n_full, remainder = divmod(image.shape[axis], chunk_size)
    partials = _empty_warp_range((n_full + bool(remainder),), image.dtype)
    full_size = n_full * chunk_size
    selection = [slice(None)] * image.ndim
    selection[axis] = slice(0, full_size)
    shape = image.shape[:axis] + (n_full, chunk_size) + image.shape[axis + 1 :]
    # Splitting a single axis preserves the view even for strided inputs.
    full_chunks = image[tuple(selection)].reshape(shape)
    reduce_axes = tuple(ax for ax in range(full_chunks.ndim) if ax != axis)
    _warp_range_kernel(
        full_chunks,
        *(out[:n_full] for out in partials),
        axis=reduce_axes,
    )
    if remainder:
        selection[axis] = slice(full_size, None)
        _warp_range_kernel(
            image[tuple(selection)],
            *(out[-1:].reshape(()) for out in partials),
        )
    return partials


def _get_warp_range(image):
    """Choose direct or staged reduction and return the range as device scalars."""
    if image.size == 0:
        raise ValueError("Cannot compute the range of an empty image")
    result = _empty_warp_range((), image.dtype)
    if image.size <= _RANGE_CHUNK_SIZE or (
        image.flags.c_contiguous and image.nbytes <= _RANGE_DIRECT_MAX_BYTES
    ):
        _warp_range_kernel(image, *result)
        return result

    if image.flags.c_contiguous and (
        image.dtype.kind in "biu" or image.dtype.char in "fd"
    ):
        _last_block_warp_range(image, result)
        return result

    # A generic scalar ReductionKernel uses a single block. Split large
    # arrays into independent ranges first to use more of the device.
    if not image.flags.c_contiguous and image.ndim > 1:
        # Reduce across existing axes to avoid copying a strided image just
        # to flatten it. Preserve the largest axis for parallel partials.
        keep_axis = max(range(image.ndim), key=lambda ax: image.shape[ax])
        if image.shape[keep_axis] <= _RANGE_CHUNK_SIZE:
            axes = tuple(ax for ax in range(image.ndim) if ax != keep_axis)
            partials = _empty_warp_range((image.shape[keep_axis],), image.dtype)
            _warp_range_kernel(image, *partials, axis=axes)
        else:
            # Bound the merge size for very long strided axes as well.
            chunk_size = (
                image.shape[keep_axis] + _RANGE_CHUNK_SIZE - 1
            ) // _RANGE_CHUNK_SIZE
            partials = _partial_warp_ranges(image, keep_axis, chunk_size)
    else:
        partials = _partial_warp_ranges(image.reshape(-1), 0, _RANGE_CHUNK_SIZE)
    _merge_warp_ranges_kernel(*partials, *result)
    return result
