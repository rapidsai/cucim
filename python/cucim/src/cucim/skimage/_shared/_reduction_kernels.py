# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Combined extrema reductions for image processing.

Each reduction carries a NaN-safe minimum, maximum, and a flag recording
whether any input was NaN. Computing these together reads each image element
once instead of scanning separately for each statistic. The accumulator also
tracks whether it contains a non-NaN value, preserving all-NaN ranges without
numeric sentinels that would complicate support for integer and complex data.

CuPy's generic scalar ReductionKernel uses a single thread block. Small inputs
use it directly to avoid the extra dispatch and launch overhead of staging:
up to 512 KiB of total C-contiguous input data, or 4096 elements for other layouts.
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
remain on the device. Callers can retain NaN-ignoring extrema with a NaN flag
or request NaN-propagating bounds, matching ordinary min/max reductions.
For min-max NRMSE, the state and staging also carry a double-precision
squared-error sum from two images, avoiding full-sized arithmetic temporaries.
"""

import cupy as cp
from cupy._core._scalar import get_typename

_RANGE_CHUNK_SIZE = 4096
# Keep the direct-reduction cutoff separate from the parallel chunk size.
# A conservative byte limit balances launch savings against the limited
# throughput of a single block.
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
struct MinMaxState {
    T minimum, maximum;
    bool has_nan, valid;

    __device__ MinMaxState()
        : minimum(0), maximum(0), has_nan(false), valid(false) {}

    __device__ MinMaxState(T x)
        : minimum(x), maximum(x), has_nan(x != x), valid(x == x) {}

    __device__ MinMaxState(T lo, T hi, bool nan)
        : minimum(lo), maximum(hi), has_nan(nan), valid(lo == lo) {}
};

template<typename T>
__device__ MinMaxState<T> merge_minmax_states(MinMaxState<T> a, MinMaxState<T> b) {
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


def _make_minmax_kernel(*, merge=False, propagate_nan=False):
    """Build reductions sharing an accumulator and the requested NaN policy."""
    post_map = (
        "minimum = a.minimum, maximum = a.maximum, contains_nan = a.has_nan"
    )
    if propagate_nan:
        post_map = (
            "minimum = a.has_nan ? T(CUDART_NAN) : a.minimum, "
            "maximum = a.has_nan ? T(CUDART_NAN) : a.maximum, contains_nan = a.has_nan"
        )
    return cp.ReductionKernel(
        "T lo, T hi, bool nan" if merge else "T x",
        "T minimum, T maximum, bool contains_nan",
        "MinMaxState<T>(lo, hi, nan)" if merge else "MinMaxState<T>(x)",
        "merge_minmax_states(a, b)",
        post_map,
        "MinMaxState<T>()",
        name="cucim_"
        + ("merge_" if merge else "")
        + "minmax"
        + ("_propagate_nan" if propagate_nan else ""),
        reduce_type="MinMaxState<T>",
        preamble=_range_preamble,
    )


_minmax_kernel = _make_minmax_kernel()
_merge_minmax_kernel = _make_minmax_kernel(merge=True)
_minmax_propagate_kernel = _make_minmax_kernel(propagate_nan=True)
_merge_minmax_propagate_kernel = _make_minmax_kernel(
    merge=True, propagate_nan=True
)


def _empty_minmax(shape, dtype):
    """Allocate device buffers for extrema and NaN flags at the given shape."""
    return (
        cp.empty(shape, dtype=dtype),
        cp.empty(shape, dtype=dtype),
        cp.empty(shape, dtype=cp.bool_),
    )


@cp.memoize(for_each_device=True)
def _get_last_block_range_kernel(dtype, propagate_nan):
    """Merge parallel ranges in the last finishing block, without a spin wait.

    Each block publishes one range through volatile stores, then fences before
    incrementing a per-call completion counter. Only the last block reads the
    partials and writes the result; every other block exits. Keeping scratch
    and counter storage private to each invocation allows concurrent streams.
    """
    ctype = get_typename(cp.dtype(dtype))
    code = (
        "#include <cupy/math_constants.h>\n"
        + _range_preamble
        + r"""
    typedef CTYPE T;

    __device__ MinMaxState<T> block_range(MinMaxState<T> value,
                                       MinMaxState<T>* shared) {
        int t = threadIdx.x;
        shared[t] = value;
        __syncthreads();
        for (int offset = blockDim.x / 2; offset; offset /= 2) {
            if (t < offset)
                shared[t] = merge_minmax_states(shared[t], shared[t + offset]);
            __syncthreads();
        }
        return shared[0];
    }

    extern "C" __global__ void cucim_minmax_last_block(
        const T* image, long long size,
        volatile T* partial_min, volatile T* partial_max,
        volatile bool* partial_nan, unsigned int* counter,
        T* minimum, T* maximum, bool* contains_nan) {
        __shared__ __align__(16) unsigned char storage[256 * sizeof(MinMaxState<T>)];
        MinMaxState<T>* shared = reinterpret_cast<MinMaxState<T>*>(storage);
        __shared__ bool last;
        MinMaxState<T> value;
        for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
             i < size; i += (long long)blockDim.x * gridDim.x)
            value = merge_minmax_states(value, MinMaxState<T>(image[i]));
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

        value = MinMaxState<T>();
        for (unsigned int i = threadIdx.x; i < gridDim.x; i += blockDim.x)
            value = merge_minmax_states(value,
                MinMaxState<T>(partial_min[i], partial_max[i], partial_nan[i]));
        value = block_range(value, shared);
        if (threadIdx.x == 0) {
            *minimum = MINIMUM;
            *maximum = MAXIMUM;
            *contains_nan = value.has_nan;
        }
    }
    """
    )
    for marker, field in (("MINIMUM", "minimum"), ("MAXIMUM", "maximum")):
        expression = f"value.{field}"
        if propagate_nan:
            expression = f"value.has_nan ? T(CUDART_NAN) : {expression}"
        code = code.replace(marker, expression)
    return cp.RawKernel(code.replace("CTYPE", ctype), "cucim_minmax_last_block")


def _last_block_minmax(image, result, *, propagate_nan=False):
    """Launch parallel scan and final merge together for contiguous real data."""
    n_blocks = min(
        (image.size + _RANGE_CHUNK_SIZE - 1) // _RANGE_CHUNK_SIZE, 4096
    )
    partials = _empty_minmax((n_blocks,), image.dtype)
    # An asynchronous memset initializes this invocation's counter. No host
    # synchronization or persistent, stream-shared counter state is needed.
    counter = cp.zeros((), dtype=cp.uint32)
    kernel = _get_last_block_range_kernel(image.dtype.str, propagate_nan)
    kernel(
        (n_blocks,), (256,), (image, image.size, *partials, counter, *result)
    )


def _partial_reduce(images, axis, chunk_size, kernel, allocate):
    """Reduce matching chunk views, including any incomplete tail."""
    image = images[0]
    n_full, remainder = divmod(image.shape[axis], chunk_size)
    partials = allocate((n_full + bool(remainder),))
    full_size = n_full * chunk_size
    selection = [slice(None)] * image.ndim
    selection[axis] = slice(0, full_size)
    shape = image.shape[:axis] + (n_full, chunk_size) + image.shape[axis + 1 :]
    # Splitting a single axis preserves views even when input strides differ.
    full_chunks = tuple(x[tuple(selection)].reshape(shape) for x in images)
    reduce_axes = tuple(ax for ax in range(len(shape)) if ax != axis)
    kernel(*full_chunks, *(out[:n_full] for out in partials), axis=reduce_axes)
    if remainder:
        selection[axis] = slice(full_size, None)
        kernel(
            *(x[tuple(selection)] for x in images),
            *(out[-1:].reshape(()) for out in partials),
        )
    return partials


def _reduce_image_statistics(
    images, result, kernel, merge_kernel, allocate, *, minmax_nan_policy=None
):
    """Stage one or more images without copying their full contents to flatten."""
    image = images[0]
    if image.size == 0:
        raise ValueError("Cannot compute the range of an empty image")
    contiguous = all(x.flags.c_contiguous for x in images)
    if image.size <= _RANGE_CHUNK_SIZE or (
        contiguous and sum(x.nbytes for x in images) <= _RANGE_DIRECT_MAX_BYTES
    ):
        kernel(*images, *result)
        return

    if (
        minmax_nan_policy is not None
        and contiguous
        and (image.dtype.kind in "biu" or image.dtype.char in "fd")
    ):
        _last_block_minmax(image, result, propagate_nan=minmax_nan_policy)
        return

    # Parallel partials avoid assigning a large scalar reduction to one block.
    if not contiguous and image.ndim > 1:
        keep_axis = max(range(image.ndim), key=lambda ax: image.shape[ax])
        if image.shape[keep_axis] <= _RANGE_CHUNK_SIZE:
            axes = tuple(ax for ax in range(image.ndim) if ax != keep_axis)
            partials = allocate((image.shape[keep_axis],))
            kernel(*images, *partials, axis=axes)
        else:
            chunk_size = (
                image.shape[keep_axis] + _RANGE_CHUNK_SIZE - 1
            ) // _RANGE_CHUNK_SIZE
            partials = _partial_reduce(
                images, keep_axis, chunk_size, kernel, allocate
            )
    else:
        partials = _partial_reduce(
            tuple(x.reshape(-1) for x in images),
            0,
            _RANGE_CHUNK_SIZE,
            kernel,
            allocate,
        )
    merge_kernel(*partials, *result)


def _reduce_minmax(image, *, propagate_nan=False):
    """Choose direct or staged reduction and return packed bounds and a NaN flag."""
    bounds = cp.empty(2, dtype=image.dtype)
    has_nan = cp.empty((), dtype=cp.bool_)
    kernel = _minmax_propagate_kernel if propagate_nan else _minmax_kernel
    merge_kernel = (
        _merge_minmax_propagate_kernel
        if propagate_nan
        else _merge_minmax_kernel
    )
    _reduce_image_statistics(
        (image,),
        (bounds[0], bounds[1], has_nan),
        kernel,
        merge_kernel,
        lambda shape: _empty_minmax(shape, image.dtype),
        minmax_nan_policy=propagate_nan,
    )
    return bounds, has_nan


def _get_minmax_with_nan(image):
    """Return NaN-ignoring device extrema and whether any input value was NaN."""
    bounds, has_nan = _reduce_minmax(image)
    return bounds[0], bounds[1], has_nan


def _get_minmax(image):
    """Return a two-element device array of NaN-propagating extrema.

    Packed bounds allow callers that need host values to transfer both in one
    readback without a separate stacking kernel. Integer bounds retain their
    dtype, including the full precision of 64-bit integers.
    """
    bounds, _ = _reduce_minmax(image, propagate_nan=image.dtype.kind in "fc")
    return bounds


_mse_preamble = (
    _range_preamble
    + r"""
template<typename T>
struct MinMaxMSEState {
    MinMaxState<T> range;
    double squared_error;

    __device__ MinMaxMSEState() : range(), squared_error(0) {}

    __device__ MinMaxMSEState(T x, T y) : range(x) {
        // Match mean_squared_error: subtract and square in the image dtype,
        // then accumulate in double precision without materializing an image.
        T difference = x - y;
        T squared = difference * difference;
        squared_error = static_cast<double>(squared);
    }

    __device__ MinMaxMSEState(T lo, T hi, bool nan, double error)
        : range(lo, hi, nan), squared_error(error) {}
};

template<typename T>
__device__ MinMaxMSEState<T> merge_minmax_mse(
        MinMaxMSEState<T> a, MinMaxMSEState<T> b) {
    a.range = merge_minmax_states(a.range, b.range);
    a.squared_error += b.squared_error;
    return a;
}
"""
)


def _make_minmax_mse_kernel(*, merge=False):
    """Combine reference extrema and squared-error sums in one image scan."""
    return cp.ReductionKernel(
        "T lo, T hi, bool nan, float64 error" if merge else "T x, T y",
        "T minimum, T maximum, bool contains_nan, float64 squared_error",
        "MinMaxMSEState<T>(lo, hi, nan, error)"
        if merge
        else "MinMaxMSEState<T>(x, y)",
        "merge_minmax_mse(a, b)",
        "minimum = a.range.minimum, maximum = a.range.maximum, "
        "contains_nan = a.range.has_nan, squared_error = a.squared_error",
        "MinMaxMSEState<T>()",
        name="cucim_" + ("merge_" if merge else "") + "minmax_mse",
        reduce_type="MinMaxMSEState<T>",
        preamble=_mse_preamble,
    )


_minmax_mse_kernel = _make_minmax_mse_kernel()
_merge_minmax_mse_kernel = _make_minmax_mse_kernel(merge=True)


def _get_minmax_mse(image_true, image_test):
    """Return extrema, a NaN flag, and squared-error sum for matching float images.

    Callers supply equally shaped arrays of the same float32/float64 dtype.
    Both images share the staging used for min/max; different strides are
    supported without making full-sized copies or arithmetic temporaries.
    """

    def allocate(shape):
        return (
            *_empty_minmax(shape, image_true.dtype),
            cp.empty(shape, dtype=cp.float64),
        )

    result = allocate(())
    _reduce_image_statistics(
        (image_true, image_test),
        result,
        _minmax_mse_kernel,
        _merge_minmax_mse_kernel,
        allocate,
    )
    return result
