# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import cupy as cp

from cucim.skimage._vendored import ndimage as ndi
from cucim.skimage.util import map_array

# Need some default includes so uint32_t, uint64_t, etc. are defined

# cucim_atomic_min and cucim_atomic_max: atomicMin and atomicMax, except for
# double. CuPy 14.2's atomicMin and atomicMax overloads for double compare
# with fminf and fmaxf, which round the values to single precision; the
# double specializations below compare in double precision.
_includes = r"""
#include <cupy/cuda_workaround.h>  // provide std:: coverage

template <typename T>
__device__ __forceinline__ void cucim_atomic_min(T* address, T val) {
  atomicMin(address, val);
}

template <typename T>
__device__ __forceinline__ void cucim_atomic_max(T* address, T val) {
  atomicMax(address, val);
}

template <>
__device__ __forceinline__ void cucim_atomic_min<double>(double* address,
                                                         double val) {
  unsigned long long* a = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *a, assumed;
  do {
    assumed = old;
    if (__longlong_as_double(static_cast<long long>(assumed)) <= val) {
      return;
    }
    old = atomicCAS(a, assumed,
                    static_cast<unsigned long long>(__double_as_longlong(val)));
  } while (assumed != old);
}

template <>
__device__ __forceinline__ void cucim_atomic_max<double>(double* address,
                                                         double val) {
  unsigned long long* a = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *a, assumed;
  do {
    assumed = old;
    if (__longlong_as_double(static_cast<long long>(assumed)) >= val) {
      return;
    }
    old = atomicCAS(a, assumed,
                    static_cast<unsigned long long>(__double_as_longlong(val)));
  } while (assumed != old);
}
"""

# Warp-level helpers for the label-run kernels (see _get_label_run_loop).
#
# cucim_shfl_down: __shfl_down_sync, also for the 8 and 16-bit label types.
# cucim_warp_reduce: reduction of `v` over the lanes lane, lane + 1, ...,
#     seg_end (which must all be in `mask`), as an inclusive Hillis-Steele scan
#     towards lower lanes: after the step with offset `off`, each lane holds
#     the reduction over up to 2 * off lanes, never past seg_end. The steps
#     stop at offset `max_off`, which must be at least half the length of the
#     longest group of lanes sharing seg_end. All lanes in `mask` must call it
#     with the same `max_off`. The result is complete in the first lane of
#     each group.
_warp_includes = (
    _includes
    + r"""
#include <cuda/std/limits>

template <typename T>
__device__ __forceinline__ T cucim_shfl_down(unsigned int mask, T v,
                                             unsigned int delta) {
  return __shfl_down_sync(mask, v, delta);
}

template <typename T>
__device__ __forceinline__ T cucim_shfl(unsigned int mask, T v,
                                        unsigned int src_lane) {
  if constexpr (sizeof(T) < 4) {
    return static_cast<T>(__shfl_sync(mask, static_cast<int>(v), src_lane));
  } else {
    return __shfl_sync(mask, v, src_lane);
  }
}

template <typename T>
__device__ __forceinline__ T cucim_shfl_down_small(unsigned int mask, T v,
                                                   unsigned int delta) {
  return static_cast<T>(__shfl_down_sync(mask, static_cast<int>(v), delta));
}

__device__ __forceinline__ signed char cucim_shfl_down(
    unsigned int mask, signed char v, unsigned int delta) {
  return cucim_shfl_down_small(mask, v, delta);
}

__device__ __forceinline__ unsigned char cucim_shfl_down(
    unsigned int mask, unsigned char v, unsigned int delta) {
  return cucim_shfl_down_small(mask, v, delta);
}

__device__ __forceinline__ short cucim_shfl_down(
    unsigned int mask, short v, unsigned int delta) {
  return cucim_shfl_down_small(mask, v, delta);
}

__device__ __forceinline__ unsigned short cucim_shfl_down(
    unsigned int mask, unsigned short v, unsigned int delta) {
  return cucim_shfl_down_small(mask, v, delta);
}

struct cucim_sum_op {
  template <typename T>
  __device__ T operator()(T a, T b) const { return a + b; }
  template <typename T>
  static __device__ T identity() { return T(0); }
};

struct cucim_min_op {
  template <typename T>
  __device__ T operator()(T a, T b) const { return min(a, b); }
  template <typename T>
  static __device__ T identity() {
    if constexpr (cuda::std::numeric_limits<T>::has_infinity) {
      return cuda::std::numeric_limits<T>::infinity();
    } else {
      return cuda::std::numeric_limits<T>::max();
    }
  }
};

struct cucim_max_op {
  template <typename T>
  __device__ T operator()(T a, T b) const { return max(a, b); }
  template <typename T>
  static __device__ T identity() {
    if constexpr (cuda::std::numeric_limits<T>::has_infinity) {
      return -cuda::std::numeric_limits<T>::infinity();
    } else {
      return cuda::std::numeric_limits<T>::lowest();
    }
  }
};

// Element at byte offset `byte` of a 16-byte chunk (any 1, 2, 4 or 8-byte
// type, selected without indexing an array so that the chunk stays in
// registers).
template <typename T>
__device__ __forceinline__ T cucim_chunk_element(const uint4& chunk,
                                                 unsigned int byte) {
  unsigned int w = byte >> 2;
  unsigned int word = w == 0 ? chunk.x
                      : w == 1 ? chunk.y
                      : w == 2 ? chunk.z
                      : chunk.w;
  T out;
  if constexpr (sizeof(T) == 8) {
    unsigned int high = w == 0 ? chunk.y : chunk.w;
    unsigned long long bits =
        (static_cast<unsigned long long>(high) << 32) | word;
    memcpy(&out, &bits, 8);
  } else if constexpr (sizeof(T) == 4) {
    memcpy(&out, &word, 4);
  } else if constexpr (sizeof(T) == 2) {
    unsigned short bits = static_cast<unsigned short>(word >> ((byte & 3u) * 8));
    memcpy(&out, &bits, 2);
  } else {
    unsigned char bits = static_cast<unsigned char>(word >> ((byte & 3u) * 8));
    memcpy(&out, &bits, 1);
  }
  return out;
}

// Reads elements of the `n` consecutive elements of `data` in increasing
// order (get(k) for increasing k < n), with 16-byte loads when the elements
// are at least 4 bytes, `n` elements are a multiple of 16 bytes, `full` is
// true and data is 16-byte aligned.
template <typename T>
struct cucim_sequential_reader {
  const T* ptr;
  bool vector_loads;
  unsigned int loaded;
  uint4 chunk;

  __device__ cucim_sequential_reader(const T* data, size_t n, bool full) {
    ptr = data;
    vector_loads = sizeof(T) >= 4 && (n * sizeof(T)) % 16 == 0 && full
                   && (reinterpret_cast<size_t>(data) & 15) == 0;
    loaded = 0xffffffffu;
  }

  __device__ __forceinline__ T get(unsigned int k) {
    if (!vector_loads) {
      return ptr[k];
    }
    unsigned int byte = k * static_cast<unsigned int>(sizeof(T));
    if ((byte >> 4) != loaded) {
      loaded = byte >> 4;
      chunk = reinterpret_cast<const uint4*>(ptr)[loaded];
    }
    return cucim_chunk_element<T>(chunk, byte & 15u);
  }
};

template <typename T, typename Op>
__device__ __forceinline__ T cucim_warp_reduce(unsigned int mask, T v,
                                               unsigned int seg_end,
                                               unsigned int max_off, Op op) {
  unsigned int lane = threadIdx.x & 31;
#pragma unroll 1
  for (unsigned int off = 1; off <= max_off; off <<= 1) {
    T y = cucim_shfl_down(mask, v, off);
    if (lane + off <= seg_end) {
      v = op(v, y);
    }
  }
  return v;
}
"""
)


def _get_count_dtype(label_image_size):
    """atomicAdd only supports int32, uint32, int64, uint64, float32, float64"""
    int32_count = label_image_size < 2**32
    count_dtype = cp.dtype(cp.uint32 if int32_count else cp.uint64)
    return count_dtype, int32_count


def _get_coord_dtype(label_image_size):
    """Coordinate type of the kernels that unravel raveled pixel indices.

    `_unravel_loop_index` unravels the raveled index in this type, so it must
    hold the raveled index of every pixel, not only each coordinate. 32-bit
    coordinates are used whenever possible because their divisions are much
    faster.
    """
    int32_coords = label_image_size < 2**32
    coord_dtype = cp.dtype(cp.uint32 if int32_coords else cp.uint64)
    return coord_dtype, int32_coords


def _get_label_run_loop(
    pixels_per_thread, pre, start, op, flush, warp_reduce=(), finish=""
):
    """Source of the loop shared by the label-run kernels.

    Each thread scans `pixels_per_thread` consecutive pixels in raveled order.
    Consecutive pixels with the same nonzero label form a run (background
    pixels are skipped and do not end a run). The statistics of the current
    run are accumulated in scalar variables (registers) and added to the
    global outputs with atomics when the label changes and after the last
    pixel. This issues the same atomics as keeping one set of accumulators per
    run in a per-thread array, but without that array in local memory.

    Parameters
    ----------
    pre : str
        Declarations, placed before the loop.
    start : str
        Executed at the start of each run. Must initialize the accumulators.
        Can use `lab`, the label of the run.
    op : str
        Executed for each pixel of the run. Can use `ii`, the raveled index of
        the pixel, and `lab`.
    flush : str
        Executed at the end of each run to write the accumulators to the
        outputs. Can use `lab`.
    finish : str, optional
        Executed at the end of each run, before its accumulators are combined
        across the warp (see `warp_reduce`) and flushed. Can use `lab`.
    warp_reduce : sequence of (str, str), optional
        The accumulators and how they are combined ("sum", "min" or "max").
        If given, the last runs of neighboring lanes of a warp that have the
        same label are combined with warp shuffles, and only the first lane
        of each such group flushes them. In a large region every thread's
        last (often only) run has the same label, so this divides the number
        of atomics on the same addresses by up to 32. The kernel must then be
        compiled with the `_warp_includes` preamble.

    Notes
    -----
    The kernel must have the inputs ``raw X labels`` and
    ``raw uint64 labels_size``, and be launched with
    ``size=math.ceil(labels_size / pixels_per_thread)``.
    """
    final_flush = f"""
      if (lab != 0) {{
        {finish}
        {flush}
      }}\n"""
    if warp_reduce:
        reduce_code = "".join(
            f"""
          {var} = cucim_warp_reduce(
              mask, {var}, seg_end, max_off, cucim_{kind}_op());"""
            for var, kind in warp_reduce
        )
        identity_code = "".join(
            f"""
          {var} = cucim_{kind}_op::identity<decltype({var})>();"""
            for var, kind in warp_reduce
        )
        # The lanes of this warp that run this iteration of the grid-stride
        # loop are those with i < num_threads (blocks are a multiple of 32
        # threads, so warps start at multiples of 32).
        final_flush = f"""
      if (lab != 0) {{
        {finish}
      }}
      {{
        unsigned int lane = threadIdx.x & 31;
        uint64_t num_threads = (labels_size + {pixels_per_thread - 1})
                               / {pixels_per_thread};
        uint64_t warp_lanes = num_threads - (i - lane);
        unsigned int mask = warp_lanes >= 32 ? 0xffffffffu
                            : (1u << warp_lanes) - 1u;
        // A lane without any labeled pixel between two lanes whose last runs
        // have the same label (e.g. in a hole of a large region) joins their
        // group with identity values, so that it does not split the group.
        // (Lanes between different labels stay apart: joining them would
        // only lengthen the groups, and so the scans.)
        unsigned int has_run = __ballot_sync(mask, lab != 0);
        if (has_run != 0 && has_run != mask) {{
          unsigned int later = has_run & ~((2u << lane) - 1u);
          unsigned int earlier = has_run & ((1u << lane) - 1u);
          bool between = lab == 0 && later != 0 && earlier != 0;
          X next_run_lab = cucim_shfl(
              mask, lab, between ? __ffs(later) - 1 : lane);
          X prev_run_lab = cucim_shfl(
              mask, lab, between ? 31 - __clz(earlier) : lane);
          if (between && next_run_lab == prev_run_lab) {{
            lab = next_run_lab;{identity_code}
          }}
        }}
        // continues: the next lane's last run has the same (nonzero) label
        X next_lab = cucim_shfl_down(mask, lab, 1);
        bool continues = (lab != 0) && (lane < 31)
                         && ((mask >> (lane + 1)) & 1u) && (next_lab == lab);
        unsigned int cont = __ballot_sync(mask, continues);
        if (cont != 0) {{
          // last lane of this lane's group
          unsigned int seg_end = __ffs(~cont & (0xffffffffu << lane)) - 1;
          // longest run of set bits in cont = (longest group length - 1), so
          // the scan needs the offsets 1, 2, 4, ... up to that
          unsigned int max_off = 0;
          for (unsigned int x = cont; x != 0; x &= x >> 1) {{
            ++max_off;
          }}
          {reduce_code}
        }}
        bool first = (lane == 0) || !((cont >> (lane - 1)) & 1u);
        if (first && lab != 0) {{
          {flush}
        }}
      }}\n"""
    # The thread's labels are read with 16-byte loads if they are 4 or 8-byte
    # labels (cucim_sequential_reader). With one-element loads, the 32 lanes
    # of a warp read 32 locations pixels_per_thread elements apart in each
    # instruction, and for 4 and 8-byte labels a warp's lines no longer fit
    # in the L1 cache. (For 1 and 2-byte labels, extracting the elements
    # costs more than it saves.) Kernels reading an image can use the same
    # reader for it; `full_thread` tells whether the thread has
    # pixels_per_thread pixels.
    return f"""
      uint64_t start_index = {pixels_per_thread}*i;
      uint64_t ii_max = min(start_index + {pixels_per_thread}, labels_size);
      bool full_thread = ii_max == start_index + {pixels_per_thread};
      {pre}
      cucim_sequential_reader<X> label_reader(
          &labels[start_index], {pixels_per_thread}, full_thread);
      X lab = 0;
      for (uint64_t ii = start_index; ii < ii_max; ii++) {{
        X current_label = label_reader.get(ii - start_index);
        if (current_label == 0) {{ continue; }}
        if (current_label != lab) {{
          if (lab != 0) {{
            {finish}
            {flush}
          }}
          lab = current_label;
          {start}
        }}
        {op}
      }}{final_flush}"""


def _get_min_integer_dtype(max_size, signed=False):
    # negate to get a signed integer type, but need to also subtract 1, due
    # to asymmetric range on positive side, e.g. we want
    #    max_sz = 127 -> int8  (signed)   uint8 (unsigned)
    #    max_sz = 128 -> int16 (signed)   uint8 (unsigned)
    func = cp.min_scalar_type
    return func(-max_size - 1) if signed else func(max_size)


def _check_intensity_image_shape(label_image, intensity_image):
    ndim = label_image.ndim
    if intensity_image.shape[:ndim] != label_image.shape:
        raise ValueError(
            "Initial dimensions of `intensity_image` must match the shape of "
            "`label_image`. (`intensity_image` may have additional trailing "
            "channels/batch dimensions)"
        )

    num_channels = (
        math.prod(intensity_image.shape[ndim:])
        if intensity_image.ndim > ndim
        else 1
    )
    return num_channels


def _unravel_loop_index_declarations(var_name, ndim, uint_t="unsigned int"):
    if ndim == 1:
        code = f"""
        {uint_t} in_coord[1];"""
        return code

    code = f"""
        // variables for unraveling a linear index to a coordinate array
        {uint_t} in_coord[{ndim}];
        {uint_t} temp_floor;"""
    for d in range(ndim):
        code += f"""
        {uint_t} dim{d}_size = {var_name}.shape()[{d}];"""
    return code


def _unravel_loop_index(
    var_name,
    ndim,
    uint_t="unsigned int",
    raveled_index="i",
    omit_declarations=False,
):
    """
    declare a multi-index array in_coord and unravel the 1D index, i into it.
    This code assumes that the array is a C-ordered array. `uint_t` must be
    able to hold the raveled index (see `_get_coord_dtype`).
    """
    code = (
        ""
        if omit_declarations
        else _unravel_loop_index_declarations(var_name, ndim, uint_t)
    )
    if ndim == 1:
        code = f"""
        in_coord[0] = {raveled_index};\n"""
        return code

    code += f"{uint_t} temp_idx = {raveled_index};"
    for d in range(ndim - 1, 0, -1):
        code += f"""
        temp_floor = temp_idx / dim{d}_size;
        in_coord[{d}] = temp_idx - temp_floor * dim{d}_size;
        temp_idx = temp_floor;"""
    code += """
        in_coord[0] = temp_idx;"""
    return code


def _reverse_label_values(label_image, max_label):
    """reverses the value of all labels (keeping background value=0 the same)"""
    dtype = label_image.dtype
    labs = cp.asarray(tuple(range(max_label + 1)), dtype=dtype)
    rev_labs = cp.asarray((0,) + tuple(range(max_label, 0, -1)), dtype=dtype)
    return map_array(label_image, labs, rev_labs)


def _find_close_labels(labels, binary_image, max_label):
    # check possibly too-close regions for which we may need to
    # manually recompute the regions perimeter in isolation
    labels_dilated2 = ndi.grey_dilation(labels, 5, mode="constant")
    labels2 = labels_dilated2 * binary_image
    rev_labels = _reverse_label_values(labels, max_label=max_label)
    rev_labels = ndi.grey_dilation(rev_labels, 5, mode="constant")
    rev_labels = rev_labels * binary_image
    labels3 = _reverse_label_values(rev_labels, max_label=max_label)
    diffs = cp.logical_or(labels != labels2, labels != labels3)
    labels_close = cp.asnumpy(cp.unique(labels[diffs]))
    return labels_close
