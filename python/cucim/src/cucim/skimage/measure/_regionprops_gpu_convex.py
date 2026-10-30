# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import warnings
from collections.abc import Sequence

import cupy as cp
import numpy as np

from cucim.skimage._shared.distance import pdist_max_blockwise
from cucim.skimage._shared.utils import _ndarray_argwhere
from cucim.skimage._vendored import ndimage as ndi
from cucim.skimage.morphology import convex_hull_image

from ._regionprops_gpu_basic_kernels import _bbox_slices, regionprops_image
from ._regionprops_gpu_utils import _warp_includes

# Store information on which other properties a given property depends on
# This information will be used by `regionprops_dict` to make sure that when
# a particular property is requested any dependent properties are computed
# first.
convex_deps = dict()
convex_deps["image_convex"] = ["bbox"]
convex_deps["area_convex"] = ["bbox"]
convex_deps["feret_diameter_max"] = ["bbox"]
convex_deps["solidity"] = ["area", "area_convex"]


def _get_row_extent_dtype(shape):
    """Type of the row extents and of the hull vertex coordinates.

    The hull vertices are stored in doubled coordinates (2 * coordinate - 1 to
    2 * coordinate + 1), which fit in 32 bits when every axis is shorter than
    2**30.
    """
    return cp.dtype(cp.int32 if max(shape) < 2**30 else cp.int64)


def _row_extent_offsets(bbox, ndim):
    """Offsets of each label's rows in the flat row-extent arrays.

    A row is a line along the last axis inside a label's bounding box, so a
    label has prod(bbox extents along axes 0, ..., ndim - 2) rows (0 for a
    label absent from the image). Its rows are stored in raveled order.

    Returns the offsets, of shape (max_label + 1,) and type int64 on the
    device, and their total on the host.
    """
    bbox_min = bbox[:, : ndim - 1].astype(cp.int64)
    bbox_max = bbox[:, ndim : 2 * ndim - 1].astype(cp.int64)
    # absent labels have bbox_min > bbox_max
    num_rows = cp.prod(cp.maximum(bbox_max - bbox_min, 0), axis=1)
    offsets = cp.zeros((bbox.shape[0] + 1,), dtype=cp.int64)
    cp.cumsum(num_rows, out=offsets[1:])
    return offsets, int(offsets[-1])


@cp.memoize(for_each_device=True)
def _get_row_extents_kernel(ndim, int32_coords, pixels_per_thread):
    """Kernel finding the first and last pixel of each region in each row.

    Each thread scans `pixels_per_thread` consecutive pixels of one row (a
    row's pixels are split into ceil(n_cols / pixels_per_thread) chunks, so
    a thread never crosses the end of a row). For each run of a label, it
    updates the label's extents for that row with one atomicMin and one
    atomicMax. The extents are columns relative to the label's bounding box.
    """
    coord_t = "int" if int32_coords else "long long"
    source = f"""
      typedef {coord_t} cucim_coord_t;"""
    for d in range(ndim):
        source += f"""
      cucim_coord_t shape{d} = labels.shape()[{d}];"""
    ppt = pixels_per_thread
    source += f"""
      cucim_coord_t chunks = (shape{ndim - 1} + {ppt - 1}) / {ppt};
      cucim_coord_t line = static_cast<cucim_coord_t>(i) / chunks;
      cucim_coord_t c0 = (static_cast<cucim_coord_t>(i) - line * chunks)
                         * {ppt};
      cucim_coord_t c1 = min(c0 + {ppt}, shape{ndim - 1});
      cucim_coord_t rem = line;"""
    # coordinates of the row along axes 0, ..., ndim - 2
    for d in range(ndim - 2, 0, -1):
        source += f"""
      cucim_coord_t coord{d} = rem % shape{d};
      rem /= shape{d};"""
    source += """
      cucim_coord_t coord0 = rem;"""
    # index of the row in the label's rows (raveled within its bbox)
    row_index = f"""
          const B* box = &bbox[(lab - 1) * {2 * ndim}];
          long long k = 0;"""
    for d in range(ndim - 1):
        row_index += f"""
          k = k * static_cast<long long>(box[{ndim + d}] - box[{d}])
              + static_cast<long long>(coord{d} - box[{d}]);"""
    flush = f"""{row_index}
          k += offsets[lab - 1];
          E col_min = static_cast<E>(box[{ndim - 1}]);
          atomicMin(&row_first[k], first - col_min);
          atomicMax(&row_last[k], last - col_min);"""
    source += f"""
      cucim_sequential_reader<X> label_reader(
          &labels[static_cast<long long>(line) * shape{ndim - 1} + c0], {ppt},
          c1 - c0 == {ppt});
      X lab = 0;
      E first = 0;
      E last = 0;
      for (cucim_coord_t c = c0; c < c1; c++) {{
        X current_label = label_reader.get(c - c0);
        if (current_label == 0) {{ continue; }}
        if (current_label != lab) {{
          if (lab != 0) {{{flush}
          }}
          lab = current_label;
          first = c;
        }}
        last = c;
      }}
      if (lab != 0) {{{flush}
      }}\n"""
    inputs = "raw X labels, raw B bbox, raw int64 offsets"
    outputs = "raw E row_first, raw E row_last"
    name = f"cucim_row_extents_{ndim}d_batch{ppt}"
    if not int32_coords:
        name += "_coords64"
    return cp.ElementwiseKernel(
        inputs, outputs, source, preamble=_warp_includes, name=name
    )


def _regionprops_row_extents(
    label_image, bbox, max_label, pixels_per_thread=16
):
    """First and last pixel of each region in each row of its bounding box.

    Parameters
    ----------
    label_image : cupy.ndarray
        C-contiguous label image with at least 2 dimensions.
    bbox : cupy.ndarray
        The "bbox" of `_regionprops_label_pass`: minimum coordinates, then
        maximum coordinates (exclusive), of shape (max_label, 2 * ndim).

    Returns
    -------
    row_first, row_last : cupy.ndarray
        For each label's rows (see `_row_extent_offsets`), the first and last
        column of the label in that row, relative to the bounding box. Rows
        without any pixel of the label have row_first > row_last.
    offsets : cupy.ndarray
        Offsets of each label's rows, of shape (max_label + 1,).
    num_rows : int
        Total number of rows (``offsets[-1]``).
    """
    ndim = label_image.ndim
    if not label_image.flags.c_contiguous:
        label_image = cp.ascontiguousarray(label_image)
    offsets, num_rows = _row_extent_offsets(bbox, ndim)
    extent_dtype = _get_row_extent_dtype(label_image.shape)
    row_first = cp.full(
        (num_rows,), cp.iinfo(extent_dtype).max, dtype=extent_dtype
    )
    row_last = cp.full((num_rows,), -1, dtype=extent_dtype)
    if num_rows == 0:
        return row_first, row_last, offsets, num_rows
    n_cols = label_image.shape[-1]
    chunks = math.ceil(n_cols / pixels_per_thread)
    num_threads = (label_image.size // n_cols) * chunks
    kernel = _get_row_extents_kernel(
        ndim, num_threads * pixels_per_thread < 2**31, pixels_per_thread
    )
    kernel(label_image, bbox, offsets, row_first, row_last, size=num_threads)
    return row_first, row_last, offsets, num_rows


# Exact integer helpers of the 2D convex-hull kernels. Points are (y, x)
# pairs in doubled coordinates (2 * row, 2 * column), so that the diamond
# points of a pixel (its center +- 0.5 along each axis) have integer
# coordinates.
_hull_2d_includes = (
    _warp_includes
    + r"""
// z component of (a - o) x (b - o), with y as the first axis: > 0 if
// o -> a -> b turns towards larger x (to the right when y points down). The
// differences are taken in T (the doubled coordinates fit in T, see
// _get_row_extent_dtype) and only the products in 64 bits.
template <typename T>
__device__ __forceinline__ long long cucim_cross(T oy, T ox, T ay, T ax,
                                                 T by, T bx) {
  return static_cast<long long>(ay - oy) * (bx - ox)
         - static_cast<long long>(ax - ox) * (by - oy);
}

// Andrew's monotone chain over points with increasing y. The last two
// vertices are kept in registers, so that a point that does not remove any
// vertex needs no read of the vertices already stored.
template <typename T>
struct cucim_hull_chain {
  T* v;  // the vertices, as (y, x) pairs
  int n;
  T y0, x0;  // the vertex before the last (if n >= 2)
  T y1, x1;  // the last vertex (if n >= 1)

  __device__ cucim_hull_chain(T* vertices) : v(vertices), n(0) {}

  // side = 1 keeps the chain convex towards smaller x (left chain), side =
  // -1 towards larger x (right chain). Collinear vertices are removed.
  __device__ __forceinline__ void push(T y, T x, int side) {
    while (n >= 2 && side * cucim_cross(y0, x0, y1, x1, y, x) <= 0) {
      n--;
      y1 = y0;
      x1 = x0;
      if (n >= 2) {
        y0 = v[2 * n - 4];
        x0 = v[2 * n - 3];
      }
    }
    v[2 * n] = y;
    v[2 * n + 1] = x;
    n++;
    y0 = y1;
    x0 = x1;
    y1 = y;
    x1 = x;
  }
};

// Adds candidate points (with increasing y) to a cucim_hull_chain, skipping
// those that are not a strict turn towards the chain's side with the
// previous and next candidate: such a point lies on or inside the triangle
// of its two neighbors and a point on the other side of the hull at the same
// y, so it is not a vertex. On straight or staircase boundaries this skips
// most points, and with them the vertex removals of the chain, which would
// read vertices back from memory. The first and last points are always
// added.
template <typename T>
struct cucim_hull_candidates {
  cucim_hull_chain<T> chain;
  int side;
  int m;  // number of candidates so far (counting up to 2)
  T ay, ax;  // the candidate before the pending one
  T by, bx;  // the pending candidate

  __device__ cucim_hull_candidates(T* vertices, int chain_side)
      : chain(vertices), side(chain_side), m(0) {}

  __device__ __forceinline__ void add(T y, T x) {
    if (m == 0) {
      chain.push(y, x, side);
      ay = y;
      ax = x;
      m = 1;
      return;
    }
    if (m == 2 && side * cucim_cross(ay, ax, by, bx, y, x) > 0) {
      chain.push(by, bx, side);
    }
    if (m == 2) {
      ay = by;
      ax = bx;
    }
    by = y;
    bx = x;
    m = 2;
  }

  __device__ __forceinline__ void finish() {
    if (m == 2) {
      chain.push(by, bx, side);
    }
  }
};

// floor(a / b) and ceil(a / b) for b > 0
__device__ __forceinline__ long long cucim_floor_div(long long a,
                                                    long long b) {
  long long q = a / b;
  return (a % b != 0 && a < 0) ? q - 1 : q;
}

__device__ __forceinline__ long long cucim_ceil_div(long long a,
                                                   long long b) {
  return -cucim_floor_div(-a, b);
}
"""
)


@cp.memoize(for_each_device=True)
def _get_hull_chains_2d_kernel(centers_only=False):
    """Kernel building the convex hull of each 2D region, one thread per label.

    The hull of a region's diamond points (each pixel center +- 0.5 along
    each axis, as in scikit-image's ``convex_hull_image``) is the hull of the
    diamond points of each row's first and last pixel. On each horizontal
    line (y = 2 * row - 1, ..., 2 * row + 1 in doubled coordinates), only the
    leftmost and rightmost of these points can be hull vertices:

        y = 2k (row k):         x = 2 * first[k] - 1 and 2 * last[k] + 1
        y = 2k + 1 (between     x = 2 * min(first[k], first[k + 1]) and
            rows k and k + 1):      2 * max(last[k], last[k + 1])

    (rows without pixels are skipped). These points come sorted by y, so
    Andrew's monotone chain builds the left chain of the hull from the
    leftmost points and the right chain from the rightmost points in one
    pass, without sorting. Collinear points are dropped, and the arithmetic
    is exact.

    A thread works alone on its label, so it waits for every memory read
    and every dependent instruction. To keep its work short:
    - A point between two rows is skipped if the two rows' ends differ by
      at most 1 (it then lies on or inside the segment between their
      points).
    - A point that is not a strict turn with its neighboring points is not
      a vertex, and is skipped before it reaches the chain
      (``cucim_hull_candidates``). On straight and staircase boundaries
      this skips most points, and with them most vertex removals.
    - The chain keeps its last two vertices in registers
      (``cucim_hull_chain``), so that it reads stored vertices back only
      when it removes one.
    - Coordinate differences are taken in 32 bits when the coordinates fit.

    Each chain runs from the top line (y = -1) to the bottom line (y = 2h - 1,
    for a bounding box of h rows), with strictly increasing y. A label's
    chains have room for 2h + 1 vertices each, at ``2 * offsets[label - 1] +
    label - 1``.

    With `centers_only`, the points are the pixel centers of each row's
    first and last pixel instead (x = 2 * first[k] and 2 * last[k] at y = 2k),
    as needed for the planes between two planes of a 3D region (see
    `_convex_hull_candidates_3d`).

    The kernel works on any sequence of "segments" of rows: segment i has
    the rows offsets[i] to offsets[i + 1] - 1.
    """
    source = """
      long long r0 = offsets[i];
      long long h = offsets[i + 1] - r0;
      long long base = 2 * r0 + i;
      cucim_hull_candidates<E> left(&left_chains[2 * base], 1);
      cucim_hull_candidates<E> right(&right_chains[2 * base], -1);
"""
    if centers_only:
        source += """
      for (E k = 0; k < h; k++) {
        E first = row_first[r0 + k];
        E last = row_last[r0 + k];
        if (first <= last) {
          left.add(2 * k, 2 * first);
          right.add(2 * k, 2 * last);
        }
      }
"""
    else:
        source += """
      // extents of the previous row (none before the first row)
      E prev_first = 1;
      E prev_last = 0;
      // row h (after the last row) is empty: only the line above it is added
      for (E k = 0; k <= h; k++) {
        E first = k < h ? row_first[r0 + k] : 1;
        E last = k < h ? row_last[r0 + k] : 0;
        // the line between rows k - 1 and k (y = 2k - 1). If both rows have
        // pixels and their ends differ by at most 1, its point lies on or
        // inside the segment between the two rows' points, so it is skipped.
        bool prev_nonempty = prev_first <= prev_last;
        bool nonempty = first <= last;
        if (!(prev_nonempty && nonempty)) {
          if (prev_nonempty || nonempty) {
            E x_min = nonempty ? first : prev_first;
            E x_max = nonempty ? last : prev_last;
            left.add(2 * k - 1, 2 * x_min);
            right.add(2 * k - 1, 2 * x_max);
          }
        } else {
          if (abs(first - prev_first) > 1) {
            left.add(2 * k - 1, 2 * min(prev_first, first));
          }
          if (abs(last - prev_last) > 1) {
            right.add(2 * k - 1, 2 * max(prev_last, last));
          }
        }
        // row k (y = 2k)
        if (nonempty) {
          left.add(2 * k, 2 * first - 1);
          right.add(2 * k, 2 * last + 1);
        }
        prev_first = first;
        prev_last = last;
      }
"""
    source += """
      left.finish();
      right.finish();
      chain_sizes[2 * i] = left.chain.n;
      chain_sizes[2 * i + 1] = right.chain.n;\n"""
    inputs = "raw E row_first, raw E row_last, raw int64 offsets"
    outputs = "raw E left_chains, raw E right_chains, raw int32 chain_sizes"
    name = "cucim_convex_hull_chains_2d"
    if centers_only:
        name += "_centers"
    return cp.ElementwiseKernel(
        inputs,
        outputs,
        source,
        preamble=_hull_2d_includes,
        name=name,
    )


@cp.memoize(for_each_device=True)
def _get_hull_rows_2d_kernel():
    """Kernel finding the pixels of each row inside the 2D convex hull.

    One thread per row of a label's bounding box. The pixel centers of row k
    inside the convex polygon form one interval [lo, hi] of columns: lo is
    the first column whose center (y = 2k, x = 2 * column) is not left of
    the left chain, and hi the last one not right of the right chain. Centers
    on an edge are inside, as in scikit-image's ``grid_points_in_poly`` with
    ``include_borders=True``.

    Also adds the number of pixels in the interval to the label's count.
    """
    source = """
      // the label of this row: the last l with offsets[l] <= i
      long long l = 0;
      long long l_end = max_label;
      while (l_end - l > 1) {
        long long mid = (l + l_end) / 2;
        if (offsets[mid] <= i) {
          l = mid;
        } else {
          l_end = mid;
        }
      }
      long long y = 2 * (static_cast<long long>(i) - offsets[l]);
      long long base = 2 * offsets[l] + l;
      long long bounds[2];
      for (int side = 0; side < 2; side++) {
        const E* chain = side == 0 ? &left_chains[2 * base]
                                   : &right_chains[2 * base];
        // the chain's edge from vertex j to j + 1 with chain y[j] <= y <
        // y[j + 1] (the first vertex is above every row center, the last one
        // below)
        int j = 0;
        int j_end = chain_sizes[2 * l + side] - 1;
        while (j_end - j > 1) {
          int mid = (j + j_end) / 2;
          if (chain[2 * mid] <= y) {
            j = mid;
          } else {
            j_end = mid;
          }
        }
        long long ay = chain[2 * j];
        long long ax = chain[2 * j + 1];
        long long by = chain[2 * j + 2];
        long long bx = chain[2 * j + 3];
        // the edge's x at y is num / (by - ay); column c is inside on this
        // side if 2 * c >= x (left) or 2 * c <= x (right)
        long long num = ax * (by - ay) + (bx - ax) * (y - ay);
        bounds[side] = side == 0 ? cucim_ceil_div(num, 2 * (by - ay))
                                 : cucim_floor_div(num, 2 * (by - ay));
      }
      row_lo = bounds[0];
      row_hi = bounds[1];
      if (bounds[1] >= bounds[0]) {
        atomicAdd(&num_pixels[l], static_cast<unsigned long long>(
            bounds[1] - bounds[0] + 1));
      }\n"""
    inputs = (
        "raw E left_chains, raw E right_chains, raw int32 chain_sizes, "
        "raw int64 offsets, int64 max_label"
    )
    outputs = "E row_lo, E row_hi, raw uint64 num_pixels"
    return cp.ElementwiseKernel(
        inputs,
        outputs,
        source,
        preamble=_hull_2d_includes,
        name="cucim_convex_hull_rows_2d",
    )


def _convex_hull_chains_2d(
    row_first, row_last, offsets, num_rows, max_label, centers_only=False
):
    """Left and right hull chains of each 2D region, and their sizes.

    Takes row extents in the layout of `_regionprops_row_extents` (of the
    regions, or of their convex images). See `_get_hull_chains_2d_kernel`.
    `max_label` is the number of segments of rows (offsets has one more
    element).
    """
    dtype = row_first.dtype
    capacity = 2 * num_rows + max_label
    left_chains = cp.empty((capacity, 2), dtype=dtype)
    right_chains = cp.empty((capacity, 2), dtype=dtype)
    chain_sizes = cp.empty((max_label, 2), dtype=cp.int32)
    _get_hull_chains_2d_kernel(centers_only)(
        row_first,
        row_last,
        offsets,
        left_chains,
        right_chains,
        chain_sizes,
        size=max_label,
    )
    return left_chains, right_chains, chain_sizes


def _convex_hull_rows_2d(row_first, row_last, offsets, num_rows, max_label):
    """Rows of each 2D region's convex image.

    Takes the output of `_regionprops_row_extents`. Returns the first and last
    column of each row of each convex image (relative to the bounding box, in
    the same layout as the row extents) and the number of pixels of each
    convex image.
    """
    chains = _convex_hull_chains_2d(
        row_first, row_last, offsets, num_rows, max_label
    )
    dtype = row_first.dtype
    row_lo = cp.empty((num_rows,), dtype=dtype)
    row_hi = cp.empty((num_rows,), dtype=dtype)
    num_pixels = cp.zeros((max_label,), dtype=cp.uint64)
    if num_rows > 0:
        _get_hull_rows_2d_kernel()(
            *chains,
            offsets,
            max_label,
            row_lo,
            row_hi,
            num_pixels,
        )
    return row_lo, row_hi, num_pixels


@cp.memoize(for_each_device=True)
def _get_feret_diameter_max_2d_kernel():
    """Kernel computing the maximum Feret diameter of each 2D region.

    One warp per label. The vertices are those of the hull chains of the
    label's convex image (in doubled coordinates); each lane takes every 32nd
    vertex and its distances to all later vertices, and the warp keeps the
    largest. The distances are scaled by the spacing.
    """
    source = """
      long long l = i / 32;
      int lane = static_cast<int>(i & 31);
      long long base = 2 * offsets[l] + l;
      const E* left = &left_chains[2 * base];
      const E* right = &right_chains[2 * base];
      int n_left = chain_sizes[2 * l];
      int n = n_left + chain_sizes[2 * l + 1];
      double best = 0.0;
      for (int p = lane; p < n; p += 32) {
        const E* vp = p < n_left ? &left[2 * p] : &right[2 * (p - n_left)];
        E py = vp[0];
        E px = vp[1];
        for (int q = p + 1; q < n; q++) {
          const E* vq = q < n_left ? &left[2 * q]
                                   : &right[2 * (q - n_left)];
          double dy = static_cast<double>(vq[0] - py) * half_spacing0;
          double dx = static_cast<double>(vq[1] - px) * half_spacing1;
          best = fmax(best, dy * dy + dx * dx);
        }
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        best = fmax(best, __shfl_xor_sync(0xffffffffu, best, offset));
      }
      if (lane == 0) {
        feret[l] = sqrt(best);
      }\n"""
    inputs = (
        "raw E left_chains, raw E right_chains, raw int32 chain_sizes, "
        "raw int64 offsets, float64 half_spacing0, float64 half_spacing1"
    )
    return cp.ElementwiseKernel(
        inputs,
        "raw float64 feret",
        source,
        preamble=_warp_includes,
        name="cucim_feret_diameter_max_2d",
    )


def _feret_diameter_max_2d(
    row_lo, row_hi, offsets, num_rows, max_label, spacing=None
):
    """Maximum Feret diameter of each 2D region, from its convex image's rows.

    scikit-image takes the largest distance between the contour points of the
    convex image at level 0.5. For a binary image these are the diamond
    points of its boundary pixels, and the largest distance between points is
    reached at two vertices of their convex hull, which is also the hull of
    the diamond points of each row's first and last pixel. So the result is
    the largest distance between the vertices of the hull chains of the
    convex image (`_get_hull_chains_2d_kernel`).
    """
    chains = _convex_hull_chains_2d(
        row_lo, row_hi, offsets, num_rows, max_label
    )
    spacing = (1.0, 1.0) if spacing is None else spacing
    feret = cp.empty((max_label,), dtype=cp.float64)
    _get_feret_diameter_max_2d_kernel()(
        *chains,
        offsets,
        0.5 * float(spacing[0]),
        0.5 * float(spacing[1]),
        feret,
        size=32 * max_label,
    )
    return feret


@cp.memoize(for_each_device=True)
def _get_merged_planes_kernel():
    """Kernel merging the row extents of each two neighboring planes.

    For a 3D label with d planes of h rows in its bounding box, merged plane m
    (m = 0, ..., d) lies between planes m - 1 and m; each of its rows has the
    smallest first and the largest last column of that row in the two planes
    (planes -1 and d are empty; an empty merged row gets first > last). One
    thread per row of the merged planes,
    whose rows are stored like the row extents, at ``merged_offsets``.
    """
    source = """
      // the label of this row: the last l with merged_offsets[l] <= i
      long long l = 0;
      long long l_end = max_label;
      while (l_end - l > 1) {
        long long mid = (l + l_end) / 2;
        if (merged_offsets[mid] <= i) {
          l = mid;
        } else {
          l_end = mid;
        }
      }
      long long local = static_cast<long long>(i) - merged_offsets[l];
      long long d = static_cast<long long>(bbox[l * 6 + 3] - bbox[l * 6]);
      long long h = static_cast<long long>(bbox[l * 6 + 4] - bbox[l * 6 + 1]);
      long long m = local / h;
      long long y = local - m * h;
      E first = cuda::std::numeric_limits<E>::max();
      E last = -1;
      for (long long plane = m - 1; plane <= m; plane++) {
        if (plane < 0 || plane >= d) { continue; }
        long long r = offsets[l] + plane * h + y;
        // (rows without pixels have row_first > row_last)
        if (row_first[r] <= row_last[r]) {
          first = min(first, row_first[r]);
          last = max(last, row_last[r]);
        }
      }
      merged_first = first;
      merged_last = last;\n"""
    inputs = (
        "raw E row_first, raw E row_last, raw int64 offsets, "
        "raw int64 merged_offsets, raw B bbox, int64 max_label"
    )
    return cp.ElementwiseKernel(
        inputs,
        "E merged_first, E merged_last",
        source,
        preamble=_warp_includes,
        name="cucim_convex_merged_planes_3d",
    )


@cp.memoize(for_each_device=True)
def _get_gather_chains_kernel():
    """Kernel copying the chain vertices of each segment of rows (a plane of
    a 3D label) to consecutive (z, y, x) points, one thread per segment."""
    source = """
      long long base = 2 * segment_offsets[i] + i;
      long long out = point_starts[i];
      E z = segment_z[i];
      int lab = segment_label[i];
      for (int side = 0; side < 2; side++) {
        const E* chain = side == 0 ? &left_chains[2 * base]
                                   : &right_chains[2 * base];
        int n = chain_sizes[2 * i + side];
        for (int j = 0; j < n; j++, out++) {
          points[3 * out] = z;
          points[3 * out + 1] = chain[2 * j];
          points[3 * out + 2] = chain[2 * j + 1];
          point_labels[out] = lab;
        }
      }\n"""
    inputs = (
        "raw E left_chains, raw E right_chains, raw int32 chain_sizes, "
        "raw int64 segment_offsets, raw int64 point_starts, raw E segment_z, "
        "raw int32 segment_label"
    )
    return cp.ElementwiseKernel(
        inputs,
        "raw E points, raw int32 point_labels",
        source,
        name="cucim_convex_gather_chains",
    )


def _gather_chain_vertices(chains, segment_offsets, segment_z, segment_label):
    """Vertices of the chains of each segment as (z, y, x) points.

    Returns the points (doubled coordinates) and the label index of each.
    """
    left_chains, right_chains, chain_sizes = chains
    point_starts = cp.zeros((segment_z.size + 1,), dtype=cp.int64)
    cp.cumsum(chain_sizes.sum(axis=1), out=point_starts[1:])
    num_points = int(point_starts[-1])
    points = cp.empty((num_points, 3), dtype=left_chains.dtype)
    point_labels = cp.empty((num_points,), dtype=cp.int32)
    if segment_z.size > 0:
        _get_gather_chains_kernel()(
            left_chains,
            right_chains,
            chain_sizes,
            cp.asarray(segment_offsets),
            point_starts,
            cp.asarray(segment_z, dtype=left_chains.dtype),
            cp.asarray(segment_label, dtype=cp.int32),
            points,
            point_labels,
            size=segment_z.size,
        )
    return points, point_labels


def _convex_hull_candidates_3d(
    row_first, row_last, offsets, num_rows, bbox, bbox_host, max_label
):
    """Candidate vertices of the convex hull of each 3D region.

    The hull of a region's diamond points is the hull of the diamond points
    of each row's first and last voxel. In doubled coordinates these lie in
    the planes z = 2 * plane (the in-plane diamond points of the plane's row
    ends) and z = 2 * plane +- 1 (the row ends' centers). A vertex of the 3D
    hull is also a vertex of the 2D hull of the points in its plane, so the
    candidates are the vertices of these 2D hulls, built by the 2D chain
    kernel for all planes of all labels at once (the planes between two
    planes of a label with ``centers_only``, on merged row extents).

    Returns the points, as (z, y, x) in doubled coordinates relative to the
    bounding box, grouped by label, and the offsets of each label's points
    (on the device).
    """
    shapes = np.maximum(bbox_host[:, 3:] - bbox_host[:, :3], 0)
    depth, height = shapes[:, 0], shapes[:, 1]
    offsets_host = np.zeros((max_label + 1,), dtype=np.int64)
    np.cumsum(depth * height, out=offsets_host[1:])
    labels = np.arange(max_label)

    def planes(num_planes):
        # label and plane index of each plane of each label
        plane_label = np.repeat(labels, num_planes)
        starts = np.zeros_like(num_planes)
        np.cumsum(num_planes[:-1], out=starts[1:])
        plane = np.arange(num_planes.sum()) - np.repeat(starts, num_planes)
        return plane_label, plane

    # planes of the regions: z = 2 * plane
    plane_label, plane = planes(depth)
    segment_offsets = np.append(
        offsets_host[plane_label] + plane * height[plane_label], num_rows
    )
    chains = _convex_hull_chains_2d(
        row_first,
        row_last,
        cp.asarray(segment_offsets),
        num_rows,
        plane.size,
    )
    points, point_labels = _gather_chain_vertices(
        chains, segment_offsets, 2 * plane, plane_label
    )

    # planes between them (and before the first and after the last):
    # z = 2 * m - 1 for merged plane m
    merged_planes = np.where(depth > 0, depth + 1, 0)
    merged_offsets = np.zeros((max_label + 1,), dtype=np.int64)
    np.cumsum(merged_planes * height, out=merged_offsets[1:])
    num_merged_rows = int(merged_offsets[-1])
    merged_first = cp.empty((num_merged_rows,), dtype=row_first.dtype)
    merged_last = cp.empty((num_merged_rows,), dtype=row_first.dtype)
    _get_merged_planes_kernel()(
        row_first,
        row_last,
        offsets,
        cp.asarray(merged_offsets),
        bbox,
        max_label,
        merged_first,
        merged_last,
    )
    plane_label, plane = planes(merged_planes)
    segment_offsets = np.append(
        merged_offsets[plane_label] + plane * height[plane_label],
        num_merged_rows,
    )
    chains = _convex_hull_chains_2d(
        merged_first,
        merged_last,
        cp.asarray(segment_offsets),
        num_merged_rows,
        plane.size,
        centers_only=True,
    )
    merged_points, merged_point_labels = _gather_chain_vertices(
        chains, segment_offsets, 2 * plane - 1, plane_label
    )

    points = cp.concatenate((points, merged_points))
    point_labels = cp.concatenate((point_labels, merged_point_labels))
    order = cp.argsort(point_labels)
    point_offsets = cp.zeros((max_label + 1,), dtype=cp.int64)
    cp.cumsum(
        cp.bincount(point_labels, minlength=max_label), out=point_offsets[1:]
    )
    return points[order], point_offsets


@cp.memoize(for_each_device=True)
def _get_hull_rows_3d_kernel():
    """Kernel finding the voxels of each row inside the 3D convex hull.

    One thread per row of a label's bounding box. A voxel center p (relative
    to the bounding box) is inside if ``n . p + b < tolerance`` for every
    facet equation (n, b) of the label's hull, as in scikit-image's
    ``convex_hull_image`` in 3D. Along a row, each facet with n_x != 0 bounds
    x on one side, so the voxels inside form one interval [lo, hi] of x.

    Also adds the number of voxels in the interval to the label's count.
    """
    source = """
      // the label of this row: the last l with offsets[l] <= i
      long long l = 0;
      long long l_end = max_label;
      while (l_end - l > 1) {
        long long mid = (l + l_end) / 2;
        if (offsets[mid] <= i) {
          l = mid;
        } else {
          l_end = mid;
        }
      }
      long long local = static_cast<long long>(i) - offsets[l];
      long long h = static_cast<long long>(bbox[l * 6 + 4] - bbox[l * 6 + 1]);
      long long w = static_cast<long long>(bbox[l * 6 + 5] - bbox[l * 6 + 2]);
      double z = static_cast<double>(local / h);
      double y = static_cast<double>(local % h);
      double lo = 0.0;
      double hi = static_cast<double>(w - 1);
      for (long long f = eq_offsets[l]; f < eq_offsets[l + 1]; f++) {
        const double* e = &equations[4 * f];
        // inside: e[2] * x < r
        double r = tolerance - (e[0] * z + e[1] * y + e[3]);
        if (e[2] > 0) {
          hi = fmin(hi, ceil(r / e[2]) - 1.0);
        } else if (e[2] < 0) {
          lo = fmax(lo, floor(r / e[2]) + 1.0);
        } else if (r <= 0) {
          hi = -1.0;
        }
      }
      // (clamped so that the conversion cannot overflow)
      lo = fmin(lo, static_cast<double>(w));
      hi = fmax(hi, -1.0);
      row_lo = static_cast<E>(lo);
      row_hi = static_cast<E>(hi);
      if (hi >= lo) {
        atomicAdd(&num_pixels[l],
                  static_cast<unsigned long long>(hi - lo + 1.0));
      }\n"""
    inputs = (
        "raw float64 equations, raw int64 eq_offsets, raw int64 offsets, "
        "raw B bbox, float64 tolerance, int64 max_label"
    )
    return cp.ElementwiseKernel(
        inputs,
        "E row_lo, E row_hi, raw uint64 num_pixels",
        source,
        preamble=_warp_includes,
        name="cucim_convex_hull_rows_3d",
    )


@cp.memoize(for_each_device=True)
def _get_feret_diameter_max_points_kernel(ndim):
    """Kernel computing the largest distance between each label's points.

    One warp per label; each lane takes every 32nd point and its distances to
    all later points of the label. The points are in doubled coordinates,
    and the distances are scaled by half the spacing.
    """
    load = "".join(
        f"""
        E p{d} = points[{ndim} * (start + p) + {d}];"""
        for d in range(ndim)
    )
    distance = "".join(
        f"""
          double t{d} = static_cast<double>(q_point[{d}] - p{d})
                        * half_spacing{d};
          d2 += t{d} * t{d};"""
        for d in range(ndim)
    )
    source = f"""
      long long l = i / 32;
      int lane = static_cast<int>(i & 31);
      long long start = point_offsets[l];
      long long n = point_offsets[l + 1] - start;
      double best = 0.0;
      for (long long p = lane; p < n; p += 32) {{{load}
        for (long long q = p + 1; q < n; q++) {{
          const E* q_point = &points[{ndim} * (start + q)];
          double d2 = 0.0;{distance}
          best = fmax(best, d2);
        }}
      }}
      for (int offset = 16; offset > 0; offset >>= 1) {{
        best = fmax(best, __shfl_xor_sync(0xffffffffu, best, offset));
      }}
      if (lane == 0) {{
        feret[l] = sqrt(best);
      }}\n"""
    inputs = "raw E points, raw int64 point_offsets, " + ", ".join(
        f"float64 half_spacing{d}" for d in range(ndim)
    )
    return cp.ElementwiseKernel(
        inputs,
        "raw float64 feret",
        source,
        name=f"cucim_feret_diameter_max_points_{ndim}d",
    )


def _feret_diameter_max_3d(
    row_lo, row_hi, offsets, num_rows, bbox, bbox_host, max_label, spacing
):
    """Maximum Feret diameter of each 3D region, from its convex image's rows.

    As in 2D (`_feret_diameter_max_2d`), scikit-image's result (the largest
    distance between the marching-cubes vertices of the convex image at
    level 0.5) is the largest distance between the diamond points of the
    convex image's row ends. The candidates of `_convex_hull_candidates_3d`
    contain all vertices of their hull and are among these points, so the
    largest distance between them is the same. No 3D hull is needed.
    """
    points, point_offsets = _convex_hull_candidates_3d(
        row_lo, row_hi, offsets, num_rows, bbox, bbox_host, max_label
    )
    spacing = (1.0, 1.0, 1.0) if spacing is None else spacing
    feret = cp.empty((max_label,), dtype=cp.float64)
    _get_feret_diameter_max_points_kernel(3)(
        points,
        point_offsets,
        *(0.5 * float(s) for s in spacing),
        feret,
        size=32 * max_label,
    )
    return feret


def _convex_hull_rows_3d(
    label_image, row_first, row_last, offsets, num_rows, bbox, max_label
):
    """Rows of each 3D region's convex image (see `_convex_hull_rows_2d`).

    The hull of each label is computed on the host with
    ``scipy.spatial.ConvexHull`` from the candidates of
    `_convex_hull_candidates_3d` (one transfer for all labels), and its facet
    equations are copied back to the device in one transfer for
    `_get_hull_rows_3d_kernel`. If qhull fails for a label, its convex image
    is computed with ``convex_hull_image`` instead.
    """
    from scipy.spatial import ConvexHull, QhullError

    bbox_host = cp.asnumpy(bbox).astype(np.int64)
    points, point_offsets = _convex_hull_candidates_3d(
        row_first, row_last, offsets, num_rows, bbox, bbox_host, max_label
    )
    points = cp.asnumpy(points)
    point_offsets = cp.asnumpy(point_offsets)
    # the offsets of each label's rows, on the host
    shapes = np.maximum(bbox_host[:, 3:] - bbox_host[:, :3], 0)
    row_offsets_host = np.zeros((max_label + 1,), dtype=np.int64)
    np.cumsum(shapes[:, 0] * shapes[:, 1], out=row_offsets_host[1:])
    points = points * 0.5
    equations = []
    num_equations = np.zeros((max_label,), dtype=np.int64)
    failed = []
    for lab in range(max_label):
        label_points = points[point_offsets[lab] : point_offsets[lab + 1]]
        if label_points.shape[0] == 0:
            continue  # absent label
        try:
            hull = ConvexHull(label_points)
        except QhullError:
            failed.append(lab)
            continue
        equations.append(hull.equations)
        num_equations[lab] = hull.equations.shape[0]
    eq_offsets = np.zeros((max_label + 1,), dtype=np.int64)
    np.cumsum(num_equations, out=eq_offsets[1:])
    equations = np.concatenate(equations) if equations else np.zeros((0, 4))

    dtype = row_first.dtype
    row_lo = cp.empty((num_rows,), dtype=dtype)
    row_hi = cp.empty((num_rows,), dtype=dtype)
    num_pixels = cp.zeros((max_label,), dtype=cp.uint64)
    if num_rows > 0:
        _get_hull_rows_3d_kernel()(
            cp.asarray(equations, dtype=cp.float64),
            cp.asarray(eq_offsets),
            offsets,
            bbox,
            1e-10,
            max_label,
            row_lo,
            row_hi,
            num_pixels,
        )
    for lab in failed:
        # one convex image with convex_hull_image, then its rows
        sl = tuple(
            slice(int(bbox_host[lab, d]), int(bbox_host[lab, d + 3]))
            for d in range(3)
        )
        image = convex_hull_image(
            label_image[sl] == lab + 1,
            omit_empty_coords_check=True,
            float64_computation=True,
        )
        rows = image.reshape(-1, image.shape[-1])
        nonempty = rows.any(axis=1)
        width = rows.shape[1]
        first = cp.where(nonempty, rows.argmax(axis=1), width)
        last = cp.where(nonempty, width - 1 - rows[:, ::-1].argmax(axis=1), -1)
        rows = slice(int(row_offsets_host[lab]), int(row_offsets_host[lab + 1]))
        row_lo[rows] = first
        row_hi[rows] = last
        num_pixels[lab] = int(image.sum())
    return row_lo, row_hi, num_pixels


@cp.memoize(for_each_device=True)
def _get_convex_image_kernel(ndim, int32_index):
    """Kernel writing the convex images of all labels into one flat buffer.

    One thread per pixel of the output. Label l's image (its bounding box,
    in C order) starts at ``image_offsets[l]``; its lines along the last
    axis are the label's rows in the row-interval arrays.
    """
    index_t = "int" if int32_index else "long long"
    source = f"""
      typedef {index_t} cucim_index_t;
      // the label of this pixel: the last l with image_offsets[l] <= i
      long long l = 0;
      long long l_end = max_label;
      while (l_end - l > 1) {{
        long long mid = (l + l_end) / 2;
        if (image_offsets[mid] <= i) {{
          l = mid;
        }} else {{
          l_end = mid;
        }}
      }}
      cucim_index_t local = static_cast<cucim_index_t>(i - image_offsets[l]);
      cucim_index_t width = static_cast<cucim_index_t>(
          bbox[l * {2 * ndim} + {2 * ndim - 1}]
          - bbox[l * {2 * ndim} + {ndim - 1}]);
      cucim_index_t line = local / width;
      cucim_index_t col = local - line * width;
      long long row = offsets[l] + line;
      image = row_lo[row] <= col && col <= row_hi[row];\n"""
    inputs = (
        "raw E row_lo, raw E row_hi, raw int64 offsets, "
        "raw int64 image_offsets, raw B bbox, int64 max_label"
    )
    name = f"cucim_convex_image_{ndim}d"
    if not int32_index:
        name += "_index64"
    return cp.ElementwiseKernel(
        inputs, "bool image", source, preamble=_warp_includes, name=name
    )


def _convex_images(row_lo, row_hi, offsets, bbox, max_label):
    """The convex image of each label, from the intervals of its rows.

    The images are written into one buffer by one kernel, and returned as a
    tuple of arrays on that buffer (an absent label gets an empty image).
    """
    ndim = bbox.shape[1] // 2
    bbox_host = cp.asnumpy(bbox).astype(np.int64)
    shapes = np.maximum(bbox_host[:, ndim:] - bbox_host[:, :ndim], 0)
    image_offsets = np.zeros((max_label + 1,), dtype=np.int64)
    np.cumsum(shapes.prod(axis=1), out=image_offsets[1:])
    total = int(image_offsets[-1])
    images = cp.empty((total,), dtype=bool)
    if total > 0:
        kernel = _get_convex_image_kernel(ndim, total < 2**31)
        kernel(
            row_lo,
            row_hi,
            offsets,
            cp.asarray(image_offsets),
            bbox,
            max_label,
            images,
        )
    # arrays on the buffer (cheaper to create than slices of it)
    memptr = images.data
    return tuple(
        cp.ndarray(tuple(shape), dtype=bool, memptr=memptr + offset)
        for shape, offset in zip(shapes.tolist(), image_offsets[:-1].tolist())
    )


def _pixel_area(spacing):
    if spacing is None:
        return 1.0
    if isinstance(spacing, cp.ndarray):
        return cp.prod(spacing)
    return math.prod(spacing)


def _regionprops_convex(
    label_image,
    bbox,
    max_label,
    *,
    spacing=None,
    compute_image=False,
    compute_area=False,
    compute_feret=False,
    props_dict=None,
):
    """Compute the convex-hull properties of each region.

    writes "image_convex" to props_dict if `compute_image`
    writes "area_convex" to props_dict if `compute_area`
    writes "feret_diameter_max" to props_dict if `compute_feret`

    In 2D and 3D, all properties are computed for all labels at once from
    the hull of each region's row extents (see `_get_hull_chains_2d_kernel`,
    `_feret_diameter_max_2d`, `_convex_hull_rows_3d` and
    `_feret_diameter_max_3d`). In other dimensions, they are computed from
    one convex image per label (`regionprops_image`).

    Parameters
    ----------
    bbox : cupy.ndarray
        The "bbox" of `_regionprops_label_pass` (exclusive maximum).
    """
    if props_dict is None:
        props_dict = {}
    if not (compute_image or compute_area or compute_feret):
        return props_dict
    ndim = label_image.ndim
    if ndim in (2, 3) and max_label > 0:
        row_first, row_last, offsets, num_rows = _regionprops_row_extents(
            label_image, bbox, max_label
        )
        if ndim == 2:
            row_lo, row_hi, num_pixels = _convex_hull_rows_2d(
                row_first, row_last, offsets, num_rows, max_label
            )
        else:
            row_lo, row_hi, num_pixels = _convex_hull_rows_3d(
                label_image,
                row_first,
                row_last,
                offsets,
                num_rows,
                bbox,
                max_label,
            )
        if compute_area:
            props_dict["area_convex"] = num_pixels.astype(cp.float64) * (
                _pixel_area(spacing)
            )
        if compute_image:
            props_dict["image_convex"] = _convex_images(
                row_lo, row_hi, offsets, bbox, max_label
            )
        if compute_feret and ndim == 2:
            props_dict["feret_diameter_max"] = _feret_diameter_max_2d(
                row_lo, row_hi, offsets, num_rows, max_label, spacing
            )
        elif compute_feret:
            props_dict["feret_diameter_max"] = _feret_diameter_max_3d(
                row_lo,
                row_hi,
                offsets,
                num_rows,
                bbox,
                cp.asnumpy(bbox).astype(np.int64),
                max_label,
                spacing,
            )
        return props_dict

    # bounding-box slices given explicitly: one convex image per label
    _, _, images_convex = regionprops_image(
        label_image,
        slices=_bbox_slices(bbox),
        max_label=max_label,
        compute_image=False,
        compute_convex=True,
        props_dict={},
    )
    if compute_area:
        regionprops_area_convex(
            images_convex,
            max_label=max_label,
            spacing=spacing,
            props_dict=props_dict,
        )
    if compute_image:
        props_dict["image_convex"] = images_convex
    if compute_feret:
        regionprops_feret_diameter_max(
            images_convex,
            spacing=spacing,
            props_dict=props_dict,
        )
    return props_dict


def regionprops_area_convex(
    images_convex,
    max_label=None,
    spacing=None,
    area_dtype=cp.float64,
    props_dict=None,
):
    """Compute the area of each convex image.

    writes "area_convex" to props_dict

    Parameters
    ----------
    images_convex : sequence of cupy.ndarray
        Convex images for each region as produced by ``regionprops_image`` with
        ``compute_convex=True``.
    """
    if max_label is None:
        max_label = len(images_convex)
    if not isinstance(images_convex, Sequence):
        raise ValueError("Expected `images_convex` to be a sequence of images")
    area_convex = cp.zeros((max_label,), dtype=area_dtype)
    for i in range(max_label):
        area_convex[i] = images_convex[i].sum()
    if spacing is not None:
        if isinstance(spacing, cp.ndarray):
            pixel_area = cp.product(spacing)
        else:
            pixel_area = math.prod(spacing)
        area_convex *= pixel_area
    if props_dict is not None:
        props_dict["area_convex"] = area_convex
    return area_convex


def _regionprops_coords_perimeter(
    image,
    connectivity=1,
):
    """
    Takes an image of a single labeled region (e.g. one element of the tuple
    resulting from regionprops_image) and returns the coordinates of the voxels
    at the edge of that region.
    """

    # remove non-boundary pixels
    binary_image = image > 0
    footprint = ndi.generate_binary_structure(
        binary_image.ndim, connectivity=connectivity
    )
    binary_image_eroded = ndi.binary_erosion(binary_image, footprint)
    binary_edges = binary_image * ~binary_image_eroded
    edge_coords = _ndarray_argwhere(binary_edges)
    return edge_coords


def _feret_diameter_max(image_convex, spacing=None, return_argmax=False):
    """Compute the maximum Feret diameter of a single convex image region."""
    if image_convex.size == 0:
        # label value absent from the label image: 0, like its area_convex
        return 0.0
    if image_convex.size == 1:
        warnings.warn(
            "single element image, returning 0 for feret diameter",
            UserWarning,
            stacklevel=2,
        )
        return 0
    coords = _regionprops_coords_perimeter(image_convex, connectivity=1)
    coords = coords.astype(cp.float32)

    if spacing is not None:
        if all(s == 1.0 for s in spacing):
            spacing = None
        else:
            spacing = cp.asarray(spacing, dtype=cp.float32).reshape(1, -1)
            coords *= spacing

    squared_dist, index_argmax = pdist_max_blockwise(
        coords,
        metric="sqeuclidean",
        compute_argmax=return_argmax,
        coords_per_block=4000,
    )
    max_diameter = math.sqrt(float(squared_dist))
    if return_argmax:
        return max_diameter, index_argmax
    return max_diameter


def regionprops_feret_diameter_max(
    images_convex, spacing=None, props_dict=None
):
    """Compute the maximum Feret diameter of the convex hull of each image in
    images_convex.

    writes "feret_diameter_max" to props_dict

    Parameters
    ----------
    images_convex : sequence of cupy.ndarray
        Convex images for each region as produced by ``regionprops_image`` with
        ``compute_convex=True``.
    spacing : tuple of float, optional
        The pixel spacing of the image.
    props_dict : dict, optional
        A dictionary to store the computed properties.

    Notes
    -----
    The maximum Feret diameter is the maximum distance between any two
    points on the convex hull of the region. The implementation here is based
    on pairwise distances of all boundary coordinates rather than using
    marching squares or marching cubes as in scikit-image. The implementation
    here is n-dimensional.

    The distance is between pixel centers and so may be approximately one pixel
    width less than the one computed by scikit-image.
    """
    if not isinstance(images_convex, Sequence):
        raise ValueError("Expected `images_convex` to be a sequence of images")
    diameters = cp.asarray(
        tuple(
            _feret_diameter_max(
                image_convex, spacing=spacing, return_argmax=False
            )
            for image_convex in images_convex
        ),
        dtype=cp.float64,
    )
    if props_dict is not None:
        props_dict["feret_diameter_max"] = diameters
    return diameters
