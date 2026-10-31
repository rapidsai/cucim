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
convex_deps["feret_diameter_min"] = ["bbox"]
convex_deps["feret_diameter_min_perpendicular"] = ["bbox"]
convex_deps["feret_angle_max"] = ["bbox"]
convex_deps["feret_angle_min"] = ["bbox"]


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
    Andrew's monotone chain [1]_ builds the left chain of the hull from the
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

    References
    ----------
    .. [1] A. M. Andrew, "Another efficient algorithm for convex hulls in
           two dimensions", Information Processing Letters 9(5), 216-219,
           1979. :DOI:`10.1016/0020-0190(79)90072-3`
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
    row_lo, row_hi, offsets, num_rows, max_label, spacing=None, chains=None
):
    """Maximum Feret diameter of each 2D region, from its convex image's rows.

    scikit-image takes the largest distance between the contour points of the
    convex image at level 0.5. For a binary image these are the diamond
    points of its boundary pixels, and the largest distance between points is
    reached at two vertices of their convex hull, which is also the hull of
    the diamond points of each row's first and last pixel. So the result is
    the largest distance between the vertices of the hull chains of the
    convex image (`_get_hull_chains_2d_kernel`), which can be given as
    `chains`.
    """
    if chains is None:
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
def _get_feret_calipers_2d_kernel():
    """Kernel computing the Feret diameters and angles of each 2D region
    with rotating calipers, one thread per label.

    The polygon is the hull of the label's convex image: its left chain from
    top to bottom, then its right chain from bottom to top (without the
    vertices the two chains share). Coordinates are scaled by the spacing.
    For each edge, the vertex farthest from the edge's line is found by
    advancing a pointer around the polygon, which moves at most twice around
    it in total (Preparata and Shamos, 1985):

    - its distance to the edge's line is the width of the polygon in the
      direction normal to the edge; the smallest width is the minimum Feret
      diameter, measured along that normal;
    - the vertices it passes, with the edge's endpoints, are the antipodal
      pairs, among which the largest distance is the maximum Feret diameter.

    The width perpendicular to the minimum one is the extent of the vertices
    along the edge of the minimum width. Angles are those of the direction
    (d0, d1) between axis 0 and axis 1, atan2(d1, d0), folded into
    (-pi/2, pi/2] (the convention of ``orientation``).
    """
    source = """
      long long base = 2 * offsets[i] + i;
      const E* left = &left_chains[2 * base];
      const E* right = &right_chains[2 * base];
      int n_left = chain_sizes[2 * i];
      int n_right = chain_sizes[2 * i + 1];
      if (n_left == 0) {
        // an absent label
        feret_min[i] = 0.0;
        feret_min_perp[i] = 0.0;
        angle_max[i] = nan("");
        angle_min[i] = nan("");
      } else {
        // the right chain, bottom to top, without the shared end vertices
        int top = (left[0] == right[0] && left[1] == right[1]) ? 1 : 0;
        int bottom = (left[2 * (n_left - 1)] == right[2 * (n_right - 1)]
                      && left[2 * (n_left - 1) + 1]
                         == right[2 * (n_right - 1) + 1]) ? 1 : 0;
        int n = n_left + n_right - top - bottom;
        auto vy = [&](int k) -> double {
          k = k % n;
          E y = k < n_left ? left[2 * k]
                           : right[2 * (n_right - 1 - bottom - (k - n_left))];
          return static_cast<double>(y) * half_spacing0;
        };
        auto vx = [&](int k) -> double {
          k = k % n;
          E x = k < n_left ? left[2 * k + 1]
                           : right[2 * (n_right - 1 - bottom - (k - n_left))
                                   + 1];
          return static_cast<double>(x) * half_spacing1;
        };
        double best_d2 = -1.0, best_dy = 0.0, best_dx = 0.0;
        auto check_pair = [&](int a, int b) {
          double dy = vy(b) - vy(a);
          double dx = vx(b) - vx(a);
          double d2 = dy * dy + dx * dx;
          if (d2 > best_d2) {
            best_d2 = d2;
            best_dy = dy;
            best_dx = dx;
          }
        };
        double min_w = 1.0 / 0.0;
        int min_edge = 0;
        int j = 1;
        for (int e = 0; e < n; e++) {
          double ay = vy(e), ax = vx(e);
          double ey = vy(e + 1) - ay;
          double ex = vx(e + 1) - ax;
          // twice the area of the triangle (edge, vertex k): proportional
          // to the distance of vertex k from the edge's line
          auto height = [&](int k) {
            return fabs(ey * (vx(k) - ax) - ex * (vy(k) - ay));
          };
          while (height(j + 1) > height(j)) {
            check_pair(e, j);
            check_pair(e + 1, j);
            j = (j + 1) % n;
          }
          check_pair(e, j);
          check_pair(e + 1, j);
          if (height(j + 1) == height(j)) {
            // an edge parallel to edge e: its other end is antipodal too
            check_pair(e, j + 1);
            check_pair(e + 1, j + 1);
          }
          double w = height(j) / sqrt(ey * ey + ex * ex);
          if (w < min_w) {
            min_w = w;
            min_edge = e;
          }
        }
        // the extent of the vertices along the edge of the minimum width
        double ey = vy(min_edge + 1) - vy(min_edge);
        double ex = vx(min_edge + 1) - vx(min_edge);
        double norm = sqrt(ey * ey + ex * ex);
        double lo = 1.0 / 0.0, hi = -1.0 / 0.0;
        for (int k = 0; k < n; k++) {
          double t = (vy(k) * ey + vx(k) * ex) / norm;
          lo = fmin(lo, t);
          hi = fmax(hi, t);
        }
        feret_min[i] = min_w;
        feret_min_perp[i] = hi - lo;
        // angles in (-pi/2, pi/2]; the minimum is measured along the normal
        // (-ex, ey) of its edge
        auto fold = [](double a) {
          const double pi = 3.141592653589793;
          if (a > 0.5 * pi) {
            a -= pi;
          } else if (a <= -0.5 * pi) {
            a += pi;
          }
          return a;
        };
        angle_max[i] = fold(atan2(best_dx, best_dy));
        angle_min[i] = fold(atan2(ey, -ex));
      }\n"""
    inputs = (
        "raw E left_chains, raw E right_chains, raw int32 chain_sizes, "
        "raw int64 offsets, float64 half_spacing0, float64 half_spacing1"
    )
    outputs = (
        "raw float64 feret_min, raw float64 feret_min_perp, "
        "raw float64 angle_max, raw float64 angle_min"
    )
    return cp.ElementwiseKernel(
        inputs, outputs, source, name="cucim_feret_calipers_2d"
    )


def _feret_calipers_2d(chains, offsets, max_label, spacing=None):
    """Minimum Feret diameter, the width perpendicular to it, and the angles
    of the maximum and minimum Feret diameters of each 2D region, from the
    hull chains of its convex image (see `_get_feret_calipers_2d_kernel`).
    """
    spacing = (1.0, 1.0) if spacing is None else spacing
    out = [cp.empty((max_label,), dtype=cp.float64) for _ in range(4)]
    _get_feret_calipers_2d_kernel()(
        *chains,
        offsets,
        0.5 * float(spacing[0]),
        0.5 * float(spacing[1]),
        *out,
        size=max_label,
    )
    return dict(
        zip(
            (
                "feret_diameter_min",
                "feret_diameter_min_perpendicular",
                "feret_angle_max",
                "feret_angle_min",
            ),
            out,
        )
    )


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


# The vertices of a segment's right chain that are not also vertices of its
# left chain: both chains start on the segment's top line and end on its
# bottom line, so they share their first (last) vertex if that line has one
# point.
_gather_chains_includes = r"""
template <typename T>
__device__ __forceinline__ void cucim_right_chain_range(
    const T* left, int nl, const T* right, int nr, int& j0, int& j1) {
  j0 = 0;
  j1 = nr;
  if (nr >= 1 && nl >= 1 && right[0] == left[0] && right[1] == left[1]) {
    j0 = 1;
  }
  if (nr >= 2 && right[2 * nr - 2] == left[2 * nl - 2]
      && right[2 * nr - 1] == left[2 * nl - 1]) {
    j1 = nr - 1;
  }
}
"""


@cp.memoize(for_each_device=True)
def _get_gather_chains_kernel(count_only):
    """Kernel copying the chain vertices of each segment of rows (a plane of
    a 3D label) to consecutive (z, y, x) points, one thread per segment.
    A vertex in both chains is copied once. With `count_only`, it only
    counts the points of each segment."""
    source = """
      long long base = 2 * segment_offsets[i] + i;
      const E* left = &left_chains[2 * base];
      const E* right = &right_chains[2 * base];
      int nl = chain_sizes[2 * i];
      int j0, j1;
      cucim_right_chain_range(left, nl, right, chain_sizes[2 * i + 1], j0,
                              j1);
"""
    if count_only:
        source += """
      num_points = nl + (j1 - j0);\n"""
        inputs = (
            "raw E left_chains, raw E right_chains, raw int32 chain_sizes, "
            "raw int64 segment_offsets"
        )
        outputs = "int64 num_points"
        name = "cucim_convex_count_chain_points"
    else:
        source += """
      long long out = point_starts[i];
      E z = segment_z[i];
      int lab = segment_label[i];
      for (int side = 0; side < 2; side++) {
        const E* chain = side == 0 ? left : right;
        int start = side == 0 ? 0 : j0;
        int end = side == 0 ? nl : j1;
        for (int j = start; j < end; j++, out++) {
          points[3 * out] = z;
          points[3 * out + 1] = chain[2 * j];
          points[3 * out + 2] = chain[2 * j + 1];
          point_labels[out] = lab;
        }
      }\n"""
        inputs = (
            "raw E left_chains, raw E right_chains, raw int32 chain_sizes, "
            "raw int64 segment_offsets, raw int64 point_starts, "
            "raw E segment_z, raw int32 segment_label"
        )
        outputs = "raw E points, raw int32 point_labels"
        name = "cucim_convex_gather_chains"
    return cp.ElementwiseKernel(
        inputs,
        outputs,
        source,
        preamble=_gather_chains_includes,
        name=name,
    )


def _gather_chain_vertices(chains, segment_offsets, segment_z, segment_label):
    """Vertices of the chains of each segment as (z, y, x) points.

    Returns the points (doubled coordinates, each once) and the label index
    of each.
    """
    left_chains, right_chains, chain_sizes = chains
    segment_offsets = cp.asarray(segment_offsets)
    point_starts = cp.zeros((segment_z.size + 1,), dtype=cp.int64)
    if segment_z.size > 0:
        _get_gather_chains_kernel(True)(
            left_chains,
            right_chains,
            chain_sizes,
            segment_offsets,
            point_starts[1:],
        )
    cp.cumsum(point_starts, out=point_starts)
    num_points = int(point_starts[-1])
    points = cp.empty((num_points, 3), dtype=left_chains.dtype)
    point_labels = cp.empty((num_points,), dtype=cp.int32)
    if segment_z.size > 0:
        _get_gather_chains_kernel(False)(
            left_chains,
            right_chains,
            chain_sizes,
            segment_offsets,
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

    References
    ----------
    .. [1] A. M. Andrew, "Another efficient algorithm for convex hulls in
           two dimensions", Information Processing Letters 9(5), 216-219,
           1979. :DOI:`10.1016/0020-0190(79)90072-3`
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


# Exact integer helpers of the 3D convex-hull kernel. Points are (z, y, x)
# triples in doubled coordinates relative to the label's bounding box (see
# `_convex_hull_candidates_3d`). For a bounding box of at most
# _HULL_3D_MAX_EXTENT voxels along each axis, coordinate differences are
# below 2**20, so the products of three of them (at most 6 * 2**60 in a 3 x 3
# determinant) fit in 64 bits. Larger labels use qhull on the host.
_HULL_3D_MAX_EXTENT = 2**19 - 2

_hull_3d_includes = r"""
struct cucim_vec3 {
  long long z, y, x;
};

__device__ __forceinline__ cucim_vec3 cucim_sub3(cucim_vec3 a,
                                                 cucim_vec3 b) {
  return {a.z - b.z, a.y - b.y, a.x - b.x};
}

__device__ __forceinline__ cucim_vec3 cucim_cross3(cucim_vec3 a,
                                                   cucim_vec3 b) {
  return {a.y * b.x - a.x * b.y, a.x * b.z - a.z * b.x,
          a.z * b.y - a.y * b.z};
}

__device__ __forceinline__ long long cucim_dot3(cucim_vec3 a, cucim_vec3 b) {
  return a.z * b.z + a.y * b.y + a.x * b.x;
}

// a . b for vectors whose components fit in 32 bits
__device__ __forceinline__ long long cucim_dot3_narrow(cucim_vec3 a,
                                                       cucim_vec3 b) {
  return static_cast<long long>(static_cast<int>(a.z)) * static_cast<int>(b.z)
         + static_cast<long long>(static_cast<int>(a.y))
               * static_cast<int>(b.y)
         + static_cast<long long>(static_cast<int>(a.x))
               * static_cast<int>(b.x);
}

template <typename T>
__device__ __forceinline__ cucim_vec3 cucim_point3(const T* points, int s) {
  return {static_cast<long long>(points[3 * s]),
          static_cast<long long>(points[3 * s + 1]),
          static_cast<long long>(points[3 * s + 2])};
}

// Is point s before point t in (z, y, x) order?
template <typename T>
__device__ __forceinline__ bool cucim_less3(const T* points, int s, int t) {
  cucim_vec3 d = cucim_sub3(cucim_point3(points, s), cucim_point3(points, t));
  return d.z < 0 || (d.z == 0 && (d.y < 0 || (d.y == 0 && d.x < 0)));
}

// Shared memory of a block working on one label.
struct cucim_hull_shared {
  int index[32];
  int point[32][3];
  int count[32];
  long long normal[3];
  int c;
  int m;
  int ea;
  int eb;
};

// The threads working on one label: one warp (W = 1), or a whole block of W
// warps. Loops over all points are strided over the group's threads; the
// rest of the work is done by the group's first warp (the leader). Values
// combined from all threads (`best`, `sum`, `any`) are returned to all of
// them. With W > 1, each of these ends with a barrier after the last read
// of shared memory, so that the next one can write it.
template <int W>
struct cucim_group {
  int rank;  // the thread's rank in the group, 0 to 32 W - 1
  int lane;
  int warp;  // the warp's rank in the group
  cucim_hull_shared* shared;  // (if W > 1)

  __device__ __forceinline__ bool any(bool pred) const {
    if (W == 1) {
      return __any_sync(0xffffffffu, pred);
    }
    return __syncthreads_or(pred) != 0;
  }

  __device__ __forceinline__ int sum(int v) const {
    for (int offset = 16; offset > 0; offset >>= 1) {
      v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    if (W > 1) {
      if (lane == 0) {
        shared->count[warp] = v;
      }
      __syncthreads();
      v = 0;
      for (int w = 0; w < W; w++) {
        v += shared->count[w];
      }
      __syncthreads();
    }
    return v;
  }

  // The best of the threads' candidates (point indices, -1 for none).
  // better(c, c2) tells whether c2 is better than c (both valid); it must
  // be a strict total order, so that all threads agree.
  template <typename F>
  __device__ __forceinline__ int best(int c, F better) const {
    for (int offset = 16; offset > 0; offset >>= 1) {
      int c2 = __shfl_xor_sync(0xffffffffu, c, offset);
      if (c2 >= 0 && c2 != c && (c < 0 || better(c, c2))) {
        c = c2;
      }
    }
    if (W > 1) {
      if (lane == 0) {
        shared->index[warp] = c;
      }
      __syncthreads();
      if (warp == 0) {
        c = lane < W ? shared->index[lane] : -1;
        for (int offset = 16; offset > 0; offset >>= 1) {
          int c2 = __shfl_xor_sync(0xffffffffu, c, offset);
          if (c2 >= 0 && c2 != c && (c < 0 || better(c, c2))) {
            c = c2;
          }
        }
        if (lane == 0) {
          shared->c = c;
        }
      }
      __syncthreads();
      c = shared->c;
      __syncthreads();
    }
    return c;
  }
};

// One step of the reduction of cucim_hull_wrap: takes candidate c2 (at
// point p2) if it is rotated further than c (at point p, with normal m).
__device__ __forceinline__ void cucim_wrap_take(int& c, cucim_vec3& p,
                                                cucim_vec3& m, int c2,
                                                cucim_vec3 p2, cucim_vec3 ab,
                                                cucim_vec3 b) {
  if (c2 >= 0 && c2 != c) {
    cucim_vec3 sb = cucim_sub3(p2, b);
    long long o = c >= 0 ? cucim_dot3(m, sb) : 1;
    if (o > 0 || (o == 0 && c2 < c)) {
      c = c2;
      p = p2;
      m = cucim_cross3(ab, sb);
    }
  }
}

__device__ __forceinline__ void cucim_wrap_butterfly(int& c, cucim_vec3& p,
                                                     cucim_vec3& m,
                                                     cucim_vec3 ab,
                                                     cucim_vec3 b) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    int c2 = __shfl_xor_sync(0xffffffffu, c, offset);
    cucim_vec3 p2 = {
        __shfl_xor_sync(0xffffffffu, static_cast<int>(p.z), offset),
        __shfl_xor_sync(0xffffffffu, static_cast<int>(p.y), offset),
        __shfl_xor_sync(0xffffffffu, static_cast<int>(p.x), offset)};
    cucim_wrap_take(c, p, m, c2, p2, ab, b);
  }
}

// Gift wrapping (Chand and Kapur) around the edge a -> b of a known face,
// whose other points are on the left of a -> b seen from outside. Returns
// the point c that is rotated furthest around the edge (a point of the
// adjacent face) and sets `normal` to that face's outward normal
// (a - b) x (c - b): normal . (s - b) <= 0 for every point s. As all points
// lie within half a turn around the edge, the sign of normal . (s - b)
// orders them by angle; ties go to the smaller index. Also sets `others` to
// the number of points in the face's plane that are not on the edge's line
// (c included): if it is 1, the face is the triangle a, c, b.
//
// With `narrow` (coordinate differences up to 2**14), the normals fit in 32
// bits and the scan takes 32-bit products.
template <int W, typename T>
__device__ int cucim_hull_wrap(const T* points, int n, cucim_vec3 a,
                               cucim_vec3 b, bool narrow,
                               const cucim_group<W>& g, cucim_vec3& normal,
                               int& others) {
  cucim_vec3 ab = cucim_sub3(a, b);
  int c = -1;
  cucim_vec3 pc = {0, 0, 0};
  cucim_vec3 m = {0, 0, 0};
  int ties = 0;  // this thread's points at the angle of c, not on the line
  // (4 points per iteration, loaded first, so that their loads overlap)
  for (int s0 = g.rank; s0 < n; s0 += 4 * 32 * W) {
    cucim_vec3 q[4];
#pragma unroll
    for (int u = 0; u < 4; u++) {
      int s = s0 + u * 32 * W;
      if (s < n) {
        q[u] = cucim_point3(points, s);
      }
    }
#pragma unroll
    for (int u = 0; u < 4; u++) {
      int s = s0 + u * 32 * W;
      if (s >= n) {
        break;
      }
      cucim_vec3 sb = cucim_sub3(q[u], b);
      long long o = 1;
      if (c >= 0) {
        o = narrow ? cucim_dot3_narrow(m, sb) : cucim_dot3(m, sb);
      }
      if (o >= 0) {
        cucim_vec3 ms = cucim_cross3(ab, sb);
        if (ms.z != 0 || ms.y != 0 || ms.x != 0) {  // not on the edge's line
          if (o > 0) {
            c = s;
            pc = q[u];
            m = ms;
            ties = 1;
          } else {
            ties++;
          }
        }
      }
    }
  }
  // the best point of all threads (with its coordinates and normal, so that
  // comparing it with another point takes one dot product)
  int own = c;
  cucim_vec3 own_p = pc;
  cucim_wrap_butterfly(c, pc, m, ab, b);
  if (W > 1) {
    if (g.lane == 0) {
      g.shared->index[g.warp] = c;
      g.shared->point[g.warp][0] = static_cast<int>(pc.z);
      g.shared->point[g.warp][1] = static_cast<int>(pc.y);
      g.shared->point[g.warp][2] = static_cast<int>(pc.x);
    }
    __syncthreads();
    if (g.warp == 0) {
      c = -1;
      if (g.lane < W) {
        int c2 = g.shared->index[g.lane];
        cucim_vec3 p2 = {g.shared->point[g.lane][0],
                         g.shared->point[g.lane][1],
                         g.shared->point[g.lane][2]};
        cucim_wrap_take(c, pc, m, c2, p2, ab, b);
      }
      cucim_wrap_butterfly(c, pc, m, ab, b);
      if (g.lane == 0) {
        g.shared->c = c;
        g.shared->normal[0] = m.z;
        g.shared->normal[1] = m.y;
        g.shared->normal[2] = m.x;
      }
    }
    __syncthreads();
    c = g.shared->c;
    m = {g.shared->normal[0], g.shared->normal[1], g.shared->normal[2]};
    __syncthreads();
  }
  normal = m;
  // A thread whose own best point is at the angle of c counted all its
  // points at that angle; the others have none.
  bool at_c = own >= 0
              && (own == c || cucim_dot3(m, cucim_sub3(own_p, b)) == 0);
  others = g.sum(at_c ? ties : 0);
  return c;
}

// Writes the points s with normal . (s - ref) == 0 to `out` and returns
// their number (in increasing order if W == 1).
template <int W, typename T>
__device__ int cucim_hull_coplanar(const T* points, int n, cucim_vec3 normal,
                                   cucim_vec3 ref, bool narrow,
                                   const cucim_group<W>& g, int* out) {
  if (W == 1) {
    int count = 0;
    for (int base = 0; base < n; base += 4 * 32) {
      bool on[4];
#pragma unroll
      for (int u = 0; u < 4; u++) {
        int s = base + u * 32 + g.lane;
        on[u] = false;
        if (s < n) {
          cucim_vec3 d = cucim_sub3(cucim_point3(points, s), ref);
          on[u] = (narrow ? cucim_dot3_narrow(normal, d)
                          : cucim_dot3(normal, d)) == 0;
        }
      }
#pragma unroll
      for (int u = 0; u < 4; u++) {
        unsigned int mask = __ballot_sync(0xffffffffu, on[u]);
        if (on[u]) {
          out[count + __popc(mask & ((1u << g.lane) - 1u))] =
              base + u * 32 + g.lane;
        }
        count += __popc(mask);
      }
    }
    __syncwarp();
    return count;
  }
  if (g.rank == 0) {
    g.shared->m = 0;
  }
  __syncthreads();
  for (int s0 = g.rank; s0 < n; s0 += 4 * 32 * W) {
    bool on[4];
#pragma unroll
    for (int u = 0; u < 4; u++) {
      int s = s0 + u * 32 * W;
      on[u] = false;
      if (s < n) {
        cucim_vec3 d = cucim_sub3(cucim_point3(points, s), ref);
        on[u] = (narrow ? cucim_dot3_narrow(normal, d)
                        : cucim_dot3(normal, d)) == 0;
      }
    }
#pragma unroll
    for (int u = 0; u < 4; u++) {
      if (on[u]) {
        out[atomicAdd(&g.shared->m, 1)] = s0 + u * 32 * W;
      }
    }
  }
  __syncthreads();
  int m = g.shared->m;
  __syncthreads();
  return m;
}

// The coordinates of a vector d in a face's plane, projected along axis k
// (the largest component of the face's normal): the other two axes in
// cyclic order after k, so that u1 * v2 - v1 * u2 is component k of the
// cross product of the two vectors.
__device__ __forceinline__ void cucim_project(cucim_vec3 d, int k,
                                              long long& u, long long& v) {
  if (k == 0) {
    u = d.y;
    v = d.x;
  } else if (k == 1) {
    u = d.x;
    v = d.z;
  } else {
    u = d.z;
    v = d.y;
  }
}

// The projection axis of a face (see cucim_project), and the sign that
// makes the projected cross products positive for counterclockwise turns
// seen from outside.
__device__ __forceinline__ void cucim_face_axis(cucim_vec3 normal, int& k,
                                                long long& sign) {
  long long az = abs(normal.z);
  long long ay = abs(normal.y);
  long long ax = abs(normal.x);
  k = (az >= ay && az >= ax) ? 0 : (ay >= ax ? 1 : 2);
  long long nk = k == 0 ? normal.z : (k == 1 ? normal.y : normal.x);
  sign = nk > 0 ? 1 : -1;
}

// Is point s (at (su, sv) from the current vertex) a better next vertex
// than point q (at (qu, qv)): further clockwise, or collinear and farther?
__device__ __forceinline__ bool cucim_jarvis_better(long long qu,
                                                    long long qv, int q,
                                                    long long su,
                                                    long long sv, int s,
                                                    long long sign) {
  long long c = sign * (qu * sv - qv * su);
  if (c != 0) {
    return c < 0;
  }
  long long dq = qu * qu + qv * qv;
  long long ds = su * su + sv * sv;
  return ds > dq || (ds == dq && s < q);
}

// The vertices of the 2D hull of a face's m points (`face`),
// counterclockwise seen from outside, starting at its vertex `start`,
// without collinear points (Jarvis march, in the projection along the
// normal's largest component): the next vertex after a vertex is the point
// furthest clockwise from it, so that all points are on the left of the
// edge between them. By one warp. Returns the number of vertices, or -1 if
// the march does not return to `start` within m vertices.
//
// For m <= 32, each lane takes one point and finds what would be its next
// vertex (meaningful for the vertices), in one pass over the points; the
// polygon then follows these from `start`. For larger m, each step of the
// march is a pass over the points.
template <typename T>
__device__ int cucim_hull_face_polygon(const T* points, const int* face,
                                       int m, cucim_vec3 normal, int start,
                                       int lane, int* polygon) {
  int k;
  long long sign;
  cucim_face_axis(normal, k, sign);
  cucim_vec3 p0 = cucim_point3(points, start);
  int p = start;
  int h = 0;
  if (m <= 32) {
    int s = lane < m ? face[lane] : -1;
    long long u = 0;
    long long v = 0;
    if (s >= 0) {
      cucim_project(cucim_sub3(cucim_point3(points, s), p0), k, u, v);
    }
    int next = -1;
    long long nu = 0;
    long long nv = 0;
    for (int t = 0; t < m; t++) {
      long long du = __shfl_sync(0xffffffffu, u, t) - u;
      long long dv = __shfl_sync(0xffffffffu, v, t) - v;
      int st = __shfl_sync(0xffffffffu, s, t);
      if (s >= 0 && st != s
          && (next < 0 || cucim_jarvis_better(nu, nv, next, du, dv, st,
                                              sign))) {
        next = st;
        nu = du;
        nv = dv;
      }
    }
    while (h < m) {
      if (lane == 0) {
        polygon[h] = p;
      }
      h++;
      int owner = __ffs(__ballot_sync(0xffffffffu, s == p)) - 1;
      p = owner >= 0 ? __shfl_sync(0xffffffffu, next, owner) : -1;
      if (p == start || p < 0) {
        break;
      }
    }
  } else {
    while (h < m) {
      if (lane == 0) {
        polygon[h] = p;
      }
      h++;
      cucim_vec3 pp = cucim_point3(points, p);
      int q = -1;
      long long qu = 0;
      long long qv = 0;
      for (int t = lane; t < m; t += 32) {
        int s = face[t];
        if (s == p) {
          continue;
        }
        long long su, sv;
        cucim_project(cucim_sub3(cucim_point3(points, s), pp), k, su, sv);
        if (q < 0 || cucim_jarvis_better(qu, qv, q, su, sv, s, sign)) {
          q = s;
          qu = su;
          qv = sv;
        }
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        int s = __shfl_xor_sync(0xffffffffu, q, offset);
        if (s >= 0 && s != q) {
          long long su, sv;
          cucim_project(cucim_sub3(cucim_point3(points, s), pp), k, su, sv);
          if (q < 0 || cucim_jarvis_better(qu, qv, q, su, sv, s, sign)) {
            q = s;
            qu = su;
            qv = sv;
          }
        }
      }
      p = q;
      if (p == start || p < 0) {
        break;
      }
    }
  }
  __syncwarp();
  return p == start ? h : -1;
}

// The pending edges of a label: a hash table of directed edges u -> v
// (keys u * 2**32 + v, 0 for an empty slot and -1 for a removed one, with
// linear probing), and a stack of the slots of the edges in the order they
// were added. The stack is not updated when an edge is removed, so its
// entries are checked against the table when popped.
struct cucim_hull_edges {
  long long* table;
  unsigned int mask;  // the table's size - 1 (a power of 2)
  int* stack;
  int size;      // of the stack
  int capacity;  // of the stack
};

__device__ __forceinline__ unsigned int cucim_edge_hash(int u, int v) {
  unsigned int h = static_cast<unsigned int>(u) * 0x9e3779b1u
                   + static_cast<unsigned int>(v) * 0x85ebca77u;
  return h ^ (h >> 15);
}

// Adds the directed edge u -> v of a new face to the pending edges, or, if
// its reverse v -> u is pending, removes that (both faces of the edge are
// then known). By one warp, each lane probing one slot. Returns false if
// there is no room.
__device__ bool cucim_hull_edge(cucim_hull_edges& edges, int u, int v,
                                int lane) {
  // An edge is added at the first free slot of its probe sequence, and
  // slots never become empty again, so a pending edge comes before any
  // empty slot of its sequence.
  long long reverse = (static_cast<long long>(v) << 32) | u;
  for (unsigned int h = cucim_edge_hash(v, u);; h += 32) {
    unsigned int slot = (h + lane) & edges.mask;
    long long key = edges.table[slot];
    unsigned int match = __ballot_sync(0xffffffffu, key == reverse);
    if (match != 0) {
      if (lane == __ffs(match) - 1) {
        edges.table[slot] = -1;
      }
      __syncwarp();
      return true;
    }
    if (__ballot_sync(0xffffffffu, key == 0) != 0) {
      break;
    }
  }
  if (edges.size >= edges.capacity) {
    return false;
  }
  long long added = (static_cast<long long>(u) << 32) | v;
  for (unsigned int h = cucim_edge_hash(u, v);; h += 32) {
    unsigned int slot = (h + lane) & edges.mask;
    long long key = edges.table[slot];
    unsigned int free = __ballot_sync(0xffffffffu, key == 0 || key == -1);
    if (free != 0) {
      if (lane == __ffs(free) - 1) {
        edges.table[slot] = added;
        edges.stack[edges.size] = static_cast<int>(slot);
      }
      edges.size++;
      __syncwarp();
      return true;
    }
  }
}

// The last pending edge on the stack (popping removed ones), as (u, v), or
// u = -1 if there is none. By one warp.
__device__ __forceinline__ void cucim_hull_next_edge(cucim_hull_edges& edges,
                                                     int& u, int& v) {
  u = -1;
  v = -1;
  while (edges.size > 0) {
    long long key = edges.table[edges.stack[edges.size - 1]];
    if (key > 0) {
      u = static_cast<int>(key >> 32);
      v = static_cast<int>(key & 0xffffffffll);
      return;
    }
    edges.size--;
  }
}

// Adds a face, by one warp: its polygon (the triangle start, c, end if
// c >= 0, else from its m points in `face`), its plane (normal and
// normal . ref, with ref a point of the face) and its edges. `end` (if >= 0)
// must be the polygon's last vertex. Returns false on failure.
template <typename T>
__device__ bool cucim_hull_add_face(const T* points, const int* face, int m,
                                    cucim_vec3 normal, cucim_vec3 ref,
                                    int start, int c, int end, int lane,
                                    int* polygon, cucim_hull_edges& edges,
                                    long long* planes, int& num_faces) {
  int h;
  if (c >= 0) {
    if (lane == 0) {
      polygon[0] = start;
      polygon[1] = c;
      polygon[2] = end;
    }
    __syncwarp();
    h = 3;
  } else {
    h = cucim_hull_face_polygon(points, face, m, normal, start, lane,
                                polygon);
    if (h < 3 || (end >= 0 && polygon[h - 1] != end)) {
      return false;
    }
  }
  if (lane == 0) {
    long long* plane = &planes[4 * num_faces];
    plane[0] = normal.z;
    plane[1] = normal.y;
    plane[2] = normal.x;
    plane[3] = cucim_dot3(normal, ref);
  }
  num_faces++;
  for (int j = 0; j < h; j++) {
    if (!cucim_hull_edge(edges, polygon[j], polygon[(j + 1) % h], lane)) {
      return false;
    }
  }
  return true;
}

// The convex hull of n distinct points spanning 3D, by a group of threads.
// Writes the plane of each face to `planes` (its outward normal N and
// N . p for its points p, with N . p smaller for the other points) and
// returns the number of faces (to the group's first thread), or -1 if some
// capacity was exceeded (2n faces, 3n edges). `face` and `polygon` need
// room for n indices, `edges` for 3n edges.
template <int W, typename T>
__device__ int cucim_convex_hull_3d(const T* points, int n, bool narrow,
                                    const cucim_group<W>& g, int* face,
                                    int* polygon, cucim_hull_edges& edges,
                                    long long* planes) {
  bool leader = g.warp == 0;
  // a: the smallest point in (z, y, x) order, a vertex of the hull
  int a = -1;
  for (int s = g.rank; s < n; s += 32 * W) {
    if (a < 0 || cucim_less3(points, s, a)) {
      a = s;
    }
  }
  a = g.best(a, [&](int s, int t) { return cucim_less3(points, t, s); });
  cucim_vec3 pa = cucim_point3(points, a);

  // The first face: the plane z = a.z supports the points, and those in it
  // have y >= a.y, so wrapping around the line through a along x (as an
  // edge of that plane) gives a plane through a that supports the points.
  cucim_vec3 normal;
  int others;
  cucim_vec3 pb = {pa.z, pa.y, pa.x + 1};
  cucim_hull_wrap(points, n, pa, pb, narrow, g, normal, others);
  int m = cucim_hull_coplanar(points, n, normal, pa, narrow, g, face);
  // If the points in that plane are collinear, the plane touches the hull in
  // an edge from a to the farthest of them, q: wrapping around that edge
  // gives a face.
  int q = -1;
  for (int t = g.rank; t < m; t += 32 * W) {
    int s = face[t];
    cucim_vec3 ds = cucim_sub3(cucim_point3(points, s), pa);
    cucim_vec3 dq = cucim_sub3(cucim_point3(points, q < 0 ? s : q), pa);
    if (q < 0 || cucim_dot3(ds, ds) > cucim_dot3(dq, dq)) {
      q = s;
    }
  }
  q = g.best(q, [&](int s, int t) {
    cucim_vec3 ds = cucim_sub3(cucim_point3(points, s), pa);
    cucim_vec3 dt = cucim_sub3(cucim_point3(points, t), pa);
    long long d2s = cucim_dot3(ds, ds);
    long long d2t = cucim_dot3(dt, dt);
    return d2t > d2s || (d2t == d2s && t < s);
  });
  cucim_vec3 aq = cucim_sub3(cucim_point3(points, q), pa);
  bool off_line = false;
  for (int t = g.rank; t < m; t += 32 * W) {
    cucim_vec3 c = cucim_cross3(aq, cucim_sub3(cucim_point3(points, face[t]),
                                               pa));
    off_line |= c.z != 0 || c.y != 0 || c.x != 0;
  }
  if (!g.any(off_line)) {
    cucim_hull_wrap(points, n, pa, cucim_point3(points, q), narrow, g,
                    normal, others);
    m = cucim_hull_coplanar(points, n, normal, pa, narrow, g, face);
  }
  // (the faces and edges are only kept by the leader)
  int num_faces = 0;
  bool ok = true;
  if (leader) {
    ok = cucim_hull_add_face(points, face, m, normal, pa, a, -1, -1, g.lane,
                             polygon, edges, planes, num_faces);
  }

  // Each pending edge u -> v belongs to one known face; the face on its
  // other side is not known yet (it would have removed the edge), so
  // wrapping around the edge finds a new face, which contains v -> u.
  while (true) {
    int ea = -1;  // -1: done, -2: failed
    int eb = -1;
    if (leader) {
      if (!ok || num_faces >= 2 * n) {
        ea = -2;
      } else {
        cucim_hull_next_edge(edges, ea, eb);
      }
    }
    if (W > 1) {
      if (g.rank == 0) {
        g.shared->ea = ea;
        g.shared->eb = eb;
      }
      __syncthreads();
      ea = g.shared->ea;
      eb = g.shared->eb;
      __syncthreads();
    }
    if (ea < 0) {
      return ea == -1 ? num_faces : -1;
    }
    pb = cucim_point3(points, eb);
    int c = cucim_hull_wrap(points, n, cucim_point3(points, ea), pb, narrow,
                            g, normal, others);
    if (c < 0) {
      return -1;
    }
    if (others != 1) {
      m = cucim_hull_coplanar(points, n, normal, pb, narrow, g, face);
    }
    if (leader) {
      // a triangle (others == 1, possibly with more points on the edge
      // ea - eb) has the vertices ea, c and eb in this order
      ok = cucim_hull_add_face(points, face, m, normal, pb, ea,
                               others == 1 ? c : -1, eb, g.lane, polygon,
                               edges, planes, num_faces);
    }
  }
}
"""


# C types of the point and bounding-box arrays of the 3D hull kernel
_hull_3d_ctypes = {
    cp.dtype(cp.int32): "int",
    cp.dtype(cp.int64): "long long",
    cp.dtype(cp.uint32): "unsigned int",
    cp.dtype(cp.uint64): "unsigned long long",
}


@cp.memoize(for_each_device=True)
def _get_convex_hulls_3d_kernel(warps_per_block, point_dtype, bbox_dtype):
    """Kernel computing the convex hull of each label's 3D points.

    On the candidates of `_convex_hull_candidates_3d` (distinct points with
    integer coordinates, spanning 3D). The hull is built face by face with
    gift wrapping [1]: wrapping around an edge of a known face finds the
    plane of the face on its other side; the points in that plane are
    collected, and their 2D hull (Jarvis march [2]) gives the face as a
    polygon. Every edge of the hull belongs to two faces, whose polygons
    traverse it in opposite directions; an edge whose reverse has not been
    seen yet is pending (kept in a hash table, `cucim_hull_edges`), and
    leads to a new face. Whole faces are found, rather than triangles,
    because the points have many coplanar subsets (they come from a voxel
    grid; about 60 % of the faces have more than 3 vertices). The wrap
    recognizes a triangle without collecting its points.

    Finding a face takes one or two passes over all the label's points, and
    the faces are found one after the other. So each of the first
    ``num_large`` blocks works on one label of `large_labels` (all its warps
    scan the points; the first warp does the rest), and the other blocks on
    one label of `small_labels` per warp.

    All tests are exact integer arithmetic (see `_HULL_3D_MAX_EXTENT`). For
    label l, the planes of its faces (normal N and N . p for the points p of
    the face, larger than for the other points) are written at
    ``2 * point_offsets[l]`` (a hull of n points has at most 2n - 4 faces),
    and their number to ``num_faces``. It is -1 if the label is too large
    for exact 64-bit arithmetic or the hull did not fit (which should not
    happen); such labels need another method.

    The kernel is launched with blocks of ``32 * warps_per_block`` threads.
    It is a raw kernel (rather than an elementwise kernel) so that
    `_convex_hulls_3d_kernel` can check that such a block fits on the device
    (``max_threads_per_block``). The arrays must be C-contiguous.

    References
    ----------
    .. [1] D. R. Chand and S. S. Kapur, "An algorithm for convex polytopes",
           Journal of the ACM 17(1), 78-86, 1970.
    .. [2] R. A. Jarvis, "On the identification of the convex hull of a
           finite set of points in the plane", Information Processing
           Letters 2(1), 18-21, 1973.
    """
    E = _hull_3d_ctypes[cp.dtype(point_dtype)]
    B = _hull_3d_ctypes[cp.dtype(bbox_dtype)]
    # (face_points, polygons, edge_table and edge_stack are scratch space;
    # edge_table must be zero)
    code = (
        _hull_3d_includes
        + f"""
extern "C" __global__ void cucim_convex_hulls_3d(
    const {E}* points, const long long* point_offsets, const {B}* bbox,
    long long max_extent, const long long* large_labels, long long num_large,
    const long long* small_labels, long long num_small, int* face_points,
    int* polygons, long long* edge_table, int* edge_stack, long long* planes,
    int* num_faces) {{
  const int W = {warps_per_block};
  __shared__ cucim_hull_shared cucim_hull_block;
  long long b = blockIdx.x;
  int rank = threadIdx.x;
  bool whole_block = b < num_large;
  long long l = -1;
  if (whole_block) {{
    l = large_labels[b];
  }} else {{
    long long j = (b - num_large) * W + rank / 32;
    if (j < num_small) {{
      l = small_labels[j];
    }}
  }}
  if (l >= 0) {{
    long long start = point_offsets[l];
    int n = static_cast<int>(point_offsets[l + 1] - start);
    long long extent = 0;
    for (int d = 0; d < 3; d++) {{
      extent = max(extent, static_cast<long long>(bbox[l * 6 + 3 + d]
                                                  - bbox[l * 6 + d]));
    }}
    // (coordinates relative to the bounding box are in [-1, 2 extent];
    // the virtual first edge reaches 2 extent + 1)
    bool narrow = extent < (1 << 13);
    int faces = 0;
    const {E}* p = &points[3 * start];
    int* face = &face_points[start];
    int* polygon = &polygons[start];
    // (the edge table has 8n slots, of which it uses the smallest power
    // of 2 >= 4n, at most 75 % full with the at most 3n edges)
    unsigned int table_size = 1;
    while (table_size < 4u * static_cast<unsigned int>(n)) {{
      table_size <<= 1;
    }}
    cucim_hull_edges edges = {{&edge_table[8 * start], table_size - 1,
                              &edge_stack[3 * start], 0, 3 * n}};
    long long* plane = &planes[8 * start];
    if (extent > max_extent) {{
      faces = -1;
    }} else if (n > 0 && whole_block) {{
      cucim_group<W> g = {{rank, rank & 31, rank >> 5, &cucim_hull_block}};
      faces = cucim_convex_hull_3d(p, n, narrow, g, face, polygon, edges,
                                   plane);
    }} else if (n > 0) {{
      cucim_group<1> g = {{rank & 31, rank & 31, 0, nullptr}};
      faces = cucim_convex_hull_3d(p, n, narrow, g, face, polygon, edges,
                                   plane);
    }}
    if ((whole_block ? rank : (rank & 31)) == 0) {{
      num_faces[l] = faces;
    }}
  }}
}}
"""
    )
    return cp.RawKernel(code, "cucim_convex_hulls_3d")


@cp.memoize(for_each_device=True)
def _get_hull_equations_kernel():
    """Kernel converting hull planes (N, N . p in doubled coordinates) to
    facet equations in the form of qhull's (unit outward normal and offset,
    in undoubled coordinates), one thread per plane."""
    source = """
      double nz = static_cast<double>(planes[4 * i]);
      double ny = static_cast<double>(planes[4 * i + 1]);
      double nx = static_cast<double>(planes[4 * i + 2]);
      double inv_norm = 1.0 / sqrt(nz * nz + ny * ny + nx * nx);
      equations[4 * i] = nz * inv_norm;
      equations[4 * i + 1] = ny * inv_norm;
      equations[4 * i + 2] = nx * inv_norm;
      equations[4 * i + 3] =
          -0.5 * static_cast<double>(planes[4 * i + 3]) * inv_norm;\n"""
    return cp.ElementwiseKernel(
        "raw int64 planes",
        "raw float64 equations",
        source,
        name="cucim_convex_hull_equations",
    )


# Labels with more points than this get a whole block of this many warps in
# `_get_convex_hulls_3d_kernel`, the others a warp. (On an RTX A6000, 16
# warps per block was about 20 % faster than 8 for labels of 10,000 points
# or more, and about 25 % slower for 512 labels of about 700 points, which
# use warps but have fewer resident warps with larger blocks.) Where a block
# of this many warps does not fit on the device, fewer warps are used (see
# `_convex_hulls_3d_kernel`).
_HULL_3D_BLOCK_POINTS = 1024
_HULL_3D_WARPS_PER_BLOCK = 16


def _convex_hulls_3d_kernel(point_dtype, bbox_dtype):
    """The 3D hull kernel with the most warps per block, at most
    `_HULL_3D_WARPS_PER_BLOCK`, whose blocks fit on the current device, and
    its number of warps per block.

    A block needs registers for all its threads, and the compiler chooses
    the number of registers per thread. As an elementwise kernel, this
    kernel had about 250 registers per thread for sm_100 and sm_120 with
    NVRTC 13.0 and 13.3 (about 120 for sm_80 to sm_90), too many for a block
    of 512 threads with 64 K registers; as a raw kernel it has 96 to 120 on
    all of these. Other compilers and devices may differ, so the kernel's
    ``max_threads_per_block`` is checked for each number of warps, halving
    it until a block fits. (A thread has at most 255 registers, so blocks of
    8 warps always fit.)
    """
    warps = _HULL_3D_WARPS_PER_BLOCK
    while True:
        kernel = _get_convex_hulls_3d_kernel(warps, point_dtype, bbox_dtype)
        if warps == 1 or kernel.max_threads_per_block >= 32 * warps:
            return kernel, warps
        warps //= 2


def _convex_hulls_3d(points, point_offsets, bbox, max_label):
    """Facet equations of the convex hull of each label's 3D points.

    See `_get_convex_hulls_3d_kernel`. Returns the equations (in the form of
    qhull's; label l's start at ``2 * point_offsets[l]``) and the number of
    each label's equations (-1 where the kernel could not compute the hull).
    """
    num_points = points.shape[0]
    bbox = cp.ascontiguousarray(bbox)
    counts = point_offsets[1:] - point_offsets[:-1]
    large = counts > _HULL_3D_BLOCK_POINTS
    large_labels = cp.flatnonzero(large)
    # largest first, so that they do not start last
    large_labels = large_labels[cp.argsort(-counts[large_labels])]
    small_labels = cp.flatnonzero(~large)
    kernel, warps = _convex_hulls_3d_kernel(points.dtype, bbox.dtype)
    num_blocks = large_labels.size + -(-small_labels.size // warps)
    planes = cp.empty((2 * num_points, 4), dtype=cp.int64)
    num_faces = cp.empty((max_label,), dtype=cp.int32)
    if num_blocks > 0:
        kernel(
            (num_blocks,),
            (32 * warps,),
            (
                cp.ascontiguousarray(points),
                point_offsets,
                bbox,
                np.int64(_HULL_3D_MAX_EXTENT),
                large_labels,
                np.int64(large_labels.size),
                small_labels,
                np.int64(small_labels.size),
                cp.empty((num_points,), dtype=cp.int32),
                cp.empty((num_points,), dtype=cp.int32),
                cp.zeros((8 * num_points,), dtype=cp.int64),
                cp.empty((3 * num_points,), dtype=cp.int32),
                planes,
                num_faces,
            ),
        )
    equations = cp.empty((2 * num_points, 4), dtype=cp.float64)
    if num_points > 0:
        _get_hull_equations_kernel()(planes, equations, size=2 * num_points)
    return equations, num_faces


@cp.memoize(for_each_device=True)
def _get_hull_rows_3d_kernel():
    """Kernel finding the voxels of each row inside the 3D convex hull.

    One thread per row of a label's bounding box. A voxel center p (relative
    to the bounding box) is inside if ``n . p + b < tolerance`` for every
    facet equation (n, b) of the label's hull (the ``eq_counts[l]`` rows of
    `equations` from ``eq_starts[l]`` on), as in scikit-image's
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
      long long f_end = eq_starts[l] + eq_counts[l];
      for (long long f = eq_starts[l]; f < f_end; f++) {
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
        "raw float64 equations, raw int64 eq_starts, raw int32 eq_counts, "
        "raw int64 offsets, raw B bbox, float64 tolerance, int64 max_label"
    )
    return cp.ElementwiseKernel(
        inputs,
        "E row_lo, E row_hi, raw uint64 num_pixels",
        source,
        preamble=_warp_includes,
        name="cucim_convex_hull_rows_3d",
    )


# upper bound of the threads launched for the 3D Feret diameters of all labels
_FERET_MAX_THREADS = 2**24


@cp.memoize(for_each_device=True)
def _get_feret_diameter_max_points_kernel(ndim):
    """Kernel computing the largest distance between each label's points.

    Each label has `threads` threads; thread t takes every `threads`-th point
    and its distances to all later points of the label, and the largest
    squared distances are combined with `atomicMax`. (The bits of a
    non-negative double order like integers.) The points are in doubled
    coordinates, and the distances are scaled by half the spacing.
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
      long long l = i / threads;
      long long start = point_offsets[l];
      long long n = point_offsets[l + 1] - start;
      double best = 0.0;
      for (long long p = i % threads; p < n; p += threads) {{{load}
        for (long long q = p + 1; q < n; q++) {{
          const E* q_point = &points[{ndim} * (start + q)];
          double d2 = 0.0;{distance}
          best = fmax(best, d2);
        }}
      }}
      if (best > 0.0) {{
        atomicMax(reinterpret_cast<unsigned long long*>(&best_bits[l]),
                  static_cast<unsigned long long>(__double_as_longlong(best)));
      }}\n"""
    inputs = (
        "raw E points, raw int64 point_offsets, int64 threads, "
        + ", ".join(f"float64 half_spacing{d}" for d in range(ndim))
    )
    return cp.ElementwiseKernel(
        inputs,
        "raw uint64 best_bits",
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
    # enough threads per label to cover its points (a multiple of 32), with
    # at most _FERET_MAX_THREADS in all, so that many small labels stay cheap
    most_points = int(cp.diff(point_offsets).max())
    threads = min(
        -(-most_points // 32) * 32,
        max(32, (_FERET_MAX_THREADS // max_label) // 32 * 32),
    )
    best_bits = cp.zeros((max_label,), dtype=cp.uint64)
    _get_feret_diameter_max_points_kernel(3)(
        points,
        point_offsets,
        threads,
        *(0.5 * float(s) for s in spacing),
        best_bits,
        size=threads * max_label,
    )
    return cp.sqrt(best_bits.view(cp.float64))


def _convex_hull_rows_3d(
    label_image,
    row_first,
    row_last,
    offsets,
    num_rows,
    bbox,
    max_label,
    use_qhull=False,
):
    """Rows of each 3D region's convex image (see `_convex_hull_rows_2d`).

    The hull of each label is computed on the device from the candidates of
    `_convex_hull_candidates_3d` (`_convex_hulls_3d`), and its facet
    equations give the rows (`_get_hull_rows_3d_kernel`). For a label the
    kernel cannot handle, or for every label if `use_qhull`, the hull is
    computed with ``scipy.spatial.ConvexHull`` (Qhull [1]_) on the host
    instead, and if qhull fails, the convex image with
    ``convex_hull_image``.

    References
    ----------
    .. [1] C. B. Barber, D. P. Dobkin and H. Huhdanpaa, "The Quickhull
           algorithm for convex hulls", ACM Transactions on Mathematical
           Software 22(4), 469-483, 1996. :DOI:`10.1145/235815.235821`
    """
    from scipy.spatial import ConvexHull, QhullError

    bbox_host = cp.asnumpy(bbox).astype(np.int64)
    points, point_offsets = _convex_hull_candidates_3d(
        row_first, row_last, offsets, num_rows, bbox, bbox_host, max_label
    )
    if use_qhull:
        equations = cp.empty((2 * points.shape[0], 4), dtype=cp.float64)
        # -1: hull still to compute (absent labels have no points or facets)
        num_faces = cp.where(
            point_offsets[1:] > point_offsets[:-1], -1, 0
        ).astype(cp.int32)
    else:
        equations, num_faces = _convex_hulls_3d(
            points, point_offsets, bbox, max_label
        )
    failed = []
    host_labels = cp.asnumpy(cp.flatnonzero(num_faces < 0)).tolist()
    if host_labels:
        # one transfer each way for all of them
        points_host = cp.asnumpy(points) * 0.5
        offsets_host = cp.asnumpy(point_offsets)
        num_faces_host = cp.asnumpy(num_faces)
        rows, values = [], []
        for lab in host_labels:
            start, end = offsets_host[lab : lab + 2].tolist()
            try:
                hull = ConvexHull(points_host[start:end])
            except QhullError:
                failed.append(lab)
                num_faces_host[lab] = 0
                continue
            # (a hull of n points has at most 2n - 4 facets)
            num = hull.equations.shape[0]
            rows.append(np.arange(2 * start, 2 * start + num))
            values.append(hull.equations)
            num_faces_host[lab] = num
        if values:
            equations[cp.asarray(np.concatenate(rows))] = cp.asarray(
                np.concatenate(values)
            )
        num_faces = cp.asarray(num_faces_host)
    # the offsets of each label's rows, on the host
    shapes = np.maximum(bbox_host[:, 3:] - bbox_host[:, :3], 0)
    row_offsets_host = np.zeros((max_label + 1,), dtype=np.int64)
    np.cumsum(shapes[:, 0] * shapes[:, 1], out=row_offsets_host[1:])

    dtype = row_first.dtype
    row_lo = cp.empty((num_rows,), dtype=dtype)
    row_hi = cp.empty((num_rows,), dtype=dtype)
    num_pixels = cp.zeros((max_label,), dtype=cp.uint64)
    if num_rows > 0:
        _get_hull_rows_3d_kernel()(
            equations,
            2 * point_offsets,
            num_faces,
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
    compute_feret_calipers=False,
    use_qhull=False,
    props_dict=None,
):
    """Compute the convex-hull properties of each region.

    writes "image_convex" to props_dict if `compute_image`
    writes "area_convex" to props_dict if `compute_area`
    writes "feret_diameter_max" to props_dict if `compute_feret`
    writes "feret_diameter_min", "feret_diameter_min_perpendicular",
        "feret_angle_max" and "feret_angle_min" to props_dict if
        `compute_feret_calipers` (2D only)

    In 2D and 3D, all properties are computed for all labels at once from
    the hull of each region's row extents (see `_get_hull_chains_2d_kernel`,
    `_feret_diameter_max_2d`, `_convex_hull_rows_3d` and
    `_feret_diameter_max_3d`). In other dimensions, they are computed from
    one convex image per label (`regionprops_image`).

    If `use_qhull`, the 3D hulls are computed with ``scipy.spatial.ConvexHull``
    on the host instead of on the device.

    Parameters
    ----------
    bbox : cupy.ndarray
        The "bbox" of `_regionprops_label_pass` (exclusive maximum).
    """
    if props_dict is None:
        props_dict = {}
    if not (
        compute_image or compute_area or compute_feret or compute_feret_calipers
    ):
        return props_dict
    ndim = label_image.ndim
    if compute_feret_calipers and ndim != 2:
        raise ValueError("the Feret angles and minimum are for 2D images only")
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
                use_qhull=use_qhull,
            )
        if compute_area:
            props_dict["area_convex"] = num_pixels.astype(cp.float64) * (
                _pixel_area(spacing)
            )
        if compute_image:
            props_dict["image_convex"] = _convex_images(
                row_lo, row_hi, offsets, bbox, max_label
            )
        if ndim == 2 and (compute_feret or compute_feret_calipers):
            # the hull chains of the convex images
            chains = _convex_hull_chains_2d(
                row_lo, row_hi, offsets, num_rows, max_label
            )
            if compute_feret:
                props_dict["feret_diameter_max"] = _feret_diameter_max_2d(
                    row_lo,
                    row_hi,
                    offsets,
                    num_rows,
                    max_label,
                    spacing,
                    chains=chains,
                )
            if compute_feret_calipers:
                props_dict.update(
                    _feret_calipers_2d(chains, offsets, max_label, spacing)
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

    if compute_feret_calipers:
        # no labels
        for name in (
            "feret_diameter_min",
            "feret_diameter_min_perpendicular",
            "feret_angle_max",
            "feret_angle_min",
        ):
            props_dict[name] = cp.zeros((0,), dtype=cp.float64)
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
