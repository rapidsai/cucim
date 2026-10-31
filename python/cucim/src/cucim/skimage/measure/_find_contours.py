# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

from collections import deque

import cupy as cp
import numpy as np

try:
    from cucim.skimage import _cucim_skimage_cpp_ext as _skimage_cpp
except ImportError:
    _skimage_cpp = None

_PARAM_OPTIONS = ("high", "low")


_KERNEL_CODE = r"""
__device__ inline double fc_fraction(double from_value, double to_value, double level)
{
    if (to_value == from_value) {
        return 0.0;
    }
    return (level - from_value) / (to_value - from_value);
}

__device__ inline int fc_square_case(double ul, double ur, double ll, double lr,
                                     double level)
{
    int square_case = 0;
    if (ul > level) {
        square_case += 1;
    }
    if (ur > level) {
        square_case += 2;
    }
    if (ll > level) {
        square_case += 4;
    }
    if (lr > level) {
        square_case += 8;
    }
    return square_case;
}

__device__ inline int fc_segment_count(int square_case)
{
    if (square_case == 0 || square_case == 15) {
        return 0;
    }
    if (square_case == 6 || square_case == 9) {
        return 2;
    }
    return 1;
}

__device__ inline bool fc_isnan(double value)
{
    return value != value;
}

__device__ inline void fc_cell_values(const double* image, int rows, int cols,
                                      int r0, int c0, double* ul, double* ur,
                                      double* ll, double* lr)
{
    int r1 = r0 + 1;
    int c1 = c0 + 1;
    *ul = image[r0 * cols + c0];
    *ur = image[r0 * cols + c1];
    *ll = image[r1 * cols + c0];
    *lr = image[r1 * cols + c1];
}

__device__ inline bool fc_masked_cell(const bool* mask, int cols, int r0, int c0)
{
    int r1 = r0 + 1;
    int c1 = c0 + 1;
    return !(mask[r0 * cols + c0] && mask[r0 * cols + c1] &&
             mask[r1 * cols + c0] && mask[r1 * cols + c1]);
}

__device__ inline void fc_write_segment(double* segments, int segment_id,
                                        double from_r, double from_c,
                                        double to_r, double to_c)
{
    int offset = 4 * segment_id;
    segments[offset + 0] = from_r;
    segments[offset + 1] = from_c;
    segments[offset + 2] = to_r;
    segments[offset + 3] = to_c;
}

// Topological endpoint keys are disjoint IDs for horizontal grid edges,
// vertical grid edges, and exact source-grid vertices. Vertex keys are used
// when interpolation lands exactly on a grid vertex so adjacent cell edges
// still stitch through the same topological endpoint.
__device__ inline long long fc_horizontal_key(int rows, int cols, int r, int c)
{
    return (long long)r * (long long)(cols - 1) + (long long)c;
}

__device__ inline long long fc_vertical_key(int rows, int cols, int r, int c)
{
    return (long long)rows * (long long)(cols - 1) +
           (long long)r * (long long)cols + (long long)c;
}

__device__ inline long long fc_vertex_key(int rows, int cols, int r, int c)
{
    long long n_horizontal = (long long)rows * (long long)(cols - 1);
    long long n_vertical = (long long)(rows - 1) * (long long)cols;
    return n_horizontal + n_vertical + (long long)r * (long long)cols + (long long)c;
}

__device__ inline long long fc_horizontal_endpoint_key(
    int rows, int cols, int r, int c, double fraction)
{
    if (fraction == 0.0) {
        return fc_vertex_key(rows, cols, r, c);
    }
    if (fraction == 1.0) {
        return fc_vertex_key(rows, cols, r, c + 1);
    }
    return fc_horizontal_key(rows, cols, r, c);
}

__device__ inline long long fc_vertical_endpoint_key(
    int rows, int cols, int r, int c, double fraction)
{
    if (fraction == 0.0) {
        return fc_vertex_key(rows, cols, r, c);
    }
    if (fraction == 1.0) {
        return fc_vertex_key(rows, cols, r + 1, c);
    }
    return fc_vertical_key(rows, cols, r, c);
}

__device__ inline void fc_write_keys(long long* segment_keys, int segment_id,
                                     long long from_key, long long to_key)
{
    int offset = 2 * segment_id;
    segment_keys[offset + 0] = from_key;
    segment_keys[offset + 1] = to_key;
}

extern "C" __global__
void fc_count_segments(const double* image, const bool* mask, int* counts,
                       int rows, int cols, double level, int use_mask)
{
    int n_cell_cols = cols - 1;
    int n_cells = (rows - 1) * n_cell_cols;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_cells) {
        return;
    }

    int r0 = idx / n_cell_cols;
    int c0 = idx - r0 * n_cell_cols;

    if (use_mask && fc_masked_cell(mask, cols, r0, c0)) {
        counts[idx] = 0;
        return;
    }

    double ul, ur, ll, lr;
    fc_cell_values(image, rows, cols, r0, c0, &ul, &ur, &ll, &lr);

    if (fc_isnan(ul) || fc_isnan(ur) || fc_isnan(ll) || fc_isnan(lr)) {
        counts[idx] = 0;
        return;
    }

    counts[idx] = fc_segment_count(fc_square_case(ul, ur, ll, lr, level));
}

extern "C" __global__
void fc_generate_segments(const double* image, const bool* mask,
                          const int* counts, const int* scan, double* segments,
                          long long* segment_keys, int rows, int cols,
                          double level, int use_mask,
                          int vertex_connect_high)
{
    int n_cell_cols = cols - 1;
    int n_cells = (rows - 1) * n_cell_cols;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_cells || counts[idx] == 0) {
        return;
    }

    int r0 = idx / n_cell_cols;
    int c0 = idx - r0 * n_cell_cols;
    int r1 = r0 + 1;
    int c1 = c0 + 1;

    if (use_mask && fc_masked_cell(mask, cols, r0, c0)) {
        return;
    }

    double ul, ur, ll, lr;
    fc_cell_values(image, rows, cols, r0, c0, &ul, &ur, &ll, &lr);
    if (fc_isnan(ul) || fc_isnan(ur) || fc_isnan(ll) || fc_isnan(lr)) {
        return;
    }

    int square_case = fc_square_case(ul, ur, ll, lr, level);
    int out_id = scan[idx] - counts[idx];

    double top_fraction = fc_fraction(ul, ur, level);
    double bottom_fraction = fc_fraction(ll, lr, level);
    double left_fraction = fc_fraction(ul, ll, level);
    double right_fraction = fc_fraction(ur, lr, level);

    double top_r = (double)r0;
    double top_c = (double)c0 + top_fraction;
    double bottom_r = (double)r1;
    double bottom_c = (double)c0 + bottom_fraction;
    double left_r = (double)r0 + left_fraction;
    double left_c = (double)c0;
    double right_r = (double)r0 + right_fraction;
    double right_c = (double)c1;

    long long top_key = fc_horizontal_endpoint_key(rows, cols, r0, c0, top_fraction);
    long long bottom_key = fc_horizontal_endpoint_key(rows, cols, r1, c0, bottom_fraction);
    long long left_key = fc_vertical_endpoint_key(rows, cols, r0, c0, left_fraction);
    long long right_key = fc_vertical_endpoint_key(rows, cols, r0, c1, right_fraction);

    if (square_case == 1) {
        fc_write_segment(segments, out_id, top_r, top_c, left_r, left_c);
        fc_write_keys(segment_keys, out_id, top_key, left_key);
    } else if (square_case == 2) {
        fc_write_segment(segments, out_id, right_r, right_c, top_r, top_c);
        fc_write_keys(segment_keys, out_id, right_key, top_key);
    } else if (square_case == 3) {
        fc_write_segment(segments, out_id, right_r, right_c, left_r, left_c);
        fc_write_keys(segment_keys, out_id, right_key, left_key);
    } else if (square_case == 4) {
        fc_write_segment(segments, out_id, left_r, left_c, bottom_r, bottom_c);
        fc_write_keys(segment_keys, out_id, left_key, bottom_key);
    } else if (square_case == 5) {
        fc_write_segment(segments, out_id, top_r, top_c, bottom_r, bottom_c);
        fc_write_keys(segment_keys, out_id, top_key, bottom_key);
    } else if (square_case == 6) {
        if (vertex_connect_high) {
            fc_write_segment(segments, out_id, left_r, left_c, top_r, top_c);
            fc_write_keys(segment_keys, out_id, left_key, top_key);
            fc_write_segment(segments, out_id + 1, right_r, right_c, bottom_r, bottom_c);
            fc_write_keys(segment_keys, out_id + 1, right_key, bottom_key);
        } else {
            fc_write_segment(segments, out_id, right_r, right_c, top_r, top_c);
            fc_write_keys(segment_keys, out_id, right_key, top_key);
            fc_write_segment(segments, out_id + 1, left_r, left_c, bottom_r, bottom_c);
            fc_write_keys(segment_keys, out_id + 1, left_key, bottom_key);
        }
    } else if (square_case == 7) {
        fc_write_segment(segments, out_id, right_r, right_c, bottom_r, bottom_c);
        fc_write_keys(segment_keys, out_id, right_key, bottom_key);
    } else if (square_case == 8) {
        fc_write_segment(segments, out_id, bottom_r, bottom_c, right_r, right_c);
        fc_write_keys(segment_keys, out_id, bottom_key, right_key);
    } else if (square_case == 9) {
        if (vertex_connect_high) {
            fc_write_segment(segments, out_id, top_r, top_c, right_r, right_c);
            fc_write_keys(segment_keys, out_id, top_key, right_key);
            fc_write_segment(segments, out_id + 1, bottom_r, bottom_c, left_r, left_c);
            fc_write_keys(segment_keys, out_id + 1, bottom_key, left_key);
        } else {
            fc_write_segment(segments, out_id, top_r, top_c, left_r, left_c);
            fc_write_keys(segment_keys, out_id, top_key, left_key);
            fc_write_segment(segments, out_id + 1, bottom_r, bottom_c, right_r, right_c);
            fc_write_keys(segment_keys, out_id + 1, bottom_key, right_key);
        }
    } else if (square_case == 10) {
        fc_write_segment(segments, out_id, bottom_r, bottom_c, top_r, top_c);
        fc_write_keys(segment_keys, out_id, bottom_key, top_key);
    } else if (square_case == 11) {
        fc_write_segment(segments, out_id, bottom_r, bottom_c, left_r, left_c);
        fc_write_keys(segment_keys, out_id, bottom_key, left_key);
    } else if (square_case == 12) {
        fc_write_segment(segments, out_id, left_r, left_c, right_r, right_c);
        fc_write_keys(segment_keys, out_id, left_key, right_key);
    } else if (square_case == 13) {
        fc_write_segment(segments, out_id, top_r, top_c, right_r, right_c);
        fc_write_keys(segment_keys, out_id, top_key, right_key);
    } else if (square_case == 14) {
        fc_write_segment(segments, out_id, left_r, left_c, top_r, top_c);
        fc_write_keys(segment_keys, out_id, left_key, top_key);
    }
}

// Contour assembly: list ranking of the linked segments.
//
// next[i] is the segment that follows segment i (-1 if none) and prev[i] the
// one before it. The segments are cut into sublists at "splitters", and one
// thread walks each sublist. Splitters are the first segments of open paths,
// and about one in 64 of the others (to bound the length of sublists). The
// loops that get no splitter in this way are cut at the segments that are lower
// numbered than both neighbors (every closed loop has one) in a second round.

extern "C" __global__
void fc_set_prev(const int* next, int* prev, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int following = next[i];
        if (following >= 0) {
            prev[following] = (int)i;
        }
    }
}

extern "C" __global__
void fc_mark_splitters(const int* next, const int* prev,
                       unsigned char* is_splitter, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        bool sampled = (((unsigned int)i * 2654435761u) >> 26) == 0;
        is_splitter[i] = prev[i] < 0 || next[i] == (int)i || sampled;
    }
}

// Splitters for the segments that no sublist has reached (those of loops
// without a splitter).
extern "C" __global__
void fc_mark_loop_splitters(const int* next, const int* prev,
                            const int* sublist, unsigned char* is_splitter,
                            long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && sublist[i] < 0) {
        is_splitter[i] = prev[i] > (int)i && next[i] > (int)i;
    }
}

extern "C" __global__
void fc_walk_sublists(const int* next, const unsigned char* is_splitter,
                      const int* splitters, int* sublist, int* position,
                      int* length, int* next_sublist, int* min_id,
                      int* max_id, int* max_position, int n_splitters,
                      int first_sublist)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_splitters) {
        return;
    }
    int s = first_sublist + t;
    int first = splitters[t];
    int current = first;
    int j = 0;
    int lowest = first;
    int highest = first;
    int highest_position = 0;
    int following_sublist = -1;
    position[first] = 0;
    while (true) {
        int following = next[current];
        if (following < 0) {
            break;
        }
        if (is_splitter[following]) {
            following_sublist = sublist[following];
            break;
        }
        ++j;
        sublist[following] = s;
        position[following] = j;
        if (following < lowest) {
            lowest = following;
        }
        if (following > highest) {
            highest = following;
            highest_position = j;
        }
        current = following;
    }
    length[t] = j + 1;
    next_sublist[t] = following_sublist;
    min_id[t] = lowest;
    max_id[t] = highest;
    max_position[t] = highest_position;
}

// Write the point that ends each segment: the junction with the next segment
// (the point of the one with the lower index) or, for the last segment of a
// contour, its own end point.
extern "C" __global__
void fc_write_points(const double* segments, const int* next,
                     const int* sublist, const int* position,
                     const long long* start, const int* cut,
                     const long long* wrap, double* points, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    int s = sublist[i];
    int j = position[i];
    int c = cut[s];
    long long target = j > c ? start[s] + (j - c - 1) : wrap[s] + j;
    int following = next[i];
    const double* point;
    if (following < 0 || j == c || (int)i < following) {
        point = segments + 4 * i + 2;
    } else {
        point = segments + 4 * (long long)following;
    }
    points[2 * target] = point[0];
    points[2 * target + 1] = point[1];
}

extern "C" __global__
void fc_write_first_points(const double* segments, const int* next,
                           const int* anchor, const unsigned char* closed,
                           const long long* offsets, double* points,
                           int n_contours)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_contours) {
        return;
    }
    int first = closed[k] ? next[anchor[k]] : anchor[k];
    points[2 * offsets[k]] = segments[4 * (long long)first];
    points[2 * offsets[k] + 1] = segments[4 * (long long)first + 1];
}
"""


@cp.memoize(for_each_device=True)
def _get_kernel(name):
    return cp.RawKernel(_KERNEL_CODE, name)


def find_contours(
    image,
    level=None,
    fully_connected="low",
    positive_orientation="low",
    *,
    mask=None,
    return_packed=False,
):
    """Find iso-valued contours in a 2D CuPy array.

    This implementation extracts marching-squares line segments on the GPU and
    links them into contours there as well. The points are then copied to the
    host, so the returned contours are NumPy arrays. (Without the optional
    compiled extension, or for images with levels that are exactly the value of
    a vertex, the contours are assembled on the host.)

    Parameters
    ----------
    image : ndarray of shape (M, N) and dtype float
        Input image in which to find contours.
    level : float, optional
        Value along which to find contours in the array. By default, the level
        is set to (max(image) + min(image)) / 2
    fully_connected : str, {'low', 'high'}
         Indicates whether array elements below the given level value are to be
         considered fully-connected (and hence elements above the value will
         only be face connected), or vice-versa. (See notes below for details.)
    positive_orientation : str, {'low', 'high'}
         Indicates whether the output contours will produce positively-oriented
         polygons around islands of low- or high-valued elements. If 'low' then
         contours will wind counter-clockwise around elements below the
         iso-value. Alternately, this means that low-valued elements are always
         on the left of the contour. (See below for details.)
    mask : ndarray of shape (M, N) and dtype bool
        A boolean mask, True where we want to draw contours.
        Note that NaN values are always excluded from the considered region
        (``mask`` is set to ``False`` wherever ``array`` is ``NaN``).
    return_packed : bool, optional
        If False (default), return the scikit-image-compatible list of NumPy
        contour arrays. If True, return a tuple ``(points, offsets)`` where
        ``points`` is one contiguous NumPy array of shape ``(K, 2)`` and
        ``offsets`` has shape ``(n_contours + 1,)``. Contour ``i`` is
        ``points[offsets[i]:offsets[i + 1]]``. Packed output requires the
        optional C++ contour assembler. If contour coordinates are needed on
        the GPU, packed output is recommended so that ``cupy.asarray(points)``
        and ``cupy.asarray(offsets)`` can transfer the data with two bulk
        copies instead of many small contour-array copies.

    Returns
    -------
    contours : list of (ndarray of shape (K, 2)) or tuple
        If ``return_packed`` is False, each contour is a ndarray of
        ``(row, column)`` coordinates along the contour. If ``return_packed``
        is True, returns ``(points, offsets)`` as described above.

    See Also
    --------
    cucim.skimage.measure.marching_cubes

    Notes
    -----
    The marching squares algorithm is a special case of the marching cubes
    algorithm [1]_.  A simple explanation is available here:

    https://users.polytech.unice.fr/~lingrand/MarchingCubes/algo.html

    There is a single ambiguous case in the marching squares algorithm: when
    a given ``2 x 2``-element square has two high-valued and two low-valued
    elements, each pair diagonally adjacent. (Where high- and low-valued is
    with respect to the contour value sought.) In this case, either the
    high-valued elements can be 'connected together' via a thin isthmus that
    separates the low-valued elements, or vice-versa. When elements are
    connected together across a diagonal, they are considered 'fully
    connected' (also known as 'face+vertex-connected' or '8-connected'). Only
    high-valued or low-valued elements can be fully-connected, the other set
    will be considered as 'face-connected' or '4-connected'. By default,
    low-valued elements are considered fully-connected; this can be altered
    with the 'fully_connected' parameter.

    Output contours are not guaranteed to be closed: contours which intersect
    the array edge or a masked-off region (either where mask is False or where
    array is NaN) will be left open. All other contours will be closed. (The
    closed-ness of a contours can be tested by checking whether the beginning
    point is the same as the end point.)

    Contours are oriented. By default, array values lower than the contour
    value are to the left of the contour and values greater than the contour
    value are to the right. This means that contours will wind
    counter-clockwise (i.e. in 'positive orientation') around islands of
    low-valued pixels. This behavior can be altered with the
    'positive_orientation' parameter.

    The order of the contours in the output list is determined by the position
    of the smallest ``x,y`` (in lexicographical order) coordinate in the
    contour.  This is a side effect of how the input array is traversed, but
    can be relied upon.

    .. warning::

       Array coordinates/values are assumed to refer to the *center* of the
       array element. Take a simple example input: ``[0, 1]``. The interpolated
       position of 0.5 in this array is midway between the 0-element (at
       ``x=0``) and the 1-element (at ``x=1``), and thus would fall at
       ``x=0.5``.

    This means that to find reasonable contours, it is best to find contours
    midway between the expected "light" and "dark" values. In particular,
    given a binarized array, *do not* choose to find contours at the low or
    high value of the array. This will often yield degenerate contours,
    especially around structures that are a single array element wide. Instead,
    choose a middle value, as above.

    References
    ----------
    .. [1] Lorensen, William and Harvey E. Cline. Marching Cubes: A High
           Resolution 3D Surface Construction Algorithm. Computer Graphics
           (SIGGRAPH 87 Proceedings) 21(4) July 1987, p. 163-170).
           :DOI:`10.1145/37401.37422`

    Examples
    --------
    >>> a = cp.zeros((3, 3))
    >>> a[0, 0] = 1
    >>> a
    array([[1., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]])
    >>> find_contours(a, 0.5)
    [array([[0. , 0.5],
           [0.5, 0. ]])]
    """
    image, level, vertex_connect_high, positive_orientation_high, mask = (
        _validate_find_contours_inputs(
            image, level, fully_connected, positive_orientation, mask
        )
    )

    segments = _get_contour_segments(image, level, vertex_connect_high, mask)
    if return_packed:
        contours = _assemble_contours_packed(*segments)
    else:
        contours = _assemble_contours(*segments)
    if positive_orientation_high:
        if return_packed:
            contours = _reverse_packed_contours(*contours)
        else:
            contours = [contour[::-1] for contour in contours]
    return contours


def _validate_find_contours_inputs(
    image, level, fully_connected, positive_orientation, mask
):
    if fully_connected not in _PARAM_OPTIONS:
        raise ValueError(
            'Parameters "fully_connected" must be either "high" or "low".'
        )
    if positive_orientation not in _PARAM_OPTIONS:
        raise ValueError(
            'Parameters "positive_orientation" must be either "high" or "low".'
        )

    if not isinstance(image, cp.ndarray) or image.ndim != 2:
        raise ValueError("Only 2D CuPy arrays are supported.")
    if image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError("Input array must be at least 2x2.")

    image = cp.ascontiguousarray(image, dtype=cp.float64)
    if level is None:
        level = float((cp.nanmin(image) + cp.nanmax(image)) / 2.0)
    else:
        level = float(level)

    if mask is not None:
        if not isinstance(mask, cp.ndarray) or mask.shape != image.shape:
            raise ValueError(
                'Parameters "array" and "mask" must have same shape.'
            )
        if not np.can_cast(mask.dtype, bool, casting="safe"):
            raise TypeError('Parameter "mask" must be a binary array.')
        mask = cp.ascontiguousarray(mask, dtype=cp.bool_)

    return (
        image,
        level,
        fully_connected == "high",
        positive_orientation == "high",
        mask,
    )


def _get_contour_segments(image, level, vertex_connect_high, mask):
    rows, cols = image.shape
    n_cells = (rows - 1) * (cols - 1)
    counts = cp.empty(n_cells, dtype=cp.int32)
    threads = 256
    blocks = ((n_cells + threads - 1) // threads,)
    mask_arg = mask
    use_mask = mask is not None
    if mask_arg is None:
        mask_arg = cp.empty(1, dtype=cp.bool_)

    _get_kernel("fc_count_segments")(
        blocks,
        (threads,),
        (
            image,
            mask_arg,
            counts,
            rows,
            cols,
            np.float64(level),
            np.int32(use_mask),
        ),
    )
    scan = cp.cumsum(counts, dtype=cp.int32)
    n_segments = int(scan[-1])
    if n_segments == 0:
        return (
            cp.empty((0, 2, 2), dtype=cp.float64),
            cp.empty((0, 2), dtype=cp.int64),
        )

    segments = cp.empty((n_segments, 2, 2), dtype=cp.float64)
    segment_keys = cp.empty((n_segments, 2), dtype=cp.int64)
    _get_kernel("fc_generate_segments")(
        blocks,
        (threads,),
        (
            image,
            mask_arg,
            counts,
            scan,
            segments,
            segment_keys,
            rows,
            cols,
            np.float64(level),
            np.int32(use_mask),
            np.int32(vertex_connect_high),
        ),
    )
    return segments, segment_keys


def _link_segments(segments, segment_keys):
    """Link every segment to the segment that starts where it ends.

    The keys of the segment endpoints identify the grid edge (or vertex) they
    lie on, so two segments are linked when the end key of one is the start key
    of the other. The links are found by sorting and a binary search on the
    device.

    Returns
    -------
    segments, next : cupy.ndarray
        The segments without the degenerate ones and, for each of them, the
        index of the segment that follows it (-1 if there is none). ``None`` is
        returned instead if a key is the start or the end of more than one
        segment (which happens when the level is exactly the value of a
        vertex). Those images need the sequential assembler.
    """
    # The sequential assembler ignores segments of zero length (vertices that
    # are exactly on the level).
    degenerate = (segments[:, 0, 0] == segments[:, 1, 0]) & (
        segments[:, 0, 1] == segments[:, 1, 1]
    )
    if bool(degenerate.any()):
        keep = ~degenerate
        segments = segments[keep]
        segment_keys = segment_keys[keep]
    n_segments = segments.shape[0]
    if n_segments == 0:
        return segments, cp.empty(0, dtype=cp.int32)

    from_keys = cp.ascontiguousarray(segment_keys[:, 0])
    to_keys = cp.ascontiguousarray(segment_keys[:, 1])
    order = cp.argsort(from_keys)
    sorted_from = from_keys[order]
    sorted_to = cp.sort(to_keys)
    repeated = (sorted_from[1:] == sorted_from[:-1]).any() | (
        sorted_to[1:] == sorted_to[:-1]
    ).any()
    if bool(repeated):
        return None
    position = cp.minimum(cp.searchsorted(sorted_from, to_keys), n_segments - 1)
    linked = sorted_from[position] == to_keys
    next_segment = cp.where(linked, order[position], -1).astype(cp.int32)
    return segments, next_segment


def _assemble_linked(segments, segment_keys):
    """Return packed ``(points, offsets)`` or ``None`` if links are ambiguous.

    The contours are put together on the device. Segments that are linked form
    paths and loops. Each path is cut into sublists at a set of splitter
    segments and one thread walks each sublist. The (much shorter) list of
    sublists is arranged into contours on the host, and a last kernel writes
    the points of every segment to its place in the packed array.
    """
    linked = _link_segments(segments, segment_keys)
    if linked is None:
        return None
    segments, next_segment = linked
    n_segments = segments.shape[0]
    if n_segments == 0:
        return np.empty((0, 2)), np.zeros(1, dtype=np.int64)
    threads = 256
    blocks = ((n_segments + threads - 1) // threads,)

    previous = cp.full(n_segments, -1, dtype=cp.int32)
    _get_kernel("fc_set_prev")(
        blocks, (threads,), (next_segment, previous, np.int64(n_segments))
    )
    is_splitter = cp.empty(n_segments, dtype=cp.uint8)
    _get_kernel("fc_mark_splitters")(
        blocks,
        (threads,),
        (next_segment, previous, is_splitter, np.int64(n_segments)),
    )
    sublist = cp.full(n_segments, -1, dtype=cp.int32)
    position = cp.empty(n_segments, dtype=cp.int32)

    def walk_sublists(first_sublist):
        splitters = cp.flatnonzero(is_splitter).astype(cp.int32)
        if first_sublist > 0:
            # only the splitters added by the second round
            splitters = splitters[sublist[splitters] < 0]
        n_new = splitters.size
        sublist[splitters] = cp.arange(
            first_sublist, first_sublist + n_new, dtype=cp.int32
        )
        info = [cp.empty(n_new, dtype=cp.int32) for _ in range(5)]
        _get_kernel("fc_walk_sublists")(
            ((n_new + threads - 1) // threads,),
            (threads,),
            (
                next_segment,
                is_splitter,
                splitters,
                sublist,
                position,
                *info,
                np.int32(n_new),
                np.int32(first_sublist),
            ),
        )
        return splitters, info

    splitters, sublist_info = walk_sublists(0)
    # Loops that were not reached have no splitter yet
    if bool((sublist < 0).any()):
        _get_kernel("fc_mark_loop_splitters")(
            blocks,
            (threads,),
            (
                next_segment,
                previous,
                sublist,
                is_splitter,
                np.int64(n_segments),
            ),
        )
        more_splitters, more_info = walk_sublists(splitters.size)
        splitters = cp.concatenate((splitters, more_splitters))
        sublist_info = [
            cp.concatenate((x, y)) for x, y in zip(sublist_info, more_info)
        ]

    # The splitters, and a few numbers about each sublist, are all that the
    # host needs to know to put the sublists in order.
    offsets, anchor, closed, start, cut, wrap = (
        _skimage_cpp.plan_linked_contours(
            cp.asnumpy(splitters), *[cp.asnumpy(x) for x in sublist_info]
        )
    )
    start = cp.asarray(start)
    cut = cp.asarray(cut)
    wrap = cp.asarray(wrap)
    n_points = int(offsets[-1])
    points = cp.empty((n_points, 2), dtype=cp.float64)
    _get_kernel("fc_write_points")(
        blocks,
        (threads,),
        (
            segments,
            next_segment,
            sublist,
            position,
            start,
            cut,
            wrap,
            points,
            np.int64(n_segments),
        ),
    )
    n_contours = anchor.size
    _get_kernel("fc_write_first_points")(
        ((n_contours + threads - 1) // threads,),
        (threads,),
        (
            segments,
            next_segment,
            cp.asarray(anchor),
            cp.asarray(closed),
            cp.asarray(offsets),
            points,
            np.int32(n_contours),
        ),
    )
    return cp.asnumpy(points), offsets


def _assemble_contours(segments, segment_keys=None):
    if _skimage_cpp is not None and segment_keys is not None:
        packed = _assemble_linked(segments, segment_keys)
        if packed is not None:
            points, offsets = packed
            return [
                points[start:stop]
                for start, stop in zip(offsets[:-1], offsets[1:])
            ]
        return _skimage_cpp.assemble_contours(
            cp.asnumpy(segments), cp.asnumpy(segment_keys)
        )
    return _assemble_contours_python(cp.asnumpy(segments))


def _assemble_contours_packed(segments, segment_keys=None):
    if _skimage_cpp is None or segment_keys is None:
        raise RuntimeError(
            "return_packed=True requires the optional "
            "cucim.skimage C++ contour assembler"
        )
    packed = _assemble_linked(segments, segment_keys)
    if packed is not None:
        return packed
    return _skimage_cpp.assemble_contours_packed(
        cp.asnumpy(segments), cp.asnumpy(segment_keys)
    )


def _reverse_packed_contours(points, offsets):
    reversed_points = np.empty_like(points)
    for start, stop in zip(offsets[:-1], offsets[1:]):
        reversed_points[start:stop] = points[start:stop][::-1]
    return reversed_points, offsets


def _assemble_contours_python(segments):
    current_index = 0
    contours = {}
    starts = {}
    ends = {}
    for from_point, to_point in segments:
        from_point = tuple(from_point)
        to_point = tuple(to_point)

        # Ignore degenerate segments. This matches scikit-image's behavior for
        # vertices exactly on the requested level.
        if from_point == to_point:
            continue

        tail, tail_num = starts.pop(to_point, (None, None))
        head, head_num = ends.pop(from_point, (None, None))

        if tail is not None and head is not None:
            if tail is head:
                head.append(to_point)
            elif tail_num > head_num:
                head.extend(tail)
                contours.pop(tail_num, None)
                starts[head[0]] = (head, head_num)
                ends[head[-1]] = (head, head_num)
            else:
                tail.extendleft(reversed(head))
                starts.pop(head[0], None)
                contours.pop(head_num, None)
                starts[tail[0]] = (tail, tail_num)
                ends[tail[-1]] = (tail, tail_num)
        elif tail is None and head is None:
            new_contour = deque((from_point, to_point))
            contours[current_index] = new_contour
            starts[from_point] = (new_contour, current_index)
            ends[to_point] = (new_contour, current_index)
            current_index += 1
        elif head is None:
            tail.appendleft(from_point)
            starts[from_point] = (tail, tail_num)
        else:
            head.append(to_point)
            ends[to_point] = (head, head_num)

    return [np.array(contour) for _, contour in sorted(contours.items())]
