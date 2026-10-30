# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import warnings
from collections.abc import Sequence

import cupy as cp

from cucim.skimage._shared.distance import pdist_max_blockwise
from cucim.skimage._shared.utils import _ndarray_argwhere
from cucim.skimage._vendored import ndimage as ndi

from ._regionprops_gpu_utils import _warp_includes

# Store information on which other properties a given property depends on
# This information will be used by `regionprops_dict` to make sure that when
# a particular property is requested any dependent properties are computed
# first.
convex_deps = dict()
convex_deps["image_convex"] = ["image"]  # computed by regionprops_image
convex_deps["area_convex"] = ["image_convex"]
convex_deps["feret_diameter_max"] = ["image_convex"]
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
