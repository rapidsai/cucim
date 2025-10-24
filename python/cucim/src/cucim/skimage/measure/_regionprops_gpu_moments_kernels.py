# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import cupy as cp

from ._regionprops_gpu_basic_kernels import get_bbox_coords_kernel
from ._regionprops_gpu_utils import (
    _check_intensity_image_shape,
    _includes,
    _unravel_loop_index,
    _unravel_loop_index_declarations,
)

__all__ = [
    "regionprops_centroid",
    "regionprops_centroid_local",
    "regionprops_centroid_weighted",
    "regionprops_inertia_tensor",
    "regionprops_inertia_tensor_eigvals",
    "regionprops_moments",
    "regionprops_moments_central",
    "regionprops_moments_hu",
    "regionprops_moments_normalized",
]


# Store information on which other properties a given property depends on
# This information will be used by `regionprops_dict` to make sure that when
# a particular property is requested any dependent properties are computed
# first.
moment_deps = dict()
moment_deps["moments"] = ["bbox"]
moment_deps["moments_weighted"] = ["bbox"]
moment_deps["eccentricity"] = ["inertia_tensor_eigvals"]
moment_deps["axis_major_length"] = ["inertia_tensor_eigvals"]
moment_deps["axis_minor_length"] = ["inertia_tensor_eigvals"]
moment_deps["inertia_tensor_eigenvectors"] = ["inertia_tensor_eigvals"]
moment_deps["inertia_tensor_eigvals"] = ["inertia_tensor"]
moment_deps["orientation"] = ["inertia_tensor"]
moment_deps["moments_hu"] = ["moments_normalized"]
moment_deps["moments_normalized"] = ["moments_central"]
moment_deps["inertia_tensor"] = ["moments_central"]
moment_deps["moments_central"] = ["moments"]
moment_deps["centroid"] = ["moments"]
moment_deps["centroid_local"] = ["moments"]
moment_deps["moments_weighted_central"] = ["moments_weighted"]
moment_deps["moments_weighted_normalized"] = ["moments_weighted_central"]
moment_deps["moments_weighted_hu"] = ["moments_weighted_normalized"]
moment_deps["centroid_weighted"] = ["moments_weighted"]
moment_deps["centroid_weighted_local"] = ["moments_weighted"]


# The minimum moment "order" required to compute each property
required_order = {
    "centroid": 1,
    "centroid_local": 1,
    "centroid_weighted": 1,
    "centroid_weighted_local": 1,
    "axis_major_length": 2,
    "axis_minor_length": 2,
    "eccentricity": 2,
    "inertia_tensor": 2,
    "inertia_tensor_eigvals": 2,
    "inertia_tensor_eigenvectors": 2,
    "moments": 2,
    "moments_central": 2,
    "moments_normalized": 2,
    "moments_weighted": 2,
    "moments_weighted_central": 2,
    "moments_weighted_normalized": 2,
    "orientation": 2,
    "moments_hu": 3,
    "moments_weighted_hu": 3,
}


def regionprops_centroid(
    label_image,
    max_label=None,
    pixels_per_thread=16,
    props_dict=None,
):
    """Compute the centroid of each labeled region in the image.

    reuses "num_pixels" from previously computed properties if present

    writes "centroid" to `props_dict`

    Returns
    -------
    centroid : cp.ndarray
        The centroid of each region.
    """
    if max_label is None:
        max_label = int(label_image.max())
    ndim = label_image.ndim

    int32_coords = max(label_image.shape) < 2**32
    if props_dict is not None and "num_pixels" in props_dict:
        centroid_counts = props_dict["num_pixels"]
        if centroid_counts.dtype != cp.uint32:
            centroid_counts = centroid_counts.astype(cp.uint32)
        compute_num_pixels = False
    else:
        centroid_counts = cp.zeros((max_label,), dtype=cp.uint32)
        compute_num_pixels = True

    bbox_coords_kernel = get_bbox_coords_kernel(
        ndim=label_image.ndim,
        int32_coords=int32_coords,
        compute_bbox=False,
        compute_num_pixels=compute_num_pixels,
        compute_coordinate_sums=True,
        pixels_per_thread=pixels_per_thread,
    )

    centroid_sums = cp.zeros((max_label, ndim), dtype=cp.uint64)

    # make a copy if the inputs are not already C-contiguous
    if not label_image.flags.c_contiguous:
        label_image = cp.ascontiguousarray(label_image)

    if compute_num_pixels:
        outputs = (centroid_counts, centroid_sums)
    else:
        outputs = centroid_sums
    bbox_coords_kernel(
        label_image,
        label_image.size,
        *outputs,
        size=math.ceil(label_image.size / pixels_per_thread),
    )

    centroid = centroid_sums / centroid_counts[:, cp.newaxis]
    if props_dict is not None:
        props_dict["centroid"] = centroid
        if "num_pixels" not in props_dict:
            props_dict["num_pixels"] = centroid_counts
    return centroid


@cp.memoize(for_each_device=True)
def get_centroid_local_kernel(coord_dtype, ndim):
    """Keep this kernel for n-dimensional support as the raw_moments kernels
    currently only support 2D and 3D data.
    """
    coord_dtype = cp.dtype(coord_dtype)
    sum_dtype = cp.dtype(cp.uint64)
    count_dtype = cp.dtype(cp.uint32)
    uint_t = (
        "unsigned int" if coord_dtype.itemsize <= 4 else "unsigned long long"
    )

    source = """
          auto L = label[i];
          if (L != 0) {"""
    source += _unravel_loop_index("label", ndim, uint_t=uint_t)
    for d in range(ndim):
        source += f"""
            atomicAdd(&centroid_sums[(L - 1) * {ndim} + {d}],
                      in_coord[{d}] - bbox[(L - 1) * {2 * ndim} + {d}]);
        """
    source += """
        atomicAdd(&centroid_counts[L - 1], 1);
          }\n"""
    inputs = f"raw X label, raw {coord_dtype.name} bbox"
    outputs = f"raw {count_dtype.name} centroid_counts, "
    outputs += f"raw {sum_dtype.name} centroid_sums"
    name = f"cucim_centroid_local_{ndim}d_{coord_dtype.name}"
    return cp.ElementwiseKernel(
        inputs, outputs, source, preamble=_includes, name=name
    )


def regionprops_centroid_local(
    label_image,
    max_label=None,
    coord_dtype=cp.uint32,
    pixels_per_thread=16,
    props_dict=None,
):
    """Compute the central moments of the labeled regions.

    dimensions supported: nD

    reuses "moments" from previously computed properties if present
    reuses "bbox" from previously computed properties if present

    writes "centroid_local" to `props_dict`
    writes "bbox" to `props_dict` if it was not already present
    writes "num_pixels" to `props_dict` if it was not already present

    Parameters
    ----------
    label_image : cp.ndarray
        Image containing labels where 0 is the background and sequential
        values > 0 are the labels.
    max_label : int or None
        The maximum label value present in label_image. Will be computed if not
        provided.
    coord_dtype : dtype, optional
        The data type to use for coordinate calculations. Should be
        ``cp.uint32`` or ``cp.uint64``.

    Returns
    -------
    counts : cp.ndarray
        The number of samples in each region.
    centroid_local : cp.ndarray
        The local centroids

    Notes
    -----
    The centroid could also be extracted from the raw moments
    computed via `regionprops_moments`. That will be more efficient than
    running this separate function if additional moment-based properties
    are also needed.

    This function is also useful for data with more than 3 dimensions as
    regionprops_moments currently only supports 2D and 3D data.
    """
    if props_dict is None:
        props_dict = {}
    if max_label is None:
        max_label = int(label_image.max())

    int32_coords = max(label_image.shape) < 2**32
    coord_dtype = cp.dtype(cp.uint32 if int32_coords else cp.uint64)

    ndim = label_image.ndim

    if "moments" in props_dict and ndim in [2, 3]:
        # already have the moments needed in previously computed properties
        moments = props_dict["moments"]
        # can't compute if only zeroth moment is present
        if moments.shape[-1] > 1:
            centroid_local = cp.empty((max_label, ndim), dtype=moments.dtype)
            if ndim == 2:
                m0 = moments[:, 0, 0]
                centroid_local[:, 0] = moments[:, 1, 0] / m0
                centroid_local[:, 1] = moments[:, 0, 1] / m0
            else:
                m0 = moments[:, 0, 0, 0]
                centroid_local[:, 0] = moments[:, 1, 0, 0] / m0
                centroid_local[:, 1] = moments[:, 0, 1, 0] / m0
                centroid_local[:, 2] = moments[:, 0, 0, 1] / m0
            props_dict["centroid_local"] = centroid_local
            return centroid_local

    if "bbox" in props_dict:
        # reuse previously computed bounding box coordinates
        bbox_coords = props_dict["bbox"]
        if bbox_coords.dtype != coord_dtype:
            bbox_coords = bbox_coords.astype(coord_dtype)

    else:
        bbox_coords_kernel = get_bbox_coords_kernel(
            ndim=label_image.ndim,
            int32_coords=int32_coords,
            compute_bbox=True,
            compute_num_pixels=False,
            compute_coordinate_sums=False,
            pixels_per_thread=pixels_per_thread,
        )

        bbox_coords = cp.zeros((max_label, 2 * ndim), dtype=coord_dtype)

        # Initialize value for atomicMin on first ndim coordinates
        # The value for atomicMax columns is already 0 as desired.
        bbox_coords[:, :ndim] = cp.iinfo(coord_dtype).max

        # make a copy if the inputs are not already C-contiguous
        if not label_image.flags.c_contiguous:
            label_image = cp.ascontiguousarray(label_image)

        bbox_coords_kernel(
            label_image,
            label_image.size,
            bbox_coords,
            size=math.ceil(label_image.size / pixels_per_thread),
        )
        if "bbox" not in props_dict:
            props_dict["bbox"] = bbox_coords

    counts = cp.zeros((max_label,), dtype=cp.uint32)
    centroids_sums = cp.zeros((max_label, ndim), dtype=cp.uint64)
    centroid_local_kernel = get_centroid_local_kernel(
        coord_dtype, label_image.ndim
    )
    centroid_local_kernel(
        label_image, bbox_coords, counts, centroids_sums, size=label_image.size
    )

    centroid_local = centroids_sums / counts[:, cp.newaxis]
    props_dict["centroid_local"] = centroid_local
    if "num_pixels" not in props_dict:
        props_dict["num_pixels"] = counts
    return centroid_local


def _get_raw_moments_code(
    coord_c_type,
    moments_c_type,
    ndim,
    order,
    array_size,
    num_channels=1,
    has_spacing=False,
    has_weights=False,
):
    """
    Notes
    -----
    Local variables created:

        - local_moments : shape (array_size, num_channels, num_moments)
            local set of moments up to the specified order (1-3 supported)

    Output variables written to:

        - moments : shape (max_label, num_channels, num_moments)
    """

    # number is for a densely populated moments matrix of size (order + 1) per
    # side (values at locations where order is greater than specified will be 0)
    num_moments = (order + 1) ** ndim

    if order > 3:
        raise ValueError("Only moments of orders 0-3 are supported")

    use_floating_point = moments_c_type in ["float", "double"]

    source_pre = f"""
    {moments_c_type} local_moments[{array_size*num_channels*num_moments}] = {{0}};
    {coord_c_type} m_offset = 0;
    {coord_c_type} local_off = 0;\n"""  # noqa: E501
    if has_weights:
        source_pre += f"""
    {moments_c_type} w = 0.0;\n"""

    # op uses external coordinate array variables:
    #    bbox : bounding box coordinates, shape (max_label, 2*ndim)
    #    in_coord[0]...in_coord[ndim - 1] : coordinates
    #        coordinates in the labeled image at the current index
    #    ii : index into labels array
    #    current_label : value of the label image at location ii
    #    spacing (optional) : pixel spacings
    #    img (optional) : intensity image
    source_operation = ""
    # using bounding box to transform the global coordinates to local ones
    # (c0 = local coordinate on axis 0, etc.)
    for d in range(ndim):
        source_operation += f"""
                {moments_c_type} c{d} = in_coord[{d}]
                            - bbox[(current_label - 1) * {2 * ndim} + {d}];"""
        if has_spacing:
            source_operation += f"""
                c{d} *= spacing[{d}];"""

    # need additional multiplication by the intensity value for weighted case
    w = "w * " if has_weights else ""
    for c in range(num_channels):
        source_operation += f"""
                local_off = {num_moments*num_channels}*offset + {c * num_moments};\n"""  # noqa: E501

        # zeroth moment
        if has_weights:
            source_operation += f"""
                w = static_cast<{moments_c_type}>(img[{num_channels} * ii + {c}]);
                local_moments[local_off] += w;\n"""  # noqa: E501
        elif use_floating_point:
            source_operation += """
                local_moments[local_off] += 1.0;\n"""
        else:
            source_operation += """
                local_moments[local_off] += 1;\n"""

        # moments for order 1-3
        if ndim == 2:
            if order == 1:
                source_operation += f"""
                local_moments[local_off + 1] += {w}c1;
                local_moments[local_off + 2] += {w}c0;\n"""
            elif order == 2:
                source_operation += f"""
                local_moments[local_off + 1] += {w}c1;
                local_moments[local_off + 2] += {w}c1 * c1;
                local_moments[local_off + 3] += {w}c0;
                local_moments[local_off + 4] += {w}c0 * c1;
                local_moments[local_off + 6] += {w}c0 * c0;\n"""
            elif order == 3:
                source_operation += f"""
                local_moments[local_off + 1] += {w}c1;
                local_moments[local_off + 2] += {w}c1 * c1;
                local_moments[local_off + 3] += {w}c1 * c1 * c1;
                local_moments[local_off + 4] += {w}c0;
                local_moments[local_off + 5] += {w}c0 * c1;
                local_moments[local_off + 6] += {w}c0 * c1 * c1;
                local_moments[local_off + 8] += {w}c0 * c0;
                local_moments[local_off + 9] += {w}c0 * c0 * c1;
                local_moments[local_off + 12] += {w}c0 * c0 * c0;\n"""
        elif ndim == 3:
            if order == 1:
                source_operation += f"""
                local_moments[local_off + 1] += {w}c2;
                local_moments[local_off + 2] += {w}c1;
                local_moments[local_off + 4] += {w}c0;\n"""
            elif order == 2:
                source_operation += f"""
                local_moments[local_off + 1] += {w}c2;
                local_moments[local_off + 2] += {w}c2 * c2;
                local_moments[local_off + 3] += {w}c1;
                local_moments[local_off + 4] += {w}c1 * c2;
                local_moments[local_off + 6] += {w}c1 * c1;
                local_moments[local_off + 9] += {w}c0;
                local_moments[local_off + 10] += {w}c0 * c2;
                local_moments[local_off + 12] += {w}c0 * c1;
                local_moments[local_off + 18] += {w}c0 * c0;\n"""
            elif order == 3:
                source_operation += f"""
                local_moments[local_off + 1] += {w}c2;
                local_moments[local_off + 2] += {w}c2 * c2;
                local_moments[local_off + 3] += {w}c2 * c2 * c2;
                local_moments[local_off + 4] += {w}c1;
                local_moments[local_off + 5] += {w}c1 * c2;
                local_moments[local_off + 6] += {w}c1 * c2 * c2;
                local_moments[local_off + 8] += {w}c1 * c1;
                local_moments[local_off + 9] += {w}c1 * c1 * c2;
                local_moments[local_off + 12] += {w}c1 * c1 * c1;
                local_moments[local_off + 16] += {w}c0;
                local_moments[local_off + 17] += {w}c0 * c2;
                local_moments[local_off + 18] += {w}c0 * c2 * c2;
                local_moments[local_off + 20] += {w}c0 * c1;
                local_moments[local_off + 21] += {w}c0 * c1 * c2;
                local_moments[local_off + 24] += {w}c0 * c1 * c1;
                local_moments[local_off + 32] += {w}c0 * c0;
                local_moments[local_off + 33] += {w}c0 * c0 * c2;
                local_moments[local_off + 36] += {w}c0 * c0 * c1;
                local_moments[local_off + 48] += {w}c0 * c0 * c0;\n"""
        else:
            raise ValueError("only ndim = 2 or 3 is supported")

    # post_operation uses external variables:
    #     ii : index into num_pixels array
    #     lab : label value that corresponds to location ii
    #     coord_sums : output with shape (max_label, ndim)
    source_post = ""
    for c in range(0, num_channels):
        source_post += f"""
                // moments outputs
                m_offset = {num_moments*num_channels}*(lab - 1) + {c * num_moments};
                local_off = {num_moments*num_channels}*ii + {c * num_moments};
                atomicAdd(&moments[m_offset], local_moments[local_off]);\n"""  # noqa: E501

        if ndim == 2:
            if order == 1:
                source_post += """
                atomicAdd(&moments[m_offset + 1], local_moments[local_off + 1]);
                atomicAdd(&moments[m_offset + 2], local_moments[local_off + 2]);\n"""  # noqa: E501
            elif order == 2:
                source_post += """
                atomicAdd(&moments[m_offset + 1], local_moments[local_off + 1]);
                atomicAdd(&moments[m_offset + 2], local_moments[local_off + 2]);
                atomicAdd(&moments[m_offset + 3], local_moments[local_off + 3]);
                atomicAdd(&moments[m_offset + 4], local_moments[local_off + 4]);
                atomicAdd(&moments[m_offset + 6], local_moments[local_off + 6]);\n"""  # noqa: E501
            elif order == 3:
                source_post += """
                atomicAdd(&moments[m_offset + 1], local_moments[local_off + 1]);
                atomicAdd(&moments[m_offset + 2], local_moments[local_off + 2]);
                atomicAdd(&moments[m_offset + 3], local_moments[local_off + 3]);
                atomicAdd(&moments[m_offset + 4], local_moments[local_off + 4]);
                atomicAdd(&moments[m_offset + 5], local_moments[local_off + 5]);
                atomicAdd(&moments[m_offset + 6], local_moments[local_off + 6]);
                atomicAdd(&moments[m_offset + 8], local_moments[local_off + 8]);
                atomicAdd(&moments[m_offset + 9], local_moments[local_off + 9]);
                atomicAdd(&moments[m_offset + 12], local_moments[local_off + 12]);\n"""  # noqa: E501
        elif ndim == 3:
            if order == 1:
                source_post += """
                atomicAdd(&moments[m_offset + 1], local_moments[local_off + 1]);
                atomicAdd(&moments[m_offset + 2], local_moments[local_off + 2]);
                atomicAdd(&moments[m_offset + 4], local_moments[local_off + 4]);\n"""  # noqa: E501
            elif order == 2:
                source_post += """
                atomicAdd(&moments[m_offset + 1], local_moments[local_off + 1]);
                atomicAdd(&moments[m_offset + 2], local_moments[local_off + 2]);
                atomicAdd(&moments[m_offset + 3], local_moments[local_off + 3]);
                atomicAdd(&moments[m_offset + 4], local_moments[local_off + 4]);
                atomicAdd(&moments[m_offset + 6], local_moments[local_off + 6]);
                atomicAdd(&moments[m_offset + 9], local_moments[local_off + 9]);
                atomicAdd(&moments[m_offset + 10], local_moments[local_off + 10]);
                atomicAdd(&moments[m_offset + 12], local_moments[local_off + 12]);
                atomicAdd(&moments[m_offset + 18], local_moments[local_off + 18]);\n"""  # noqa: E501
            elif order == 3:
                source_post += """
                atomicAdd(&moments[m_offset + 1], local_moments[local_off + 1]);
                atomicAdd(&moments[m_offset + 2], local_moments[local_off + 2]);
                atomicAdd(&moments[m_offset + 3], local_moments[local_off + 3]);
                atomicAdd(&moments[m_offset + 4], local_moments[local_off + 4]);
                atomicAdd(&moments[m_offset + 5], local_moments[local_off + 5]);
                atomicAdd(&moments[m_offset + 6], local_moments[local_off + 6]);
                atomicAdd(&moments[m_offset + 8], local_moments[local_off + 8]);
                atomicAdd(&moments[m_offset + 9], local_moments[local_off + 9]);
                atomicAdd(&moments[m_offset + 12], local_moments[local_off + 12]);
                atomicAdd(&moments[m_offset + 16], local_moments[local_off + 16]);
                atomicAdd(&moments[m_offset + 17], local_moments[local_off + 17]);
                atomicAdd(&moments[m_offset + 18], local_moments[local_off + 18]);
                atomicAdd(&moments[m_offset + 20], local_moments[local_off + 20]);
                atomicAdd(&moments[m_offset + 21], local_moments[local_off + 21]);
                atomicAdd(&moments[m_offset + 24], local_moments[local_off + 24]);
                atomicAdd(&moments[m_offset + 32], local_moments[local_off + 32]);
                atomicAdd(&moments[m_offset + 33], local_moments[local_off + 33]);
                atomicAdd(&moments[m_offset + 36], local_moments[local_off + 36]);
                atomicAdd(&moments[m_offset + 48], local_moments[local_off + 48]);\n"""  # noqa: E501
    return source_pre, source_operation, source_post


@cp.memoize(for_each_device=True)
def get_raw_moments_kernel(
    ndim,
    order,
    moments_dtype=cp.float64,
    int32_coords=True,
    spacing=None,
    weighted=False,
    num_channels=1,
    pixels_per_thread=8,
):
    moments_dtype = cp.dtype(moments_dtype)

    array_size = pixels_per_thread

    coord_dtype = cp.dtype(cp.uint32 if int32_coords else cp.uint64)
    if coord_dtype.itemsize <= 4:
        coord_c_type = "unsigned int"
    else:
        coord_c_type = "unsigned long long"

    use_floating_point = moments_dtype.kind == "f"
    has_spacing = spacing is not None
    if (weighted or has_spacing) and not use_floating_point:
        raise ValueError(
            "`moments_dtype` must be a floating point type for weighted "
            "moments calculations or moment calculations using spacing."
        )
    moments_c_type = "double" if use_floating_point else "unsigned long long"
    if spacing is not None:
        if len(spacing) != ndim:
            raise ValueError("len(spacing) must equal len(shape)")
        if moments_dtype.kind != "f":
            raise ValueError("moments must have a floating point data type")

    moments_pre, moments_op, moments_post = _get_raw_moments_code(
        coord_c_type=coord_c_type,
        moments_c_type=moments_c_type,
        ndim=ndim,
        order=order,
        array_size=array_size,
        has_weights=weighted,
        has_spacing=spacing is not None,
        num_channels=num_channels,
    )

    # store only counts for label > 0  (label = 0 is the background)
    source = f"""
      uint64_t start_index = {pixels_per_thread}*i;
    """
    source += moments_pre

    inner_op = ""

    source += _unravel_loop_index_declarations(
        "labels", ndim, uint_t=coord_c_type
    )

    inner_op += _unravel_loop_index(
        "labels",
        ndim=ndim,
        uint_t=coord_c_type,
        raveled_index="ii",
        omit_declarations=True,
    )
    inner_op += moments_op

    source += f"""
      X encountered_labels[{array_size}] = {{0}};
      X current_label;
      X prev_label = labels[start_index];
      int offset = 0;
      encountered_labels[0] = prev_label;
      uint64_t ii_max = min(start_index + {pixels_per_thread}, labels_size);
      for (uint64_t ii = start_index; ii < ii_max; ii++) {{
        current_label = labels[ii];
        if (current_label == 0) {{ continue; }}
        if (current_label != prev_label) {{
            offset += 1;
            prev_label = current_label;
            encountered_labels[offset] = current_label;
        }}
        {inner_op}
      }}"""
    source += """
      for (size_t ii = 0; ii <= offset; ii++) {
        X lab = encountered_labels[ii];
        if (lab != 0) {"""

    source += moments_post
    source += """
        }
      }\n"""

    # print(source)
    inputs = (
        f"raw X labels, raw uint64 labels_size, raw {coord_dtype.name} bbox"
    )
    if spacing:
        inputs += ", raw float64 spacing"
    if weighted:
        inputs += ", raw Y img"
    outputs = f"raw {moments_dtype.name} moments"
    weighted_str = "_weighted" if weighted else ""
    spacing_str = "_sp" if spacing else ""
    name = f"cucim_moments{weighted_str}{spacing_str}_order{order}_{ndim}d"
    name += f"_{coord_dtype.char}_{moments_dtype.char}_batch{pixels_per_thread}"
    return cp.ElementwiseKernel(
        inputs, outputs, source, preamble=_includes, name=name
    )


def regionprops_moments(
    label_image,
    intensity_image=None,
    max_label=None,
    order=2,
    spacing=None,
    pixels_per_thread=10,
    props_dict=None,
):
    """Compute the raw moments of the labeled regions.

    reuses "bbox" from previously computed properties if present

    writes "moments" to `props_dict` if `intensity_image` is not provided
    writes "moments_weighted" to `props_dict` if `intensity_image` is provided

    Parameters
    ----------
    label_image : cp.ndarray
        Image containing labels where 0 is the background and sequential
        values > 0 are the labels.
    intensity_image : cp.ndarray, optional
        Image of intensities. If provided, weighted moments are computed. If
        this is a multi-channel image, moments are computed independently for
        each channel.
    max_label : int or None, optional
        The maximum label value present in label_image. Will be computed if not
        provided.

    Returns
    -------
    moments : cp.ndarray
        The moments up to the specified order. Will be stored in an
        ``(order + 1, ) * ndim`` matrix where any elements corresponding to
        order greater than that specified will be set to 0.  For example, for
        the 2D case, the last two axes represent the 2D moments matrix, ``M``
        where each matrix would have the following sizes and non-zero entries:

            ```py
            # for a 2D image with order = 1
            M = [
               [m00, m01],
               [m10,   0],
            ]

            # for a 2D image with order = 2
            M = [
               [m00, m01, m02],
               [m10, m11,   0],
               [m20,   0,   0],
            ]

            # for a 2D image with order = 3
            M = [
               [m00, m01, m02, m03],
               [m10, m11, m12,   0],
               [m20, m21,   0,   0],
               [m30,   0,   0,   0],
            ]
            ```

        When there is no `intensity_image` or the `intensity_image` is single
        channel, the shape of the moments output is
        ``shape = (max_label, ) + (order + 1, ) * ndim``.
        When the ``intensity_image`` is multichannel a channel axis will be
        present in the `moments` output at position 1 to give
        ``shape = (max_label, ) + (num_channels, ) + (order + 1,) * ndim``.
    """

    if props_dict is None:
        props_dict = {}

    if max_label is None:
        max_label = int(label_image.max())

    # make a copy if the inputs are not already C-contiguous
    if not label_image.flags.c_contiguous:
        label_image = cp.ascontiguousarray(label_image)
    ndim = label_image.ndim

    int32_coords = max(label_image.shape) < 2**32
    coord_dtype = cp.dtype(cp.uint32 if int32_coords else cp.uint64)
    if "bbox" in props_dict:
        bbox_coords = props_dict["bbox"]
        if bbox_coords.dtype != coord_dtype:
            bbox_coords = bbox_coords.astype(coord_dtype)
    else:
        bbox_kernel = get_bbox_coords_kernel(
            ndim=ndim,
            int32_coords=int32_coords,
            compute_bbox=True,
            compute_num_pixels=False,
            compute_coordinate_sums=False,
            pixels_per_thread=pixels_per_thread,
        )

        bbox_coords = cp.zeros((max_label, 2 * ndim), dtype=coord_dtype)

        # Initialize value for atomicMin on first ndim coordinates
        # The value for atomicMax columns is already 0 as desired.
        bbox_coords[:, :ndim] = cp.iinfo(coord_dtype).max

        bbox_kernel(
            label_image,
            label_image.size,
            bbox_coords,
            size=math.ceil(label_image.size / pixels_per_thread),
        )
        if props_dict is not None:
            props_dict["bbox"] = bbox_coords

    moments_shape = (max_label,) + (order + 1,) * ndim
    if intensity_image is not None:
        if not intensity_image.flags.c_contiguous:
            intensity_image = cp.ascontiguousarray(intensity_image)

        num_channels = _check_intensity_image_shape(
            label_image, intensity_image
        )
        if num_channels > 1:
            moments_shape = (max_label,) + (num_channels,) + (order + 1,) * ndim
        weighted = True
    else:
        num_channels = 1
        weighted = False

    # total number of elements in the moments matrix
    moments = cp.zeros(moments_shape, dtype=cp.float64)
    moments_kernel = get_raw_moments_kernel(
        ndim=label_image.ndim,
        order=order,
        moments_dtype=moments.dtype,
        int32_coords=int32_coords,
        spacing=spacing,
        weighted=weighted,
        num_channels=num_channels,
        pixels_per_thread=pixels_per_thread,
    )
    input_args = (
        label_image,
        label_image.size,
        bbox_coords,
    )
    if spacing:
        input_args = input_args + (cp.asarray(spacing, dtype=cp.float64),)
    if weighted:
        input_args = input_args + (intensity_image,)
    size = math.ceil(label_image.size / pixels_per_thread)
    moments_kernel(*input_args, moments, size=size)
    if weighted:
        props_dict["moments_weighted"] = moments
    else:
        props_dict["moments"] = moments
    return moments


@cp.memoize(for_each_device=True)
def get_moments_central_kernel(
    moments_dtype,
    ndim,
    order,
):
    """Applies analytical formulas to convert raw moments to central moments

    These are as in `_moments_raw_to_central_fast` from
    `_moments_analytical.py` but that kernel is scalar, while this one will be
    applied to all labeled regions (and any channels dimension) at once.
    """
    moments_dtype = cp.dtype(moments_dtype)

    uint_t = "unsigned int"

    # number is for a densely populated moments matrix of size (order + 1) per
    # side (values at locations where order is greater than specified will be 0)
    num_moments = (order + 1) ** ndim

    if moments_dtype.kind != "f":
        raise ValueError(
            "`moments_dtype` must be a floating point type for central moments "
            "calculations."
        )

    # floating point type used for the intermediate computations
    float_type = "double"

    source = f"""
            {uint_t} offset = i * {num_moments};\n"""
    if ndim == 2:
        if order <= 1:
            # only zeroth moment is non-zero for central moments
            source += """
            out[offset] = moments_raw[offset];\n"""
        elif order == 2:
            source += f"""
            // retrieve the 2nd order raw moments
            {float_type} m00 = moments_raw[offset];
            {float_type} m01 = moments_raw[offset + 1];
            {float_type} m02 = moments_raw[offset + 2];
            {float_type} m10 = moments_raw[offset + 3];
            {float_type} m11 = moments_raw[offset + 4];
            {float_type} m20 = moments_raw[offset + 6];

            // compute centroids
            // (TODO: add option to output the centroids as well?)
            {float_type} cx = m10 / m00;
            {float_type} cy = m01 / m00;

            // analytical expressions for the central moments
            out[offset] = m00;                  // out[0, 0]
            // 2nd order central moments
            out[offset + 2] = m02 - cy * m01;   // out[0, 2]
            out[offset + 4] = m11 - cx * m01;   // out[1, 1]
            out[offset + 6] = m20 - cx * m10;   // out[2, 0]\n"""
        elif order == 3:
            source += f"""
            // retrieve the 2nd order raw moments
            {float_type} m00 = moments_raw[offset];
            {float_type} m01 = moments_raw[offset + 1];
            {float_type} m02 = moments_raw[offset + 2];
            {float_type} m03 = moments_raw[offset + 3];
            {float_type} m10 = moments_raw[offset + 4];
            {float_type} m11 = moments_raw[offset + 5];
            {float_type} m12 = moments_raw[offset + 6];
            {float_type} m20 = moments_raw[offset + 8];
            {float_type} m21 = moments_raw[offset + 9];
            {float_type} m30 = moments_raw[offset + 12];

            // compute centroids
            {float_type} cx = m10 / m00;
            {float_type} cy = m01 / m00;

            // zeroth moment
            out[offset] = m00;                                                  // out[0, 0]
            // 2nd order central moments
            out[offset + 2] = m02 - cy * m01;                                   // out[0, 2]
            out[offset + 5] = m11 - cx * m01;                                   // out[1, 1]
            out[offset + 8] = m20 - cx * m10;                                   // out[2, 0]
            // 3rd order central moments
            out[offset + 3] = m03 - 3*cy*m02 + 2*cy*cy*m01;                     // out[0, 3]
            out[offset + 6] = m12 - 2*cy*m11 - cx*m02 + 2*cy*cx*m01;            // out[1, 2]
            out[offset + 9] = m21 - 2*cx*m11 - cy*m20 + cx*cx*m01 + cy*cx*m10;  // out[2, 1]
            out[offset + 12] = m30 - 3*cx*m20 + 2*cx*cx*m10;                    // out[3, 0]\n"""  # noqa: E501
        else:
            raise ValueError("only order <= 3 is supported")
    elif ndim == 3:
        if order <= 1:
            # only zeroth moment is non-zero for central moments
            source += """
            out[offset] = moments_raw[offset];\n"""
        elif order == 2:
            source += f"""
             // retrieve the 2nd order raw moments
            {float_type} m000 = moments_raw[offset];
            {float_type} m001 = moments_raw[offset + 1];
            {float_type} m002 = moments_raw[offset + 2];
            {float_type} m010 = moments_raw[offset + 3];
            {float_type} m011 = moments_raw[offset + 4];
            {float_type} m020 = moments_raw[offset + 6];
            {float_type} m100 = moments_raw[offset + 9];
            {float_type} m101 = moments_raw[offset + 10];
            {float_type} m110 = moments_raw[offset + 12];
            {float_type} m200 = moments_raw[offset + 18];

            // compute centroids
            {float_type} cx = m100 / m000;
            {float_type} cy = m010 / m000;
            {float_type} cz = m001 / m000;

            // zeroth moment
            out[offset] = m000;                  // out[0, 0, 0]
            // 2nd order central moments
            out[offset + 2] = -cz*m001 + m002;   // out[0, 0, 2]
            out[offset + 4] = -cy*m001 + m011;   // out[0, 1, 1]
            out[offset + 6] = -cy*m010 + m020;   // out[0, 2, 0]
            out[offset + 10] = -cx*m001 + m101;  // out[1, 0, 1]
            out[offset + 12] = -cx*m010 + m110;  // out[1, 1, 0]
            out[offset + 18] = -cx*m100 + m200;  // out[2, 0, 0]\n"""
        elif order == 3:
            source += f"""
             // retrieve the 3rd order raw moments
            {float_type} m000 = moments_raw[offset];
            {float_type} m001 = moments_raw[offset + 1];
            {float_type} m002 = moments_raw[offset + 2];
            {float_type} m003 = moments_raw[offset + 3];
            {float_type} m010 = moments_raw[offset + 4];
            {float_type} m011 = moments_raw[offset + 5];
            {float_type} m012 = moments_raw[offset + 6];
            {float_type} m020 = moments_raw[offset + 8];
            {float_type} m021 = moments_raw[offset + 9];
            {float_type} m030 = moments_raw[offset + 12];
            {float_type} m100 = moments_raw[offset + 16];
            {float_type} m101 = moments_raw[offset + 17];
            {float_type} m102 = moments_raw[offset + 18];
            {float_type} m110 = moments_raw[offset + 20];
            {float_type} m111 = moments_raw[offset + 21];
            {float_type} m120 = moments_raw[offset + 24];
            {float_type} m200 = moments_raw[offset + 32];
            {float_type} m201 = moments_raw[offset + 33];
            {float_type} m210 = moments_raw[offset + 36];
            {float_type} m300 = moments_raw[offset + 48];

            // compute centroids
            {float_type} cx = m100 / m000;
            {float_type} cy = m010 / m000;
            {float_type} cz = m001 / m000;

            // zeroth moment
            out[offset] = m000;
            // 2nd order central moments
            out[offset + 2] = -cz*m001 + m002;     // out[0, 0, 2]
            out[offset + 5] = -cy*m001 + m011;     // out[0, 1, 1]
            out[offset + 8] = -cy*m010 + m020;     // out[0, 2, 0]
            out[offset + 17] = -cx*m001 + m101;    // out[1, 0, 1]
            out[offset + 20] = -cx*m010 + m110;    // out[1, 1, 0]
            out[offset + 32] = -cx*m100 + m200;    // out[2, 0, 0]
            // 3rd order central moments
            out[offset + 3] = 2*cz*cz*m001 - 3*cz*m002 + m003;                               // out[0, 0, 3]
            out[offset + 6] = -cy*m002 + 2*cz*(cy*m001 - m011) + m012;                       // out[0, 1, 2]
            out[offset + 9] = cy*cy*m001 - 2*cy*m011 + cz*(cy*m010 - m020) + m021;           // out[0, 2, 1]
            out[offset + 12] = 2*cy*cy*m010 - 3*cy*m020 + m030;                              // out[0, 3, 0]
            out[offset + 18] = -cx*m002 + 2*cz*(cx*m001 - m101) + m102;                      // out[1, 0, 2]
            out[offset + 21] = -cx*m011 + cy*(cx*m001 - m101) + cz*(cx*m010 - m110) + m111;  // out[1, 1, 1]
            out[offset + 24] = -cx*m020 - 2*cy*(-cx*m010 + m110) + m120;                     // out[1, 2, 0]
            out[offset + 33] = cx*cx*m001 - 2*cx*m101 + cz*(cx*m100 - m200) + m201;          // out[2, 0, 1]
            out[offset + 36] = cx*cx*m010 - 2*cx*m110 + cy*(cx*m100 - m200) + m210;          // out[2, 1, 0]
            out[offset + 48] = 2*cx*cx*m100 - 3*cx*m200 + m300;                              // out[3, 0, 0]\n"""  # noqa: E501
        else:
            raise ValueError("only order <= 3 is supported")
    else:
        # note: ndim here is the number of spatial image dimensions
        raise ValueError("only ndim = 2 or 3 is supported")
    inputs = "raw X moments_raw"
    outputs = "raw X out"
    name = f"cucim_moments_central_order{order}_{ndim}d"
    return cp.ElementwiseKernel(
        inputs, outputs, source, preamble=_includes, name=name
    )


def regionprops_moments_central(
    moments_raw, ndim, weighted=False, props_dict=None
):
    """Compute the central moments of the labeled regions.

    Computes the central moments from raw moments.

    Writes "moments_central" to `props_dict` if `weighted` is ``False``.

    Writes "moments_weighted_central" to `props_dict` if `weighted` is ``True``.
    """
    if props_dict is None:
        props_dict = {}

    if moments_raw.ndim == 2 + ndim:
        num_channels = moments_raw.shape[1]
    elif moments_raw.ndim == 1 + ndim:
        num_channels = 1
    else:
        raise ValueError(
            f"{moments_raw.shape=} does not have expected length of `ndim + 1`"
            " (or `ndim + 2` for the multi-channel weighted moments case)."
        )
    order = moments_raw.shape[-1] - 1
    max_label = moments_raw.shape[0]

    if moments_raw.dtype.kind != "f":
        float_dtype = cp.promote_types(cp.float32, moments_raw.dtype)
        moments_raw = moments_raw.astype(float_dtype)

    # make a copy if the inputs are not already C-contiguous
    if not moments_raw.flags.c_contiguous:
        moments_raw = cp.ascontiguousarray(moments_raw)

    moments_kernel = get_moments_central_kernel(moments_raw.dtype, ndim, order)
    moments_central = cp.zeros_like(moments_raw)
    # kernel loops over moments so size is max_label * num_channels
    moments_kernel(moments_raw, moments_central, size=max_label * num_channels)
    if props_dict is not None:
        if weighted:
            props_dict["moments_weighted_central"] = moments_central
        else:
            props_dict["moments_central"] = moments_central
    return moments_central


@cp.memoize(for_each_device=True)
def get_moments_normalize_kernel(
    moments_dtype, ndim, order, unit_scale=False, pixel_correction=False
):
    """Normalizes central moments of order >=2"""
    moments_dtype = cp.dtype(moments_dtype)

    uint_t = "unsigned int"

    # number is for a densely populated moments matrix of size (order + 1) per
    # side (values at locations where order is greater than specified will be 0)
    num_moments = (order + 1) ** ndim

    if moments_dtype.kind != "f":
        raise ValueError(
            "`moments_dtype` must be a floating point type for central moments "
            "calculations."
        )

    # floating point type used for the intermediate computations
    float_type = "double"
    source = f"""
            {uint_t} offset = i * {num_moments};\n"""
    if ndim == 2:
        if order == 2:
            source += f"""
            // retrieve zeroth moment
            {float_type} m00 = moments_central[offset];\n"""

            # compute normalization factor
            source += f"""
            {float_type} norm_order2 = pow(m00, 2.0 / {ndim} + 1.0);"""
            if not unit_scale:
                source += """
                norm_order2 *= scale * scale;\n"""

            # normalize
            source += """
            // normalize the 2nd order central moments
            out[offset + 2] = moments_central[offset + 2] / norm_order2;  // out[0, 2]
            out[offset + 4] = moments_central[offset + 4] / norm_order2;  // out[1, 1]
            out[offset + 6] = moments_central[offset + 6] / norm_order2;  // out[2, 0]\n"""  # noqa: E501
        elif order == 3:
            source += f"""
            // retrieve zeroth moment
            {float_type} m00 = moments_central[offset];\n"""

            # compute normalization factor
            source += f"""
            {float_type} norm_order2 = pow(m00, 2.0 / {ndim} + 1.0);
            {float_type} norm_order3 = pow(m00, 3.0 / {ndim} + 1.0);"""
            if not unit_scale:
                source += """
                norm_order2 *= scale * scale;
                norm_order3 *= scale * scale * scale;\n"""

            # normalize
            source += """
            // normalize the 2nd order central moments
            out[offset + 2] = moments_central[offset + 2] / norm_order2;  // out[0, 2]
            out[offset + 5] = moments_central[offset + 5] / norm_order2;  // out[1, 1]
            out[offset + 8] = moments_central[offset + 8] / norm_order2;  // out[2, 0]
            // normalize the 3rd order central moments
            out[offset + 3] = moments_central[offset + 3] / norm_order3;    // out[0, 3]
            out[offset + 6] = moments_central[offset + 6] / norm_order3;    // out[1, 2]
            out[offset + 9] = moments_central[offset + 9] / norm_order3;    // out[2, 1]
            out[offset + 12] = moments_central[offset + 12] / norm_order3;  // out[3, 0]\n"""  # noqa: E501
        else:
            raise ValueError("only order = 2 or 3 is supported")
    elif ndim == 3:
        if order == 2:
            source += f"""
            // retrieve the zeroth moment
            {float_type} m000 = moments_central[offset];\n"""

            # compute normalization factor
            source += f"""
            {float_type} norm_order2 = pow(m000, 2.0 / {ndim} + 1.0);"""
            if not unit_scale:
                source += """
                norm_order2 *= scale * scale;\n"""

            # normalize
            source += """
            // normalize the 2nd order central moments
            out[offset + 2] = moments_central[offset + 2] / norm_order2;    // out[0, 0, 2]
            out[offset + 4] = moments_central[offset + 4] / norm_order2;    // out[0, 1, 1]
            out[offset + 6] = moments_central[offset + 6] / norm_order2;    // out[0, 2, 0]
            out[offset + 10] = moments_central[offset + 10] / norm_order2;  // out[1, 0, 1]
            out[offset + 12] = moments_central[offset + 12] / norm_order2;  // out[1, 1, 0]
            out[offset + 18] = moments_central[offset + 18] / norm_order2;  // out[2, 0, 0]\n"""  # noqa: E501
        elif order == 3:
            source += f"""
            // retrieve the zeroth moment
            {float_type} m000 = moments_central[offset];\n"""

            # compute normalization factor
            source += f"""
            {float_type} norm_order2 = pow(m000, 2.0 / {ndim} + 1.0);
            {float_type} norm_order3 = pow(m000, 3.0 / {ndim} + 1.0);"""
            if not unit_scale:
                source += """
                norm_order2 *= scale * scale;
                norm_order3 *= scale * scale * scale;\n"""

            # normalize
            source += """
            // normalize the 2nd order central moments
            out[offset + 2] = moments_central[offset + 2] / norm_order2;    // out[0, 0, 2]
            out[offset + 5] = moments_central[offset + 5] / norm_order2;    // out[0, 1, 1]
            out[offset + 8] = moments_central[offset + 8] / norm_order2;    // out[0, 2, 0]
            out[offset + 17] = moments_central[offset + 17] / norm_order2;  // out[1, 0, 1]
            out[offset + 20] = moments_central[offset + 20] / norm_order2;  // out[1, 1, 0]
            out[offset + 32] = moments_central[offset + 32] / norm_order2;  // out[2, 0, 0]
            // normalize the 3rd order central moments
            out[offset + 3] = moments_central[offset + 3] / norm_order3;    // out[0, 0, 3]
            out[offset + 6] = moments_central[offset + 6] / norm_order3;    // out[0, 1, 2]
            out[offset + 9] = moments_central[offset + 9] / norm_order3;    // out[0, 2, 1]
            out[offset + 12] = moments_central[offset + 12] / norm_order3;  // out[0, 3, 0]
            out[offset + 18] = moments_central[offset + 18] / norm_order3;  // out[1, 0, 2]
            out[offset + 21] = moments_central[offset + 21] / norm_order3;  // out[1, 1, 1]
            out[offset + 24] = moments_central[offset + 24] / norm_order3;  // out[1, 2, 0]
            out[offset + 33] = moments_central[offset + 33] / norm_order3;  // out[2, 0, 1]
            out[offset + 36] = moments_central[offset + 36] / norm_order3;  // out[2, 1, 0]
            out[offset + 48] = moments_central[offset + 48] / norm_order3;  // out[3, 0, 0]\n"""  # noqa: E501
        else:
            raise ValueError("only order = 2 or 3 is supported")
    else:
        # note: ndim here is the number of spatial image dimensions
        raise ValueError("only ndim = 2 or 3 is supported")
    inputs = "raw X moments_central"
    if not unit_scale:
        inputs += ", float64 scale"
    outputs = "raw X out"
    name = f"cucim_moments_normalized_order{order}_{ndim}d"
    return cp.ElementwiseKernel(
        inputs, outputs, source, preamble=_includes, name=name
    )


def regionprops_moments_normalized(
    moments_central,
    ndim,
    spacing=None,
    pixel_correction=False,
    weighted=False,
    props_dict=None,
):
    """Compute the normalized central moments of the labeled regions.

    Computes normalized central moments from central moments.

    Writes "moments_normalized" to `props_dict` if `weighted` is ``False``.

    Writes "moments_weighted_normalized" to `props_dict` if `weighted` is
    ``True``.

    Notes
    -----
    Default setting of `pixel_correction=False` matches the scikit-image
    behavior (as of v0.25).

    The `pixel_correction` is to account for pixel/voxel size and is only
    implemented for 2nd order moments currently based on the derivation in:

    The correction should need to be updated to take 'spacing' into account as
    it currently assumes unit size.

    Padfield D., Miller J. "A Label Geometry Image Filter for Multiple Object
    Measurement". The Insight Journal. 2013 Mar.
    https://doi.org/10.54294/saa3nn
    """
    if moments_central.ndim == 2 + ndim:
        num_channels = moments_central.shape[1]
    elif moments_central.ndim == 1 + ndim:
        num_channels = 1
    else:
        raise ValueError(
            f"{moments_central.shape=} does not have expected length of "
            " `ndim + 1` (or `ndim + 2` for the multi-channel weighted moments "
            "case)."
        )
    order = moments_central.shape[-1] - 1
    if order < 2 or order > 3:
        raise ValueError(
            "normalized moment calculations only implemented for order=2 "
            "and order=3"
        )
    if ndim < 2 or ndim > 3:
        raise ValueError(
            "moment normalization only implemented for 2D and 3D images"
        )
    max_label = moments_central.shape[0]

    if moments_central.dtype.kind != "f":
        raise ValueError("moments_central must have a floating point dtype")

    # make a copy if the inputs are not already C-contiguous
    if not moments_central.flags.c_contiguous:
        moments_central = cp.ascontiguousarray(moments_central)

    if spacing is None:
        unit_scale = True
        inputs = (moments_central,)
    else:
        if spacing:
            if isinstance(spacing, cp.ndarray):
                scale = spacing.min()
            else:
                scale = float(min(spacing))
        unit_scale = False
        inputs = (moments_central, scale)

    moments_norm_kernel = get_moments_normalize_kernel(
        moments_central.dtype,
        ndim,
        order,
        unit_scale=unit_scale,
        pixel_correction=pixel_correction,
    )
    # output is NaN except for locations with orders in range [2, order]
    moments_norm = cp.full(
        moments_central.shape, cp.nan, dtype=moments_central.dtype
    )

    # kernel loops over moments so size is max_label * num_channels
    moments_norm_kernel(*inputs, moments_norm, size=max_label * num_channels)
    if props_dict is not None:
        if weighted:
            props_dict["moments_weighted_normalized"] = moments_norm
        else:
            props_dict["moments_normalized"] = moments_norm
    return moments_norm


@cp.memoize(for_each_device=True)
def get_moments_hu_kernel(moments_dtype):
    """Normalizes central moments of order >=2"""
    moments_dtype = cp.dtype(moments_dtype)

    uint_t = "unsigned int"

    # number is for a densely populated moments matrix of size (order + 1) per
    # side (values at locations where order is greater than specified will be 0)
    num_moments = 16

    if moments_dtype.kind != "f":
        raise ValueError(
            "`moments_dtype` must be a floating point type for central moments "
            "calculations."
        )

    # floating point type used for the intermediate computations
    float_type = "double"

    # compute offset to the current moment matrix and hu moment vector
    source = f"""
            {uint_t} offset_normalized = i * {num_moments};
            {uint_t} offset_hu = i * 7;\n"""

    source += f"""
    // retrieve 2nd and 3rd order normalized moments
    {float_type} m02 = moments_normalized[offset_normalized + 2];
    {float_type} m03 = moments_normalized[offset_normalized + 3];
    {float_type} m12 = moments_normalized[offset_normalized + 6];
    {float_type} m11 = moments_normalized[offset_normalized + 5];
    {float_type} m20 = moments_normalized[offset_normalized + 8];
    {float_type} m21 = moments_normalized[offset_normalized + 9];
    {float_type} m30 = moments_normalized[offset_normalized + 12];

    {float_type} t0 = m30 + m12;
    {float_type} t1 = m21 + m03;
    {float_type} q0 = t0 * t0;
    {float_type} q1 = t1 * t1;
    {float_type} n4 = 4 * m11;
    {float_type} s = m20 + m02;
    {float_type} d = m20 - m02;
    hu[offset_hu] = s;
    hu[offset_hu + 1] = d * d + n4 * m11;
    hu[offset_hu + 3] = q0 + q1;
    hu[offset_hu + 5] = d * (q0 - q1) + n4 * t0 * t1;
    t0 *= q0 - 3 * q1;
    t1 *= 3 * q0 - q1;
    q0 = m30- 3 * m12;
    q1 = 3 * m21 - m03;
    hu[offset_hu + 2] = q0 * q0 + q1 * q1;
    hu[offset_hu + 4] = q0 * t0 + q1 * t1;
    hu[offset_hu + 6] = q1 * t0 - q0 * t1;\n"""

    inputs = f"raw {moments_dtype.name} moments_normalized"
    outputs = f"raw {moments_dtype.name} hu"
    name = f"cucim_moments_hu_order_{moments_dtype.name}"
    return cp.ElementwiseKernel(
        inputs, outputs, source, preamble=_includes, name=name
    )


def regionprops_moments_hu(moments_normalized, weighted=False, props_dict=None):
    """Compute the 2D Hu invariant moments from 3rd order normalized central
    moments.

    Writes "moments_hu" to `props_dict` if `weighted` is ``False``

    Writes "moments_weighted_hu" to `props_dict` if `weighted` is ``True``.
    """
    if props_dict is None:
        props_dict = {}

    if moments_normalized.ndim == 4:
        num_channels = moments_normalized.shape[1]
    elif moments_normalized.ndim == 3:
        num_channels = 1
    else:
        raise ValueError(
            "Hu's moments are only defined for 2D images. Expected "
            "`moments_normalized to have 3 dimensions (or 4 for the "
            "multi-channel `intensity_image` case)."
        )
    order = moments_normalized.shape[-1] - 1
    if order < 3:
        raise ValueError(
            "Calculating Hu's moments requires normalized moments of "
            "order >= 3 to be provided as input"
        )
    elif order > 3:
        # truncate any unused moments
        moments_normalized = cp.ascontiguousarray(
            moments_normalized[..., :4, :4]
        )
    max_label = moments_normalized.shape[0]

    if moments_normalized.dtype.kind != "f":
        raise ValueError("moments_normalized must have a floating point dtype")

    # make a copy if the inputs are not already C-contiguous
    if not moments_normalized.flags.c_contiguous:
        moments_normalized = cp.ascontiguousarray(moments_normalized)

    moments_hu_kernel = get_moments_hu_kernel(moments_normalized.dtype)
    # Hu's moments are a set of 7 moments stored instead of a moment matrix
    hu_shape = moments_normalized.shape[:-2] + (7,)
    moments_hu = cp.full(hu_shape, cp.nan, dtype=moments_normalized.dtype)

    # kernel loops over moments so size is max_label * num_channels
    moments_hu_kernel(
        moments_normalized, moments_hu, size=max_label * num_channels
    )
    if props_dict is not None:
        if weighted:
            props_dict["moments_weighted_hu"] = moments_hu
        else:
            props_dict["moments_hu"] = moments_hu
    return moments_hu


@cp.memoize(for_each_device=True)
def get_inertia_tensor_kernel(moments_dtype, ndim, compute_orientation):
    """Normalizes central moments of order >=2"""
    moments_dtype = cp.dtype(moments_dtype)

    # assume moments input was truncated to only hold order<=2 moments
    num_moments = 3**ndim

    # size of the inertia_tensor matrix
    num_out = ndim * ndim

    if moments_dtype.kind != "f":
        raise ValueError(
            "`moments_dtype` must be a floating point type for central moments "
            "calculations."
        )

    source = f"""
            unsigned int offset = i * {num_moments};
            unsigned int offset_out = i * {num_out};\n"""
    if ndim == 2:
        source += """
        F mu0 = moments_central[offset];
        F mxx = moments_central[offset + 6];
        F myy = moments_central[offset + 2];
        F mxy = moments_central[offset + 4];

        F a = myy / mu0;
        F b = -mxy / mu0;
        F c = mxx / mu0;
        out[offset_out + 0] = a;
        out[offset_out + 1] = b;
        out[offset_out + 2] = b;
        out[offset_out + 3] = c;
        """
        if compute_orientation:
            source += """
        if (a - c == 0) {
          // had to use <= 0 to get same result as Python's atan2 with < 0
          if (b < 0) {
            orientation[i] = -M_PI / 4.0;
          } else {
            orientation[i] = M_PI / 4.0;
          }
        } else {
          orientation[i] = 0.5 * atan2(-2 * b, c - a);
        }\n"""
    elif ndim == 3:
        if compute_orientation:
            raise ValueError("orientation can only be computed for 2d images")
        source += """
        F mu0 = moments_central[offset];       // [0, 0, 0]
        F mxx = moments_central[offset + 18];  // [2, 0, 0]
        F myy = moments_central[offset + 6];   // [0, 2, 0]
        F mzz = moments_central[offset + 2];   // [0, 0, 2]

        F mxy = moments_central[offset + 12];  // [1, 1, 0]
        F mxz = moments_central[offset + 10];  // [1, 0, 1]
        F myz = moments_central[offset + 4];   // [0, 1, 1]

        out[offset_out + 0] = (myy + mzz) / mu0;
        out[offset_out + 4] = (mxx + mzz) / mu0;
        out[offset_out + 8] = (mxx + myy) / mu0;
        out[offset_out + 1] = -mxy / mu0;
        out[offset_out + 3] = -mxy / mu0;
        out[offset_out + 2] = -mxz / mu0;
        out[offset_out + 6] = -mxz / mu0;
        out[offset_out + 5] = -myz / mu0;
        out[offset_out + 7] = -myz / mu0;\n"""
    else:
        # note: ndim here is the number of spatial image dimensions
        raise ValueError("only ndim = 2 or 3 is supported")
    inputs = "raw F moments_central"
    outputs = "raw F out"
    if compute_orientation:
        outputs += ", raw F orientation"
    name = f"cucim_inertia_tensor_{ndim}d"
    return cp.ElementwiseKernel(
        inputs, outputs, source, preamble=_includes, name=name
    )


def regionprops_inertia_tensor(
    moments_central, ndim, compute_orientation=False, props_dict=None
):
    """ "Compute the inertia tensor from the central moments.

    The input to this function is the output of `regionprops_moments_central`.

    Writes "inertia_tensor" to `props_dict`.
    Writes "orientation" to `props_dict` if `compute_orientation` is ``True``.
    """
    if ndim < 2 or ndim > 3:
        raise ValueError("inertia tensor only implemented for 2D and 3D images")
    if compute_orientation and ndim != 2:
        raise ValueError("orientation can only be computed for ndim=2")

    nbatch = math.prod(moments_central.shape[:-ndim])

    if moments_central.dtype.kind != "f":
        raise ValueError("moments_central must have a floating point dtype")

    # make a copy if the inputs are not already C-contiguous
    if not moments_central.flags.c_contiguous:
        moments_central = cp.ascontiguousarray(moments_central)

    order = moments_central.shape[-1] - 1
    if order < 2:
        raise ValueError(
            f"inertia tensor calculation requires order>=2, found {order}"
        )
    if order > 2:
        # truncate to only the 2nd order moments
        slice_kept = (Ellipsis,) + (slice(0, 3),) * ndim
        moments_central = cp.ascontiguousarray(moments_central[slice_kept])

    kernel = get_inertia_tensor_kernel(
        moments_central.dtype, ndim, compute_orientation=compute_orientation
    )
    itensor_shape = moments_central.shape[:-ndim] + (ndim, ndim)
    itensor = cp.zeros(itensor_shape, dtype=moments_central.dtype)
    if compute_orientation:
        orientation = cp.zeros(
            moments_central.shape[:-ndim], dtype=moments_central.dtype
        )
        kernel(moments_central, itensor, orientation, size=nbatch)
        if props_dict is not None:
            props_dict["inertia_tensor"] = itensor
            props_dict["orientation"] = orientation
        return itensor, orientation

    kernel(moments_central, itensor, size=nbatch)
    if props_dict is not None:
        props_dict["inertia_tensor"] = itensor
    return itensor


@cp.memoize(for_each_device=True)
def get_spd_matrix_eigvals_kernel(
    rank,
    compute_axis_lengths=False,
    compute_eccentricity=False,
):
    """Compute symmetric positive definite (SPD) matrix eigenvalues

    Implements closed-form analytical solutions for 2x2 and 3x3 matrices.

    C. Deledalle, L. Denis, S. Tabti, F. Tupin. Closed-form expressions
    of the eigen decomposition of 2 x 2 and 3 x 3 Hermitian matrices.

    [Research Report] Université de Lyon. 2017.
    https://hal.archives-ouvertes.fr/hal-01501221/file/matrix_exp_and_log_formula.pdf
    """  # noqa: E501

    # assume moments input was truncated to only hold order<=2 moments
    num_elements = rank * rank

    # size of the inertia_tensor matrix
    source = f"""
            unsigned int offset = i * {num_elements};
            unsigned int offset_evals = i * {rank};\n"""
    if rank == 2:
        source += """
            F tmp1, tmp2;
            double m00 = static_cast<double>(spd_matrix[offset]);
            double m01 = static_cast<double>(spd_matrix[offset + 1]);
            double m11 = static_cast<double>(spd_matrix[offset + 3]);
            tmp1 = m01 * m01;
            tmp1 *= 4;

            tmp2 = m00 - m11;
            tmp2 *= tmp2;
            tmp2 += tmp1;
            tmp2 = sqrt(tmp2);
            tmp2 /= 2;

            tmp1 = m00 + m11;
            tmp1 /= 2;

            // store in "descending" order and clip to positive values
            // (matrix is spd, so negatives values can only be due to
            //  numerical errors)
            F lam1 = max(tmp1 + tmp2, 0.0);
            F lam2 = max(tmp1 - tmp2, 0.0);
            evals[offset_evals] = lam1;
            evals[offset_evals + 1] = lam2;\n"""
        if compute_axis_lengths:
            source += """
            axis_lengths[offset_evals] = 4.0 * sqrt(lam1);
            axis_lengths[offset_evals + 1] = 4.0 * sqrt(lam2);\n"""
        if compute_eccentricity:
            source += """
            eccentricity[i] =  sqrt(1.0 - lam2 / lam1);\n"""
    elif rank == 3:
        if compute_eccentricity:
            raise ValueError("eccentricity only supported for 2D images")

        source += """
            double x1, x2, phi;
            // extract triangle of (spd) inertia tensor values
            // [a, d, f]
            // [-, b, e]
            // [-, -, c]
            double a = static_cast<double>(spd_matrix[offset]);
            double b = static_cast<double>(spd_matrix[offset + 4]);
            double c = static_cast<double>(spd_matrix[offset + 8]);
            double d = static_cast<double>(spd_matrix[offset + 1]);
            double e = static_cast<double>(spd_matrix[offset + 5]);
            double f = static_cast<double>(spd_matrix[offset + 2]);
            double d_sq = d * d;
            double e_sq = e * e;
            double f_sq = f * f;
            double tmpa = (2*a - b - c);
            double tmpb = (2*b - a - c);
            double tmpc = (2*c - a - b);
            x2 = - tmpa * tmpb * tmpc;
            x2 += 9 * (tmpc*d_sq + tmpb*f_sq + tmpa*e_sq);
            x2 -= 54 * (d * e * f);
            x1 = a*a + b*b + c*c - a*b - a*c - b*c + 3 * (d_sq + e_sq + f_sq);

            // grlee77: added max() here for numerical stability
            // (avoid NaN values in ridge filter test cases)
            x1 = max(x1, 0.0);

            if (x2 == 0.0) {
                phi = M_PI / 2.0;
            } else {
                // grlee77: added max() here for numerical stability
                // (avoid NaN values in test_hessian_matrix_eigvals_3d)
                double arg = max(4*x1*x1*x1 - x2*x2, 0.0);
                phi = atan(sqrt(arg)/x2);
                if (x2 < 0) {
                    phi += M_PI;
                }
            }
            double x1_term = (2.0 / 3.0) * sqrt(x1);
            double abc = (a + b + c) / 3.0;
            F lam1 = abc - x1_term * cos(phi / 3.0);
            F lam2 = abc + x1_term * cos((phi - M_PI) / 3.0);
            F lam3 = abc + x1_term * cos((phi + M_PI) / 3.0);

            // abc = 141.94321771
            // x1_term = 1279.25821493
            // M_PI = 3.14159265
            // phi = 1.91643394
            // cos(phi/3.0) = 0.80280507
            // cos((phi - M_PI) / 3.0) = 0.91776289

            F stmp;
            if (lam3 > lam2) {
                stmp = lam2;
                lam2 = lam3;
                lam3 = stmp;
            }
            if (lam3 > lam1) {
                stmp = lam1;
                lam1 = lam3;
                lam3 = stmp;
            }
            if (lam2 > lam1) {
                stmp = lam1;
                lam1 = lam2;
                lam2 = stmp;
            }
            // clip to positive values
            // (matrix is spd, so negatives values can only be due to
            //  numerical errors)
            lam1 = max(lam1, 0.0);
            lam2 = max(lam2, 0.0);
            lam3 = max(lam3, 0.0);
            evals[offset_evals] = lam1;
            evals[offset_evals + 1] = lam2;
            evals[offset_evals + 2] = lam3;\n"""
        if compute_axis_lengths:
            """
            Notes
            -----
            Let a >= b >= c be the ellipsoid semi-axes and s1 >= s2 >= s3 be the
            inertia tensor eigenvalues.

            The inertia tensor eigenvalues are given for a solid ellipsoid in [1]_.
            s1 = 1 / 5 * (a**2 + b**2)
            s2 = 1 / 5 * (a**2 + c**2)
            s3 = 1 / 5 * (b**2 + c**2)

            Rearranging to solve for a, b, c in terms of s1, s2, s3 gives
            a = math.sqrt(5 / 2 * ( s1 + s2 - s3))
            b = math.sqrt(5 / 2 * ( s1 - s2 + s3))
            c = math.sqrt(5 / 2 * (-s1 + s2 + s3))

            We can then simply replace sqrt(5/2) by sqrt(10) to get the full axes
            lengths rather than the semi-axes lengths.

            References
            ----------
            ..[1] https://en.wikipedia.org/wiki/List_of_moments_of_inertia#List_of_3D_inertia_tensors
            """  # noqa: E501
            source += """
            // formula reference:
            //   https://github.com/scikit-image/scikit-image/blob/v0.25.0/skimage/measure/_regionprops.py#L275-L295
            // note: added max to clip possible small (e.g. 1e-7) negative value due to numerical error
            axis_lengths[offset_evals] = sqrt(10.0 * (lam1 + lam2 - lam3));
            axis_lengths[offset_evals + 1] = sqrt(10.0 * (lam1 - lam2 + lam3));
            axis_lengths[offset_evals + 2] = sqrt(10.0 * max(-lam1 + lam2 + lam3, 0.0));\n"""  # noqa: E501
    else:
        # note: ndim here is the number of spatial image dimensions
        raise ValueError("only rank = 2 or 3 is supported")
    inputs = "raw F spd_matrix"
    outputs = ["raw F evals"]
    name = f"cucim_spd_matrix_eigvals_{rank}d"
    if compute_axis_lengths:
        outputs.append("raw F axis_lengths")
        name += "_with_axis"
    if compute_eccentricity:
        outputs.append("raw F eccentricity")
        name += "_eccen"
    outputs = ", ".join(outputs)
    return cp.ElementwiseKernel(
        inputs, outputs, source, preamble=_includes, name=name
    )


def regionprops_inertia_tensor_eigvals(
    inertia_tensor,
    compute_eigenvectors=False,
    compute_axis_lengths=False,
    compute_eccentricity=False,
    props_dict=None,
):
    """Compute the inertia tensor eigenvalues (and eigenvectors) from the
    inertia tensor of each labeled region.

    The input to this function is the output of `regionprops_inertia_tensor`.

    writes "inertia_tensor_eigvals" to `props_dict`
    if compute_eigenvectors:
        - writes "inertia_tensor_eigenvectors" to `props_dict`
    if compute_axis_lengths:
        - writes "axis_major_length" to `props_dict`
        - writes "axis_minor_length" to `props_dict`
        - writes "axis_lengths" to `props_dict`
    if compute_eccentricity:
        - writes "eccentricity" to `props_dict`
    """
    # inertia tensor should have shape (ndim, ndim) on last two axes
    ndim = inertia_tensor.shape[-1]
    if ndim < 2 or ndim > 3:
        raise ValueError("inertia tensor only implemented for 2D and 3D images")
    nbatch = math.prod(inertia_tensor.shape[:-2])

    if compute_eccentricity and ndim != 2:
        raise ValueError("eccentricity is only supported for 2D images")

    if inertia_tensor.dtype.kind != "f":
        raise ValueError("moments_central must have a floating point dtype")

    if not inertia_tensor.flags.c_contiguous:
        inertia_tensor = cp.ascontiguousarray(inertia_tensor)

    # don't use this kernel for eigenvectors as it is not robust to 0 entries
    kernel = get_spd_matrix_eigvals_kernel(
        rank=ndim,
        compute_axis_lengths=compute_axis_lengths,
        compute_eccentricity=compute_eccentricity,
    )
    eigvals_shape = inertia_tensor.shape[:-2] + (ndim,)
    eigvals = cp.empty(eigvals_shape, dtype=inertia_tensor.dtype)
    outputs = [eigvals]
    if compute_axis_lengths:
        axis_lengths = cp.empty(eigvals_shape, dtype=inertia_tensor.dtype)
        outputs.append(axis_lengths)
    if compute_eccentricity:
        eccentricity = cp.empty(
            inertia_tensor.shape[:-2], dtype=inertia_tensor.dtype
        )
        outputs.append(eccentricity)
    kernel(inertia_tensor, *outputs, size=nbatch)
    if compute_eigenvectors:
        # eigenvectors computed by the kernel are not robust to 0 entries, so
        # use slightly slow cp.linalg.eigh instead
        eigvals, eigvecs = cp.linalg.eigh(inertia_tensor)
        # swap from ascending to descending order
        eigvals = eigvals[:, ::-1]
        eigvecs = eigvecs[:, ::-1]
    if props_dict is None:
        props_dict = {}
    props_dict["inertia_tensor_eigvals"] = eigvals
    if compute_eccentricity:
        props_dict["eccentricity"] = eccentricity
    if compute_axis_lengths:
        props_dict["axis_lengths"] = axis_lengths
        props_dict["axis_major_length"] = axis_lengths[..., 0]
        props_dict["axis_minor_length"] = axis_lengths[..., -1]
    if compute_eigenvectors:
        props_dict["inertia_tensor_eigenvectors"] = eigvecs
    return props_dict


@cp.memoize(for_each_device=True)
def get_centroid_weighted_kernel(
    moments_dtype,
    ndim,
    compute_local=True,
    compute_global=False,
    unit_spacing=True,
    num_channels=1,
):
    """Centroid (in global or local coordinates) from 1st order moment matrix"""
    moments_dtype = cp.dtype(moments_dtype)

    # assume moments input was truncated to only hold order<=2 moments
    num_moments = 2**ndim
    if moments_dtype.kind != "f":
        raise ValueError(
            "`moments_dtype` must be a floating point type for central moments "
            "calculations."
        )
    source = ""
    if compute_global:
        source += f"""
        unsigned int offset_coords = i * {2 * ndim};\n"""

    if num_channels > 1:
        source += f"""
        uint32_t num_channels = moments_raw.shape()[1];
        for (int c = 0; c < num_channels; c++) {{
            unsigned int offset = i * {num_moments} * num_channels + c * {num_moments};
            unsigned int offset_out = i * {ndim} * num_channels + c * {ndim};
            F m0 = moments_raw[offset];\n"""  # noqa: E501
    else:
        source += f"""
            unsigned int offset = i * {num_moments};
            unsigned int offset_out = i * {ndim};
            F m0 = moments_raw[offset];\n"""

    # general formula for the n-dimensional case
    #
    #   in 2D it gives:
    #     out[offset_out + 1] = moments_raw[offset + 1] / m0;  // m[0, 1]
    #     out[offset_out] = moments_raw[offset + 2] / m0;      // m[1, 0]
    #
    #   in 3D it gives:
    #     out[offset_out + 2] = moments_raw[offset + 1] / m0;  // m[0, 0, 1]
    #     out[offset_out + 1] = moments_raw[offset + 2] / m0;  // m[0, 1, 0]
    #     out[offset_out] = moments_raw[offset + 4] / m0;      // m[1, 0, 0]
    axis_offset = 1
    for d in range(ndim - 1, -1, -1):
        if compute_local:
            source += f"""
            out_local[offset_out + {d}] = moments_raw[offset + {axis_offset}] / m0;"""  # noqa: E501
        if compute_global:
            spc = "" if unit_spacing else f" * spacing[{d}]"
            source += f"""
            out_global[offset_out + {d}] = moments_raw[offset + {axis_offset}] / m0 + bbox[offset_coords + {d}]{spc};"""  # noqa: E501
        axis_offset *= 2
    if num_channels > 1:
        source += """
        }  // channels loop\n"""
    name = f"cucim_centroid_weighted_{ndim}d"
    inputs = ["raw F moments_raw"]
    outputs = []
    if compute_global:
        name += "_global"
        outputs.append("raw F out_global")
        # bounding box coordinates
        inputs.append("raw Y bbox")
        if not unit_spacing:
            inputs.append("raw float64 spacing")
            name += "_spacing"
    if compute_local:
        name += "_local"
        outputs.append("raw F out_local")
    inputs = ", ".join(inputs)
    outputs = ", ".join(outputs)
    return cp.ElementwiseKernel(
        inputs, outputs, source, preamble=_includes, name=name
    )


def regionprops_centroid_weighted(
    moments_raw,
    ndim,
    bbox=None,
    compute_local=True,
    compute_global=False,
    weighted=True,
    spacing=None,
    props_dict=None,
):
    """Centroid (in global or local coordinates) from 1st order moment matrix

    If `compute_local` the centroid is in local coordinates, otherwise it is in
    global coordinates.

    `bbox` property must be provided either via kwarg or within `props_dict` if
    `compute_global` is ``True``.

    if weighted:
        if compute_global:
            writes "centroid_weighted" to `props_dict`
        if compute_local:
            writes "centroid_weighted_local" to `props_dict`
    else:
        if compute_global:
            writes "centroid" to `props_dict`
        if compute_local:
            writes "centroid_local" to `props_dict`
    """
    max_label = moments_raw.shape[0]
    if moments_raw.ndim == ndim + 2:
        num_channels = moments_raw.shape[1]
    elif moments_raw.ndim == ndim + 1:
        num_channels = 1
    else:
        raise ValueError("moments_raw has unexpected shape")

    if compute_global and bbox is None:
        if "bbox" in props_dict:
            bbox = props_dict["bbox"]
        else:
            raise ValueError(
                "bbox coordinates must be provided to get the non-local "
                "centroid"
            )

    if not (compute_local or compute_global):
        raise ValueError(
            "nothing to compute: either compute_global and/or compute_local "
            "must be true"
        )
    if moments_raw.dtype.kind != "f":
        raise ValueError("moments_raw must have a floating point dtype")
    order = moments_raw.shape[-1] - 1
    if order < 1:
        raise ValueError(
            f"inertia tensor calculation requires order>=1, found {order}"
        )
    if order >= 1:
        # truncate to only the 1st order moments
        slice_kept = (Ellipsis,) + (slice(0, 2),) * ndim
        moments_raw = cp.ascontiguousarray(moments_raw[slice_kept])

    # make a copy if the inputs are not already C-contiguous
    if not moments_raw.flags.c_contiguous:
        moments_raw = cp.ascontiguousarray(moments_raw)

    unit_spacing = spacing is None

    if compute_local and not compute_global:
        inputs = (moments_raw,)
    else:
        if not bbox.flags.c_contiguous:
            bbox = cp.ascontiguousarray(bbox)
        inputs = (moments_raw, bbox)
        if not unit_spacing:
            inputs = inputs + (cp.asarray(spacing),)
    kernel = get_centroid_weighted_kernel(
        moments_raw.dtype,
        ndim,
        compute_local=compute_local,
        compute_global=compute_global,
        unit_spacing=unit_spacing,
        num_channels=num_channels,
    )
    centroid_shape = moments_raw.shape[:-ndim] + (ndim,)
    outputs = []
    if compute_global:
        centroid_global = cp.zeros(centroid_shape, dtype=moments_raw.dtype)
        outputs.append(centroid_global)
    if compute_local:
        centroid_local = cp.zeros(centroid_shape, dtype=moments_raw.dtype)
        outputs.append(centroid_local)
    # Note: order of inputs and outputs here must match
    #       get_centroid_weighted_kernel
    kernel(*inputs, *outputs, size=max_label)
    if props_dict is None:
        props_dict = {}
    if compute_local:
        if weighted:
            props_dict["centroid_weighted_local"] = centroid_local
        else:
            props_dict["centroid_local"] = centroid_local
    if compute_global:
        if weighted:
            props_dict["centroid_weighted"] = centroid_global
        else:
            props_dict["centroid"] = centroid_global

    return props_dict
