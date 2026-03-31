# SPDX-FileCopyrightText: Copyright (c) 2015 Preferred Infrastructure, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2015 Preferred Networks, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT
from __future__ import annotations

import math
from typing import NamedTuple

import cupy
import cupy._core.internal
import numpy

from cucim.skimage._vendored import (
    _ndimage_spline_kernel_weights as _spline_kernel_weights,
    _ndimage_spline_prefilter_core as _spline_prefilter_core,
    _ndimage_util as _util,
)


class KernelInfo(NamedTuple):
    """Kernel info returned by kernel getter functions."""

    kernel: cupy.ElementwiseKernel
    size: int | None  # None means use output array size


math_constants_preamble = r"""
// workaround for HIP: line begins with #include
#include <cupy/math_constants.h>
"""

spline_weights_inline = _spline_kernel_weights.spline_weights_inline

# Empirical threshold above which using loop_batch_axis=True begins to become
# disadvantageous.
loop_batch_max_channels = 12


def _get_coord_map(
    ndim, nprepad=0, float_type=None, batch_axes=None, loop_batch_axis=False
):
    """Extract target coordinate from coords array (for map_coordinates).

    Notes
    -----
    Assumes the following variables have been initialized on the device::

        coords (ndarray): array of shape (ncoords, ndim) containing the target
            coordinates.
        c_j: variables to hold the target coordinates

    computes::

        c_j = coords[i + j * ncoords];

    ncoords is determined by the size of the output array, y.
    y will be indexed by the CIndexer, _ind.
    Thus ncoords = _ind.size();

    For batch axes (identity mapping), the output index is used directly:

        c_j = in_coord[j]

    When loop_batch_axis=True, the kernel loops over the last batch axis, so:
    - The spatial coords are read once using out_base_idx (first batch element)
    - The coords are assumed to be the same for all batch elements at the
      same spatial position
    - The last axis (batch axis) is not generated here - it's handled in the
      batch loop

    """
    if batch_axes is None:
        batch_axes = ()
    ops = []

    # When loop_batch_axis is True, skip the last axis (it's the batch axis)
    axes_to_process = range(ndim - 1) if loop_batch_axis else range(ndim)

    # only need ncoords if there are non-batch axes to process
    non_batch_axes_to_process = [
        j for j in axes_to_process if j not in batch_axes
    ]
    if non_batch_axes_to_process:
        if loop_batch_axis:
            # When looping, ncoords =  spatial_size * batch_size
            ops.append("ptrdiff_t ncoords = _ind.size() * batch_size;")
        else:
            ops.append("ptrdiff_t ncoords = _ind.size();")
    pre = f" + (W){nprepad}" if nprepad > 0 else ""
    for j in axes_to_process:
        if j in batch_axes:
            # batch axis: use identity (output index = input coordinate)
            ops.append(
                f"""
    W c_{j} = (W)in_coord[{j}]{pre};"""
            )
        else:
            if loop_batch_axis:
                # Read coords at out_base_idx (first batch element at this spatial pos)
                # The spatial coords are the same for all batch elements
                ops.append(
                    f"""
    W c_{j} = coords[out_base_idx + {j} * ncoords]{pre};"""
                )
            else:
                ops.append(
                    f"""
    W c_{j} = coords[i + {j} * ncoords]{pre};"""
                )
    return ops


def _get_coord_zoom_and_shift(
    ndim, nprepad=0, float_type=None, batch_axes=None, loop_batch_axis=False
):
    """Compute target coordinate based on a shift followed by a zoom.

    This version zooms from the center of the edge pixels.

    Notes
    -----
    Assumes the following variables have been initialized on the device::

        in_coord[ndim]: array containing the source coordinate
        zoom[ndim]: array containing the zoom for each axis
        shift[ndim]: array containing the shift for each axis

    computes::

        c_j = zoom[j] * (in_coord[j] - shift[j])

    For batch axes (zoom == 1 and shift == 0), the identity is used:

        c_j = in_coord[j]

    When loop_batch_axis=True, the last axis is handled separately in the
    batch loop, so no c_j is generated for it here.

    """
    if batch_axes is None:
        batch_axes = ()
    ops = []
    pre = f" + (W){nprepad}" if nprepad > 0 else ""
    # When loop_batch_axis is True, skip the last axis (it's the batch axis)
    axes_to_process = range(ndim - 1) if loop_batch_axis else range(ndim)
    for j in axes_to_process:
        if j in batch_axes:
            # identity transform for batch axes
            ops.append(
                f"""
    W c_{j} = (W)in_coord[{j}]{pre};"""
            )
        else:
            ops.append(
                f"""
    W c_{j} = zoom[{j}] * ((W)in_coord[{j}] - shift[{j}]){pre};"""
            )
    return ops


def _get_coord_zoom_and_shift_grid(
    ndim, nprepad=0, float_type=None, batch_axes=None, loop_batch_axis=False
):
    """Compute target coordinate based on a shift followed by a zoom.

    This version zooms from the outer edges of the grid pixels.

    Notes
    -----
    Assumes the following variables have been initialized on the device::

        in_coord[ndim]: array containing the source coordinate
        zoom[ndim]: array containing the zoom for each axis
        shift[ndim]: array containing the shift for each axis

    computes::

        c_j = zoom[j] * (in_coord[j] - shift[j] + 0.5) - 0.5

    For batch axes (zoom == 1 and shift == 0), the identity is used:

        c_j = in_coord[j]

    When loop_batch_axis=True, the last axis is handled separately in the
    batch loop, so no c_j is generated for it here.

    """
    if batch_axes is None:
        batch_axes = ()
    ops = []
    pre = f" + (W){nprepad}" if nprepad > 0 else ""
    # When loop_batch_axis is True, skip the last axis (it's the batch axis)
    axes_to_process = range(ndim - 1) if loop_batch_axis else range(ndim)
    for j in axes_to_process:
        if j in batch_axes:
            # identity transform for batch axes
            ops.append(
                f"""
    W c_{j} = (W)in_coord[{j}]{pre};"""
            )
        else:
            ops.append(
                f"""
    W c_{j} = zoom[{j}] * ((W)in_coord[{j}] - shift[j] + ({float_type})0.5)
              - (W)0.5{pre};"""
            )
    return ops


def _get_coord_zoom(
    ndim, nprepad=0, float_type=None, batch_axes=None, loop_batch_axis=False
):
    """Compute target coordinate based on a zoom.

    This version zooms from the center of the edge pixels.

    Notes
    -----
    Assumes the following variables have been initialized on the device::

        in_coord[ndim]: array containing the source coordinate
        zoom[ndim]: array containing the zoom for each axis

    computes::

        c_j = zoom[j] * in_coord[j]

    For batch axes (zoom == 1), the identity is used:

        c_j = in_coord[j]

    When loop_batch_axis=True, the last axis is handled separately in the
    batch loop, so no c_j is generated for it here.

    """
    if batch_axes is None:
        batch_axes = ()
    ops = []
    pre = f" + (W){nprepad}" if nprepad > 0 else ""
    # When loop_batch_axis is True, skip the last axis (it's the batch axis)
    axes_to_process = range(ndim - 1) if loop_batch_axis else range(ndim)
    for j in axes_to_process:
        if j in batch_axes:
            # identity transform for batch axes
            ops.append(
                f"""
    W c_{j} = (W)in_coord[{j}]{pre};"""
            )
        else:
            ops.append(
                f"""
    W c_{j} = zoom[{j}] * (W)in_coord[{j}]{pre};"""
            )
    return ops


def _get_coord_zoom_grid(
    ndim, nprepad=0, float_type=None, batch_axes=None, loop_batch_axis=False
):
    """Compute target coordinate based on a zoom (grid_mode=True version).

    This version zooms from the outer edges of the grid pixels.

    Notes
    -----
    Assumes the following variables have been initialized on the device::

        in_coord[ndim]: array containing the source coordinate
        zoom[ndim]: array containing the zoom for each axis

    computes::

        c_j = zoom[j] * (in_coord[j] + 0.5) - 0.5

    For batch axes (zoom == 1), the identity is used:

        c_j = in_coord[j]

    When loop_batch_axis=True, the last axis is handled separately in the
    batch loop, so no c_j is generated for it here.

    """
    if batch_axes is None:
        batch_axes = ()
    ops = []
    pre = f" + (W){nprepad}" if nprepad > 0 else ""
    # When loop_batch_axis is True, skip the last axis (it's the batch axis)
    axes_to_process = range(ndim - 1) if loop_batch_axis else range(ndim)
    for j in axes_to_process:
        if j in batch_axes:
            # identity transform for batch axes
            ops.append(
                f"""
    W c_{j} = (W)in_coord[{j}]{pre};"""
            )
        else:
            ops.append(
                f"""
    W c_{j} = zoom[{j}] * ((W)in_coord[{j}] + ({float_type})0.5)
                           - ({float_type})0.5{pre};"""
            )
    return ops


def _get_coord_shift(
    ndim, nprepad=0, float_type=None, batch_axes=None, loop_batch_axis=False
):
    """Compute target coordinate based on a shift.

    Notes
    -----
    Assumes the following variables have been initialized on the device::

        in_coord[ndim]: array containing the source coordinate
        shift[ndim]: array containing the shift for each axis

    computes::

        c_j = in_coord[j] - shift[j]

    For batch axes (shift == 0), the identity is used:

        c_j = in_coord[j]

    When loop_batch_axis=True, the last axis is handled separately in the
    batch loop, so no c_j is generated for it here.

    """
    if batch_axes is None:
        batch_axes = ()
    ops = []
    pre = f" + (W){nprepad}" if nprepad > 0 else ""
    # When loop_batch_axis is True, skip the last axis (it's the batch axis)
    axes_to_process = range(ndim - 1) if loop_batch_axis else range(ndim)
    for j in axes_to_process:
        if j in batch_axes:
            # identity transform for batch axes
            ops.append(
                f"""
    W c_{j} = (W)in_coord[{j}]{pre};"""
            )
        else:
            ops.append(
                f"""
    W c_{j} = (W)in_coord[{j}] - shift[{j}]{pre};"""
            )
    return ops


def _get_coord_affine(
    ndim, nprepad=0, float_type=None, batch_axes=None, loop_batch_axis=False
):
    """Compute target coordinate based on a homogeneous transformation matrix.

    The homogeneous matrix has shape (ndim, ndim + 1). It corresponds to
    affine matrix where the last row of the affine is assumed to be:
    ``[0] * ndim + [1]``.

    Notes
    -----
    Assumes the following variables have been initialized on the device::

        mat(array): array containing the (ndim, ndim + 1) transform matrix.
        in_coords(array): coordinates of the input

    For example, in 2D:

        c_0 = mat[0] * in_coords[0] + mat[1] * in_coords[1] + aff[2];
        c_1 = mat[3] * in_coords[0] + mat[4] * in_coords[1] + aff[5];

    For batch axes (identity row with zero offset), the identity is used:

        c_j = in_coord[j]

    When loop_batch_axis=True, the last axis is handled separately in the
    batch loop, so no c_j is generated for it here.

    """
    if batch_axes is None:
        batch_axes = ()
    ops = []
    pre = f" + (W){nprepad}" if nprepad > 0 else ""
    ncol = ndim + 1
    # When loop_batch_axis is True, skip the last axis (it's the batch axis)
    axes_to_process = range(ndim - 1) if loop_batch_axis else range(ndim)
    # When loop_batch_axis is True, skip multiplying by in_coord for batch axis
    # since it's not initialized yet (it's set in the batch loop)
    # For batch axes, the matrix column should be 0 for non-batch rows anyway
    coord_axes = range(ndim - 1) if loop_batch_axis else range(ndim)

    for j in axes_to_process:
        if j in batch_axes:
            # identity transform for batch axes
            ops.append(
                f"""
            W c_{j} = (W)in_coord[{j}]{pre};"""
            )
        else:
            ops.append(
                f"""
            W c_{j} = (W)0.0;"""
            )
            for k in coord_axes:
                ops.append(
                    f"""
            c_{j} += mat[{ncol * j + k}] * (W)in_coord[{k}];"""
                )
            ops.append(
                f"""
            c_{j} += mat[{ncol * j + ndim}]{pre};"""
            )
    return ops


def _unravel_loop_index(shape, uint_t="unsigned int", array_size=None):
    """
    declare a multi-index array in_coord and unravel the 1D index, i into it.
    This code assumes that the array is a C-ordered array.

    Args:
        shape: tuple of dimension sizes to unravel
        uint_t: unsigned integer type to use
        array_size: size of in_coord array (defaults to len(shape), but can be
            larger if additional dimensions will be set separately)
    """
    ndim = len(shape)
    if array_size is None:
        array_size = ndim
    code = [
        f"""
        {uint_t} in_coord[{array_size}];
        {uint_t} s, t, idx = i;"""
    ]
    for j in range(ndim - 1, 0, -1):
        code.append(
            f"""
        s = {shape[j]};
        t = idx / s;
        in_coord[{j}] = idx - t * s;
        idx = t;"""
        )
    code.append(
        """
        in_coord[0] = idx;"""
    )
    return "\n".join(code)


def _generate_interp_custom(
    coord_func,
    ndim,
    large_int,
    yshape,
    mode,
    cval,
    order,
    name="",
    integer_output=False,
    nprepad=0,
    float_dtype=cupy.float64,
    omit_in_coord=False,
    batch_axes=None,
    loop_batch_axis=False,
):
    """
    Args:
        coord_func (function): generates code to do the coordinate
            transformation. See for example, `_get_coord_shift`.
        ndim (int): The number of dimensions.
        large_int (bool): If true use Py_ssize_t instead of int for indexing.
        yshape (tuple): Shape of the output array. Can be None if not needed
            in the kernel.
        mode (str): Signal extension mode to use at the array boundaries
        cval (float): constant value used when `mode == 'constant'`.
        name (str): base name for the interpolation kernel
        integer_output (bool): boolean indicating whether the output has an
            integer type.
        nprepad (int): integer indicating the amount of prepadding at the
            boundaries.
        float_dtype (cupy.dtype): the floating point precision to use
            internally. Should be cupy.float32 or cupy.float64.
        omit_in_coord (bool): omit computation of unraveled input coordinates
            when not needed. This is an optimization for use by
            ``map_coordinates`` where the coordinates are already known.
        batch_axes (tuple or None): tuple of axis indices that are "batch"
            dimensions where no interpolation is needed (zoom == 1 and
            shift == 0). For these axes, the identity transform is used and
            interpolation is skipped.
        loop_batch_axis (bool): If True, the last axis is a contiguous batch
            axis and the kernel will loop over it internally. This requires
            using raw output and reduces kernel launch overhead.

    Returns:
        operation (str): code body for the ElementwiseKernel
        name (str): name for the ElementwiseKernel
    """
    if batch_axes is None:
        batch_axes = ()

    if large_int:
        uint_t = "size_t"
        int_t = "ptrdiff_t"
    else:
        uint_t = "unsigned int"
        int_t = "int"

    ops = []

    # indicate the input and output arrays don't overlap so the compiler
    # can use LDG (read-only cache) instead of potentially slow LD cache.
    # x is guaranteed C-contiguous by (_filter_input / ascontiguousarray)
    ops.append("const X* __restrict__ x_ptr = &x[0];")

    # When looping over the last batch axis, we process all batch elements
    # in a single kernel thread. Spatial coordinates are computed once,
    # and the batch loop is placed as the innermost loop.
    if loop_batch_axis:
        batch_size = yshape[-1]
        batch_axis = ndim - 1
        reduced_yshape = yshape[:-1]
        ops.append(f"const {uint_t} batch_size = {batch_size};")
        ops.append(f"const {uint_t} out_base_idx = i * batch_size;")
    else:
        reduced_yshape = yshape

    float_type = cupy._core._scalar.get_typename(float_dtype)
    internal_dtype = float_type if integer_output else "Y"

    # determine strides for x along each axis
    for j in range(ndim):
        ops.append(f"const {int_t} xsize_{j} = x.shape()[{j}];")
    ops.append(f"const {uint_t} sx_{ndim - 1} = 1;")
    for j in range(ndim - 1, 0, -1):
        ops.append(f"const {uint_t} sx_{j - 1} = sx_{j} * xsize_{j};")

    if loop_batch_axis:
        # Only unravel to ndim-1 dimensions (spatial only)
        if not omit_in_coord:
            ops.append(_unravel_loop_index(reduced_yshape, uint_t, ndim))
        # Don't start batch loop here - it will be the innermost loop
    else:
        if not omit_in_coord:
            ops.append(_unravel_loop_index(reduced_yshape, uint_t))

    if not loop_batch_axis:
        ops.append(f"{internal_dtype} out = 0.0;")

    # compute the transformed (target) coordinates, c_j
    # For loop_batch_axis, this only computes spatial coords (batch coord set later)
    ops = ops + coord_func(
        ndim,
        nprepad,
        float_type=float_type,
        batch_axes=batch_axes,
        loop_batch_axis=loop_batch_axis,
    )

    if cval is numpy.nan:
        cval = "(Y)CUDART_NAN"
    elif cval == numpy.inf:
        cval = "(Y)CUDART_INF"
    elif cval == -numpy.inf:
        cval = "(Y)(-CUDART_INF)"
    else:
        cval = f"({internal_dtype}){cval}"

    if mode == "constant":
        # use cval if coordinate is outside the bounds of x
        if loop_batch_axis:
            # Only check spatial bounds (batch bounds are always valid)
            spatial_axes = [j for j in range(ndim) if j != batch_axis]
            _cond = " || ".join(
                [
                    f"(c_{j} < 0) || (c_{j} > xsize_{j} - 1)"
                    for j in spatial_axes
                ]
            )
            # If spatial out of bounds, fill all batch elements with cval
            ops.append(
                f"""
        if ({_cond})
        {{
            #pragma unroll 4
            for ({uint_t} batch_idx = 0; batch_idx < batch_size; batch_idx++) {{
                y[out_base_idx + batch_idx] = {cval};
            }}
        }}
        else
        {{"""
            )
        else:
            _cond = " || ".join(
                [f"(c_{j} < 0) || (c_{j} > xsize_{j} - 1)" for j in range(ndim)]
            )
            ops.append(
                f"""
        if ({_cond})
        {{
            out = {cval};
        }}
        else
        {{"""
            )

    if order == 0:
        if mode == "wrap":
            # mode 'wrap' requires this to work
            ops.append(f"{float_type} dcoord;")

        # Compute spatial indices (outside batch loop)
        spatial_axes = [j for j in range(ndim) if j not in batch_axes]
        for j in spatial_axes:
            # determine nearest neighbor
            if mode == "wrap":
                ops.append(
                    f"""
                dcoord = c_{j};"""
                )
            else:
                ops.append(
                    f"""
                {int_t} cf_{j} = ({int_t})floor(({float_type})c_{j}
                                                + ({float_type})0.5);"""
                )

            # handle boundary
            if mode != "constant":
                if mode == "wrap":
                    ixvar = "dcoord"
                    float_ix = True
                else:
                    ixvar = f"cf_{j}"
                    float_ix = False
                ops.append(
                    _util._generate_boundary_condition_ops(
                        mode, ixvar, f"xsize_{j}", int_t, float_ix
                    )
                )
                if mode == "wrap":
                    ops.append(
                        f"""
                {int_t} cf_{j} = ({int_t})floor(dcoord + ({float_type})0.5);"""
                    )

            # sum over ic_j will give the raveled coordinate in the input
            ops.append(
                f"""
            {int_t} ic_{j} = cf_{j} * sx_{j};"""
            )

        # Compute base index from spatial coords
        _spatial_coord_idx = " + ".join([f"ic_{j}" for j in spatial_axes])

        if loop_batch_axis:
            # Batch loop is innermost - iterate over batch elements
            if mode == "grid-constant":
                _cond = " || ".join([f"(ic_{j} < 0)" for j in spatial_axes])
                ops.append(
                    f"""
            if ({_cond}) {{
                #pragma unroll 4
                for ({uint_t} batch_idx = 0; batch_idx < batch_size; batch_idx++) {{
                    y[out_base_idx + batch_idx] = {cval};
                }}
            }} else {{
                #pragma unroll 4
                for ({uint_t} batch_idx = 0; batch_idx < batch_size; batch_idx++) {{
                    y[out_base_idx + batch_idx] = ({internal_dtype})x_ptr[{_spatial_coord_idx} + batch_idx];
                }}
            }}"""  # noqa: E501
                )
            else:
                ops.append(
                    f"""
            #pragma unroll 4
            for ({uint_t} batch_idx = 0; batch_idx < batch_size; batch_idx++) {{
                y[out_base_idx + batch_idx] = ({internal_dtype})x_ptr[{_spatial_coord_idx} + batch_idx];
            }}"""  # noqa: E501
                )
        else:
            # Original non-looped code path
            for j in batch_axes:
                ops.append(
                    f"""
            {int_t} cf_{j} = ({int_t})in_coord[{j}];
            {int_t} ic_{j} = cf_{j} * sx_{j};"""
                )
            _coord_idx = " + ".join([f"ic_{j}" for j in range(ndim)])
            if mode == "grid-constant":
                _cond = " || ".join([f"(ic_{j} < 0)" for j in range(ndim)])
                ops.append(
                    f"""
            if ({_cond}) {{
                out = {cval};
            }} else {{
                out = ({internal_dtype})x_ptr[{_coord_idx}];
            }}"""
                )
            else:
                ops.append(
                    f"""
            out = ({internal_dtype})x_ptr[{_coord_idx}];"""
                )

    elif order == 1:
        spatial_axes = [j for j in range(ndim) if j not in batch_axes]

        # Compute interpolation setup for spatial axes (outside any batch loop)
        for j in spatial_axes:
            # get coordinates for linear interpolation along axis j
            ops.append(
                f"""
            {int_t} cf_{j} = ({int_t})floor(({float_type})c_{j});
            {int_t} cc_{j} = cf_{j} + 1;
            {int_t} n_{j} = (c_{j} == cf_{j}) ? 1 : 2;  // points needed
            """
            )

            if mode == "wrap":
                ops.append(
                    f"""
                {float_type} dcoordf_{j} = c_{j};
                {float_type} dcoordc_{j} = c_{j} + 1;"""
                )
            else:
                # handle boundaries for extension modes.
                ops.append(
                    f"""
                {int_t} cf_bounded_{j} = cf_{j};
                {int_t} cc_bounded_{j} = cc_{j};"""
                )

            if mode != "constant":
                if mode == "wrap":
                    ixvar = f"dcoordf_{j}"
                    float_ix = True
                else:
                    ixvar = f"cf_bounded_{j}"
                    float_ix = False
                ops.append(
                    _util._generate_boundary_condition_ops(
                        mode, ixvar, f"xsize_{j}", int_t, float_ix
                    )
                )

                ixvar = f"dcoordc_{j}" if mode == "wrap" else f"cc_bounded_{j}"
                ops.append(
                    _util._generate_boundary_condition_ops(
                        mode, ixvar, f"xsize_{j}", int_t, float_ix
                    )
                )
                if mode == "wrap":
                    ops.append(
                        f"""
                    {int_t} cf_bounded_{j} = ({int_t})floor(dcoordf_{j});
                    {int_t} cc_bounded_{j} = ({int_t})floor(dcoordf_{j} + ({float_type})1.0);
                    """
                    )

        if loop_batch_axis:
            # Optimized path: spatial interpolation loops outside, batch loop innermost
            # Initialize output array for all batch elements
            ops.append(
                f"""
            {internal_dtype} out_batch[{batch_size}];
            #pragma unroll 4
            for ({uint_t} b = 0; b < batch_size; b++) {{ out_batch[b] = 0.0; }}"""
            )

            # Generate nested loops for spatial interpolation
            for j in spatial_axes:
                ops.append(
                    f"""
            for (int s_{j} = 0; s_{j} < n_{j}; s_{j}++)
                {{
                    W w_{j};
                    {int_t} ic_{j};
                    if (s_{j} == 0)
                    {{
                        w_{j} = (W)cc_{j} - c_{j};
                        ic_{j} = cf_bounded_{j} * sx_{j};
                    }} else
                    {{
                        w_{j} = c_{j} - (W)cf_{j};
                        ic_{j} = cc_bounded_{j} * sx_{j};
                    }}"""
                )

            # Compute spatial weight and base index
            _spatial_weight = " * ".join([f"w_{j}" for j in spatial_axes])
            _spatial_coord_idx = " + ".join([f"ic_{j}" for j in spatial_axes])

            # Innermost batch loop - with bounds check for grid-constant mode
            if mode == "grid-constant":
                _spatial_cond = " || ".join(
                    [f"(ic_{j} < 0)" for j in spatial_axes]
                )
                ops.append(
                    f"""
                    W spatial_weight = {_spatial_weight};
                    {int_t} ic_base = {_spatial_coord_idx};
                    if ({_spatial_cond}) {{
                        #pragma unroll 4
                        for ({uint_t} batch_idx = 0; batch_idx < batch_size; batch_idx++) {{
                            out_batch[batch_idx] += {cval} * ({internal_dtype})spatial_weight;
                        }}
                    }} else {{
                        #pragma unroll 4
                        for ({uint_t} batch_idx = 0; batch_idx < batch_size; batch_idx++) {{
                            {internal_dtype} val = ({internal_dtype})x_ptr[ic_base + batch_idx];
                            out_batch[batch_idx] += val * ({internal_dtype})spatial_weight;
                        }}
                    }}"""
                )
            else:
                ops.append(
                    f"""
                    W spatial_weight = {_spatial_weight};
                    {int_t} ic_base = {_spatial_coord_idx};
                    #pragma unroll 4
                    for ({uint_t} batch_idx = 0; batch_idx < batch_size; batch_idx++) {{
                        {internal_dtype} val = ({internal_dtype})x_ptr[ic_base + batch_idx];
                        out_batch[batch_idx] += val * ({internal_dtype})spatial_weight;
                    }}"""
                )

            # Close spatial interpolation loops
            ops.append("}" * len(spatial_axes))

            # Write results for all batch elements
            if integer_output:
                ops.append(
                    f"""
            #pragma unroll 4
            for ({uint_t} batch_idx = 0; batch_idx < batch_size; batch_idx++) {{
                y[out_base_idx + batch_idx] = (Y)rint(({float_type})out_batch[batch_idx]);
            }}"""
                )
            else:
                ops.append(
                    f"""
            #pragma unroll 4
            for ({uint_t} batch_idx = 0; batch_idx < batch_size; batch_idx++) {{
                y[out_base_idx + batch_idx] = (Y)out_batch[batch_idx];
            }}"""
                )
        else:
            # Original non-looped code path
            for j in range(ndim):
                if j in batch_axes:
                    ops.append(
                        f"""
            W w_{j} = (W)1.0;
            {int_t} ic_{j} = (({int_t})in_coord[{j}]) * sx_{j};
            {{  // dummy scope for batch axis {j}"""
                    )
                elif j in spatial_axes:
                    # spatial axes already have their setup code, just add the loop
                    ops.append(
                        f"""
            for (int s_{j} = 0; s_{j} < n_{j}; s_{j}++)
                {{
                    W w_{j};
                    {int_t} ic_{j};
                    if (s_{j} == 0)
                    {{
                        w_{j} = (W)cc_{j} - c_{j};
                        ic_{j} = cf_bounded_{j} * sx_{j};
                    }} else
                    {{
                        w_{j} = c_{j} - (W)cf_{j};
                        ic_{j} = cc_bounded_{j} * sx_{j};
                    }}"""
                    )
    elif order > 1:
        if mode == "grid-constant":
            spline_mode = "constant"
        elif mode == "nearest":
            spline_mode = "nearest"
        else:
            spline_mode = _spline_prefilter_core._get_spline_mode(mode)

        spatial_axes = [j for j in range(ndim) if j not in batch_axes]

        # wx, wy are temporary variables used during spline weight computation
        if spatial_axes:
            ops.append(
                f"""
            W wx, wy;
            {int_t} start;"""
            )

        # Compute spline weights and indices for spatial axes (outside batch loop)
        for j in spatial_axes:
            # determine weights along the current axis
            ops.append(
                f"""
            W weights_{j}[{order + 1}];"""
            )
            ops.append(
                spline_weights_inline[order].format(
                    j=j, order=order, F=float_type
                )
            )

            # get starting coordinate for spline interpolation along axis j
            if mode in ["wrap"]:
                ops.append(f"{float_type} dcoord_{j} = c_{j};")
                coord_var = f"dcoord_{j}"
                ops.append(
                    _util._generate_boundary_condition_ops(
                        mode, coord_var, f"xsize_{j}", int_t, True
                    )
                )
            else:
                coord_var = f"({float_type})c_{j}"

            if order & 1:
                op_str = """
                start = ({int_t})floor({coord_var}) - {order_2};"""
            else:
                op_str = """
                start = ({int_t})floor({coord_var} + ({float_type})0.5) - {order_2};"""
            ops.append(
                op_str.format(
                    int_t=int_t,
                    coord_var=coord_var,
                    float_type=float_type,
                    order_2=order // 2,
                )
            )

            # set of coordinate values within spline footprint along axis j
            ops.append(f"""{int_t} ci_{j}[{order + 1}];""")
            for k in range(order + 1):
                ixvar = f"ci_{j}[{k}]"
                ops.append(
                    f"""
                {ixvar} = start + {k};"""
                )
                ops.append(
                    _util._generate_boundary_condition_ops(
                        spline_mode, ixvar, f"xsize_{j}", int_t
                    )
                )

        if loop_batch_axis:
            # Optimized path: spatial interpolation loops outside, batch loop innermost
            # Initialize output array for all batch elements
            ops.append(
                f"""
            {internal_dtype} out_batch[{batch_size}];
            for ({uint_t} b = 0; b < batch_size; b++) {{ out_batch[b] = 0.0; }}"""
            )

            # Generate nested loops for spatial interpolation
            for j in spatial_axes:
                ops.append(
                    f"""
            W w_{j};
            {int_t} ic_{j};
            for (int k_{j} = 0; k_{j} <= {order}; k_{j}++)
                {{
                    w_{j} = weights_{j}[k_{j}];
                    ic_{j} = ci_{j}[k_{j}] * sx_{j};
            """
                )

            # Compute spatial weight and base index
            _spatial_weight = " * ".join([f"w_{j}" for j in spatial_axes])
            _spatial_coord_idx = " + ".join([f"ic_{j}" for j in spatial_axes])

            # Innermost batch loop
            # For order > 1, we need bounds check for both constant and grid-constant modes
            if mode == "grid-constant" or mode == "constant":
                _spatial_cond = " || ".join(
                    [f"(ic_{j} < 0)" for j in spatial_axes]
                )
                ops.append(
                    f"""
                    W spatial_weight = {_spatial_weight};
                    {int_t} ic_base = {_spatial_coord_idx};
                    if ({_spatial_cond}) {{
                        #pragma unroll 4
                        for ({uint_t} batch_idx = 0; batch_idx < batch_size; batch_idx++) {{
                            out_batch[batch_idx] += {cval} * ({internal_dtype})spatial_weight;
                        }}
                    }} else {{
                        #pragma unroll 4
                        for ({uint_t} batch_idx = 0; batch_idx < batch_size; batch_idx++) {{
                            {internal_dtype} val = ({internal_dtype})x_ptr[ic_base + batch_idx];
                            out_batch[batch_idx] += val * ({internal_dtype})spatial_weight;
                        }}
                    }}"""
                )
            else:
                ops.append(
                    f"""
                    W spatial_weight = {_spatial_weight};
                    {int_t} ic_base = {_spatial_coord_idx};
                    #pragma unroll 4
                    for ({uint_t} batch_idx = 0; batch_idx < batch_size; batch_idx++) {{
                        {internal_dtype} val = ({internal_dtype})x_ptr[ic_base + batch_idx];
                        out_batch[batch_idx] += val * ({internal_dtype})spatial_weight;
                    }}"""
                )

            # Close spatial interpolation loops
            ops.append("}" * len(spatial_axes))

            # Write results for all batch elements
            if integer_output:
                ops.append(
                    f"""
            #pragma unroll 4
            for ({uint_t} batch_idx = 0; batch_idx < batch_size; batch_idx++) {{
                y[out_base_idx + batch_idx] = (Y)rint(({float_type})out_batch[batch_idx]);
            }}"""
                )
            else:
                ops.append(
                    f"""
            #pragma unroll 4
            for ({uint_t} batch_idx = 0; batch_idx < batch_size; batch_idx++) {{
                y[out_base_idx + batch_idx] = (Y)out_batch[batch_idx];
            }}"""
                )
        else:
            # Original non-looped code path
            for j in range(ndim):
                if j in batch_axes:
                    ops.append(
                        f"""
            W w_{j} = (W)1.0;
            {int_t} ic_{j} = (({int_t})in_coord[{j}]) * sx_{j};
            {{  // dummy scope for batch axis {j}"""
                    )
                elif j in spatial_axes:
                    # spatial axes already have weights/indices computed, just add the loop
                    ops.append(
                        f"""
            W w_{j};
            {int_t} ic_{j};
            for (int k_{j} = 0; k_{j} <= {order}; k_{j}++)
                {{
                    w_{j} = weights_{j}[k_{j}];
                    ic_{j} = ci_{j}[k_{j}] * sx_{j};
            """
                    )

    # Final accumulation and output (for non-looped cases and order > 0)
    if order > 0 and not (loop_batch_axis and order >= 1):
        _weight = " * ".join([f"w_{j}" for j in range(ndim)])
        _coord_idx = " + ".join([f"ic_{j}" for j in range(ndim)])
        if mode == "grid-constant" or (order > 1 and mode == "constant"):
            _cond = " || ".join([f"(ic_{j} < 0)" for j in range(ndim)])
            ops.append(
                f"""
            if ({_cond}) {{
                out += {cval} * ({internal_dtype})({_weight});
            }} else {{
                {internal_dtype} val = ({internal_dtype})x_ptr[{_coord_idx}];
                out += val * ({internal_dtype})({_weight});
            }}"""
            )
        else:
            ops.append(
                f"""
            {internal_dtype} val = ({internal_dtype})x_ptr[{_coord_idx}];
            out += val * ({internal_dtype})({_weight});"""
            )

        ops.append("}" * ndim)

    if mode == "constant":
        ops.append("}")

    # Output writing (for non-looped cases - looped cases write inside their batch loops)
    if not loop_batch_axis:
        if integer_output:
            ops.append(f"y = (Y)rint(({float_type})out);")
        else:
            ops.append("y = (Y)out;")
    operation = "\n".join(ops)

    mode_str = mode.replace("-", "_")  # avoid hyphen in kernel name
    name = "cupyx_scipy_ndimage_interpolate_{}_order{}_{}_{}d_y{}".format(
        name,
        order,
        mode_str,
        ndim,
        "_".join([f"{j}" for j in yshape]),
    )
    if uint_t == "size_t":
        name += "_i64"
    if batch_axes:
        name += "_batch_" + "_".join([str(j) for j in sorted(batch_axes)])
    if loop_batch_axis:
        name += "_looped"
    return operation, name


def use_loop_batch(batch_axes, ndim, yshape):
    """Whether batch optimization over the final axis should be applied.

    If the last axis (contiguous axis) is a batch axis, we can compute the
    weights once and then loop over this axis in an inner loop.

    Only enabled when there is at least one spatial axis (ndim > 1).

    Empirically, restrict this optimization to batch size <= 12. Performance
    degradation was observed at higher batch sizes.
    """
    return (
        batch_axes == (ndim - 1,)
        and ndim > 1
        and yshape[-1] <= loop_batch_max_channels
    )


@cupy._util.memoize(for_each_device=True)
def _get_map_kernel(
    ndim,
    large_int,
    yshape,
    mode,
    cval=0.0,
    order=1,
    integer_output=False,
    nprepad=0,
    float_dtype=cupy.double,
    batch_axes=None,
):
    in_params = "raw X x, raw W coords"

    # Optimize for contiguous batch axis at the end
    # Only enable when there's at least one spatial axis (ndim > 1)
    loop_batch_axis = use_loop_batch(batch_axes, ndim, yshape)
    if loop_batch_axis:
        out_params = "raw Y y"
        size = math.prod(yshape[:-1])
    else:
        out_params = "Y y"
        size = None

    # if there are batch axes, we need in_coord to be computed
    # (batch axes use in_coord[j] instead of reading from coords)
    omit_in_coord = not batch_axes
    operation, name = _generate_interp_custom(
        coord_func=_get_coord_map,
        ndim=ndim,
        large_int=large_int,
        yshape=yshape,
        mode=mode,
        cval=cval,
        order=order,
        name="map",
        integer_output=integer_output,
        nprepad=nprepad,
        float_dtype=float_dtype,
        omit_in_coord=omit_in_coord,
        batch_axes=batch_axes,
        loop_batch_axis=loop_batch_axis,
    )
    kernel = cupy.ElementwiseKernel(
        in_params, out_params, operation, name, preamble=math_constants_preamble
    )
    return KernelInfo(kernel, size)


@cupy._util.memoize(for_each_device=True)
def _get_shift_kernel(
    ndim,
    large_int,
    yshape,
    mode,
    cval=0.0,
    order=1,
    integer_output=False,
    nprepad=0,
    float_dtype=cupy.double,
    batch_axes=None,
):
    in_params = "raw X x, raw W shift"

    # Optimize for contiguous batch axis at the end
    # Only enable when there's at least one spatial axis (ndim > 1)
    loop_batch_axis = use_loop_batch(batch_axes, ndim, yshape)
    if loop_batch_axis:
        out_params = "raw Y y"
        size = math.prod(yshape[:-1])
    else:
        out_params = "Y y"
        size = None

    operation, name = _generate_interp_custom(
        coord_func=_get_coord_shift,
        ndim=ndim,
        large_int=large_int,
        yshape=yshape,
        mode=mode,
        cval=cval,
        order=order,
        name="shift",
        integer_output=integer_output,
        nprepad=nprepad,
        float_dtype=float_dtype,
        batch_axes=batch_axes,
        loop_batch_axis=loop_batch_axis,
    )
    kernel = cupy.ElementwiseKernel(
        in_params, out_params, operation, name, preamble=math_constants_preamble
    )
    return KernelInfo(kernel, size)


@cupy._util.memoize(for_each_device=True)
def _get_zoom_shift_kernel(
    ndim,
    large_int,
    yshape,
    mode,
    cval=0.0,
    order=1,
    integer_output=False,
    grid_mode=False,
    nprepad=0,
    float_dtype=cupy.double,
    batch_axes=None,
):
    in_params = "raw X x, raw W shift, raw W zoom"

    # Optimize for contiguous batch axis at the end
    # Only enable when there's at least one spatial axis (ndim > 1)
    loop_batch_axis = use_loop_batch(batch_axes, ndim, yshape)
    if loop_batch_axis:
        out_params = "raw Y y"
        size = math.prod(yshape[:-1])
    else:
        out_params = "Y y"
        size = None

    if grid_mode:
        zoom_shift_func = _get_coord_zoom_and_shift_grid
    else:
        zoom_shift_func = _get_coord_zoom_and_shift
    operation, name = _generate_interp_custom(
        coord_func=zoom_shift_func,
        ndim=ndim,
        large_int=large_int,
        yshape=yshape,
        mode=mode,
        cval=cval,
        order=order,
        name="zoom_shift_grid" if grid_mode else "zoom_shift",
        integer_output=integer_output,
        nprepad=nprepad,
        float_dtype=float_dtype,
        batch_axes=batch_axes,
        loop_batch_axis=loop_batch_axis,
    )
    kernel = cupy.ElementwiseKernel(
        in_params, out_params, operation, name, preamble=math_constants_preamble
    )
    return KernelInfo(kernel, size)


@cupy._util.memoize(for_each_device=True)
def _get_zoom_kernel(
    ndim,
    large_int,
    yshape,
    mode,
    cval=0.0,
    order=1,
    integer_output=False,
    grid_mode=False,
    nprepad=0,
    float_dtype=cupy.double,
    batch_axes=None,
):
    in_params = "raw X x, raw W zoom"

    # Optimize for contiguous batch axis at the end
    # Only enable when there's at least one spatial axis (ndim > 1)
    loop_batch_axis = use_loop_batch(batch_axes, ndim, yshape)
    if loop_batch_axis:
        out_params = "raw Y y"
        size = math.prod(yshape[:-1])
    else:
        out_params = "Y y"
        size = None

    operation, name = _generate_interp_custom(
        coord_func=_get_coord_zoom_grid if grid_mode else _get_coord_zoom,
        ndim=ndim,
        large_int=large_int,
        yshape=yshape,
        mode=mode,
        cval=cval,
        order=order,
        name="zoom_grid" if grid_mode else "zoom",
        integer_output=integer_output,
        nprepad=nprepad,
        float_dtype=float_dtype,
        batch_axes=batch_axes,
        loop_batch_axis=loop_batch_axis,
    )
    kernel = cupy.ElementwiseKernel(
        in_params, out_params, operation, name, preamble=math_constants_preamble
    )
    return KernelInfo(kernel, size)


@cupy._util.memoize(for_each_device=True)
def _get_affine_kernel(
    ndim,
    large_int,
    yshape,
    mode,
    cval=0.0,
    order=1,
    integer_output=False,
    nprepad=0,
    float_dtype=cupy.double,
    batch_axes=None,
):
    in_params = "raw X x, raw W mat"

    # Optimize for contiguous batch axis at the end
    # Only enable when there's at least one spatial axis (ndim > 1)
    loop_batch_axis = use_loop_batch(batch_axes, ndim, yshape)
    if loop_batch_axis:
        out_params = "raw Y y"
        size = math.prod(yshape[:-1])
    else:
        out_params = "Y y"
        size = None

    operation, name = _generate_interp_custom(
        coord_func=_get_coord_affine,
        ndim=ndim,
        large_int=large_int,
        yshape=yshape,
        mode=mode,
        cval=cval,
        order=order,
        name="affine",
        integer_output=integer_output,
        nprepad=nprepad,
        float_dtype=float_dtype,
        batch_axes=batch_axes,
        loop_batch_axis=loop_batch_axis,
    )
    kernel = cupy.ElementwiseKernel(
        in_params, out_params, operation, name, preamble=math_constants_preamble
    )
    return KernelInfo(kernel, size)
