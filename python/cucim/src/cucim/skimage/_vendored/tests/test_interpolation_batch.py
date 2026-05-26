# SPDX-FileCopyrightText: Copyright (c) 2015 Preferred Infrastructure, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2015 Preferred Networks, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extensions to CuPy's own test cases for new features in the vendored code.

Follows the style of the following file in the CuPy repository:
  tests/cupyx_tests/scipy_tests/ndimage_tests/test_interpolation.py

These additional tests verify new batch kernel launch functionality introduced
in cuCIM's vendored ndimage interpolation code. These test cases verify that
using batched kernels gives equivalent results to running multiple separate
(non-batched) kernels in a loop.

Note that results are NOT equivalent to SciPy in such cases as SciPy would also
apply spline prefiltering along the batch axes. To get the same results with
SciPy when prefilter=True, one would have to manually loop over the batched
axes as done in the reference implementation here.
"""

from __future__ import annotations

import numpy
import pytest

import cupy
from cupy.cuda import runtime
from cupy import testing
import cucim.skimage._vendored.ndimage as vendored_ndimage
from cucim.skimage._vendored._internal import AxisError
from cucim.skimage._vendored._ndimage_interp_kernels import (
    _get_coord_zoom_and_shift_grid,
    _get_shift_kernel,
    loop_batch_max_channels,
)

try:
    import scipy
    import scipy.ndimage

    scipy_version = numpy.lib.NumpyVersion(scipy.__version__)
except ImportError:
    pass

try:
    import cv2
except ImportError:
    pass


# testing these modes can only be tested against SciPy >= 1.6.0+
scipy16_modes = ["wrap", "grid-wrap", "reflect", "grid-mirror", "grid-constant"]
# these modes are okay to test on older SciPy
legacy_modes = ["constant", "nearest", "mirror"]
all_modes = legacy_modes + scipy16_modes
batch_modes = ["constant", "nearest", "wrap", "grid-constant"]
order_prefilter_pairs = [(0, False), (1, False), (3, False), (3, True)]


class OrderPrefilterMixin:
    @property
    def order(self):
        return self.order_prefilter[0]

    @property
    def prefilter(self):
        return self.order_prefilter[1]


def test_zoom_shift_grid_codegen_indexes_shift_by_axis():
    code = "\n".join(_get_coord_zoom_and_shift_grid(3, float_type="float"))

    assert "shift[j]" not in code
    for axis in range(3):
        assert f"shift[{axis}]" in code


def test_loop_batch_selected_when_last_axis_is_one_of_multiple_batch_axes():
    kern_info = _get_shift_kernel(
        4,
        False,
        (5, 6, 7, 3),
        "nearest",
        order=1,
        batch_axes=(0, 3),
        output_c_contiguous=True,
    )

    assert kern_info.size == 5 * 6 * 7


@testing.parameterize(
    *(
        testing.product(
            {
                "shape_and_axes": [
                    # contiguous (last) axis as batch to test loop_batch_axis case
                    ((48, 64, 4), (1, 0)),
                    ((48, 64, loop_batch_max_channels + 1), (1, 0)),
                    # first axis as batch
                    ((8, 48, 64), (2, 1)),
                    # both first and last axes as batch
                    ((17, 48, 64, 3), (2, 1)),
                ],
                "angle": [-15],
                "reshape": [False],
                "output": [None],
                "order_prefilter": order_prefilter_pairs,
                "mode": batch_modes,
                "cval": [1.0],
            }
        )
    )
)
@testing.with_requires("scipy")
class TestRotateBatch(OrderPrefilterMixin):
    _multiprocess_can_split = True

    def _rotate(self, a, axes):
        """Compare against manually looping over batch axes."""
        rotate = vendored_ndimage.rotate
        axes = tuple(ax % a.ndim for ax in axes)
        if a.ndim == 3 and len(axes) == 2 and 2 not in axes:
            # loop over last axis
            expected = cupy.stack(
                [
                    rotate(
                        a[..., i],
                        self.angle,
                        axes,
                        self.reshape,
                        None,
                        self.order,
                        self.mode,
                        self.cval,
                        self.prefilter,
                    )
                    for i in range(a.shape[-1])
                ],
                axis=-1,
            )
        elif a.ndim == 3 and len(axes) == 2 and 0 not in axes:
            # loop over first axis
            _axes = tuple(ax - 1 for ax in axes)
            expected = cupy.stack(
                [
                    rotate(
                        a[i, ...],
                        self.angle,
                        _axes,
                        self.reshape,
                        None,
                        self.order,
                        self.mode,
                        self.cval,
                        self.prefilter,
                    )
                    for i in range(a.shape[0])
                ],
                axis=0,
            )
        elif a.ndim == 4 and 0 not in axes and 3 not in axes:
            # loop over first and last axes
            _axes = tuple(ax - 1 for ax in axes)
            expected = cupy.stack(
                [
                    cupy.stack(
                        [
                            rotate(
                                a[i, ..., j],
                                self.angle,
                                _axes,
                                self.reshape,
                                None,
                                self.order,
                                self.mode,
                                self.cval,
                                self.prefilter,
                            )
                            for i in range(a.shape[0])
                        ],
                        axis=0,
                    )
                    for j in range(a.shape[-1])
                ],
                axis=-1,
            )
        else:
            raise ValueError("unsupported test case")
        result = rotate(
            a,
            self.angle,
            axes,
            self.reshape,
            None,
            self.order,
            self.mode,
            self.cval,
            self.prefilter,
        )

        # for integer output, allow ±1 due to rint() boundary
        atol = 1 if numpy.dtype(a.dtype).kind in "iu" else 1e-5

        cupy.testing.assert_allclose(result, expected, rtol=1e-5, atol=atol)
        return result

    @testing.for_float_dtypes(no_float16=True)
    def test_rotate_float(self, dtype):
        shape, axes = self.shape_and_axes
        a = testing.shaped_random(shape, cupy, dtype, scale=1)
        return self._rotate(a, axes)

    @testing.for_complex_dtypes()
    def test_rotate_complex_float(self, dtype):
        shape, axes = self.shape_and_axes
        if self.output == numpy.float64:
            self.output = numpy.complex128
        a = testing.shaped_random(shape, cupy, dtype, scale=1)
        return self._rotate(a, axes)

    @testing.for_float_dtypes(no_float16=True)
    def test_rotate_fortran_order(self, dtype):
        shape, axes = self.shape_and_axes
        a = testing.shaped_random(shape, cupy, dtype, scale=1)
        a = cupy.asfortranarray(a)
        return self._rotate(a, axes)

    @testing.for_int_dtypes(no_bool=True)
    def test_rotate_int(self, dtype):
        shape, axes = self.shape_and_axes

        if numpy.lib.NumpyVersion(scipy.__version__) < "1.0.0":
            if dtype in (numpy.dtype("l"), numpy.dtype("q")):
                dtype = numpy.int64
            elif dtype in (numpy.dtype("L"), numpy.dtype("Q")):
                dtype = numpy.uint64

        a = testing.shaped_random(shape, cupy, dtype)
        return self._rotate(a, axes)


@testing.parameterize(
    *(
        testing.product(
            {
                "shape_and_shift": [
                    # Any axis with shift==0 is effectively a batch dimension.
                    #
                    # contiguous (last) axis as batch to test loop_batch_axis case
                    ((48, 64, 4), (2, -3.2, 0)),
                    ((48, 64, loop_batch_max_channels + 1), (2, -3.2, 0)),
                    # first axis as batch
                    ((8, 48, 64), (0, 2, -3.2)),
                    # both first and last axes as batch
                    ((17, 48, 64, 3), (0, 2, -3.2, 0)),
                ],
                "output": [None],
                "order_prefilter": order_prefilter_pairs,
                "mode": batch_modes,
                "cval": [1.0],
            }
        )
    )
)
class TestShiftBatch(OrderPrefilterMixin):
    _multiprocess_can_split = True

    def _shift(self, a, shift_val):
        """Compare against manually looping over axes where shift is 0."""
        shift = vendored_ndimage.shift
        if len(shift_val) == 3 and shift_val[-1] == 0:
            # loop over last axis
            expected = cupy.stack(
                [
                    shift(
                        a[..., i],
                        shift_val[:-1],
                        self.output,
                        self.order,
                        self.mode,
                        self.cval,
                        self.prefilter,
                    )
                    for i in range(a.shape[-1])
                ],
                axis=-1,
            )
        elif len(shift_val) == 3 and shift_val[0] == 0:
            # loop over first axis
            expected = cupy.stack(
                [
                    shift(
                        a[i, ...],
                        shift_val[1:],
                        self.output,
                        self.order,
                        self.mode,
                        self.cval,
                        self.prefilter,
                    )
                    for i in range(a.shape[0])
                ],
                axis=0,
            )
        elif len(shift_val) == 4 and shift_val[0] == 0 and shift_val[-1] == 0:
            # loop over first and last axes
            expected = cupy.stack(
                [
                    cupy.stack(
                        [
                            shift(
                                a[i, ..., j],
                                shift_val[1:-1],
                                self.output,
                                self.order,
                                self.mode,
                                self.cval,
                                self.prefilter,
                            )
                            for i in range(a.shape[0])
                        ],
                        axis=0,
                    )
                    for j in range(a.shape[-1])
                ],
                axis=-1,
            )
        result = shift(
            a,
            shift_val,
            self.output,
            self.order,
            self.mode,
            self.cval,
            self.prefilter,
        )

        # for integer output, allow ±1 due to rint() boundary
        atol = 1 if numpy.dtype(a.dtype).kind in "iu" else 1e-5

        cupy.testing.assert_allclose(result, expected, rtol=1e-5, atol=atol)
        return result

    @testing.for_float_dtypes(no_float16=True)
    def test_shift_float(self, dtype):
        shape, shift = self.shape_and_shift
        a = testing.shaped_random(shape, cupy, dtype, scale=1)
        return self._shift(a, shift)

    @testing.for_complex_dtypes()
    def test_shift_complex_float(self, dtype):
        shape, shift = self.shape_and_shift
        if self.output == numpy.float64:
            self.output = numpy.complex128
        a = testing.shaped_random(shape, cupy, dtype, scale=1)
        return self._shift(a, shift)

    @testing.for_float_dtypes(no_float16=True)
    def test_shift_fortran_order(self, dtype):
        shape, shift = self.shape_and_shift
        a = testing.shaped_random(shape, cupy, dtype, scale=1)
        a = cupy.asfortranarray(a)
        return self._shift(a, shift)

    @testing.for_int_dtypes(no_bool=True)
    def test_shift_int(self, dtype):
        shape, shift = self.shape_and_shift

        if self.mode == "constant" and not cupy.isfinite(self.cval):
            if self.output is None or self.output == "empty":
                # Non-finite cval with integer output array is not supported
                # CuPy exception is tested in TestInterpolationInvalidCval
                return cupy.asarray([])

        if numpy.lib.NumpyVersion(scipy.__version__) < "1.0.0":
            if dtype in (numpy.dtype("l"), numpy.dtype("q")):
                dtype = numpy.int64
            elif dtype in (numpy.dtype("L"), numpy.dtype("Q")):
                dtype = numpy.uint64

        a = testing.shaped_random(shape, cupy, dtype)
        return self._shift(a, shift)


@testing.parameterize(
    *testing.product(
        {
            "shape_and_zoom": [
                # Any axis with zoom == 1 is effectively a batch dimension.
                #
                # contiguous (last) axis as batch to test loop_batch_axis case
                ((32, 64, 4), (2.2, 0.3, 1)),
                ((32, 64, loop_batch_max_channels + 1), (2.2, 0.3, 1)),
                # first axis as batch
                ((8, 32, 64), (1, 2.2, 0.3)),
                # both first and last axes as batch
                ((17, 32, 64, 3), (1, 2.2, 0.3, 1)),
            ],
            "output": [None],
            "order_prefilter": order_prefilter_pairs,
            "mode": batch_modes,
            "cval": [1.0],
        }
    )
)
@testing.with_requires("scipy")
class TestZoomBatch(OrderPrefilterMixin):
    _multiprocess_can_split = True

    def _zoom(self, a, zm):
        """Compare against manually looping over axes where zoom is 1."""
        zoom = vendored_ndimage.zoom
        if a.ndim == 3 and zm[-1] == 1:
            # loop over last axis
            expected = cupy.stack(
                [
                    zoom(
                        a[..., i],
                        zm[:-1],
                        self.output,
                        self.order,
                        self.mode,
                        self.cval,
                        self.prefilter,
                    )
                    for i in range(a.shape[-1])
                ],
                axis=-1,
            )
        elif a.ndim == 3 and zm[0] == 1:
            # loop over first axis
            expected = cupy.stack(
                [
                    zoom(
                        a[i, ...],
                        zm[1:],
                        self.output,
                        self.order,
                        self.mode,
                        self.cval,
                        self.prefilter,
                    )
                    for i in range(a.shape[0])
                ],
                axis=0,
            )
        elif a.ndim == 4 and zm[0] == 1 and zm[-1] == 1:
            # loop over first and last axes
            expected = cupy.stack(
                [
                    cupy.stack(
                        [
                            zoom(
                                a[i, ..., j],
                                zm[1:-1],
                                self.output,
                                self.order,
                                self.mode,
                                self.cval,
                                self.prefilter,
                            )
                            for i in range(a.shape[0])
                        ],
                        axis=0,
                    )
                    for j in range(a.shape[-1])
                ],
                axis=-1,
            )
        else:
            raise ValueError("unsupported test case")
        result = zoom(
            a, zm, self.output, self.order, self.mode, self.cval, self.prefilter
        )

        # for integer output, allow ±1 due to rint() boundary
        atol = 1 if numpy.dtype(a.dtype).kind in "iu" else 1e-5

        cupy.testing.assert_allclose(result, expected, rtol=1e-5, atol=atol)
        return result

    @testing.for_float_dtypes(no_float16=True)
    def test_zoom_float(self, dtype):
        shape, zoom = self.shape_and_zoom
        a = testing.shaped_random(shape, cupy, dtype, scale=1)
        return self._zoom(a, zoom)

    @testing.for_complex_dtypes()
    def test_zoom_complex_float(self, dtype):
        shape, zoom = self.shape_and_zoom
        if self.output == numpy.float64:
            self.output = numpy.complex128
        a = testing.shaped_random(shape, cupy, dtype, scale=1)
        return self._zoom(a, zoom)

    @testing.for_float_dtypes(no_float16=True)
    def test_zoom_fortran_order(self, dtype):
        shape, zoom = self.shape_and_zoom
        a = testing.shaped_random(shape, cupy, dtype)
        a = cupy.asfortranarray(a)
        return self._zoom(a, zoom)

    @testing.for_int_dtypes(no_bool=True)
    def test_zoom_int(self, dtype):
        shape, zoom = self.shape_and_zoom
        if numpy.lib.NumpyVersion(scipy.__version__) < "1.0.0":
            if dtype in (numpy.dtype("l"), numpy.dtype("q")):
                dtype = numpy.int64
            elif dtype in (numpy.dtype("L"), numpy.dtype("Q")):
                dtype = numpy.uint64
        a = testing.shaped_random(shape, cupy, dtype)
        return self._zoom(a, zoom)


@testing.parameterize(
    *testing.product(
        {
            "operation": ["rotate", "shift", "zoom"],
            "mode": all_modes,
        }
    )
)
class TestBatchAllModesSmoke:
    def test_all_modes_last_axis_batch(self):
        a = testing.shaped_random((7, 8, 3), cupy, cupy.float32, scale=1)
        cval = 1.0

        if self.operation == "rotate":
            rotate = vendored_ndimage.rotate
            expected = cupy.stack(
                [
                    rotate(
                        a[..., i],
                        -15,
                        (1, 0),
                        False,
                        None,
                        1,
                        self.mode,
                        cval,
                        False,
                    )
                    for i in range(a.shape[-1])
                ],
                axis=-1,
            )
            result = rotate(
                a, -15, (1, 0), False, None, 1, self.mode, cval, False
            )
        elif self.operation == "shift":
            shift = vendored_ndimage.shift
            expected = cupy.stack(
                [
                    shift(
                        a[..., i],
                        (0.4, -0.3),
                        None,
                        1,
                        self.mode,
                        cval,
                        False,
                    )
                    for i in range(a.shape[-1])
                ],
                axis=-1,
            )
            result = shift(a, (0.4, -0.3, 0), None, 1, self.mode, cval, False)
        else:
            zoom = vendored_ndimage.zoom
            expected = cupy.stack(
                [
                    zoom(
                        a[..., i],
                        (1.2, 0.8),
                        None,
                        1,
                        self.mode,
                        cval,
                        False,
                    )
                    for i in range(a.shape[-1])
                ],
                axis=-1,
            )
            result = zoom(a, (1.2, 0.8, 1), None, 1, self.mode, cval, False)

        cupy.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "operation", ["map_coordinates", "shift", "zoom", "affine"]
)
@pytest.mark.parametrize("mode", all_modes)
def test_prefilter_boundary_modes_with_batch_axes(operation, mode):
    shape = (2, 5, 6, 3)
    a = cupy.arange(numpy.prod(shape), dtype=cupy.float32).reshape(shape)
    a = a / 10 + 0.125
    cval = -7.25

    if operation == "map_coordinates":
        coordinates = cupy.indices(shape, dtype=cupy.float32)
        coordinates[1] -= 1.2
        coordinates[2] += 0.85
        result = vendored_ndimage.map_coordinates(
            a,
            coordinates,
            order=3,
            mode=mode,
            cval=cval,
            prefilter=True,
            batch_axes=(0, 3),
        )
        expected = cupy.stack(
            [
                cupy.stack(
                    [
                        vendored_ndimage.map_coordinates(
                            a[n, :, :, c],
                            coordinates[1:3, n, :, :, c],
                            order=3,
                            mode=mode,
                            cval=cval,
                            prefilter=True,
                        )
                        for c in range(shape[-1])
                    ],
                    axis=-1,
                )
                for n in range(shape[0])
            ],
            axis=0,
        )
    elif operation == "shift":
        result = vendored_ndimage.shift(
            a,
            shift=(0, 1.2, -0.85, 0),
            order=3,
            mode=mode,
            cval=cval,
            prefilter=True,
        )
        expected = cupy.stack(
            [
                cupy.stack(
                    [
                        vendored_ndimage.shift(
                            a[n, :, :, c],
                            shift=(1.2, -0.85),
                            order=3,
                            mode=mode,
                            cval=cval,
                            prefilter=True,
                        )
                        for c in range(shape[-1])
                    ],
                    axis=-1,
                )
                for n in range(shape[0])
            ],
            axis=0,
        )
    elif operation == "zoom":
        result = vendored_ndimage.zoom(
            a,
            zoom=(1, 1.4, 0.7, 1),
            order=3,
            mode=mode,
            cval=cval,
            prefilter=True,
        )
        expected = cupy.stack(
            [
                cupy.stack(
                    [
                        vendored_ndimage.zoom(
                            a[n, :, :, c],
                            zoom=(1.4, 0.7),
                            order=3,
                            mode=mode,
                            cval=cval,
                            prefilter=True,
                        )
                        for c in range(shape[-1])
                    ],
                    axis=-1,
                )
                for n in range(shape[0])
            ],
            axis=0,
        )
    else:
        matrix = cupy.eye(a.ndim, dtype=cupy.float32)
        offset = (0, 1.2, -0.85, 0)
        result = vendored_ndimage.affine_transform(
            a,
            matrix,
            offset=offset,
            order=3,
            mode=mode,
            cval=cval,
            prefilter=True,
        )
        expected = cupy.stack(
            [
                cupy.stack(
                    [
                        vendored_ndimage.affine_transform(
                            a[n, :, :, c],
                            cupy.eye(2, dtype=cupy.float32),
                            offset=offset[1:3],
                            order=3,
                            mode=mode,
                            cval=cval,
                            prefilter=True,
                        )
                        for c in range(shape[-1])
                    ],
                    axis=-1,
                )
                for n in range(shape[0])
            ],
            axis=0,
        )

    cupy.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@testing.parameterize(
    *testing.product(
        {
            "matrix_kind": ["diagonal", "full"],
            "mode": ["constant", "nearest"],
        }
    )
)
@testing.with_requires("scipy")
class TestAffineTransformBatchOutputShape:
    def test_identity_batch_axis_output_larger_than_input(self):
        input_shape = (3, 4, 2)
        # Safe to request a different identity-axis extent: the optimized
        # batch kernel is only selected when the input/output extents match.
        output_shape = (3, 4, 5)
        cval = -7.0
        a_np = numpy.arange(numpy.prod(input_shape), dtype=numpy.float32)
        a_np = a_np.reshape(input_shape)
        a = cupy.asarray(a_np)

        if self.matrix_kind == "diagonal":
            matrix_np = numpy.ones(a_np.ndim, dtype=numpy.float32)
        else:
            matrix_np = numpy.eye(a_np.ndim, dtype=numpy.float32)
        matrix = cupy.asarray(matrix_np)

        result = vendored_ndimage.affine_transform(
            a,
            matrix,
            offset=0.0,
            output_shape=output_shape,
            order=1,
            mode=self.mode,
            cval=cval,
            prefilter=False,
        )
        expected = scipy.ndimage.affine_transform(
            a_np,
            matrix_np,
            offset=0.0,
            output_shape=output_shape,
            order=1,
            mode=self.mode,
            cval=cval,
            prefilter=False,
        )

        cupy.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@testing.with_requires("scipy")
def test_affine_transform_cross_term_from_last_axis_is_not_batch_axis():
    input_shape = (5, 6, 4)
    a_np = numpy.arange(numpy.prod(input_shape), dtype=numpy.float32)
    a_np = a_np.reshape(input_shape)
    a = cupy.asarray(a_np)

    matrix_np = numpy.asarray(
        [
            [1.0, 0.0, 0.25],
            [0.0, 0.9, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=numpy.float32,
    )
    matrix = cupy.asarray(matrix_np)
    offset = (0.1, -0.2, 0.0)

    result = vendored_ndimage.affine_transform(
        a,
        matrix,
        offset=offset,
        order=1,
        mode="nearest",
        prefilter=False,
    )
    expected = scipy.ndimage.affine_transform(
        a_np,
        matrix_np,
        offset=offset,
        order=1,
        mode="nearest",
        prefilter=False,
    )

    cupy.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_map_coordinates_batch_axes_use_same_spatial_map_for_all_channels():
    a = testing.shaped_random((5, 6, 3), cupy, cupy.float32, scale=1)
    coordinates = cupy.indices(a.shape, dtype=cupy.float32)
    coordinates[0] += 0.2
    coordinates[1] -= 0.3

    # These channel-varying spatial coordinate planes intentionally violate
    # the batch_axes contract. With batch_axes=(2,), the same spatial
    # transform is applied to all channels, so only channel 0's spatial
    # coordinate planes are used by the looped batch path.
    coordinates[0, :, :, 1] += 0.5
    coordinates[0, :, :, 2] += 1.0
    coordinates[1, :, :, 1] -= 0.25
    coordinates[1, :, :, 2] -= 0.5

    invariant_coordinates = coordinates.copy()
    for axis in (0, 1):
        invariant_coordinates[axis] = invariant_coordinates[axis, :, :, 0][
            :, :, cupy.newaxis
        ]

    result = vendored_ndimage.map_coordinates(
        a,
        coordinates,
        order=1,
        mode="nearest",
        prefilter=False,
        batch_axes=(2,),
    )
    expected = vendored_ndimage.map_coordinates(
        a,
        invariant_coordinates,
        order=1,
        mode="nearest",
        prefilter=False,
        batch_axes=(2,),
    )

    cupy.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_map_coordinates_batch_axes_negative_axis_is_normalized():
    a = testing.shaped_random((5, 6, 3), cupy, cupy.float32, scale=1)
    coordinates = cupy.indices(a.shape, dtype=cupy.float32)
    coordinates[0] += 0.2
    coordinates[1] -= 0.3

    result = vendored_ndimage.map_coordinates(
        a,
        coordinates,
        order=1,
        mode="nearest",
        prefilter=False,
        batch_axes=(-1,),
    )
    expected = vendored_ndimage.map_coordinates(
        a,
        coordinates,
        order=1,
        mode="nearest",
        prefilter=False,
        batch_axes=(2,),
    )

    cupy.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "batch_axes, match",
    [
        ((3,), "out of bounds"),
        ((-4,), "out of bounds"),
        ((2, -1), "unique"),
    ],
)
def test_map_coordinates_batch_axes_invalid_axes(batch_axes, match):
    a = testing.shaped_random((5, 6, 3), cupy, cupy.float32, scale=1)
    coordinates = cupy.indices(a.shape, dtype=cupy.float32)

    with pytest.raises((ValueError, AxisError), match=match):
        vendored_ndimage.map_coordinates(
            a,
            coordinates,
            order=1,
            mode="nearest",
            prefilter=False,
            batch_axes=batch_axes,
        )


def test_map_coordinates_batch_axes_require_matching_output_extent():
    a = testing.shaped_random((5, 6, 3), cupy, cupy.float32, scale=1)
    coordinates = cupy.indices((5, 6, 4), dtype=cupy.float32)

    with pytest.raises(ValueError, match="output shape must match input shape"):
        vendored_ndimage.map_coordinates(
            a,
            coordinates,
            order=1,
            mode="nearest",
            prefilter=False,
            batch_axes=(2,),
        )


@testing.parameterize(
    *testing.product(
        {
            "operation": [
                "map_coordinates",
                "shift",
                "zoom",
                "affine_diagonal",
                "affine_full",
            ],
        }
    )
)
class TestLoopBatchNonContiguousOutput:
    def _run_operation(self, a, output=None):
        if self.operation == "map_coordinates":
            coordinates = cupy.indices(a.shape, dtype=cupy.float32)
            coordinates[0] += 0.3
            coordinates[1] -= 0.2
            return vendored_ndimage.map_coordinates(
                a,
                coordinates,
                output=output,
                order=1,
                mode="nearest",
                prefilter=False,
                batch_axes=(2,),
            )
        if self.operation == "shift":
            return vendored_ndimage.shift(
                a,
                shift=(0.3, -0.2, 0),
                output=output,
                order=1,
                mode="nearest",
                prefilter=False,
            )
        if self.operation == "zoom":
            return vendored_ndimage.zoom(
                a,
                zoom=(1.2, 0.8, 1),
                output=output,
                order=1,
                mode="nearest",
                prefilter=False,
            )

        offset = (0.3, -0.2, 0)
        if self.operation == "affine_diagonal":
            matrix = cupy.ones(a.ndim, dtype=cupy.float32)
        else:
            matrix = cupy.eye(a.ndim, dtype=cupy.float32)
        return vendored_ndimage.affine_transform(
            a,
            matrix,
            offset=offset,
            output=output,
            order=1,
            mode="nearest",
            prefilter=False,
        )

    def test_loop_batch_falls_back_for_non_contiguous_output(self):
        a = testing.shaped_random((5, 6, 3), cupy, cupy.float32, scale=1)
        expected = self._run_operation(a)

        sentinel = cupy.asarray(-12345, dtype=expected.dtype)
        base_shape = (
            expected.shape[0],
            expected.shape[1] * 2,
            expected.shape[2],
        )
        base = cupy.full(base_shape, sentinel, dtype=expected.dtype)
        output = base[:, ::2, :]
        assert not output.flags.c_contiguous

        # The loop-batch kernel writes raw contiguous output, so strided output
        # views must fall back to the normal strided ElementwiseKernel output.
        result = self._run_operation(a, output=output)

        assert result is output
        cupy.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-5)
        cupy.testing.assert_array_equal(base[:, 1::2, :], sentinel)


class TestLoopBatchIntegerOutputRounding:
    def test_constant_cval_uses_rint_for_integer_output(self):
        a = testing.shaped_random((5, 6, 3), cupy, cupy.float32, scale=1)
        result = vendored_ndimage.shift(
            a,
            shift=(20, 0.1, 0),
            output=cupy.uint8,
            order=1,
            mode="constant",
            cval=1.6,
            prefilter=False,
        )

        expected = cupy.full(a.shape, 2, dtype=cupy.uint8)
        cupy.testing.assert_array_equal(result, expected)

    def test_grid_constant_cval_uses_rint_for_integer_output(self):
        a = testing.shaped_random((5, 6, 3), cupy, cupy.float32, scale=1)
        result = vendored_ndimage.shift(
            a,
            shift=(20, 0.1, 0),
            output=cupy.uint8,
            order=0,
            mode="grid-constant",
            cval=1.6,
            prefilter=False,
        )

        expected = cupy.full(a.shape, 2, dtype=cupy.uint8)
        cupy.testing.assert_array_equal(result, expected)

    def test_order0_samples_use_rint_for_integer_output(self):
        a = cupy.asarray(
            [
                [[1.6, 2.6, 3.6], [4.6, 5.6, 6.6]],
                [[7.6, 8.6, 9.6], [10.6, 11.6, 12.6]],
            ],
            dtype=cupy.float32,
        )
        result = vendored_ndimage.shift(
            a,
            shift=(0.2, -0.2, 0),
            output=cupy.uint8,
            order=0,
            mode="nearest",
            prefilter=False,
        )
        expected = cupy.stack(
            [
                vendored_ndimage.shift(
                    a[..., i],
                    shift=(0.2, -0.2),
                    output=cupy.uint8,
                    order=0,
                    mode="nearest",
                    prefilter=False,
                )
                for i in range(a.shape[-1])
            ],
            axis=-1,
        )

        cupy.testing.assert_array_equal(result, expected)
