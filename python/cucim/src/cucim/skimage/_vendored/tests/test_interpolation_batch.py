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
from cucim.skimage._vendored._ndimage_interp_kernels import (
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
                "order": [0, 1, 3],
                "mode": legacy_modes + scipy16_modes,
                "cval": [1.0],
                "prefilter": [False, True],
            }
        )
    )
)
@testing.with_requires("scipy")
class TestRotateBatch:
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
                "order": [0, 1, 3],
                "mode": legacy_modes + scipy16_modes,
                "cval": [1.0],
                "prefilter": [False, True],
            }
        )
    )
)
class TestShiftBatch:
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
            "order": [0, 1, 3],
            "mode": legacy_modes + scipy16_modes,
            "cval": [1.0],
            "prefilter": [False, True],
        }
    )
)
@testing.with_requires("scipy")
class TestZoomBatch:
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
