# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import os

import cupy
import numpy as np

from cucim.skimage._vendored import pad
from cucim.skimage._vendored._ndimage_util import _get_inttype

from ._pba_2d import (
    _check_distances,
    _check_indices,
    _distance_tranform_arg_check,
    _generate_indices_ops,
    _generate_shape,
    _get_block_size,
    _guard_no_site,
)

pba3d_defines_template = """

#define MARKER     {marker}
#define MAX_INT    {max_int}
#define BLOCKSIZE  {block_size_3d}
#define SPACING_T  {spacing_t}  // type of the pixel spacing and of the distances computed with it
#define IMAGE_T    {image_t}  // type of the elements of the input image

"""

# For efficiency, the original PBA+ packs three 10-bit integers and two binary
# flags into a single 32-bit integer. Here, the number of bits of the
# coordinates follows from the shape of the array: `_get_encoding` returns
# the number of bits of the x and y coordinates (the same for both, because the
# coloring kernel exchanges the two), the number of bits of the z coordinate and
# whether a 64-bit integer is needed.
#
#   | NOTSITE | HASNEXT | ... | x (bits_xy) | y (bits_xy) | z (bits_z) |
#
# The flags are the highest two bits of the 32-bit or of the 64-bit integer.
pba3d_defines_encode = """
// Sites     : ENCODE(x, y, z, 0, 0)
// Not sites : ENCODE(0, 0, 0, 1, 0) or MARKER
#define ENCODED_INT_TYPE {int_type}
#define ZERO {zero}
#define ONE {one}
#define Z_BITS {bits_z}
#define XY_BITS {bits_xy}
#define NOTSITE_BIT {notsite_bit}
#define HASNEXT_BIT {hasnext_bit}
#define Z_MASK ((ONE << Z_BITS) - ONE)
#define XY_MASK ((ONE << XY_BITS) - ONE)
#define ENCODE(x, y, z, a, b)  (((x) << (XY_BITS + Z_BITS)) | ((y) << Z_BITS) | (z) | ((a) << NOTSITE_BIT) | ((b) << HASNEXT_BIT))
#define DECODE(value, x, y, z) \
    x = ((value) >> (XY_BITS + Z_BITS)) & XY_MASK; \
    y = ((value) >> Z_BITS) & XY_MASK; \
    z = (value) & Z_MASK

#define NOTSITE(value)  (((value) >> NOTSITE_BIT) & 1)
#define HASNEXT(value)  (((value) >> HASNEXT_BIT) & 1)

#define GET_X(value)    (((value) >> (XY_BITS + Z_BITS)) & XY_MASK)
#define GET_Y(value)    (((value) >> Z_BITS) & XY_MASK)
#define GET_Z(value)    ((NOTSITE((value))) ? MAX_INT : ((value) & Z_MASK))

"""  # noqa


def _get_encoding(shape):
    """Return (bits_xy, bits_z, wide) for packing coordinates in `shape`.

    `wide` is True if a 64-bit integer is needed (more than 30 bits for the
    coordinates), otherwise a 32-bit integer is used.
    """
    size_z, size_y, size_x = shape
    bits_xy = max(1, (max(size_y, size_x) - 1).bit_length())
    bits_z = max(1, (size_z - 1).bit_length())
    n_bits = 2 * bits_xy + bits_z
    if n_bits > 60:
        raise ValueError(f"the shape {tuple(shape)} is too large")
    return bits_xy, bits_z, n_bits > 30


def _get_encoded_dtype(encoding):
    return np.int64 if encoding[2] else np.int32


@cupy.memoize(True)
def get_pba3d_src(
    block_size_3d=32,
    marker=-2147483648,
    max_int=2147483647,
    encoding=(10, 10, False),
    spacing_t="double",
    image_t="unsigned char",
):
    pba3d_code = pba3d_defines_template.format(
        block_size_3d=block_size_3d,
        marker=marker,
        max_int=max_int,
        spacing_t=spacing_t,
        image_t=image_t,
    )
    bits_xy, bits_z, wide = encoding
    pba3d_code += pba3d_defines_encode.format(
        int_type="long long" if wide else "int",
        zero="0L" if wide else "0",
        one="1L" if wide else "1",
        bits_xy=bits_xy,
        bits_z=bits_z,
        notsite_bit=61 if wide else 31,
        hasnext_bit=60 if wide else 30,
    )
    kernel_directory = os.path.join(os.path.dirname(__file__), "cuda")
    with open(os.path.join(kernel_directory, "pba_kernels_3d.h")) as f:
        pba3d_kernels = "\n".join(f.readlines())
    pba3d_code += pba3d_kernels
    return pba3d_code


@cupy.memoize(for_each_device=True)
def _get_pba3d_kernels(
    block_size,
    encoding,
    with_spacing,
    spacing_t="double",
    image_t="unsigned char",
):
    """Return the kernels of the 3D algorithm as a dictionary.

    The kernels that take a pixel spacing are returned under the names of the
    kernels without a spacing.
    """
    module = cupy.RawModule(
        code=get_pba3d_src(
            block_size_3d=block_size,
            encoding=encoding,
            spacing_t=spacing_t,
            image_t=image_t,
        )
    )
    suffix = "WithSpacing" if with_spacing else ""
    kernels = {"kernelFloodZ": module.get_function("kernelFloodZ")}
    for name in ["kernelMaurerAxis", "kernelColorAxis"]:
        kernels[name] = module.get_function(name + suffix)
    return kernels


# C types of the dtypes that the kernels read directly. The kernels only test
# if an element is 0.
_image_ctypes = {
    "bool": "bool",
    "int8": "signed char",
    "uint8": "unsigned char",
    "int16": "short",
    "uint16": "unsigned short",
    "int32": "int",
    "uint32": "unsigned int",
    "int64": "long long",
    "uint64": "unsigned long long",
    "float32": "float",
    "float64": "double",
}


def _get_decode3d_code(encoding, int_type=""):
    # bit shifts here must match those of the ENCODE macro in the kernels
    bits_xy, bits_z, wide = encoding
    notsite_bit = 61 if wide else 31
    # The NOTSITE flag is set if there is no nearest site. The coordinates are
    # then set to -1 (otherwise they would be arbitrary values).
    return f"""
        {int_type} x = (encoded >> {bits_xy + bits_z}) & {(1 << bits_xy) - 1};
        {int_type} y = (encoded >> {bits_z}) & {(1 << bits_xy) - 1};
        {int_type} z = encoded & {(1 << bits_z) - 1};
        if ((encoded >> {notsite_bit}) & 1) {{ x = -1; y = -1; z = -1; }}
        """


@cupy.memoize(for_each_device=True)
def _get_decode3d_kernel(encoding):
    """Unpack 3 coordinates encoded as a single integer."""

    # int_type = "" here because x, y, z were already allocated externally
    code = _get_decode3d_code(encoding, int_type="")

    return cupy.ElementwiseKernel(
        in_params="E encoded",
        out_params="I x, I y, I z",
        operation=code,
    )


def decode3d(encoded, encoding):
    x = cupy.empty_like(encoded, dtype=cupy.int32)
    y = cupy.empty_like(x)
    z = cupy.empty_like(x)
    kern = _get_decode3d_kernel(encoding)
    kern(encoded, x, y, z)
    return (x, y, z)


def _determine_padding(shape, block_size, blockx, blocky):
    """Padding needed so that every kernel launch divides its axes exactly.

    The kernels take a separate extent for each axis, so each axis is padded
    only to the multiple that the launch configuration needs rather than to a
    common size:

    - x is covered by blocks of ``blockx`` threads and by tiles of
      ``block_size`` in the coloring kernel.
    - y is covered by blocks of ``blocky`` threads (flood pass), by tiles of
      ``block_size`` (coloring kernel) and, after the coloring kernel
      transposes y and x, by blocks of ``blockx`` threads.
    - z is covered by blocks of ``blocky`` threads (Maurer passes).

    Returns the ``pad`` width for each axis or None if no padding is needed.
    """
    orig_sz, orig_sy, orig_sx = shape
    x_multiple = math.lcm(block_size, blockx)
    y_multiple = math.lcm(block_size, blockx, blocky)
    sx = -(-orig_sx // x_multiple) * x_multiple
    sy = -(-orig_sy // y_multiple) * y_multiple
    sz = -(-orig_sz // blocky) * blocky
    if (sz, sy, sx) == tuple(shape):
        return None
    return ((0, sz - orig_sz), (0, sy - orig_sy), (0, sx - orig_sx))


def _generate_distance_computation(int_type, dist_int_type):
    """
    Compute euclidean distance from current coordinate (ind_0, ind_1, ind_2) to
    the coordinates of the nearest point (z, y, x)."""
    return f"""
    {int_type} tmp = z - ind_0;
    {dist_int_type} sq_dist = tmp * tmp;
    tmp = y - ind_1;
    sq_dist += tmp * tmp;
    tmp = x - ind_2;
    sq_dist += tmp * tmp;
    dist[i] = sqrt(static_cast<F>(sq_dist));
    """


def _get_distance_kernel_code(int_type, dist_int_type, raw_out_var=True):
    code = _generate_shape(
        ndim=3, int_type=int_type, var_name="dist", raw_var=raw_out_var
    )
    code += _generate_indices_ops(ndim=3, int_type=int_type)
    code += _generate_distance_computation(int_type, dist_int_type)
    return _guard_no_site(code, "x < 0")


@cupy.memoize(for_each_device=True)
def _get_distance_kernel(int_type, large_dist=False):
    """Returns kernel computing the Euclidean distance from coordinates."""
    dist_int_type = "ptrdiff_t" if large_dist else "int"
    operation = _get_distance_kernel_code(
        int_type, dist_int_type, raw_out_var=True
    )
    return cupy.ElementwiseKernel(
        in_params="I z, I y, I x",
        out_params="raw F dist",
        operation=operation,
    )


def _generate_aniso_distance_computation():
    """
    Compute euclidean distance from current coordinate (ind_0, ind_1, ind_2) to
    the coordinates of the nearest point (z, y, x)."""
    return """
    F tmp = static_cast<F>(z - ind_0) * sampling[0];
    F sq_dist = tmp * tmp;
    tmp = static_cast<F>(y - ind_1) * sampling[1];
    sq_dist += tmp * tmp;
    tmp = static_cast<F>(x - ind_2) * sampling[2];
    sq_dist += tmp * tmp;
    dist[i] = sqrt(static_cast<F>(sq_dist));
    """


def _get_aniso_distance_kernel_code(int_type, raw_out_var=True):
    code = _generate_shape(
        ndim=3, int_type=int_type, var_name="dist", raw_var=raw_out_var
    )
    code += _generate_indices_ops(ndim=3, int_type=int_type)
    code += _generate_aniso_distance_computation()
    return _guard_no_site(code, "x < 0")


@cupy.memoize(for_each_device=True)
def _get_aniso_distance_kernel(int_type):
    """Returns kernel computing the Euclidean distance from coordinates with
    axis spacing != 1."""
    operation = _get_aniso_distance_kernel_code(int_type, raw_out_var=True)
    return cupy.ElementwiseKernel(
        in_params="I z, I y, I x, raw F sampling",
        out_params="raw F dist",
        operation=operation,
    )


@cupy.memoize(for_each_device=True)
def _get_decode_as_distance_kernel(encoding, large_dist=False, sampling=None):
    """Fused decode3d and distance computation.

    This kernel is for use when `return_distances=True`, but
    `return_indices=False`. It replaces the separate calls to
    `_get_decode3d_kernel` and `_get_distance_kernel`, avoiding the overhead of
    generating full arrays containing the coordinates since the coordinate
    arrays are not going to be returned.
    """
    if sampling is None:
        dist_int_type = "ptrdiff_t" if large_dist else "int"
    int_type = "int"

    # Step 1: decode the (z, y, x) coordinate
    code = _get_decode3d_code(encoding, int_type=int_type)

    # Step 2: compute the Euclidean distance based on this (z, y, x).
    code_dist = _generate_shape(
        ndim=3, int_type=int_type, var_name="dist", raw_var=True
    )
    code_dist += _generate_indices_ops(ndim=3, int_type=int_type)
    if sampling is None:
        code_dist += _generate_distance_computation(int_type, dist_int_type)
        in_params = "E encoded"
    else:
        code_dist += _generate_aniso_distance_computation()
        in_params = "E encoded, raw F sampling"
    code += _guard_no_site(code_dist, "x < 0")
    return cupy.ElementwiseKernel(
        in_params=in_params,
        out_params="raw F dist",
        operation=code,
    )


def _pba_3d(
    arr,
    sampling=None,
    return_distances=True,
    return_indices=False,
    block_params=None,
    check_warp_size=False,
    *,
    float64_distances=False,
    float32_spacing=False,
    distances=None,
    indices=None,
):
    indices_inplace = isinstance(indices, cupy.ndarray)
    dt_inplace = isinstance(distances, cupy.ndarray)
    _distance_tranform_arg_check(
        dt_inplace, indices_inplace, return_distances, return_indices
    )

    if arr.ndim != 3:
        raise ValueError(f"expected a 3D array, got {arr.ndim}D")

    if block_params is None:
        m3 = 2
    else:
        # Only m3 is used. Unlike in 2D, there is no banding in phases 1 and 2
        # (m1 = m2 = 1), so the first two values are accepted for consistency
        # with the 2D API, but have no effect.
        _, _, m3 = block_params

    # reduce blockx for small inputs
    s_min = min(arr.shape)
    if s_min <= 4:
        blockx = 4
    elif s_min <= 8:
        blockx = 8
    elif s_min <= 16:
        blockx = 16
    else:
        blockx = 32
    blocky = 4

    block_size = _get_block_size(check_warp_size)

    if arr.dtype.name not in _image_ctypes:
        # other types (e.g. float16 or complex) are converted
        arr = arr != 0
    orig_sz, orig_sy, orig_sx = arr.shape
    padding_width = _determine_padding(arr.shape, block_size, blockx, blocky)
    if padding_width is not None:
        arr = pad(arr, padding_width, mode="constant", constant_values=1)
    sz, sy, sx = arr.shape
    if sz * sy * sx >= 2**31:
        # the kernels index the array with 32-bit integers
        raise ValueError(
            f"the padded array shape {arr.shape} has too many elements: "
            "at most 2**31 - 1 are supported"
        )

    # The coordinates are packed in 32-bit or 64-bit integers. The two arrays
    # are written before they are read: the first kernel reads the image and
    # writes the second array.
    encoding = _get_encoding(arr.shape)
    input_arr = cupy.empty(arr.shape, dtype=_get_encoded_dtype(encoding))
    buffer_idx = 0
    output = cupy.empty_like(input_arr)
    pba_images = [input_arr, output]
    # strides of the image in elements
    image_strides = tuple(np.int64(st // arr.itemsize) for st in arr.strides)

    block = (blockx, blocky, 1)
    spacing_t = "float" if float32_spacing else "double"
    kernels = _get_pba3d_kernels(
        block_size,
        encoding,
        sampling is not None,
        spacing_t,
        _image_ctypes[arr.dtype.name],
    )
    kernelFloodZ = kernels["kernelFloodZ"]
    kernelMaurerAxis = kernels["kernelMaurerAxis"]
    kernelColorAxis = kernels["kernelColorAxis"]
    if sampling is None:
        sampling_args = ()
    else:
        sampling = tuple(map(float, sampling))
        # the type of the arguments has to match the one in the kernels
        spacing_type = np.float32 if float32_spacing else np.float64
        sampling_args = tuple(spacing_type(sampling[i]) for i in (2, 1, 0))

    # Each launch is given the extents (nx, ny, nz) of the layout that it
    # reads. The coloring kernel writes its output with the y and x axes
    # exchanged, so the extents (and the spacing) are exchanged for the second
    # pair of launches.
    kernelFloodZ(
        (sx // blockx, sy // blocky, 1),
        block,
        (arr,) + image_strides + (pba_images[1 - buffer_idx], sx, sy, sz),
    )
    buffer_idx = 1 - buffer_idx

    kernelMaurerAxis(
        (sx // blockx, sz // blocky, 1),
        block,
        (pba_images[buffer_idx], pba_images[1 - buffer_idx], sx, sy, sz)
        + sampling_args,  # noqa
    )

    kernelColorAxis(
        (sx // block_size, sz, 1),
        (block_size, m3, 1),
        (pba_images[1 - buffer_idx], pba_images[buffer_idx], sx, sy, sz)
        + sampling_args,  # noqa
    )

    if sampling is not None:
        # kernelColorAxis transposes the first two axis, so have to reorder
        # the sampling_args tuple correspondingly
        sampling_args = tuple(spacing_type(sampling[i]) for i in (1, 2, 0))

    kernelMaurerAxis(
        (sy // blockx, sz // blocky, 1),
        block,
        (pba_images[buffer_idx], pba_images[1 - buffer_idx], sy, sx, sz)
        + sampling_args,  # noqa
    )

    kernelColorAxis(
        (sy // block_size, sz, 1),
        (block_size, m3, 1),
        (pba_images[1 - buffer_idx], pba_images[buffer_idx], sy, sx, sz)
        + sampling_args,  # noqa
    )
    output = pba_images[buffer_idx]

    if return_distances:
        out_shape = (orig_sz, orig_sy, orig_sx)
        dtype_out = cupy.float64 if float64_distances else cupy.float32
        if dt_inplace:
            _check_distances(distances, out_shape, dtype_out)
        else:
            distances = cupy.zeros(out_shape, dtype=dtype_out)

        # make sure maximum possible distance doesn't overflow
        max_possible_dist = sum((s - 1) ** 2 for s in out_shape)
        large_dist = max_possible_dist >= 2**31

        if not return_indices:
            # Compute distances without forming explicit coordinate arrays.
            kern = _get_decode_as_distance_kernel(
                encoding=encoding, large_dist=large_dist, sampling=sampling
            )
            if sampling is None:
                kern(output[:orig_sz, :orig_sy, :orig_sx], distances)
            else:
                sampling = cupy.asarray(sampling, dtype=distances.dtype)
                kern(output[:orig_sz, :orig_sy, :orig_sx], sampling, distances)
            return (distances,)

    if return_indices:
        x, y, z = decode3d(output[:orig_sz, :orig_sy, :orig_sx], encoding)
    vals = ()
    if return_distances:
        if sampling is None:
            kern = _get_distance_kernel(
                int_type=_get_inttype(distances),
                large_dist=large_dist,
            )
            kern(z, y, x, distances)
        else:
            kern = _get_aniso_distance_kernel(int_type=_get_inttype(distances))
            sampling = cupy.asarray(sampling, dtype=distances.dtype)
            kern(z, y, x, sampling, distances)
        vals = vals + (distances,)
    if return_indices:
        if indices_inplace:
            # `arr` may have been padded, so use the shape of the cropped result
            _check_indices(indices, (arr.ndim,) + x.shape, x.dtype.itemsize)
            indices[0, ...] = z
            indices[1, ...] = y
            indices[2, ...] = x
        else:
            indices = cupy.stack((z, y, x), axis=0)
        vals = vals + (indices,)
    return vals
