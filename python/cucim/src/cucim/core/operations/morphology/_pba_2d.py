# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import os

import cupy

from cucim.skimage._vendored import pad
from cucim.skimage._vendored._ndimage_util import _get_inttype

# An image is processed transposed if its first axis has at most
# _TRANSPOSE_MAX_SIZE entries and its second axis is at least _TRANSPOSE_RATIO
# times longer. For larger sizes there are enough threads in either orientation
# and reading the transposed views is slower.
_TRANSPOSE_MAX_SIZE = 1024
_TRANSPOSE_RATIO = 2

pba2d_defines_template = """

// MARKER is used to mark blank pixels in the texture.
// Any uncolored pixels will have x = MARKER.
// Input texture should have x = MARKER for all pixels other than sites
#define MARKER      {marker}
#define BLOCKSIZE   {block_size_2d}
#define pixel_int2_t {pixel_int2_t}                // typically short2 (int2 for images with > 32k pixels per side)
#define make_pixel(x, y)  {make_pixel_func}(x, y)  // typically make_short2 (make_int2 images with > 32k pixels per side

"""  # noqa: E501


def _init_marker(int_dtype):
    """use a minimum value that is appropriate to the integer dtype"""
    if int_dtype == cupy.int16:
        # marker = cupy.iinfo(int_dtype).min
        marker = -32768
    elif int_dtype == cupy.int32:
        # divide by two so we don't have to promote other intermediate int
        # variables to 64-bit int
        marker = -2147483648 // 2
    else:
        raise ValueError(
            "expected int_dtype to be either cupy.int16 or cupy.int32"
        )
    return marker


@cupy.memoize(True)
def get_pba2d_src(block_size_2d=64, marker=-32768, pixel_int2_t="short2"):
    make_pixel_func = "make_" + pixel_int2_t

    pba2d_code = pba2d_defines_template.format(
        block_size_2d=block_size_2d,
        marker=marker,
        pixel_int2_t=pixel_int2_t,
        make_pixel_func=make_pixel_func,
    )
    kernel_directory = os.path.join(os.path.dirname(__file__), "cuda")
    with open(os.path.join(kernel_directory, "pba_kernels_2d.h")) as f:
        pba2d_kernels = "\n".join(f.readlines())

    pba2d_code += pba2d_kernels
    return pba2d_code


def _get_block_size(check_warp_size=False):
    if check_warp_size:
        dev = cupy.cuda.runtime.getDevice()
        device_properties = cupy.cuda.runtime.getDeviceProperties(dev)
        return int(device_properties["warpSize"])
    else:
        return 32


@cupy.memoize(for_each_device=True)
def _get_pack_kernel(int_type, marker=-32768):
    """Pack coordinates into array of type short2 (or int2).

    This kernel works with 2D input data, `arr` (typically boolean).

    The output array, `out` will be 3D with a signed integer dtype.
    It will have size 2 on the last axis so that it can be viewed as a CUDA
    vector type such as `int2` or `float2`.
    """
    code = f"""
    if (arr[i]) {{
        out[2*i] = {marker};
        out[2*i + 1] = {marker};
    }} else {{
        int shape_1 = arr.shape()[1];
        int _i = i;
        int ind_1 = _i % shape_1;
        _i /= shape_1;
        out[2*i] = ind_1;   // out.x
        out[2*i + 1] = _i;  // out.y
    }}
    """
    return cupy.ElementwiseKernel(
        in_params="raw B arr",
        out_params="raw I out",
        operation=code,
    )


def _pack_int2(arr, marker=-32768, int_dtype=cupy.int16):
    if arr.ndim != 2:
        raise ValueError("only 2d arr supported")
    int2_dtype = cupy.dtype({"names": ["x", "y"], "formats": [int_dtype] * 2})
    out = cupy.zeros(arr.shape + (2,), dtype=int_dtype)
    assert out.size == 2 * arr.size
    pack_kernel = _get_pack_kernel(
        int_type="short" if int_dtype == cupy.int16 else "int", marker=marker
    )
    pack_kernel(arr, out, size=arr.size)
    out = cupy.squeeze(out.view(int2_dtype))
    return out


def _unpack_int2(img, make_copy=False, int_dtype=cupy.int16):
    temp = img.view(int_dtype).reshape(img.shape + (2,))
    if make_copy:
        temp = temp.copy()
    return temp


def _default_block_params(shape, block_size):
    """Default (m1, m2, m3) for an image of the given shape."""
    # number of block_size tiles along each axis, rounding up
    n_y = math.ceil(shape[0] / block_size)
    n_x = math.ceil(shape[1] / block_size)
    # m1: bands in phase 1 have block_size rows. This is applied along the
    # first axis.
    m1 = n_y
    # m2: number of bands for phase 2. This must be a power of two. It is
    # applied along the second axis.
    m2 = max(1, min(n_x, block_size))
    m2 = 2 ** math.floor(math.log2(m2))
    # m3: number of threads that share a column in phase 3. The best value
    # grows with the size, but a value above 16 is not faster, and it can not
    # be larger than the number of tiles along the second axis.
    m3 = min(16, 2 ** math.floor(math.log2(n_x)))
    return m1, m2, m3


def _determine_padding(shape, block_size, m1, m2):
    """Padding so that the kernels' bands and tiles divide each axis exactly.

    Each axis is padded separately: the kernels take a separate size for each
    axis, so the image does not have to be square.

    - The first axis is divided into `m1` bands that have to be a multiple of
      `block_size` rows (the size of the tiles used to transpose the image).
    - The second axis is divided into `m2` bands and has to be a multiple of
      `block_size` (the size of the tiles used to transpose the image).

    Returns the ``pad`` width for each axis or None if no padding is needed.
    """
    orig_sy, orig_sx = shape
    y_multiple = block_size * m1
    x_multiple = math.lcm(block_size, m2)
    sy = -(-orig_sy // y_multiple) * y_multiple
    sx = -(-orig_sx // x_multiple) * x_multiple
    if (sy, sx) == tuple(shape):
        return None
    return ((0, sy - orig_sy), (0, sx - orig_sx))


def _generate_shape(ndim, int_type, var_name="out", raw_var=True):
    code = ""
    if not raw_var:
        var_name = "_raw_" + var_name
    for i in range(ndim):
        code += f"{int_type} shape_{i} = {var_name}.shape()[{i}];\n"
    return code


def _generate_indices_ops(ndim, int_type):
    code = f"{int_type} _i = i;\n"
    for j in range(ndim - 1, 0, -1):
        code += f"{int_type} ind_{j} = _i % shape_{j};\n_i /= shape_{j};\n"
    code += f"{int_type} ind_0 = _i;"
    return code


def _guard_no_site(code, no_site_condition):
    """Make a distance kernel output infinity if there is no nearest site.

    This is the case for every pixel if the image does not contain a
    background pixel. The coordinates of the nearest site are then negative.
    """
    return (
        f"if ({no_site_condition}) {{\n"
        "    dist[i] = static_cast<F>(CUDART_INF);\n"
        "} else {\n"
        f"{code}\n"
        "}\n"
    )


def _get_distance_kernel_code(int_type, dist_int_type, raw_out_var=True):
    code = _generate_shape(
        ndim=2, int_type=int_type, var_name="dist", raw_var=raw_out_var
    )
    code += _generate_indices_ops(ndim=2, int_type=int_type)
    code += f"""
    {int_type} tmp;
    {dist_int_type} sq_dist;
    tmp = y[i] - ind_0;
    sq_dist = tmp * tmp;
    tmp = x[i] - ind_1;
    sq_dist += tmp * tmp;
    dist[i] = sqrt(static_cast<F>(sq_dist));
    """
    return _guard_no_site(code, "y[i] < 0 || x[i] < 0")


@cupy.memoize(for_each_device=True)
def _get_distance_kernel(int_type, dist_int_type):
    """Returns kernel computing the Euclidean distance from coordinates."""
    operation = _get_distance_kernel_code(
        int_type, dist_int_type, raw_out_var=True
    )
    return cupy.ElementwiseKernel(
        in_params="raw I y, raw I x",
        out_params="raw F dist",
        operation=operation,
    )


def _get_aniso_distance_kernel_code(int_type, raw_out_var=True):
    code = _generate_shape(
        ndim=2, int_type=int_type, var_name="dist", raw_var=raw_out_var
    )
    code += _generate_indices_ops(ndim=2, int_type=int_type)
    code += """
    F tmp;
    F sq_dist;
    tmp = static_cast<F>(y[i] - ind_0) * sampling[0];
    sq_dist = tmp * tmp;
    tmp = static_cast<F>(x[i] - ind_1) * sampling[1];
    sq_dist += tmp * tmp;
    dist[i] = sqrt(sq_dist);
    """
    return _guard_no_site(code, "y[i] < 0 || x[i] < 0")


@cupy.memoize(for_each_device=True)
def _get_aniso_distance_kernel(int_type):
    """Returns kernel computing the Euclidean distance from coordinates."""
    operation = _get_aniso_distance_kernel_code(int_type, raw_out_var=True)
    return cupy.ElementwiseKernel(
        in_params="raw I y, raw I x, raw F sampling",
        out_params="raw F dist",
        operation=operation,
    )


def _distance_tranform_arg_check(
    distances_out, indices_out, return_distances, return_indices
):
    """Raise a RuntimeError if the arguments are invalid"""
    error_msgs = []
    if (not return_distances) and (not return_indices):
        error_msgs.append(
            "at least one of return_distances/return_indices must be True"
        )
    if distances_out and not return_distances:
        error_msgs.append(
            "return_distances must be True if distances is supplied"
        )
    if indices_out and not return_indices:
        error_msgs.append("return_indices must be True if indices is supplied")
    if error_msgs:
        raise RuntimeError(", ".join(error_msgs))


def _check_distances(distances, shape, dtype):
    if distances.shape != shape:
        raise RuntimeError("distances array has wrong shape")
    if distances.dtype != dtype:
        raise RuntimeError(f"distances array must have dtype: {dtype}")


def _check_indices(indices, shape, itemsize):
    if indices.shape != shape:
        raise RuntimeError("indices array has wrong shape")
    if indices.dtype.kind not in "iu":
        raise RuntimeError("indices array must have an integer dtype")
    elif indices.dtype.itemsize < itemsize:
        raise RuntimeError(f"indices dtype must have itemsize > {itemsize}")


def _pba_2d(
    arr,
    sampling=None,
    return_distances=True,
    return_indices=False,
    block_params=None,
    check_warp_size=False,
    *,
    float64_distances=False,
    distances=None,
    indices=None,
):
    indices_inplace = isinstance(indices, cupy.ndarray)
    dt_inplace = isinstance(distances, cupy.ndarray)
    _distance_tranform_arg_check(
        dt_inplace, indices_inplace, return_distances, return_indices
    )

    # input_arr: a 2D image
    #    For each site at (x, y), the pixel at coordinate (x, y) should contain
    #    the pair (x, y). Pixels that are not sites should contain the pair
    #    (MARKER, MARKER)

    # Note: could query warp size here, but for now just assume 32 to avoid
    #       overhead of querying properties
    block_size = _get_block_size(check_warp_size)

    # Phases 2 and 3 have one thread block per block_size entries of the first
    # axis, and each thread walks along the whole second axis. Much less work
    # can be done in parallel if the first axis is short, so a short first axis
    # in an image that is much longer along the second axis is exchanged with
    # the second axis. The transposition is only a view: it is undone by
    # swapping the coordinates when reading the result. This is not done if the
    # user chose the block parameters since they refer to the axes of the
    # image. It is also not done if the indices are returned: the distances do
    # not depend on the order of the axes, but the choice between equidistant
    # background elements does.
    transposed = (
        block_params is None
        and not return_indices
        and arr.shape[0] <= _TRANSPOSE_MAX_SIZE
        and arr.shape[1] >= _TRANSPOSE_RATIO * arr.shape[0]
    )
    if transposed:
        arr = arr.T

    orig_sy, orig_sx = arr.shape
    if block_params is None:
        m1, m2, m3 = _default_block_params(arr.shape, block_size)
    else:
        if any(p < 1 for p in block_params):
            raise ValueError("(m1, m2, m3) in blockparams must be >= 1")
        m1, m2, m3 = block_params
        if math.log2(m2) % 1 > 1e-5:
            raise ValueError("m2 must be a power of 2")

    # the bands can't be smaller than block_size
    n_y = math.ceil(orig_sy / block_size)
    n_x = math.ceil(orig_sx / block_size)
    if m1 > n_y:
        raise ValueError(
            f"m1 too large. must be <= arr.shape[0] // {block_size} "
            "(rounded up)"
        )
    if m2 > n_x:
        raise ValueError(
            f"m2 too large. must be <= arr.shape[1] // {block_size} "
            "(rounded up)"
        )
    if m3 > n_x:
        raise ValueError(
            f"m3 too large. must be <= arr.shape[1] // {block_size} "
            "(rounded up)"
        )

    padding_width = _determine_padding(arr.shape, block_size, m1, m2)
    if padding_width is not None:
        arr = pad(arr, padding_width, mode="constant", constant_values=1)
    sy, sx = arr.shape
    if sy * sx >= 2**31:
        # the kernels index the array with 32-bit integers
        raise ValueError(
            f"the padded array shape {arr.shape} has too many elements: "
            "at most 2**31 - 1 are supported"
        )

    shape_max = max(sy, sx)
    if shape_max <= 32768:
        int_dtype = cupy.int16
        pixel_int2_type = "short2"
    else:
        if shape_max > (1 << 24):
            # limit to coordinate range to 2**24 due to use of __mul24 in
            # coordinate TOID macro
            raise ValueError(
                f"maximum axis size of {1 << 24} exceeded, for image with "
                f"shape {arr.shape}"
            )
        int_dtype = cupy.int32
        pixel_int2_type = "int2"

    marker = _init_marker(int_dtype)

    input_arr = _pack_int2(arr, marker=marker, int_dtype=int_dtype)
    output = cupy.zeros_like(input_arr)

    int2_dtype = cupy.dtype({"names": ["x", "y"], "formats": [int_dtype] * 2})
    margin = cupy.empty((2 * m1 * sx,), dtype=int2_dtype)

    # phase 1 of PBA. m1 must divide texture size and be <= 64
    pba2d = cupy.RawModule(
        code=get_pba2d_src(
            block_size_2d=block_size,
            marker=marker,
            pixel_int2_t=pixel_int2_type,
        )
    )
    kernelFloodDown = pba2d.get_function("kernelFloodDown")
    kernelFloodUp = pba2d.get_function("kernelFloodUp")
    kernelPropagateInterband = pba2d.get_function("kernelPropagateInterband")
    kernelUpdateVertical = pba2d.get_function("kernelUpdateVertical")
    kernelCreateForwardPointers = pba2d.get_function(
        "kernelCreateForwardPointers"
    )
    kernelDoubleToSingleList = pba2d.get_function("kernelDoubleToSingleList")

    if sampling is None:
        kernelProximatePoints = pba2d.get_function("kernelProximatePoints")
        kernelMergeBands = pba2d.get_function("kernelMergeBands")
        kernelColor = pba2d.get_function("kernelColor")
    else:
        kernelProximatePoints = pba2d.get_function(
            "kernelProximatePointsWithSpacing"
        )
        kernelMergeBands = pba2d.get_function("kernelMergeBandsWithSpacing")
        kernelColor = pba2d.get_function("kernelColorWithSpacing")

    # Phase 1 works along the first axis of the (sy, sx) image, in bands of
    # bandSize1 rows. Each launch gets the size of the image that it reads.
    block = (block_size, 1, 1)
    grid = (sx // block_size, m1, 1)
    bandSize1 = sy // m1
    # kernelFloodDown modifies input_arr in-place
    kernelFloodDown(grid, block, (input_arr, input_arr, sx, bandSize1))
    # kernelFloodUp modifies input_arr in-place
    kernelFloodUp(grid, block, (input_arr, input_arr, sx, bandSize1))
    # kernelFloodUp fills values into margin
    kernelPropagateInterband(
        grid, block, (input_arr, margin, sx, sy, bandSize1)
    )
    # kernelUpdateVertical stores output into an intermediate array of
    # transposed shape (sx, sy)
    kernelUpdateVertical(
        grid, block, (input_arr, margin, output, sx, sy, bandSize1)
    )

    # phase 2 works along the first axis of the transposed image (sx rows of
    # length sy)
    block = (block_size, 1, 1)
    grid = (sy // block_size, m2, 1)
    bandSize2 = sx // m2
    if sampling is None:
        sampling_args = ()
    else:
        # Originally the shape is (y, x) and sampling[1] corresponds to y.
        # However, kernelUpdateVertical transposed the image, so
        # we are now working with (x, y) instead. Need sampling ordered
        # accordingly.
        sampling = tuple(map(float, sampling))
        # reverse the order if the image was transposed
        kernel_sampling = sampling[::-1] if transposed else sampling
        sampling_args = (kernel_sampling[0], kernel_sampling[1])
    kernelProximatePoints(
        grid,
        block,
        (output, input_arr, sy, bandSize2) + sampling_args,
    )
    kernelCreateForwardPointers(
        grid,
        block,
        (input_arr, input_arr, sy, bandSize2),
    )
    # Repeatedly merging two bands into one
    noBand = m2
    while noBand > 1:
        grid = (sy // block_size, noBand // 2)
        kernelMergeBands(
            grid,
            block,
            (output, input_arr, input_arr, sy, sx // noBand) + sampling_args,
        )
        noBand //= 2
    # Replace the forward link with the X coordinate of the seed to remove
    # the need of looking at the other texture. We need it for coloring.
    grid = (sy // block_size, sx)
    kernelDoubleToSingleList(
        grid,
        block,
        (output, input_arr, input_arr, sy),
    )

    # Phase 3 of PBA: this transposes back to the original shape
    block = (block_size, m3, 1)
    grid = (sy // block_size, 1, 1)
    kernelColor(
        grid,
        block,
        (input_arr, output, sy, sx) + sampling_args,
    )

    output = _unpack_int2(output, make_copy=False, int_dtype=int_dtype)
    # make sure to crop any padding that was added here!
    x = output[:orig_sy, :orig_sx, 0]
    y = output[:orig_sy, :orig_sx, 1]
    if transposed:
        # the roles of the axes were exchanged: transpose to get views with
        # the shape of the input
        x, y = y.T, x.T

    vals = ()
    if return_distances:
        dtype_out = cupy.float64 if float64_distances else cupy.float32
        if dt_inplace:
            _check_distances(distances, y.shape, dtype_out)
        else:
            distances = cupy.zeros(y.shape, dtype=dtype_out)

        # make sure maximum possible distance doesn"t overflow
        max_possible_dist = sum((s - 1) ** 2 for s in y.shape)
        dist_int_type = "int" if max_possible_dist < 2**31 else "ptrdiff_t"

        if sampling is None:
            distance_kernel = _get_distance_kernel(
                int_type=_get_inttype(distances),
                dist_int_type=dist_int_type,
            )
            distance_kernel(y, x, distances, size=distances.size)
        else:
            distance_kernel = _get_aniso_distance_kernel(
                int_type=_get_inttype(distances),
            )
            sampling = cupy.asarray(sampling, dtype=dtype_out)
            distance_kernel(y, x, sampling, distances, size=distances.size)

        vals = vals + (distances,)
    if return_indices:
        if indices_inplace:
            # `arr` may have been padded, so use the shape of the cropped result
            _check_indices(indices, (arr.ndim,) + y.shape, x.dtype.itemsize)
            indices[0, ...] = y
            indices[1, ...] = x
        else:
            indices = cupy.stack((y, x), axis=0)
        vals = vals + (indices,)
    return vals
