import functools
import math
import numbers
import os


import cupy

try:
    # math.lcm was introduced in Python 3.9
    from math import lcm
except ImportError:

    """Fallback implementation of least common multiple (lcm)

    TODO: remove once minimum Python requirement is >= 3.9
    """

    def _lcm(a, b):
        return abs(b * (a // math.gcd(a, b)))

    @functools.lru_cache()
    def lcm(*args):
        nargs = len(args)
        if not all(isinstance(a, numbers.Integral) for a in args):
            raise TypeError("all arguments must be integers")
        if nargs == 0:
            return 1
        res = int(args[0])
        if nargs == 1:
            return abs(res)
        for i in range(1, nargs):
            x = int(args[i])
            res = _lcm(res, x)
        return res


pba2d_defines_template = """

// MARKER is used to mark blank pixels in the texture.
// Any uncolored pixels will have x = MARKER.
// Input texture should have x = MARKER for all pixels other than sites
#define MARKER      {marker}
#define BLOCKSIZE   {block_size_2d}
#define pixel_int2_t {pixel_int2_t}                // typically short2 (int2 for images with > 32k pixels per side)
#define make_pixel(x, y)  {make_pixel_func}(x, y)  // typically make_short2 (make_int2 images with > 32k pixels per side

"""  # noqa


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
def get_pba2d_src(block_size_2d=64, marker=-32768, pixel_int2_t='short2'):
    make_pixel_func = 'make_' + pixel_int2_t

    pba2d_code = pba2d_defines_template.format(
        block_size_2d=block_size_2d,
        marker=marker,
        pixel_int2_t=pixel_int2_t,
        make_pixel_func=make_pixel_func
    )
    kernel_directory = os.path.join(os.path.dirname(__file__), 'cuda')
    with open(os.path.join(kernel_directory, 'pba_kernels_2d.h'), 'rt') as f:
        pba2d_kernels = '\n'.join(f.readlines())

    pba2d_code += pba2d_kernels
    return pba2d_code


def _get_block_size(check_warp_size=False):
    if check_warp_size:
        dev = cupy.cuda.runtime.getDevice()
        device_properties = cupy.cuda.runtime.getDeviceProperties(dev)
        return int(device_properties['warpSize'])
    else:
        return 32


def _pack_int2(arr, marker=-32768, int_dtype=cupy.int16):
    if arr.ndim != 2:
        raise ValueError("only 2d arr suppported")
    input_x = cupy.zeros(arr.shape, dtype=int_dtype)
    input_y = cupy.zeros(arr.shape, dtype=int_dtype)
    # TODO: create custom kernel for setting values in input_x, input_y
    cond = arr == 0
    y, x = cupy.where(cond)
    input_x[cond] = x
    mask = arr != 0
    input_x[mask] = marker  # 1 << 32
    input_y[cond] = y
    input_y[mask] = marker  # 1 << 32
    int2_dtype = cupy.dtype({'names': ['x', 'y'], 'formats': [int_dtype] * 2})
    # in C++ code x is the contiguous axis and corresponds to width
    #             y is the non-contiguous axis and corresponds to height
    # given that, store input_x as the last axis here
    return cupy.squeeze(
        cupy.stack((input_x, input_y), axis=-1).view(int2_dtype)
    )


def _unpack_int2(img, make_copy=False, int_dtype=cupy.int16):
    temp = img.view(int_dtype).reshape(img.shape + (2,))
    if make_copy:
        temp = temp.copy()
    return temp


def _determine_padding(shape, padded_size, block_size):
    # all kernels assume equal size along both axes, so pad up to equal size if
    # shape is not isotropic
    orig_sy, orig_sx = shape
    if orig_sx != padded_size or orig_sy != padded_size:
        padding_width = (
            (0, padded_size - orig_sy), (0, padded_size - orig_sx)
        )
    else:
        padding_width = None
    return padding_width


def _pba_2d(arr, sampling=None, return_distances=True, return_indices=False,
            block_params=None, check_warp_size=False, *,
            float64_distances=False):

    # input_arr: a 2D image
    #    For each site at (x, y), the pixel at coordinate (x, y) should contain
    #    the pair (x, y). Pixels that are not sites should contain the pair
    #    (MARKER, MARKER)

    # Note: could query warp size here, but for now just assume 32 to avoid
    #       overhead of querying properties
    block_size = _get_block_size(check_warp_size)

    if sampling is not None:
        raise NotImplementedError("sampling not yet supported")
        # if len(sampling) != 2:
        #     raise ValueError("sampling must be a sequence of two values.")

    if block_params is None:
        padded_size = math.ceil(max(arr.shape) / block_size) * block_size

        # should be <= size / block_size. sy must be a multiple of m1
        m1 = padded_size // block_size
        # size must be a multiple of m2
        m2 = max(1, min(padded_size // block_size, block_size))
        # m2 must also be a power of two
        m2 = 2**math.floor(math.log2(m2))
        if padded_size % m2 != 0:
            raise RuntimeError("error in setting default m2")
        m3 = min(min(m1, m2), 2)
    else:
        if any(p < 1 for p in block_params):
            raise ValueError("(m1, m2, m3) in blockparams must be >= 1")
        m1, m2, m3 = block_params
        if math.log2(m2) % 1 > 1e-5:
            raise ValueError("m2 must be a power of 2")
        multiple = lcm(block_size, m1, m2, m3)
        padded_size = math.ceil(max(arr.shape) / multiple) * multiple

    if m1 > padded_size // block_size:
        raise ValueError(
            f"m1 too large. must be <= padded arr.shape[0] // {block_size}"
        )
    if m2 > padded_size // block_size:
        raise ValueError(
            f"m2 too large. must be <= padded arr.shape[1] // {block_size}"
        )
    if m3 > padded_size // block_size:
        raise ValueError(
            f"m3 too large. must be <= padded arr.shape[1] // {block_size}"
        )
    for m in (m1, m2, m3):
        if padded_size % m != 0:
            raise ValueError(
                f"Largest dimension of image ({padded_size}) must be evenly "
                f"disivible by each element of block_params: {(m1, m2, m3)}."
            )

    shape_max = max(arr.shape)
    if shape_max <= 32768:
        int_dtype = cupy.int16
        pixel_int2_type = 'short2'
    else:
        if shape_max > (1 << 24):
            # limit to coordinate range to 2**24 due to use of __mul24 in
            # coordinate TOID macro
            raise ValueError(
                f"maximum axis size of {1 << 24} exceeded, for image with "
                f"shape {arr.shape}"
            )
        int_dtype = cupy.int32
        pixel_int2_type = 'int2'

    marker = _init_marker(int_dtype)

    orig_sy, orig_sx = arr.shape
    padding_width = _determine_padding(arr.shape, padded_size, block_size)
    if padding_width is not None:
        arr = cupy.pad(arr, padding_width, mode='constant', constant_values=1)
    size = arr.shape[0]

    input_arr = _pack_int2(arr, marker=marker, int_dtype=int_dtype)
    output = cupy.zeros_like(input_arr)

    int2_dtype = cupy.dtype({'names': ['x', 'y'], 'formats': [int_dtype] * 2})
    margin = cupy.empty((2 * m1 * size,), dtype=int2_dtype)

    # phase 1 of PBA. m1 must divide texture size and be <= 64
    pba2d = cupy.RawModule(
        code=get_pba2d_src(
            block_size_2d=block_size,
            marker=marker,
            pixel_int2_t=pixel_int2_type,
        )
    )
    kernelFloodDown = pba2d.get_function('kernelFloodDown')
    kernelFloodUp = pba2d.get_function('kernelFloodUp')
    kernelPropagateInterband = pba2d.get_function('kernelPropagateInterband')
    kernelUpdateVertical = pba2d.get_function('kernelUpdateVertical')
    kernelProximatePoints = pba2d.get_function('kernelProximatePoints')
    kernelCreateForwardPointers = pba2d.get_function(
        'kernelCreateForwardPointers'
    )
    kernelMergeBands = pba2d.get_function('kernelMergeBands')
    kernelDoubleToSingleList = pba2d.get_function('kernelDoubleToSingleList')
    kernelColor = pba2d.get_function('kernelColor')

    block = (block_size, 1, 1)
    grid = (math.ceil(size / block[0]), m1, 1)
    bandSize1 = size // m1
    # kernelFloodDown modifies input_arr in-place
    kernelFloodDown(
        grid,
        block,
        (input_arr, input_arr, size, bandSize1),
    )
    # kernelFloodUp modifies input_arr in-place
    kernelFloodUp(
        grid,
        block,
        (input_arr, input_arr, size, bandSize1),
    )
    # kernelFloodUp fills values into margin
    kernelPropagateInterband(
        grid,
        block,
        (input_arr, margin, size, bandSize1),
    )
    # kernelUpdateVertical stores output into an intermediate array of
    # transposed shape
    kernelUpdateVertical(
        grid,
        block,
        (input_arr, margin, output, size, bandSize1),
    )

    # phase 2
    block = (block_size, 1, 1)
    grid = (math.ceil(size / block[0]), m2, 1)
    bandSize2 = size // m2
    kernelProximatePoints(
        grid,
        block,
        (output, input_arr, size, bandSize2),
    )
    kernelCreateForwardPointers(
        grid,
        block,
        (input_arr, input_arr, size, bandSize2),
    )
    # Repeatly merging two bands into one
    noBand = m2
    while noBand > 1:
        grid = (math.ceil(size / block[0]), noBand // 2)
        kernelMergeBands(
            grid,
            block,
            (output, input_arr, input_arr, size, size // noBand),
        )
        noBand //= 2
    # Replace the forward link with the X coordinate of the seed to remove
    # the need of looking at the other texture. We need it for coloring.
    grid = (math.ceil(size / block[0]), size)
    kernelDoubleToSingleList(
        grid,
        block,
        (output, input_arr, input_arr, size),
    )

    # Phase 3 of PBA
    block = (block_size, m3, 1)
    grid = (math.ceil(size / block[0]), 1, 1)
    kernelColor(
        grid,
        block,
        (input_arr, output, size),
    )

    output = _unpack_int2(output, make_copy=False, int_dtype=int_dtype)
    # make sure to crop any padding that was added here!
    x = output[:orig_sy, :orig_sx, 0]
    y = output[:orig_sy, :orig_sx, 1]

    # raise NotImplementedError("TODO")
    vals = ()
    if return_distances:
        # TODO: custom kernel for more efficient distance computation
        y0, x0 = cupy.meshgrid(
            *(
                cupy.arange(s, dtype=cupy.int32)
                for s in (orig_sy, orig_sx)
            ),
            indexing='ij',
            sparse=True,
        )
        tmp = (x - x0)
        dist = tmp * tmp
        tmp = (y - y0)
        dist += tmp * tmp
        if float64_distances:
            dist = cupy.sqrt(dist)
        else:
            dist = dist.astype(cupy.float32)
            cupy.sqrt(dist, out=dist)
        vals = vals + (dist,)
    if return_indices:
        indices = cupy.stack((y, x), axis=0)
        vals = vals + (indices,)
    return vals
