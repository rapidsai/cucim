import math
import os

import cupy
import numpy as np

from ._pba_2d import _get_block_size, lcm


pba3d_defines_template = """

#define MARKER     {marker}
#define MAX_INT    {max_int}
#define BLOCKSIZE  {block_size_3d}

"""

# For efficiency, the original PBA+ packs three 10-bit integers and two binary
# flags into a single 32-bit integer. The defines in
# `pba3d_defines_encode_32bit` handle this format.
pba3d_defines_encode_32bit = """
// Sites     : ENCODE(x, y, z, 0, 0)
// Not sites : ENCODE(0, 0, 0, 1, 0) or MARKER
#define ENCODED_INT_TYPE int
#define ZERO 0
#define ONE 1
#define ENCODE(x, y, z, a, b)  (((x) << 20) | ((y) << 10) | (z) | ((a) << 31) | ((b) << 30))
#define DECODE(value, x, y, z) \
    x = ((value) >> 20) & 0x3ff; \
    y = ((value) >> 10) & 0x3ff; \
    z = (value) & 0x3ff

#define NOTSITE(value)  (((value) >> 31) & 1)
#define HASNEXT(value)  (((value) >> 30) & 1)

#define GET_X(value)    (((value) >> 20) & 0x3ff)
#define GET_Y(value)    (((value) >> 10) & 0x3ff)
#define GET_Z(value)    ((NOTSITE((value))) ? MAX_INT : ((value) & 0x3ff))

""" # noqa


# 64bit version of ENCODE/DECODE to allow a 20-bit integer per coordinate axis.
pba3d_defines_encode_64bit = """
// Sites     : ENCODE(x, y, z, 0, 0)
// Not sites : ENCODE(0, 0, 0, 1, 0) or MARKER
#define ENCODED_INT_TYPE long long
#define ZERO 0L
#define ONE 1L
#define ENCODE(x, y, z, a, b)  (((x) << 40) | ((y) << 20) | (z) | ((a) << 61) | ((b) << 60))
#define DECODE(value, x, y, z) \
    x = ((value) >> 40) & 0xfffff; \
    y = ((value) >> 20) & 0xfffff; \
    z = (value) & 0xfffff

#define NOTSITE(value)  (((value) >> 61) & 1)
#define HASNEXT(value)  (((value) >> 60) & 1)

#define GET_X(value)    (((value) >> 40) & 0xfffff)
#define GET_Y(value)    (((value) >> 20) & 0xfffff)
#define GET_Z(value)    ((NOTSITE((value))) ? MAX_INT : ((value) & 0xfffff))

""" # noqa


@cupy.memoize(True)
def get_pba3d_src(block_size_3d=32, marker=-2147483648, max_int=2147483647,
                  size_max=1024):
    pba3d_code = pba3d_defines_template.format(
        block_size_3d=block_size_3d, marker=marker, max_int=max_int
    )
    if size_max > 1024:
        pba3d_code += pba3d_defines_encode_64bit
    else:
        pba3d_code += pba3d_defines_encode_32bit
    kernel_directory = os.path.join(os.path.dirname(__file__), 'cuda')
    with open(os.path.join(kernel_directory, 'pba_kernels_3d.h'), 'rt') as f:
        pba3d_kernels = '\n'.join(f.readlines())
    pba3d_code += pba3d_kernels
    return pba3d_code


# TODO: custom kernel for encode3d
def encode3d(arr, marker=-2147483648, bit_depth=32, size_max=1024):
    if arr.ndim != 3:
        raise ValueError("only 3d arr suppported")
    if bit_depth not in [32, 64]:
        raise ValueError("only bit_depth of 32 or 64 is supported")
    if size_max > 1024:
        dtype = np.int64
    else:
        dtype = np.int32
    image = cupy.zeros(arr.shape, dtype=dtype, order='C')
    cond = arr == 0
    z, y, x = cupy.where(cond)
    # z, y, x so that x is the contiguous axis
    # (must match TOID macro in the C++ code!)
    if size_max > 1024:
        image[cond] = (((x) << 40) | ((y) << 20) | (z))
    else:
        image[cond] = (((x) << 20) | ((y) << 10) | (z))
    image[arr != 0] = marker  # 1 << 32
    return image


# TODO: custom kernel for decode3d
def decode3d(output, size_max=1024):
    if size_max > 1024:
        x = (output >> 40) & 0xfffff
        y = (output >> 20) & 0xfffff
        z = output & 0xfffff
    else:
        x = (output >> 20) & 0x3ff
        y = (output >> 10) & 0x3ff
        z = output & 0x3ff
    return (x, y, z)


def _determine_padding(shape, block_size, m1, m2, m3, blockx, blocky):
    # TODO: can possibly revise to consider only particular factors for LCM on
    #       a given axis
    LCM = lcm(block_size, m1, m2, m3, blockx, blocky)
    orig_sz, orig_sy, orig_sx = shape
    round_up = False
    if orig_sx % LCM != 0:
        # round up size to a multiple of the band size
        round_up = True
        sx = LCM * math.ceil(orig_sx / LCM)
    else:
        sx = orig_sx
    if orig_sy % LCM != 0:
        # round up size to a multiple of the band size
        round_up = True
        sy = LCM * math.ceil(orig_sy / LCM)
    else:
        sy = orig_sy
    if orig_sz % LCM != 0:
        # round up size to a multiple of the band size
        round_up = True
        sz = LCM * math.ceil(orig_sz / LCM)
    else:
        sz = orig_sz

    aniso = not (sx == sy == sz)
    if aniso or round_up:
        smax = max(sz, sy, sx)
        padding_width = (
            (0, smax - orig_sz), (0, smax - orig_sy), (0, smax - orig_sx)
        )
    else:
        padding_width = None
    return padding_width


def _pba_3d(arr, sampling=None, return_distances=True, return_indices=False,
            block_params=None, check_warp_size=False, *,
            float64_distances=False):
    if arr.ndim != 3:
        raise ValueError(f"expected a 3D array, got {arr.ndim}D")

    if sampling is not None:
        raise NotImplementedError("sampling not yet supported")
        # if len(sampling) != 3:
        #     raise ValueError("sampling must be a sequence of three values.")

    if block_params is None:
        m1 = 1
        m2 = 1
        m3 = 2
    else:
        m1, m2, m3 = block_params

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

    orig_sz, orig_sy, orig_sx = arr.shape
    padding_width = _determine_padding(
        arr.shape, block_size, m1, m2, m3, blockx, blocky
    )
    if padding_width is not None:
        arr = cupy.pad(arr, padding_width, mode='constant', constant_values=1)
    size = arr.shape[0]

    # pba algorithm was implemented to use 32-bit integer to store compressed
    # coordinates. input_arr will be C-contiguous, int32
    size_max = max(arr.shape)
    input_arr = encode3d(arr, size_max=size_max)
    buffer_idx = 0
    output = cupy.zeros_like(input_arr)
    pba_images = [input_arr, output]

    block = (blockx, blocky, 1)
    grid = (size // block[0], size // block[1], 1)
    pba3d = cupy.RawModule(
        code=get_pba3d_src(block_size_3d=block_size, size_max=size_max)
    )

    kernelFloodZ = pba3d.get_function('kernelFloodZ')
    kernelMaurerAxis = pba3d.get_function('kernelMaurerAxis')
    kernelColorAxis = pba3d.get_function('kernelColorAxis')

    kernelFloodZ(
        grid,
        block,
        (pba_images[buffer_idx], pba_images[1 - buffer_idx], size)
    )
    buffer_idx = 1 - buffer_idx

    block = (blockx, blocky, 1)
    grid = (size // block[0], size // block[1], 1)
    kernelMaurerAxis(
        grid,
        block,
        (pba_images[buffer_idx], pba_images[1 - buffer_idx], size),
    )

    block = (block_size, m3, 1)
    grid = (size // block[0], size, 1)
    kernelColorAxis(
        grid,
        block,
        (pba_images[1 - buffer_idx], pba_images[buffer_idx], size),
    )

    block = (blockx, blocky, 1)
    grid = (size // block[0], size // block[1], 1)
    kernelMaurerAxis(
        grid,
        block,
        (pba_images[buffer_idx], pba_images[1 - buffer_idx], size),
    )

    block = (block_size, m3, 1)
    grid = (size // block[0], size, 1)
    kernelColorAxis(
        grid,
        block,
        (pba_images[1 - buffer_idx], pba_images[buffer_idx], size),
    )

    output = pba_images[buffer_idx]
    if return_distances or return_indices:
        x, y, z = decode3d(output[:orig_sz, :orig_sy, :orig_sx],
                           size_max=size_max)

    vals = ()
    if return_distances:
        # TODO: custom kernel for more efficient distance computation
        orig_shape = (orig_sz, orig_sy, orig_sx)
        z0, y0, x0 = cupy.meshgrid(
            *(cupy.arange(s, dtype=cupy.int32) for s in orig_shape),
            indexing='ij',
            sparse=True
        )
        tmp = (x - x0)
        dist = tmp * tmp
        tmp = (y - y0)
        dist += tmp * tmp
        tmp = (z - z0)
        dist += tmp * tmp
        if float64_distances:
            dist = cupy.sqrt(dist)
        else:
            dist = dist.astype(cupy.float32)
            cupy.sqrt(dist, out=dist)
        vals = vals + (dist,)
    if return_indices:
        indices = cupy.stack((z, y, x), axis=0)
        vals = vals + (indices,)
    return vals
