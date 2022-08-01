import math

import cupy as cp

from cucim.skimage._vendored import _ndimage_util as util
from cucim.skimage._vendored._internal import _normalize_axis_index, prod
from cucim.skimage._vendored._ndimage_filters_core import (
    _ndimage_CAST_FUNCTION, _ndimage_includes)


def _get_constants(ndim, axis, kernel_size, anchor, patch_per_block=None):
    if anchor is None:
        anchor = kernel_size // 2
    halo_pixels_needed = max(kernel_size - anchor, anchor)
    if patch_per_block is None:
        patch_per_block = 4

    if ndim == 2:
        # note, in 2d axis 0 = "y"
        #             axis 1 = "x"
        # for simplicity, keeping same halo size at both start and end
        if axis == 1:
            # as in OpenCV's column_filter.hpp
            block_x = 16
            block_y = 16
            halo_size = math.ceil(halo_pixels_needed / block_x)
        elif axis == 0:
            # as in OpenCV's row_filter.hpp
            block_x = 32  # 16 in CUDA example
            block_y = 8  # 4 in CUDA example
            halo_size = math.ceil(halo_pixels_needed / block_y)
        # can have out of bounds access unless patch_per_block >= halo_size
        patch_per_block = max(patch_per_block, halo_size)
        block_z = 1
    elif ndim == 3:
        # note, in 3d axis 0 = "z"
        #             axis 1 = "y"
        #             axis 2 = "x"
        # for simplicity, keeping same halo size at both start and end
        if axis == 2:
            # as in OpenCV's column_filter.hpp
            block_x = 16
            block_y = 4
            block_z = 4
            halo_size = math.ceil(halo_pixels_needed / block_x)
        elif axis == 1:
            # as in OpenCV's column_filter.hpp
            block_x = 32
            block_y = 4
            block_z = 4
            halo_size = math.ceil(halo_pixels_needed / block_y)
        elif axis == 0:
            # as in OpenCV's row_filter.hpp
            block_x = 32
            block_y = 4
            block_z = 4
            halo_size = math.ceil(halo_pixels_needed / block_z)
        # can have out of bounds access unless patch_per_block >= halo_size
        patch_per_block = max(patch_per_block, halo_size)
    else:
        raise NotImplementedError("Only 2D and 3D are currently supported")
    block = (block_x, block_y, block_z)
    return block, patch_per_block, halo_size


def _get_smem_shape(ndim, axis, block, patch_per_block, halo_size, anchor=None,
                    image_dtype=cp.float32):
    bx, by, bz = block
    if ndim == 2:
        if axis == 0:
            shape = ((patch_per_block + 2 * halo_size) * by, bx)
        elif axis == 1:
            shape = (by, (patch_per_block + 2 * halo_size) * bx)
    elif ndim == 3:
        if axis == 0:
            shape = ((patch_per_block + 2 * halo_size) * bz, by, bx)
        elif axis == 1:
            shape = (bz, (patch_per_block + 2 * halo_size) * by, bx)
        elif axis == 2:
            shape = (bz, by, (patch_per_block + 2 * halo_size) * bx)
    else:
        raise NotImplementedError("TODO")
    nbytes = cp.dtype(image_dtype).itemsize * prod(shape)
    return shape, nbytes


def _get_warp_size(device_id=None):
    if device_id is None:
        device_id = cp.cuda.runtime.getDevice()
    device_props = cp.cuda.runtime.getDeviceProperties(device_id)
    return device_props['warpSize']


def _get_shmem_limits(device_id=None):
    if device_id is None:
        device_id = cp.cuda.runtime.getDevice()
    device_props = cp.cuda.runtime.getDeviceProperties(device_id)
    shared_mp = device_props.get('sharedMemPerMultiprocessor', None)
    shared_block = device_props.get('sharedMemPerBlock', None)
    shared_block_optin = device_props.get('sharedMemPerBlockOptin', None)
    global_l1_cache_supported = device_props.get('globalL1CacheSupported', None)
    local_l1_cache_supported = device_props.get('localL1CacheSupported', None)
    l2_size = device_props.get('l2CacheSize', None)
    warp_size = device_props.get('warpSize', None)
    regs_per_block = device_props.get('regsPerBlock', None)
    return {
        'device_id': device_id,
        'shared_mp': shared_mp,
        'shared_block': shared_block,
        'shared_block_optin': shared_block_optin,
        'global_l1_cache_supported': global_l1_cache_supported,
        'local_l1_cache_supported': local_l1_cache_supported,
        'l2_size': l2_size,
        'warp_size': warp_size,
        'regs_per_block': regs_per_block,
    }


class ResourceLimitError(RuntimeError):
    pass


@cp.memoize(for_each_device=True)
def _check_smem_availability(ndim, axis, kernel_size, anchor=None,
                             patch_per_block=None, image_dtype=cp.float32,
                             device_id=None):
    block, patch_per_block, halo_size = _get_constants(
        ndim, axis, kernel_size, anchor=anchor, patch_per_block=patch_per_block
    )
    shape, nbytes = _get_smem_shape(
        ndim, axis, block, patch_per_block, halo_size, image_dtype
    )
    props = _get_shmem_limits(device_id=device_id)
    if nbytes > props['shared_block']:
        raise ResourceLimitError("inadequate shared memory available")


_dtype_char_to_c_types = {
    'e': 'float16',
    'f': 'float',
    'd': 'double',
    'F': 'complex<float>',
    'D': 'complex<double>',
    '?': 'char',
    'b': 'char',
    'h': 'short',
    'i': 'int',
    'l': 'long long',
    'B': 'unsigned char',
    'H': 'unsigned short',
    'I': 'unsigned int',
    'L': 'unsigned long long',
}


def _get_code_stage1_shared_memory_load_2d(ndim, axis, mode, cval):
    """Generates the first stage of the function body.

    This involves just copying from the `src` array into the `smem` shared
    memory array followed by a call to __syncthreads(). All boundary
    handling also occurs within this function.
    """

    if ndim == 2 and axis == 0:
        if mode not in ['constant', 'grid-constant']:
            boundary_code_lower, boundary_code_upper = util._generate_boundary_condition_ops(mode, 'row', 'n_rows', separate=True)  # noqa

        # as in OpenCV's column_filter.hpp
        code = """
        __shared__ T smem[(PATCH_PER_BLOCK + 2 * HALO_SIZE) * BLOCK_DIM_Y][BLOCK_DIM_X];
        const int x = blockIdx.x * BLOCK_DIM_X + threadIdx.x;
        if (x >= n_cols){
            return;
        }
        const T* src_col = &src[x];
        const int yStart = blockIdx.y * (BLOCK_DIM_Y * PATCH_PER_BLOCK) + threadIdx.y;

        // memory is contiguous along last (columns) axis
        const int row_stride = n_cols;  // stride (in elements) along axis 0
        int row;

        if (blockIdx.y > 0)
        {
            //Upper halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j)
                smem[threadIdx.y + j * BLOCK_DIM_Y][threadIdx.x] = src_col[(yStart - (HALO_SIZE - j) * BLOCK_DIM_Y) * row_stride];
        }
        else
        {
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j) {
                row = yStart - (HALO_SIZE - j) * BLOCK_DIM_Y;
        """  # noqa
        if mode == 'constant':
            code += f"""
                if (row < 0)
                    smem[threadIdx.y + j * BLOCK_DIM_Y][threadIdx.x] = static_cast<T>({cval});
                else
            """  # noqa
        else:
            code += boundary_code_lower
        code += """
                    smem[threadIdx.y + j * BLOCK_DIM_Y][threadIdx.x] = src_col[row * row_stride];
            }
        }

        if (blockIdx.y + 2 < gridDim.y)  // Note: +2 here assumes HALO_SIZE <= PATCH_PER_BLOCK so we ensure that elsewhere
        {
            //Main data
            #pragma unroll
            for (int j = 0; j < PATCH_PER_BLOCK; ++j)
                smem[threadIdx.y + (HALO_SIZE + j) * BLOCK_DIM_Y][threadIdx.x] = src_col[(yStart + j * BLOCK_DIM_Y) * row_stride];

            //Lower halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j)
                smem[threadIdx.y + (PATCH_PER_BLOCK + HALO_SIZE + j) * BLOCK_DIM_Y][threadIdx.x] = src_col[(yStart + (PATCH_PER_BLOCK + j) * BLOCK_DIM_Y) * row_stride];
        }
        else
        {
            //Main data
            #pragma unroll
            for (int j = 0; j < PATCH_PER_BLOCK; ++j) {
                row = yStart + j * BLOCK_DIM_Y;
        """  # noqa
        if mode == 'constant':
            code += f"""
                if (row >= n_rows)
                    smem[threadIdx.y + (HALO_SIZE + j) * BLOCK_DIM_Y][threadIdx.x] = static_cast<T>({cval});
                else
            """  # noqa
        else:
            code += boundary_code_upper
        code += """
                    smem[threadIdx.y + (HALO_SIZE + j) * BLOCK_DIM_Y][threadIdx.x] = src_col[row * row_stride];
            }

            //Lower halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j)
            {
                row = yStart + (PATCH_PER_BLOCK + j) * BLOCK_DIM_Y;
        """  # noqa
        if mode == 'constant':
            code += f"""
                if (row >= n_rows)
                    smem[threadIdx.y + (PATCH_PER_BLOCK + HALO_SIZE + j) * BLOCK_DIM_Y][threadIdx.x] = static_cast<T>({cval});
                else
            """  # noqa
        else:
            code += boundary_code_upper
        code += """
                    smem[threadIdx.y + (PATCH_PER_BLOCK + HALO_SIZE + j) * BLOCK_DIM_Y][threadIdx.x] = src_col[row * row_stride];
            }
        }
        """  # noqa
    elif ndim == 2 and axis == 1:
        if mode not in ['constant', 'grid-constant']:
            boundary_code_lower, boundary_code_upper = util._generate_boundary_condition_ops(mode, 'col', 'n_cols', separate=True)  # noqa

        # as in OpenCV's row_filter.hpp
        code = """
        __shared__ T smem[BLOCK_DIM_Y][(PATCH_PER_BLOCK + 2 * HALO_SIZE) * BLOCK_DIM_X];
        const int y = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;
        if (y >= n_rows) {
            return;
        }
        const int row_stride = n_cols;  // stride (in elements) along axis 0
        int col;
        const T* src_row = &src[y * row_stride];
        const int xStart = blockIdx.x * (PATCH_PER_BLOCK * BLOCK_DIM_X) + threadIdx.x;

        if (blockIdx.x > 0)
        {
            //Load left halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j)
                smem[threadIdx.y][threadIdx.x + j * BLOCK_DIM_X] = src_row[xStart - (HALO_SIZE - j) * BLOCK_DIM_X];
        }
        else
        {
            //Load left halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j){
                col = xStart - (HALO_SIZE - j) * BLOCK_DIM_X;
        """  # noqa
        if mode == 'constant':
            code += f"""
                if (col < 0)
                    smem[threadIdx.y][threadIdx.x + j * BLOCK_DIM_X] = static_cast<T>({cval});
                else
            """  # noqa
        else:
            code += boundary_code_lower
        code += """
                    smem[threadIdx.y][threadIdx.x + j * BLOCK_DIM_X] = src_row[col];
            }
        }
        if (blockIdx.x + 2 < gridDim.x)  // Note: +2 here assumes HALO_SIZE <= PATCH_PER_BLOCK so we ensure that elsewhere
        {
            //Load main data
            #pragma unroll
            for (int j = 0; j < PATCH_PER_BLOCK; ++j)
                smem[threadIdx.y][threadIdx.x + (HALO_SIZE + j)* BLOCK_DIM_X] = src_row[xStart + j * BLOCK_DIM_X];

            //Load right halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j)
                smem[threadIdx.y][threadIdx.x + (PATCH_PER_BLOCK + HALO_SIZE + j) * BLOCK_DIM_X] = src_row[xStart + (PATCH_PER_BLOCK + j) * BLOCK_DIM_X];
        }
        else
        {
            //Load main data
            #pragma unroll
            for (int j = 0; j < PATCH_PER_BLOCK; ++j) {
                col = xStart + j * BLOCK_DIM_X;
        """  # noqa
        if mode == 'constant':
            code += f"""
                if (col >= n_cols)
                    smem[threadIdx.y][threadIdx.x + (HALO_SIZE + j) * BLOCK_DIM_X] = static_cast<T>({cval});
                else
            """  # noqa
        else:
            code += boundary_code_upper
        code += """
                    smem[threadIdx.y][threadIdx.x + (HALO_SIZE + j) * BLOCK_DIM_X] = src_row[col];
            }

            //Load right halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j){
                col = xStart + (PATCH_PER_BLOCK + j) * BLOCK_DIM_X;
        """  # noqa
        if mode == 'constant':
            code += f"""
                if (col >= n_cols)
                    smem[threadIdx.y][threadIdx.x + (PATCH_PER_BLOCK + HALO_SIZE + j) * BLOCK_DIM_X] = static_cast<T>({cval});
                else
            """  # noqa
        else:
            code += boundary_code_upper
        code += """
                    smem[threadIdx.y][threadIdx.x + (PATCH_PER_BLOCK + HALO_SIZE + j) * BLOCK_DIM_X] = src_row[col];
            }
        }
        """  # noqa

    code += """
        __syncthreads();
    """
    return code


def _get_code_stage1_shared_memory_load_3d(ndim, axis, mode, cval):
    """Generates the first stage of the function body.

    This involves just copying from the `src` array into the `smem` shared
    memory array followed by a call to __syncthreads(). All boundary
    handling also occurs within this function.
    """

    if ndim == 3 and axis == 0:
        if mode not in ['constant', 'grid-constant']:
            boundary_code_lower, boundary_code_upper = util._generate_boundary_condition_ops(mode, 'row', 's_0', separate=True)  # noqa

        # as in OpenCV's column_filter.hpp
        code = """
        __shared__ T smem[(PATCH_PER_BLOCK + 2 * HALO_SIZE) * BLOCK_DIM_Z][BLOCK_DIM_Y][BLOCK_DIM_X];
        const int x = blockIdx.x * BLOCK_DIM_X + threadIdx.x;
        const int y = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;
        if ((x >= s_2) || (y >= s_1)) {
            return;
        }
        // memory is contiguous along last (columns) axis
        const int stride_0 = s_1 * s_2;  // stride (in elements) along axis 0
        const int stride_1 = s_2;  // stride (in elements) along axis 1

        const T* src_col = &src[x + stride_1 * y];
        const int zStart = blockIdx.z * (BLOCK_DIM_Z * PATCH_PER_BLOCK) + threadIdx.z;

        int row;

        if (blockIdx.z > 0)
        {
            //Upper halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j)
                smem[threadIdx.z + j * BLOCK_DIM_Z][threadIdx.y][threadIdx.x] = src_col[(zStart - (HALO_SIZE - j) * BLOCK_DIM_Z) * stride_0];
        }
        else
        {
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j) {
                row = zStart - (HALO_SIZE - j) * BLOCK_DIM_Z;
        """  # noqa
        if mode == 'constant':
            code += f"""
                if (row < 0)
                    smem[threadIdx.z + j * BLOCK_DIM_Z][threadIdx.y][threadIdx.x] = static_cast<T>({cval});
                else
            """  # noqa
        else:
            code += boundary_code_lower
        code += """
                    smem[threadIdx.z + j * BLOCK_DIM_Z][threadIdx.y][threadIdx.x] = src_col[row * stride_0];
            }
        }

        if (blockIdx.z + 2 < gridDim.z)  // Note: +2 here assumes HALO_SIZE <= PATCH_PER_BLOCK so we ensure that elsewhere
        {
            //Main data
            #pragma unroll
            for (int j = 0; j < PATCH_PER_BLOCK; ++j)
                smem[threadIdx.z + (HALO_SIZE + j) * BLOCK_DIM_Z][threadIdx.y][threadIdx.x] = src_col[(zStart + j * BLOCK_DIM_Z) * stride_0];

            //Lower halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j)
                smem[threadIdx.z + (PATCH_PER_BLOCK + HALO_SIZE + j) * BLOCK_DIM_Z][threadIdx.y][threadIdx.x] = src_col[(zStart + (PATCH_PER_BLOCK + j) * BLOCK_DIM_Z) * stride_0];
        }
        else
        {
            //Main data
            #pragma unroll
            for (int j = 0; j < PATCH_PER_BLOCK; ++j) {
                row = zStart + j * BLOCK_DIM_Z;
        """  # noqa
        if mode == 'constant':
            code += f"""
                if (row >= s_0)
                    smem[threadIdx.z + (HALO_SIZE + j) * BLOCK_DIM_Z][threadIdx.y][threadIdx.x] = static_cast<T>({cval});
                else
            """  # noqa
        else:
            code += boundary_code_upper
        code += """
                    smem[threadIdx.z + (HALO_SIZE + j) * BLOCK_DIM_Z][threadIdx.y][threadIdx.x] = src_col[row * stride_0];
            }

            //Lower halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j)
            {
                row = zStart + (PATCH_PER_BLOCK + j) * BLOCK_DIM_Z;
        """  # noqa
        if mode == 'constant':
            code += f"""
                if (row >= s_0)
                    smem[threadIdx.z + (PATCH_PER_BLOCK + HALO_SIZE + j) * BLOCK_DIM_Z][threadIdx.y][threadIdx.x] = static_cast<T>({cval});
                else
            """  # noqa
        else:
            code += boundary_code_upper
        code += """
                    smem[threadIdx.z + (PATCH_PER_BLOCK + HALO_SIZE + j) * BLOCK_DIM_Z][threadIdx.y][threadIdx.x] = src_col[row * stride_0];
            }
        }
        """  # noqa
    elif ndim == 3 and axis == 1:
        if mode not in ['constant', 'grid-constant']:
            boundary_code_lower, boundary_code_upper = util._generate_boundary_condition_ops(mode, 'row', 's_1', separate=True)  # noqa

        # as in OpenCV's column_filter.hpp
        code = """
        __shared__ T smem[BLOCK_DIM_Z][(PATCH_PER_BLOCK + 2 * HALO_SIZE) * BLOCK_DIM_Y][BLOCK_DIM_X];
        const int x = blockIdx.x * BLOCK_DIM_X + threadIdx.x;
        const int z = blockIdx.z * BLOCK_DIM_Z + threadIdx.z;
        if ((x >= s_2) || (z >= s_0)) {
            return;
        }
        // memory is contiguous along last (columns) axis
        const int stride_0 = s_1 * s_2;  // stride (in elements) along axis 0
        const int stride_1 = s_2;  // stride (in elements) along axis 1

        const T* src_col = &src[x + stride_0 * z];
        const int yStart = blockIdx.y * (BLOCK_DIM_Y * PATCH_PER_BLOCK) + threadIdx.y;

        int row;

        if (blockIdx.y > 0)
        {
            //Upper halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j)
                smem[threadIdx.z][threadIdx.y + j * BLOCK_DIM_Y][threadIdx.x] = src_col[(yStart - (HALO_SIZE - j) * BLOCK_DIM_Y) * stride_1];
        }
        else
        {
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j) {
                row = yStart - (HALO_SIZE - j) * BLOCK_DIM_Y;
        """  # noqa
        if mode == 'constant':
            code += f"""
                if (row < 0)
                    smem[threadIdx.z][threadIdx.y + j * BLOCK_DIM_Y][threadIdx.x] = static_cast<T>({cval});
                else
            """  # noqa
        else:
            code += boundary_code_lower
        code += """
                    smem[threadIdx.z][threadIdx.y + j * BLOCK_DIM_Y][threadIdx.x] = src_col[row * stride_1];
            }
        }

        if (blockIdx.y + 2 < gridDim.y)  // Note: +2 here assumes HALO_SIZE <= PATCH_PER_BLOCK so we ensure that elsewhere
        {
            //Main data
            #pragma unroll
            for (int j = 0; j < PATCH_PER_BLOCK; ++j)
                smem[threadIdx.z][threadIdx.y + (HALO_SIZE + j) * BLOCK_DIM_Y][threadIdx.x] = src_col[(yStart + j * BLOCK_DIM_Y) * stride_1];

            //Lower halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j)
                smem[threadIdx.z][threadIdx.y + (PATCH_PER_BLOCK + HALO_SIZE + j) * BLOCK_DIM_Y][threadIdx.x] = src_col[(yStart + (PATCH_PER_BLOCK + j) * BLOCK_DIM_Y) * stride_1];
        }
        else
        {
            //Main data
            #pragma unroll
            for (int j = 0; j < PATCH_PER_BLOCK; ++j) {
                row = yStart + j * BLOCK_DIM_Y;
        """  # noqa
        if mode == 'constant':
            code += f"""
                if (row >= s_1)
                    smem[threadIdx.z][threadIdx.y + (HALO_SIZE + j) * BLOCK_DIM_Y][threadIdx.x] = static_cast<T>({cval});
                else
            """  # noqa
        else:
            code += boundary_code_upper
        code += """
                    smem[threadIdx.z][threadIdx.y + (HALO_SIZE + j) * BLOCK_DIM_Y][threadIdx.x] = src_col[row * stride_1];
            }

            //Lower halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j)
            {
                row = yStart + (PATCH_PER_BLOCK + j) * BLOCK_DIM_Y;
        """  # noqa
        if mode == 'constant':
            code += f"""
                if (row >= s_1)
                    smem[threadIdx.z][threadIdx.y + (PATCH_PER_BLOCK + HALO_SIZE + j) * BLOCK_DIM_Y][threadIdx.x] = static_cast<T>({cval});
                else
            """  # noqa
        else:
            code += boundary_code_upper
        code += """
                    smem[threadIdx.z][threadIdx.y + (PATCH_PER_BLOCK + HALO_SIZE + j) * BLOCK_DIM_Y][threadIdx.x] = src_col[row * stride_1];
            }
        }
        """  # noqa
    elif ndim == 3 and axis == 2:
        if mode not in ['constant', 'grid-constant']:
            boundary_code_lower, boundary_code_upper = util._generate_boundary_condition_ops(mode, 'col', 's_2', separate=True)  # noqa

        # as in OpenCV's row_filter.hpp
        code = """
        __shared__ T smem[BLOCK_DIM_Z][BLOCK_DIM_Y][(PATCH_PER_BLOCK + 2 * HALO_SIZE) * BLOCK_DIM_X];
        const int y = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;
        const int z = blockIdx.z * BLOCK_DIM_Z + threadIdx.z;
        if ((y >= s_1) || (z >= s_0)) {
            return;
        }
        const int stride_0 = s_1 * s_2;  // stride (in elements) along axis 0
        const int stride_1 = s_2;  // stride (in elements) along axis 1
        int col;
        const T* src_row = &src[z * stride_0 + y * stride_1];
        const int xStart = blockIdx.x * (PATCH_PER_BLOCK * BLOCK_DIM_X) + threadIdx.x;

        if (blockIdx.x > 0)
        {
            //Load left halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j)
                smem[threadIdx.z][threadIdx.y][threadIdx.x + j * BLOCK_DIM_X] = src_row[xStart - (HALO_SIZE - j) * BLOCK_DIM_X];
        }
        else
        {
            //Load left halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j){
                col = xStart - (HALO_SIZE - j) * BLOCK_DIM_X;
        """  # noqa
        if mode == 'constant':
            code += f"""
                if (col < 0)
                    smem[threadIdx.z][threadIdx.y][threadIdx.x + j * BLOCK_DIM_X] = static_cast<T>({cval});
                else
            """  # noqa
        else:
            code += boundary_code_lower
        code += """
                    smem[threadIdx.z][threadIdx.y][threadIdx.x + j * BLOCK_DIM_X] = src_row[col];
            }
        }
        if (blockIdx.x + 2 < gridDim.x)  // Note: +2 here assumes HALO_SIZE <= PATCH_PER_BLOCK so we ensure that elsewhere
        {
            //Load main data
            #pragma unroll
            for (int j = 0; j < PATCH_PER_BLOCK; ++j)
                smem[threadIdx.z][threadIdx.y][threadIdx.x + (HALO_SIZE + j)* BLOCK_DIM_X] = src_row[xStart + j * BLOCK_DIM_X];

            //Load right halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j)
                smem[threadIdx.z][threadIdx.y][threadIdx.x + (PATCH_PER_BLOCK + HALO_SIZE + j) * BLOCK_DIM_X] = src_row[xStart + (PATCH_PER_BLOCK + j) * BLOCK_DIM_X];
        }
        else
        {
            //Load main data
            #pragma unroll
            for (int j = 0; j < PATCH_PER_BLOCK; ++j) {
                col = xStart + j * BLOCK_DIM_X;
        """  # noqa
        if mode == 'constant':
            code += f"""
                if (col >= s_2)
                    smem[threadIdx.z][threadIdx.y][threadIdx.x + (HALO_SIZE + j) * BLOCK_DIM_X] = static_cast<T>({cval});
                else
            """  # noqa
        else:
            code += boundary_code_upper
        code += """
                    smem[threadIdx.z][threadIdx.y][threadIdx.x + (HALO_SIZE + j) * BLOCK_DIM_X] = src_row[col];
            }

            //Load right halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j){
                col = xStart + (PATCH_PER_BLOCK + j) * BLOCK_DIM_X;
        """  # noqa
        if mode == 'constant':
            code += f"""
                if (col >= s_2)
                    smem[threadIdx.z][threadIdx.y][threadIdx.x + (PATCH_PER_BLOCK + HALO_SIZE + j) * BLOCK_DIM_X] = static_cast<T>({cval});
                else
            """  # noqa
        else:
            code += boundary_code_upper
        code += """
                    smem[threadIdx.z][threadIdx.y][threadIdx.x + (PATCH_PER_BLOCK + HALO_SIZE + j) * BLOCK_DIM_X] = src_row[col];
            }
        }
        """  # noqa

    code += """
        __syncthreads();
    """
    return code


@cp.memoize(for_each_device=False)
def _get_code_stage1_shared_memory_load(ndim, axis, mode, cval):
    if ndim == 2:
        return _get_code_stage1_shared_memory_load_2d(ndim, axis, mode, cval)
    elif ndim == 3:
        return _get_code_stage1_shared_memory_load_3d(ndim, axis, mode, cval)


def _get_code_stage2_convolve_2d(ndim, axis, flip_kernel):
    code = """
    #pragma unroll
    for (int j = 0; j < PATCH_PER_BLOCK; ++j)
    {
    """
    if flip_kernel:
        kernel_idx = "KSIZE - 1 - k"
    else:
        kernel_idx = "k"

    if ndim == 2 and axis == 0:
        code += """
        const int y = yStart + j * BLOCK_DIM_Y;

        if (y < n_rows)
        {
        """
        inner = f"""
                sum = sum + static_cast<W>(smem[threadIdx.y + (HALO_SIZE + j) * BLOCK_DIM_Y - anchor + k][threadIdx.x]) * kernel[{kernel_idx}];
        """  # noqa
    elif ndim == 2 and axis == 1:
        code += """
        const int x = xStart + j * BLOCK_DIM_X;

        if (x < n_cols)
        {
        """
        inner = f"""
                sum = sum + static_cast<W>(smem[threadIdx.y][threadIdx.x + (HALO_SIZE + j) * BLOCK_DIM_X - anchor + k]) * kernel[{kernel_idx}];
        """  # noqa
    code += f"""
            W sum = static_cast<W>(0);

            #pragma unroll
            for (int k = 0; k < KSIZE; ++k) {{
               {inner}
            }}
            dst[y * row_stride + x] = cast<D>(sum);
        }}
    }}
    """
    return code


def _get_code_stage2_convolve_3d(ndim, axis, flip_kernel):
    code = """
    #pragma unroll
    for (int j = 0; j < PATCH_PER_BLOCK; ++j)
    {
    """
    if flip_kernel:
        kernel_idx = "KSIZE - 1 - k"
    else:
        kernel_idx = "k"

    if ndim == 3 and axis == 0:
        code += """
        const int z = zStart + j * BLOCK_DIM_Z;

        if (z < s_0)
        {
        """
        inner = f"""
                sum = sum + static_cast<W>(smem[threadIdx.z + (HALO_SIZE + j) * BLOCK_DIM_Z - anchor + k][threadIdx.y][threadIdx.x]) * kernel[{kernel_idx}];
        """  # noqa
    elif ndim == 3 and axis == 1:
        code += """
        const int y = yStart + j * BLOCK_DIM_Y;

        if (y < s_1)
        {
        """
        inner = f"""
                sum = sum + static_cast<W>(smem[threadIdx.z][threadIdx.y + (HALO_SIZE + j) * BLOCK_DIM_Y - anchor + k][threadIdx.x]) * kernel[{kernel_idx}];
        """  # noqa
    elif ndim == 3 and axis == 2:
        code += """
        const int x = xStart + j * BLOCK_DIM_X;

        if (x < s_2)
        {
        """
        inner = f"""
                sum = sum + static_cast<W>(smem[threadIdx.z][threadIdx.y][threadIdx.x + (HALO_SIZE + j) * BLOCK_DIM_X - anchor + k]) * kernel[{kernel_idx}];
        """  # noqa
    code += f"""
            W sum = static_cast<W>(0);

            #pragma unroll
            for (int k = 0; k < KSIZE; ++k) {{
               {inner}
            }}
            dst[z * stride_0 + y * stride_1 + x] = cast<D>(sum);
        }}
    }}
    """
    return code


@cp.memoize(for_each_device=False)
def _get_code_stage2_convolve(ndim, axis, flip_kernel):
    if ndim == 2:
        return _get_code_stage2_convolve_2d(ndim, axis, flip_kernel)
    elif ndim == 3:
        return _get_code_stage2_convolve_3d(ndim, axis, flip_kernel)


@cp.memoize(for_each_device=True)
def _get_separable_conv_kernel_src(
    kernel_size, axis, ndim, anchor, image_c_type, kernel_c_type,
    output_c_type, mode, cval, patch_per_block=None, flip_kernel=False
):
    blocks, patch_per_block, halo_size = _get_constants(
        ndim, axis, kernel_size, anchor, patch_per_block
    )
    block_x, block_y, block_z = blocks

    mode_str = mode
    if 'constant' in mode_str:
        mode_str += f'_{cval:0.2f}'.replace('.', '_')
    mode_str = mode_str.replace('-', '_')
    if flip_kernel:
        func_name = f'convolve_s{kernel_size}_{ndim}d_ax{axis}_{mode_str}'
    else:
        func_name = f'correlate_s{kernel_size}_{ndim}d_ax{axis}_{mode_str}'
    func_name += f"_T{image_c_type}_W{kernel_c_type}_D{output_c_type}".replace('complex<', 'c').replace('>', '').replace('long ', 'l').replace('unsigned ', 'u')  # noqa
    func_name += f"_patch{patch_per_block}_halo{halo_size}"
    # func_name += f"_bx{block_x}_by{block_y}" // these are fixed per axis

    code = """
    #include "cupy/carray.cuh"  // for float16
    #include "cupy/complex.cuh"  // for complex<float>
    """

    # SciPy-style float -> unsigned integer casting for the output
    # (use cast<D>(sum) instead of static_cast<D>(sum) for the output)
    code += _ndimage_includes + _ndimage_CAST_FUNCTION

    code += f"""
    const int KSIZE = {kernel_size};
    const int BLOCK_DIM_X = {block_x};
    const int BLOCK_DIM_Y = {block_y};
    const int BLOCK_DIM_Z = {block_z};
    const int PATCH_PER_BLOCK = {patch_per_block};
    const int HALO_SIZE = {halo_size};
    typedef {image_c_type}  T;
    typedef {output_c_type} D;
    typedef {kernel_c_type} W;
    """

    if ndim == 2:
        code += f"""
        extern "C"{{
        __global__ void {func_name}(const T *src, D *dst, const W* kernel, const int anchor, int n_rows, int n_cols)
        {{
        """  # noqa
    elif ndim == 3:
        code += f"""
        extern "C"{{
        __global__ void {func_name}(const T *src, D *dst, const W* kernel, const int anchor, int s_0, int s_1, int s_2)
        {{
        """  # noqa
    code += _get_code_stage1_shared_memory_load(ndim, axis, mode, cval)
    code += _get_code_stage2_convolve(ndim, axis, flip_kernel)
    code += """
    }  // end of function
    }  // extern "C"
    """
    return func_name, blocks, patch_per_block, code


@cp.memoize(for_each_device=True)
def _get_separable_conv_kernel(kernel_size, axis, ndim, image_c_type,
                               kernel_c_type, output_c_type, anchor=None,
                               mode='nearest', cval=0,
                               patch_per_block=None, flip_kernel=False):
    func_name, block, patch_per_block, code = _get_separable_conv_kernel_src(
        kernel_size=kernel_size,
        axis=axis,
        ndim=ndim,
        image_c_type=image_c_type,
        kernel_c_type=kernel_c_type,
        output_c_type=output_c_type,
        anchor=anchor,
        mode=mode,
        cval=cval,
        patch_per_block=patch_per_block,
        flip_kernel=flip_kernel,
    )
    options = ('--std=c++11', '-DCUPY_USE_JITIFY')
    m = cp.RawModule(code=code, options=options)
    return m.get_function(func_name), block, patch_per_block


def _get_grid(shape, block, axis, patch_per_block):
    """Determine grid size from image shape and block parameters"""
    ndim = len(shape)
    if ndim == 2:
        if axis == 0:
            # column filter
            grid = (
                math.ceil(shape[1] / block[0]),
                math.ceil(shape[0] / (block[1] * patch_per_block)),
                1,
            )
        elif axis == 1:
            # row filter
            grid = (
                math.ceil(shape[1] / (block[0] * patch_per_block)),
                math.ceil(shape[0] / block[1]),
                1,
            )
        else:
            raise ValueError(f"invalid axis: {axis}")
    elif ndim == 3:
        if axis == 0:
            # column filter
            grid = (
                math.ceil(shape[2] / block[0]),
                math.ceil(shape[1] / block[1]),
                math.ceil(shape[0] / (block[2] * patch_per_block)),
            )
        elif axis == 1:
            # row filter
            grid = (
                math.ceil(shape[2] / block[0]),
                math.ceil(shape[1] / (block[1] * patch_per_block)),
                math.ceil(shape[0] / block[2]),
            )
        elif axis == 2:
            # row filter
            grid = (
                math.ceil(shape[2] / (block[0] * patch_per_block)),
                math.ceil(shape[1] / block[1]),
                math.ceil(shape[0] / block[2]),
            )
        else:
            raise ValueError(f"invalid axis: {axis}")
    else:
        raise NotImplementedError(f"unsupported ndim: {ndim}")
    return grid


def _shmem_convolve1d(image, weights, axis=-1, output=None, mode="reflect",
                      cval=0.0, origin=0, convolution=False):

    ndim = image.ndim
    if weights.ndim != 1:
        raise ValueError("expected 1d weight array")
    axis = _normalize_axis_index(axis, ndim)
    origin = util._check_origin(origin, weights.size)
    if weights.size == 0:
        return cp.zeros_like(input)
    util._check_mode(mode)

    if convolution:
        # use flip_kernel to avoid cp.ascontiguousarray(weights[::-1]))
        origin = -origin
        if weights.size % 2 == 0:
            origin -= 1
    elif weights.dtype.kind == "c":
        # numpy.correlate conjugates weights rather than input.
        weights = weights.conj()

    anchor = weights.size // 2 + origin

    if weights.size > 32:
        # For large kernels, make sure we have adequate shared memory
        _check_smem_availability(ndim, axis, weights.size, anchor=anchor,
                                 patch_per_block=None, image_dtype=image.dtype,
                                 device_id=None)

    # CUDA kernels assume C-contiguous memory layout
    if not image.flags.c_contiguous:
        image = cp.ascontiguousarray(image)

    complex_output = image.dtype.kind == 'c'
    # Note: important to set use_cucim_casting=True for performance with
    #       8 and 16-bit integer types. This causes the weights to get cast to
    #       float32 rather than float64.
    weights_dtype = util._get_weights_dtype(
        image, weights, use_cucim_casting=True
    )
    if not weights.flags.c_contiguous or weights.dtype != weights_dtype:
        weights = cp.ascontiguousarray(weights, weights_dtype)

    # promote output to nearest complex dtype if necessary
    complex_output = complex_output or weights.dtype.kind == 'c'
    output = util._get_output(output, image, None, complex_output)

    # handle potential overlap between input and output arrays
    needs_temp = cp.shares_memory(output, image, 'MAY_SHARE_BOUNDS')
    if needs_temp:
        output, temp = util._get_output(output.dtype, input), output

    # index_c_type = util._get_inttype(image)
    image_c_type = _dtype_char_to_c_types[image.dtype.char]
    weights_c_type = _dtype_char_to_c_types[weights.dtype.char]
    output_c_type = _dtype_char_to_c_types[output.dtype.char]

    conv_axis_kernel, block, patch_per_block = _get_separable_conv_kernel(
        weights.size,
        axis=axis,
        ndim=ndim,
        anchor=anchor,
        image_c_type=image_c_type,
        kernel_c_type=weights_c_type,
        output_c_type=output_c_type,
        mode=mode,
        cval=cval,
        patch_per_block=None,
        flip_kernel=convolution,
    )
    grid = _get_grid(image.shape, block, axis, patch_per_block)
    args = (image, output, weights, anchor) + image.shape[:ndim]
    conv_axis_kernel(
        grid,
        block,
        args,
    )
    if needs_temp:
        output[:] = temp
        output = temp
    return output
