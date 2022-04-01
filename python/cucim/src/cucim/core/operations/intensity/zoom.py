# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
from typing import Any, Sequence, Union

import cupy
import numpy as np

from .kernel.cuda_kernel_source import cuda_kernel_code

_logger = logging.getLogger("zoom_cucim")
CUDA_KERNELS = cupy.RawModule(code=cuda_kernel_code)


def zoom(
    img: Any,
    zoom_factor: Sequence[float]
):
    """Zooms an ND image

    Parameters
    ----------
    img : channel first, cupy.ndarray or numpy.ndarray
        Input data of shape (C, H, W). Can also batch process input of shape
        (N, C, H, W). Can be a numpy.ndarray or cupy.ndarray.
    zoom_factor: Sequence[float]
        The zoom factor along the spatial axes.
        Zoom factor should contain one value for each spatial axis.
    Returns
    -------
    out : cupy.ndarray or numpy.ndarray
        Output data. Same dimensions and type as input.

    Raises
    ------
    TypeError
        If input 'img' is not cupy.ndarray or numpy.ndarray

    Examples
    --------
    >>> import cucim.core.operations.intensity as its
    >>> # input is channel first 3d array
    >>> output_array = its.zoom(input_arr,[1.1,1.1])
    """
    try:
        to_cupy = False

        if isinstance(img, np.ndarray):
            to_cupy = True
            cupy_img = cupy.asarray(img, dtype=cupy.float32, order="C")
        elif not isinstance(img, cupy.ndarray):
            raise TypeError("img must be a cupy.ndarray or numpy.ndarray")
        else:
            cupy_img = cupy.ascontiguousarray(img)

        if cupy_img.dtype != cupy.float32:
            if cupy.can_cast(img.dtype, cupy.float32) is False:
                raise ValueError(
                    "Cannot safely cast type {cupy_img.dtype.name} \
                     to 'float32'"
                )
            else:
                cupy_img = cupy_img.astype(cupy.float32)

        if img.ndim not in (3, 4):
            raise ValueError(
                f"Unsupported img.ndim={img.ndim}. Expected `img` with "
                "dimensions (C, H, W) or (N, C, H, W)."
            )

        if len(img.shape) == 4:
            N, C, H, W = img.shape
        elif len(img.shape) == 3:
            C, H, W = img.shape
            N = 1

        output_size_cu = [N, C, int(math.floor(H * zoom_factor[0])),
                          int(math.floor(W * zoom_factor[1]))]

        if output_size_cu[2] == H and output_size_cu[3] == W:
            return img

        def get_block_size(output_size_cu, H, W):
            max_smem = 48 * 1024
            cu_block_options = [(16, 16, 1), (16, 8, 1), (8, 8, 1), (8, 4, 1)]
            # compare for 48KB for standard CC optimal occupancy
            # array is H, W but kernel is x--> W, y-->H
            for param in cu_block_options:
                h_stretch = [math.floor((0 * H) / output_size_cu[2]),
                             math.ceil((param[1] * H) / output_size_cu[2])]
                w_stretch = [math.floor((0 * W) / output_size_cu[3]),
                             math.ceil((param[0] * W) / output_size_cu[3])]

                smem_size = (h_stretch[1] + 1) * (w_stretch[1] + 1) * 4
                if smem_size < max_smem:
                    return param, smem_size

            raise Exception("Random Zoom couldnt find a \
                             shared memory configuration")

        # input pitch
        pitch = H * W

        # get block size
        block_config, smem_size = get_block_size(output_size_cu, H, W)
        grid = (int((output_size_cu[3] - 1) / block_config[0] + 1),
                int((output_size_cu[2] - 1) / block_config[1] + 1), C * N)

        is_zoom_out = output_size_cu[2] < H and output_size_cu[3] < W
        is_zoom_in = output_size_cu[2] > H and output_size_cu[3] > W

        pad_dims = [[0, 0]] * 2  # zoom out
        slice_dims = [[0, 0]] * 2  # zoom in
        for idx, (orig, zoom) in enumerate(zip((H, W),
                                           (output_size_cu[2],
                                            output_size_cu[3]))):
            diff = orig - zoom
            half = abs(diff) // 2
            if diff > 0:
                pad_dims[idx] = [half, diff - half]
            elif diff < 0:
                slice_dims[idx] = [half, half + orig]

        result = cupy.ndarray(cupy_img.shape, dtype=cupy.float32)

        if is_zoom_in:
            # slice
            kernel = CUDA_KERNELS.get_function("zoom_in_kernel")
            kernel(grid, block_config,
                   args=(cupy_img, result, np.int32(H), np.int32(W),
                         np.int32(output_size_cu[2]),
                         np.int32(output_size_cu[3]),
                         np.int32(pitch), np.int32(slice_dims[0][0]),
                         np.int32(slice_dims[0][1]),
                         np.int32(slice_dims[1][0]),
                         np.int32(slice_dims[1][1])),
                   shared_mem=smem_size)
        elif is_zoom_out:
            # pad
            kernel = CUDA_KERNELS.get_function("zoom_out_kernel")
            kernel(grid, block_config,
                   args=(cupy_img, result, np.int32(H), np.int32(W),
                         np.int32(output_size_cu[2]),
                         np.int32(output_size_cu[3]),
                         np.int32(pitch), np.int32(pad_dims[0][0]),
                         np.int32(pad_dims[0][1]),
                         np.int32(pad_dims[1][0]),
                         np.int32(pad_dims[1][1])),
                   shared_mem=smem_size)
            # padding kernel
            kernel = CUDA_KERNELS.get_function("zoomout_edge_pad")
            grid = (int((W - 1) / block_config[0] + 1),
                    int((H - 1) / block_config[1] + 1),
                    C * N)
            kernel(grid, block_config,
                   args=(result, np.int32(H), np.int32(W), np.int32(pitch),
                         np.int32(pad_dims[0][0]), np.int32(pad_dims[1][0]),
                         np.int32(pad_dims[0][0] + output_size_cu[2]),
                         np.int32(pad_dims[1][0] + output_size_cu[3])))

        else:
            raise Exception("Can only handle simultaneous \
                            expansion(or shrinkage) in both H,W dimension, \
                            check zoom factors")

        if img.dtype != np.float32:
            result = result.astype(img.dtype)

        if to_cupy is True:
            result = cupy.asnumpy(result)

        return result

    except Exception as e:
        _logger.error("[cucim] " + str(e), exc_info=True)
        _logger.info("Error executing random zoom on GPU")
        raise


def get_zoom_factor(
    min_zoom: Union[Sequence[float], float] = 0.9,
    max_zoom: Union[Sequence[float], float] = 1.1,
):
    R = np.random.RandomState()
    try:
        zoom_factor = [R.uniform(low, high)
                       for low, high in zip(min_zoom, max_zoom)]
    except Exception:
        zoom_factor = [R.uniform(min_zoom, max_zoom)]

    if len(zoom_factor) != 2:
        zoom_factor = [zoom_factor[0] for _ in range(2)]

    return zoom_factor


def rand_zoom(
    img: Any,
    min_zoom: Union[Sequence[float], float] = 0.9,
    max_zoom: Union[Sequence[float], float] = 1.1,
    prob: float = 0.1,
    whole_batch: bool = False
):
    """
    Randomly Calls zoom with random zoom factor

    Parameters
    ----------
    img : channel first, cupy.ndarray or numpy.ndarray
        Input data of shape (C, H, W). Can also batch process input of shape
        (N, C, H, W). Can be a numpy.ndarray or cupy.ndarray.
    min_zoom: Min zoom factor. Can be float or sequence same size as image.
        If a float, select a random factor from `[min_zoom, max_zoom]` then
        apply to all spatial dims to keep the original spatial shape ratio.
        If a sequence, min_zoom should contain one value for each spatial axis.
        If 2 values provided for 3D data, use the first value for both H & W
        dims to keep the same zoom ratio.
    max_zoom: Max zoom factor. Can be float or sequence same size as image.
        If a float, select a random factor from `[min_zoom, max_zoom]` then
        apply to all spatial dims to keep the original spatial shape ratio.
        If a sequence, max_zoom should contain one value for each spatial axis.
        If 2 values provided for 3D data, use the first value for both H & W
        dims to keep the same zoom ratio.
    prob: Probability of zooming.
    whole_batch: Flag to apply transform on whole batch.
        If False, each image in the batch is randomly transformed
        It True, entire batch is transformed randomly.

    Returns
    -------
    out : cupy.ndarray or numpy.ndarray
        Output data. Same dimensions and type as input.

    Raises
    ------
    TypeError
        If input 'img' is not cupy.ndarray or numpy.ndarray

    Examples
    --------
    >>> import cucim.core.operations.intensity as its
    >>> # input is channel first 3d array
    >>> output_array = its.rand_zoom(input_arr)
    """
    R = np.random.RandomState()

    shape = img.shape

    if whole_batch is False and len(shape) == 4:
        image_wise_probs = R.rand(shape[0])
        for i in range(shape[0]):
            if image_wise_probs[i] < prob:
                zoom_factor = get_zoom_factor(min_zoom, max_zoom)
                img[i] = zoom(img[i], zoom_factor)
        return img
    elif R.rand() < prob:
        zoom_factor = get_zoom_factor(min_zoom, max_zoom)
        return zoom(img, zoom_factor)
    else:
        return img
