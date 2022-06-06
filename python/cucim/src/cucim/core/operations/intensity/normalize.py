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

from typing import Any

import cupy
import numpy as np

from .kernel.cuda_kernel_source import cuda_kernel_code

CUDA_KERNELS = cupy.RawModule(code=cuda_kernel_code)


def normalize_data(
    img: Any,
    norm_constant: float,
    min_value: float,
    max_value: float,
    type: str = 'range'
) -> Any:
    """
    Apply intensity normalization to the input array.
    Normalize intensities to the range of [0, norm_constant].

    Parameters
    ----------
    img : channel first, cupy.ndarray or numpy.ndarray
        Input data of shape (C, H, W). Can also batch process input of shape
        (N, C, H, W). Can be a numpy.ndarray or cupy.ndarray.
    norm_constant: float
        Normalization range of the input data. [0, norm_constant]
    min_value : float
        Minimum intensity value in input data.
    max_value : float
        Maximum intensity value in input data.
    type : {'range', 'atan'}
        Type of normalization.

    Returns
    -------
    out : cupy.ndarray or numpy.ndarray
        Output data. Same dimensions and type as input.

    Raises
    ------
    TypeError
        If input 'img' is not cupy.ndarray or numpy.ndarray
    ValueError
        If input original intensity min and max are same
    ValueError
        If incorrect normalization type is invoked

    Examples
    --------
    >>> import cucim.core.operations.intensity as its
    >>> # input is channel first 3d array
    >>> output_array = its.normalize_data(input_arr,
                                          10, 0 , 255)
    """
    if max_value - min_value == 0.0:
        raise ValueError("Minimum and Maximum intensity \
                          same in input data")

    if type not in ['range', 'atan']:
        raise ValueError("Incorrect normalization type. \
                          Supported types are: \
                              range based: 1,\
                              atangent based: 2")

    to_numpy = False
    if isinstance(img, np.ndarray):
        to_numpy = True
        cupy_img = cupy.asarray(img, dtype=cupy.float32, order='C')
    elif not isinstance(img, cupy.ndarray):
        raise TypeError("img must be a cupy.ndarray or numpy.ndarray")
    else:
        cupy_img = cupy.ascontiguousarray(img)

    if cupy_img.dtype != cupy.float32:
        if cupy.can_cast(img.dtype, cupy.float32) is False:
            raise ValueError(
                "Cannot safely cast type {cupy_img.dtype.name} to \
                'float32'"
            )
        else:
            cupy_img = cupy_img.astype(cupy.float32)

    normalize = CUDA_KERNELS.get_function("normalize_data_by_range")

    if type == 'atan':
        normalize = CUDA_KERNELS.get_function("normalize_data_by_atan")

    value_range = max_value - min_value
    norm_factor = norm_constant / value_range

    total_size = int(np.prod(img.shape))
    blockx = 128
    gridx = int((total_size - 1) / blockx + 1)

    result = cupy.empty(img.shape, dtype=cupy_img.dtype)

    normalize((gridx, 1, 1), (blockx, 1, 1),
              (cupy_img, result, np.float32(norm_factor),
               np.float32(min_value),
               np.int32(total_size)))

    if img.dtype != cupy.float32:
        result = result.astype(img.dtype)

    if to_numpy:
        result = cupy.asnumpy(result)
    return result
