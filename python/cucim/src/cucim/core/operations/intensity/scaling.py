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

import os
import sys
from typing import Any
from warnings import warn

import cupy
import numpy as np
import scipy.ndimage as ndimage

from .kernel.cuda_kernel_source import cuda_kernel_code

CUDA_KERNELS = cupy.RawModule(code=cuda_kernel_code)

import logging

_logger = logging.getLogger("scaling_cucim")

def scale_intensity_range(
    img: Any, 
    b_max: float, 
    b_min: float, 
    a_max: float, 
    a_min: float, 
    clip: bool = False
    )-> Any:
    """
    Apply intensity scaling to the input array.
    Scaling from [a_min, a_max] to [b_min, b_max] with clip option.

    Parameters
    ----------
    img : channel first, cupy.ndarray or numpy.ndarray
        Input data. Can be numpy.ndarray or cupy.ndarray
    b_min : float
        intensity target range min.
    b_max : float
        intensity target range max.
    a_min : float
        intensity original range min.
    a_max : float
        intensity original range max.
    clip : float
        whether to perform clip after scaling.

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

    Examples
    --------
    >>> import cucim.core.operations.intensity as its
    >>> # input is channel first 3d array
    >>> output_array = its.scale_intensity_range(input_arr,0.0,255.0,-1.0,1.0,False)
    """
    try:
        if a_max - a_min == 0.0:
            raise ValueError("Original intensity range min and max are same")
        
        iscupy = False
        image = img
        if isinstance(img, np.ndarray):
            iscupy = True
            image = cupy.asarray(img.astype(img.dtype))    
        
        if isinstance(image, cupy.ndarray) is False:
          raise TypeError("Input must be a cupy.ndarray or numpy.ndarray")

        scale = CUDA_KERNELS.get_function("scaleVolume")
        
        x = (b_max - b_min) / (a_max - a_min)
        y = a_min * x - b_min
        if clip is False:
            b_max = float('inf')
            b_min = float('-inf')

        sh = img.shape
        total_size = np.prod(sh)
        blockx = 128
        gridx = int((total_size - 1) / blockx + 1)

        result = cupy.empty(image.shape, dtype=img.dtype)

        scale((gridx, 1, 1), (blockx, 1, 1), 
              (image, result, np.float32(x), np.float32(y), 
              np.float32(b_min), np.float32(b_max), 
              np.int32(total_size)))
        
        if iscupy is True:
            result = cupy.asnumpy(result.astype(result.dtype))
        
    except Exception as e:
        _logger.error("[cucim] " + str(e), exc_info=True)
        raise
        
    return result

