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
import scipy.ndimage as ndimage


def image_flip(
    img: Any, 
    spatial_axis: tuple()
    )-> Any:
    """
    Shape preserving order reversal of elements in input array along the given spatial axis

    Parameters
    ----------
    img : cupy.ndarray or numpy.ndarray
        Input data. Can be numpy.ndarray or cupy.ndarray
    spatial_axis : tuple
        spatial axes along which to flip over the input array
        
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
    >>> import cucim.core.operations.spatial as spt
    >>> # input is channel first 3d array
    >>> output_array = spt.image_flip(input_arr,(1,2))
    """
    try:
        iscupy = False
        cupy_img = img
        if isinstance(img, np.ndarray):
            iscupy = True
            cupy_img = cupy.asarray(img.astype(img.dtype))    

        if isinstance(cupy_img, cupy.ndarray) is False:
          raise TypeError("Input must be a cupy.ndarray or numpy.ndarray")

        result = cupy.flip(cupy_img, spatial_axis)
        if iscupy is True:
            result = cupy.asnumpy(result.astype(result.dtype))
        return result
    except Exception as e:
        _logger.error("[cucim] " + str(e), exc_info=True)
        _logger.info("Error executing image flip on GPU")
        raise

def image_rotate_90(
    img: Any, 
    k: int, 
    axis: tuple() 
    )-> Any:
    """
    Rotate input array by 90 degress along the given axis

    Parameters
    ----------
    img : cupy.ndarray or numpy.ndarray
        Input data. Can be numpy.ndarray or cupy.ndarray
    k : int
        number of times to rotate
    axis : tuple
        spatial axes along which to rotate the input array by 90 degrees
        
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
    >>> import cucim.core.operations.spatial as spt
    >>> # input is channel first 3d array
    >>> output_array = spt.image_rotate(input_arr,1,(1,2))
    """
    try:
        iscupy = False
        cupy_img = img
        if isinstance(img, np.ndarray):
            iscupy = True
            cupy_img = cupy.asarray(img.astype(img.dtype))    
        
        if isinstance(cupy_img, cupy.ndarray) is False:
          raise TypeError("Input must be a cupy.ndarray or numpy.ndarray")

        result = cupy.rot90(cupy_img, k, axis)
        if iscupy is True:
            result = cupy.asnumpy(result.astype(result.dtype))
        return result
    except Exception as e:
        _logger.error("[cucim] " + str(e), exc_info=True)
        _logger.info("Error executing image rotation on GPU")
        raise
