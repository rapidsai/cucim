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
        spatial axis along which to flip over the input array
        
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
    spatial_axis: tuple() 
    )-> Any:
    """
    Rotate input array by 90 degress along the given axis

    Parameters
    ----------
    img : cupy.ndarray or numpy.ndarray
        Input data. Can be numpy.ndarray or cupy.ndarray
    k : int
        number of times to rotate
    spatial_axis : tuple
        spatial axis along which to rotate the input array by 90 degrees
        
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
    >>> output_array = spt.image_rotate_90(input_arr,1,(1,2))
    """
    try:
        iscupy = False
        cupy_img = img
        if isinstance(img, np.ndarray):
            iscupy = True
            cupy_img = cupy.asarray(img.astype(img.dtype))    
        
        if isinstance(cupy_img, cupy.ndarray) is False:
          raise TypeError("Input must be a cupy.ndarray or numpy.ndarray")

        result = cupy.rot90(cupy_img, k, spatial_axis)
        if iscupy is True:
            result = cupy.asnumpy(result.astype(result.dtype))
        return result
    except Exception as e:
        _logger.error("[cucim] " + str(e), exc_info=True)
        _logger.info("Error executing image rotation on GPU")
        raise

def rand_image_flip(
    img: Any,
    spatial_axis: tuple(),
    prob: float = 0.1
    )-> Any:
    """
    Randomly flips the image along axis. 

    Parameters
    ----------
    img : cupy.ndarray or numpy.ndarray
        Input data. Can be numpy.ndarray or cupy.ndarray
    prob: Probability of flipping.
    spatial_axis : tuple
        spatial axis along which to flip over the input array
        
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
    >>> output_array = spt.rand_image_flip(input_arr,spatial_axis=(1,2))
    """
    R = np.random.RandomState()

    if R.rand() < prob:
        return image_flip(img, spatial_axis)
    else:
        return img

def rand_image_rotate_90(
    img: Any,
    spatial_axis: tuple(),
    prob: float = 0.1,
    max_k: int = 3
    ) -> Any:
    """
    With probability `prob`, input arrays are rotated by 90 degrees
    in the plane specified by `spatial_axis`.

    Parameters
    ----------
    img : cupy.ndarray or numpy.ndarray
        Input data. Can be numpy.ndarray or cupy.ndarray
    prob: probability of rotating.
        (Default 0.1, with 10% probability it returns a rotated array)
    max_k: number of rotations will be sampled from `np.random.randint(max_k) + 1`, (Default 3).    
    spatial_axis : tuple
        spatial axis along which to rotate the input array by 90 degrees
        
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
    >>> output_array = spt.rand_image_rotate_90(input_arr, spatial_axis=(1,2))
    """
    R = np.random.RandomState()

    _rand_k = R.randint(max_k) + 1

    if R.rand() < prob:
        return image_rotate_90(img, _rand_k, spatial_axis)
    else:
        return img
