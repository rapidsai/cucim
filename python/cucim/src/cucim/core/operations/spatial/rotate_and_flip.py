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


def image_flip(
    img: Any,
    spatial_axis: tuple()
) -> Any:
    """
    Shape preserving order reversal of elements in input array
    along the given spatial axis

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
    >>> output_array = spt.image_flip(input_arr, (1, 2))
    """
    to_numpy = False
    if isinstance(img, np.ndarray):
        to_numpy = True
        cupy_img = cupy.asarray(img, order="C")
    elif not isinstance(img, cupy.ndarray):
        raise TypeError("img must be a cupy.ndarray or numpy.ndarray")
    else:
        cupy_img = cupy.ascontiguousarray(img)

    result = cupy.flip(cupy_img, spatial_axis)
    if to_numpy:
        result = cupy.asnumpy(result)

    return result


def image_rotate_90(
    img: Any,
    k: int,
    spatial_axis: tuple()
) -> Any:
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
    to_numpy = False
    if isinstance(img, np.ndarray):
        to_numpy = True
        cupy_img = cupy.asarray(img, order="C")
    elif not isinstance(img, cupy.ndarray):
        raise TypeError("img must be a cupy.ndarray or numpy.ndarray")
    else:
        cupy_img = cupy.ascontiguousarray(img)

    result = cupy.rot90(cupy_img, k, spatial_axis)
    if to_numpy:
        result = cupy.asnumpy(result)
    return result


def rand_image_flip(
    img: Any,
    spatial_axis: tuple(),
    prob: float = 0.1,
    whole_batch: bool = False
) -> Any:
    """
    Randomly flips the image along axis.

    Parameters
    ----------
    img : cupy.ndarray or numpy.ndarray
        Input data. Can be numpy.ndarray or cupy.ndarray
    prob: Probability of flipping.
    spatial_axis : tuple
        spatial axis along which to flip over the input array
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
    >>> import cucim.core.operations.spatial as spt
    >>> # input is channel first 3d array
    >>> output_array = spt.rand_image_flip(input_arr,spatial_axis=(1,2))
    """
    R = np.random.RandomState()

    shape = img.shape

    if whole_batch is False and len(shape) == 4:
        image_wise_probs = R.rand(shape[0])
        for i in range(shape[0]):
            if image_wise_probs[i] < prob:
                img[i] = image_flip(img[i], spatial_axis)
        return img
    elif R.rand() < prob:
        return image_flip(img, spatial_axis)
    else:
        return img


def rand_image_rotate_90(
    img: Any,
    spatial_axis: tuple(),
    prob: float = 0.1,
    max_k: int = 3,
    whole_batch: bool = False
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
    max_k: number of rotations
        will be sampled from `np.random.randint(max_k) + 1`, (Default 3).
    spatial_axis : tuple
        spatial axis along which to rotate the input array by 90 degrees
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
    >>> import cucim.core.operations.spatial as spt
    >>> # input is channel first 3d array
    >>> output_array = spt.rand_image_rotate_90(input_arr, spatial_axis=(1, 2))
    """
    R = np.random.RandomState()

    shape = img.shape

    if whole_batch is False and len(shape) == 4:
        image_wise_probs = R.rand(shape[0])
        for i in range(shape[0]):
            if image_wise_probs[i] < prob:
                _rand_k = R.randint(max_k) + 1
                img[i] = image_rotate_90(img[i], _rand_k, spatial_axis)
        return img
    elif R.rand() < prob:
        _rand_k = R.randint(max_k) + 1
        return image_rotate_90(img, _rand_k, spatial_axis)
    else:
        return img
