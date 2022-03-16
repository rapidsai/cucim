import os

import cupy
import numpy as np
import skimage.data
from PIL import Image

import cucim.core.operations.spatial as spt


def get_input_arr():
    img = skimage.data.astronaut()
    arr = np.asarray(img)
    arr = np.transpose(arr)
    return arr


def get_flipped_data():
    dirname = os.path.dirname(__file__)
    img1 = Image.open(os.path.join(os.path.abspath(dirname), "flipped.png"))
    arr_o = np.asarray(img1)
    arr_o = np.transpose(arr_o)
    return arr_o


def test_rand_flip_numpy_input():
    arr = get_input_arr()
    flip_arr = get_flipped_data()
    output = spt.rand_image_flip(arr, prob=1.0, spatial_axis=(1, 2))
    assert np.allclose(output, flip_arr)


def test_rand_flip_zero_prob():
    arr = get_input_arr()
    output = spt.rand_image_flip(arr, prob=0.0, spatial_axis=(1, 2))
    assert np.allclose(output, arr)


def test_rand_flip_cupy_input():
    arr = get_input_arr()
    flip_arr = get_flipped_data()
    cupy_arr = cupy.asarray(arr)
    cupy_output = spt.rand_image_flip(cupy_arr, prob=1.0, spatial_axis=(1, 2))
    np_output = cupy.asnumpy(cupy_output)
    assert np.allclose(np_output, flip_arr)


def test_rand_flip_batchinput():
    arr = get_input_arr()
    flip_arr = get_flipped_data()
    arr_batch = np.stack((arr,) * 8, axis=0)
    np_output = spt.rand_image_flip(arr_batch,
                                    prob=1.0,
                                    spatial_axis=(2, 3),
                                    whole_batch=True)

    assert np_output.shape[0] == 8

    for i in range(np_output.shape[0]):
        assert np.allclose(np_output[i], flip_arr)
