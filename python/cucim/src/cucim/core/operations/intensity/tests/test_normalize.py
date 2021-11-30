import os

import cupy
import numpy as np
import pytest
import skimage.data
from PIL import Image

import cucim.core.operations.intensity as its


def get_input_arr():
    img = skimage.data.astronaut()
    arr = np.asarray(img)
    arr = np.transpose(arr)
    return arr


def get_norm_data():
    dirname = os.path.dirname(__file__)
    img1 = Image.open(os.path.join(os.path.abspath(dirname),
                                   "normalized.png"))
    arr_o = np.asarray(img1)
    arr_o = np.transpose(arr_o)
    return arr_o


def get_norm_atan_data():
    dirname = os.path.dirname(__file__)
    img1 = Image.open(os.path.join(os.path.abspath(dirname),
                                   "normalized_atan.png"))
    arr_o = np.asarray(img1)
    arr_o = np.transpose(arr_o)
    return arr_o


def test_norm_param():
    arr = get_input_arr()
    with pytest.raises(ValueError):
        its.normalize_data(arr, 10.0, 255, 255)
    with pytest.raises(ValueError):
        its.normalize_data(arr, 10.0, 0, 255, 'invalid')
    with pytest.raises(TypeError):
        img = Image.fromarray(arr.T, 'RGB')
        its.normalize_data(img, 10.0, 0, 255)


def test_norm_numpy_input():
    arr = get_input_arr()
    norm_arr = get_norm_data()
    output = its.normalize_data(arr, 10.0, 0, 255)
    assert np.allclose(output, norm_arr)

    norm_atan_arr = get_norm_atan_data()
    output = its.normalize_data(arr, 10000, 0, 255, 'atan')
    assert np.allclose(output, norm_atan_arr)


def test_norm_cupy_input():
    arr = get_input_arr()
    norm_arr = get_norm_data()
    cupy_arr = cupy.asarray(arr)
    cupy_output = its.normalize_data(cupy_arr, 10.0, 0, 255)
    np_output = cupy.asnumpy(cupy_output)
    assert np.allclose(np_output, norm_arr)


def test_norm_batchinput():
    arr = get_input_arr()
    norm_arr = get_norm_data()
    arr_batch = np.stack((arr,) * 8, axis=0)
    output = its.normalize_data(arr_batch, 10.0, 0, 255)

    assert output.shape[0] == 8

    for i in range(output.shape[0]):
        assert np.allclose(output[i], norm_arr)
