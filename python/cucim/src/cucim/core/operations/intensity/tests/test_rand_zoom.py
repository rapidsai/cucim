import os

import cupy
import numpy as np
import skimage
from PIL import Image

import cucim.core.operations.intensity as its


def get_input_arr():
    img = skimage.data.astronaut()
    arr = np.asarray(img)
    arr = np.transpose(arr)
    return arr

def get_zoomed_data():
    img1 = Image.open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "zoomed.png"))
    arr_o = np.asarray(img1)
    arr_o = np.transpose(arr_o)
    return arr_o

def test_rand_zoom_numpy_input():
    arr = get_input_arr()
    zoomed_arr = get_zoomed_data()
    output = its.rand_zoom(arr,prob=1.0,min_zoom=1.1,max_zoom=1.1)
    assert np.allclose(output, zoomed_arr)

def test_rand_zoom_zero_prob():
    arr = get_input_arr()
    output = its.rand_zoom(arr,prob=0.0,min_zoom=1.1,max_zoom=1.1)
    assert np.allclose(output,arr)

def test_rand_zoom_cupy_input():
    arr = get_input_arr()
    zoomed_arr = get_zoomed_data()
    cupy_arr = cupy.asarray(arr)
    cupy_output = its.rand_zoom(cupy_arr,prob=1.0,min_zoom=1.1,max_zoom=1.1)
    np_output = cupy.asnumpy(cupy_output)
    
    assert np.allclose(np_output,zoomed_arr)

def test_rand_zoom_batchinput():
    arr = get_input_arr()
    zoomed_arr = get_zoomed_data()

    arr_batch = np.stack((arr,)*8, axis=0)
    np_output = its.rand_zoom(arr_batch,prob=1.0,min_zoom=1.1,max_zoom=1.1)
    assert np_output.shape[0] == 8

    for i in range(np_output.shape[0]):
        assert np.allclose(np_output[i],zoomed_arr)

