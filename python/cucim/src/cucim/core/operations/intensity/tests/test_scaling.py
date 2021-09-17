import os

import cupy
import numpy as np
from PIL import Image
import pytest
import cucim.core.operations.intensity as its
import skimage

def get_input_arr():
    img = skimage.data.astronaut()
    arr = np.asarray(img)
    arr = np.transpose(arr)
    return arr

def get_scaled_data():
    img1 = Image.open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "scaled.png"))
    arr_o = np.asarray(img1)
    arr_o = np.transpose(arr_o)
    return arr_o

def test_scale_param():
    arr = get_input_arr()
    with pytest.raises(ValueError):
        output = its.scale_intensity_range(arr,0.0,255.0,1.0,1.0,False)
    with pytest.raises(TypeError):
        img = Image.fromarray(arr.T, 'RGB')
        output = its.scale_intensity_range(img,0.0,255.0,-1.0,1.0,False)

def test_scale_numpy_input():

    arr = get_input_arr()
    scaled_arr = get_scaled_data()
    output = its.scale_intensity_range(arr,0.0,255.0,-1.0,1.0,False)
    assert np.allclose(output,scaled_arr)

def test_scale_cupy_input():

    arr = get_input_arr()
    scaled_arr = get_scaled_data()
    cupy_arr = cupy.asarray(arr)
    cupy_output = its.scale_intensity_range(cupy_arr,0.0,255.0,-1.0,1.0,False)
    np_output = cupy.asnumpy(cupy_output)
    
    assert np.allclose(np_output,scaled_arr)

def test_scale_batchinput():
    
    arr = get_input_arr()
    scaled_arr = get_scaled_data()

    arr_batch = np.stack((arr,)*8, axis=0)
    np_output = its.scale_intensity_range(arr_batch,0.0,255.0,-1.0,1.0,False)

    assert np_output.shape[0] == 8

    for i in range(np_output.shape[0]):
        assert np.allclose(np_output[i],scaled_arr)

