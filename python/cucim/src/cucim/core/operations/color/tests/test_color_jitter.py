import os

import cupy
import numpy as np
from PIL import Image
import pytest
import cucim.core.operations.color as ccl

def get_image_array():
    img = Image.open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "1.png"))
    arr = np.asarray(img)
    arr = np.transpose(arr)

    return arr

def verify_result(output, input):
    diff_im = output - input
    diff_total_value = np.abs(np.sum(diff_im))

    assert diff_total_value >= 0

def test_color_jitter_param():
    arr = get_image_array()
    with pytest.raises(ValueError):
        arr1 = arr.flatten()
        np_output = ccl.color_jitter(arr1,.25,.75,.25,.04)
    with pytest.raises(TypeError):
        img = Image.fromarray(arr.T, 'RGB')
        np_output = ccl.color_jitter(img,.25,.75,.25,.04)

def test_color_jitter_numpyinput():
    
    arr = get_image_array()
    np_output = ccl.color_jitter(arr,.25,.75,.25,.04)
    verify_result(np_output, arr)

def test_color_jitter_cupyinput():
    arr = get_image_array()
    cupy_arr = cupy.asarray(arr)
    
    cupy_output = ccl.color_jitter(cupy_arr,.25,.75,.25,.04)
    np_output = cupy.asnumpy(cupy_output)

    verify_result(np_output, arr)

def test_color_jitter_cupy_cast():
    arr = get_image_array()
    cupy_arr = cupy.asarray(arr)
    cupy_arr = cupy_arr.astype(cupy.float32)
    cupy_output = ccl.color_jitter(cupy_arr,.25,.75,.25,.04)
    np_output = cupy.asnumpy(cupy_output)
    assert cupy_output.dtype == cupy.float32

def test_color_jitter_factor():
    arr = get_image_array()
    np_output = ccl.color_jitter(arr,0,0,0,0)
    verify_result(np_output, arr)

def test_color_jitter_batchinput():
    arr = get_image_array()
    arr_batch = np.stack((arr,)*8, axis=0)
    np_output = ccl.color_jitter(arr_batch,.25,.75,.25,.04)
    
    assert np_output.shape[0] == 8
    verify_result(np_output, arr_batch)

