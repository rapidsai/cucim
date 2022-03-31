import cupy
import numpy as np
import pytest
import skimage.data
from PIL import Image

import cucim.core.operations.color as ccl


def get_image_array():
    img = skimage.data.astronaut()
    arr = np.asarray(img)
    arr = np.transpose(arr)
    return arr


def verify_result(output, input):
    diff_im = output - input
    diff_total_value = np.abs(np.sum(diff_im))
    assert diff_total_value >= 0


def test_color_jitter_bad_params():
    arr = get_image_array()
    with pytest.raises(ValueError):
        arr1 = arr.flatten()
        ccl.color_jitter(arr1, .25, .75, .25, .04)
    with pytest.raises(TypeError):
        img = Image.fromarray(arr.T, 'RGB')
        ccl.color_jitter(img, .25, .75, .25, .04)


def test_color_jitter_numpyinput():
    arr = get_image_array()
    np_output = ccl.color_jitter(arr, .25, .75, .25, .04)
    verify_result(np_output, arr)


def test_color_jitter_cupyinput():
    arr = get_image_array()
    cupy_arr = cupy.asarray(arr)
    cupy_output = ccl.color_jitter(cupy_arr, .25, .75, .25, .04)
    np_output = cupy.asnumpy(cupy_output)
    verify_result(np_output, arr)


def test_color_jitter_cupy_cast():
    arr = get_image_array()
    cupy_arr = cupy.asarray(arr)
    cupy_arr = cupy_arr.astype(cupy.float32)
    cupy_output = ccl.color_jitter(cupy_arr, .25, .75, .25, .04)
    assert cupy_output.dtype == cupy.float32


def test_color_jitter_factor():
    arr = get_image_array()
    np_output = ccl.color_jitter(arr, 0, 0, 0, 0)
    verify_result(np_output, arr)


def test_color_jitter_batchinput():
    arr = get_image_array()
    arr_batch = np.stack((arr,) * 8, axis=0)
    np_output = ccl.color_jitter(arr_batch, .25, .75, .25, .04)
    assert np_output.shape[0] == 8
    verify_result(np_output, arr_batch)


def test_rand_color_jitter_batchinput():
    arr = get_image_array()
    arr_batch = np.stack((arr,) * 8, axis=0)
    np_output = ccl.rand_color_jitter(arr_batch,
                                      .25, .75, .25, .04,
                                      prob=1.0,
                                      whole_batch=True)
    assert np_output.shape[0] == 8
    verify_result(np_output, arr_batch)
