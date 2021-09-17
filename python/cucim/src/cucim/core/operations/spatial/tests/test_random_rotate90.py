import os
import cupy
import numpy as np
from PIL import Image
import skimage
import cucim.core.operations.spatial as spt

def get_input_arr():
    img = skimage.data.astronaut()
    arr = np.asarray(img)
    arr = np.transpose(arr)
    return arr

def get_rotated_data():
    img1 = Image.open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "rotated.png"))
    arr_o = np.asarray(img1)
    arr_o = np.transpose(arr_o)
    return arr_o

def test_rand_rotate90_numpy_input():

    arr = get_input_arr()
    rotate90_arr = get_rotated_data()
    output = spt.rand_image_rotate_90(arr,max_k=1,prob=1.0,spatial_axis=[1,2])
    assert np.allclose(output, rotate90_arr)

def test_rand_rotate90_zero_prob():

    arr = get_input_arr()
    output = spt.rand_image_rotate_90(arr,max_k=1,prob=0.0,spatial_axis=[1,2])
    assert np.allclose(output, arr)

def test_rand_rotate90_cupy_input():

    arr = get_input_arr()
    rotate90_arr = get_rotated_data()
    cupy_arr = cupy.asarray(arr)
    cupy_output = spt.rand_image_rotate_90(cupy_arr,max_k=1,prob=1.0,spatial_axis=[1,2])
    np_output = cupy.asnumpy(cupy_output)
    assert np.allclose(np_output, rotate90_arr)

def test_rand_rotate90_batchinput():
    
    arr = get_input_arr()
    rotate90_arr = get_rotated_data()
    arr_batch = np.stack((arr,)*8, axis=0)
    np_output = spt.rand_image_rotate_90(arr_batch,max_k=1,prob=1.0,spatial_axis=[2,3])
    assert np_output.shape[0] == 8

    for i in range(np_output.shape[0]):
        assert np.allclose(np_output[i], rotate90_arr)
        


