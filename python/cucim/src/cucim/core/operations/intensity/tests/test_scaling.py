import cucim.core.operations.intensity as its
import numpy as np
import cupy
from PIL import Image
from PIL import ImageChops

img = Image.open('1.png')
arr = np.asarray(img)
arr = np.transpose(arr)

img1 = Image.open('2.png')
arr_o = np.asarray(img1)
arr_o = np.transpose(arr_o)

cupy_arr = cupy.asarray(arr)
cupy_output = its.scale_intensity_range(cupy_arr,0.0,255.0,-1.0,1.0,False)
np_output = cupy.asnumpy(cupy_output)

scaled_sum = np.sum(np_output)
expected_sum = np.sum(arr_o)
error_percent = float(abs(scaled_sum-expected_sum))/float(expected_sum)
error_threshold = .01

assert error_percent < error_threshold