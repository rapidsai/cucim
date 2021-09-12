import cucim.core.operations.spatial as spt
import numpy as np
import cupy
from PIL import Image

img = Image.open('1.png')
arr = np.asarray(img)
arr = np.transpose(arr)

img1 = Image.open('2f.png')
arr_o = np.asarray(img1)
arr_o = np.transpose(arr_o)

cupy_arr = cupy.asarray(arr)
cupy_output = spt.image_flip(cupy_arr,(1,2))
np_output = cupy.asnumpy(cupy_output)

diff_im = np_output-arr_o
error_threshold = 1000.01

assert np.sum(diff_im) < error_threshold