import os

import cupy
import numpy as np
from PIL import Image

import cucim.core.operations.spatial as spt

img = Image.open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "1.png"))
arr = np.asarray(img)
arr = np.transpose(arr)

img1 = Image.open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "2f.png"))
arr_o = np.asarray(img1)
arr_o = np.transpose(arr_o)

cupy_arr = cupy.asarray(arr)
cupy_output = spt.image_flip(cupy_arr,(1,2))
np_output = cupy.asnumpy(cupy_output)

diff_im = np_output-arr_o
error_threshold = 1000.01

assert np.sum(diff_im) < error_threshold