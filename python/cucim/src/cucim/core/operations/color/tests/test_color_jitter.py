import os

import cupy
import numpy as np
from PIL import Image

import cucim.core.operations.color as ccl

img = Image.open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "1.png"))
arr = np.asarray(img)
arr = np.transpose(arr)

cupy_arr = cupy.asarray(arr)
cupy_output = ccl.color_jitter(cupy_arr,.25,.75,.25,.04)
np_output_ch = cupy.asnumpy(cupy_output)
np_output = np_output_ch[0]

diff_im = np_output - arr
diff_total_value = np.abs(np.sum(diff_im))

assert diff_total_value > 0
