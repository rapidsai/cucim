import cucim.core.operations.color as ccl
import numpy as np
import cupy
from PIL import Image

img = Image.open('1.png')
arr = np.asarray(img)
arr = np.transpose(arr)

cupy_arr = cupy.asarray(arr)
cupy_output = ccl.color_jitter(cupy_arr,.25,.75,.25,.04)
np_output_ch = cupy.asnumpy(cupy_output)
np_output = np_output_ch[0]

diff_im = np_output - arr
diff_total_value = np.abs(np.sum(diff_im))

assert diff_total_value > 0