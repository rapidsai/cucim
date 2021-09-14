import os

import cupy
import numpy as np
from PIL import Image

import cucim.core.operations.intensity as its

img = Image.open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "1.png"))
arr = np.asarray(img)
arr = np.transpose(arr)

R = np.random.RandomState(100)
min_zoom = 0.9
max_zoom = 1.1
try:
    zoom_factor = [R.uniform(low, high) for low, high in zip(min_zoom, max_zoom)]
except:
    zoom_factor = [R.uniform(min_zoom, max_zoom)]

if len(zoom_factor) != 2:
    zoom_factor = [zoom_factor[0] for _ in range(2)]

cupy_arr = cupy.asarray(arr)
cupy_output = its.zoom(cupy_arr,zoom_factor)
np_output_ch = cupy.asnumpy(cupy_output)
np_output = np_output_ch[0]

expected_sum = 7531080.0
zoom_sum = np.sum(np_output)
error_percent = float(abs(zoom_sum-expected_sum))/float(expected_sum)
error_threshold = .01

assert error_percent < error_threshold
