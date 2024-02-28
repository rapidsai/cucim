#
# Copyright (c) 2021, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import json

import numpy as np
from cucim import CuImage
from PIL import Image

img = CuImage("image.tif")

print(img.is_loaded)  # True if image data is loaded & available.
print(img.device)  # A device type.
print(img.ndim)  # The number of dimensions.
print(img.dims)  # A string containing a list of dimensions being requested.
print(img.shape)  # A tuple of dimension sizes (in the order of `dims`).
print(img.size("XYC"))  # Returns size as a tuple for the given dimension order.
print(img.dtype)  # The data type of the image.
print(img.channel_names)  # A channel name list.
print(img.spacing())  # Returns physical size in tuple.
print(
    img.spacing_units()
)  # Units for each spacing element (size is same with `ndim`).
print(img.origin)  # Physical location of (0, 0, 0) (size is always 3).
print(img.direction)  # Direction cosines (size is always 3x3).
print(img.coord_sys)  # Coordinate frame in which the direction cosines are
# measured. Available Coordinate frame is not finalized yet.

# Returns a set of associated image names.
print(img.associated_images)
# Returns a dict that includes resolution information.
print(json.dumps(img.resolutions, indent=2))
# A metadata object as `dict`
print(json.dumps(img.metadata, indent=2))
# A raw metadata string.
print(img.raw_metadata)

# Read whole slide at the lowest resolution
resolutions = img.resolutions
level_count = resolutions["level_count"]
# Note: 'level' is at 3rd parameter (OpenSlide has it at 2nd parameter)
region = img.read_region(
    location=(10000, 10000), size=(512, 512), level=level_count - 1
)

region.save("test.ppm")

Image.fromarray(np.asarray(region))
