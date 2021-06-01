#
# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

"""cuCIM module

This project contains core modules and CuPy-based implementations of functions
from scikit-image that are not currently available in CuPy itself.

Most functions are not provided via the top level-import. Instead, individual
subpackages should be imported instead.

Subpackages
-----------

clara
    Functions for image IO and operations.
skimage
    Functions from scikit-image.

"""

# Try to import cupy first.
# If cucim.clara package is imported first, you may see the following error when running on CUDA 10.x (#44)
#   python3: Relink `/usr/lib/x86_64-linux-gnu/libnccl.so.2.8.3' with `/lib/x86_64-linux-gnu/librt.so.1' for IFUNC symbol `clock_gettime'
#   Segmentation fault
try:
    import cupy
except ImportError:
    pass

try:
    from .clara import __version__, CuImage, cli
except ImportError:
    from ._version import get_versions
    __version__ = get_versions()['version']
    del get_versions
    del _version
