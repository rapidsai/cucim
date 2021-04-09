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

"""CuPy Extensions

This project contains CuPy-based implementations of functions from NumPy,
SciPy and scikit-image that are not currently available in CuPy itself.

Most functions are not provided via the top level-import. Instead, individual
subpackages should be imported instead.

Subpackages
-----------
numpy
    Functions from NumPy which are not available via CuPy.
scipy
    Functions from SciPy which are not available via CuPy.
skimage
    Functions from scikit-image.

Additional documentation and usage examples for the functions can be found
at the main documentation pages of the various packges:

"""

try:
    import cupy

    try:
        memoize = cupy.util.memoize
    except AttributeError:
        memoize = cupy.memoize

    del cupy
except ImportError:
    import sys
    print('[warning] CuPy is not available. cucim.skimage package may not work correctly.', file=sys.stderr)

# from ._version import get_versions

# __version__ = get_versions()['version']
# del get_versions

from .clara import CuImage
from .clara import __version__
from .clara import cli
