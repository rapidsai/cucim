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

submodules = []

try:
    import cupy
    _is_cupy_available = True
    submodules += ['core', 'skimage']
except ImportError:
    pass

try:
    from .clara import CuImage, __version__, cli
    _is_clara_available = True
    submodules += ['clara']
except ImportError:
    from ._version import get_versions
    __version__ = get_versions()['version']
    del get_versions
    del _version

__all__ = submodules + ['__version__', 'is_available']
