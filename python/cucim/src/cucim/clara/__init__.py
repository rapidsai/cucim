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

import os

from . import cli, converter
# import hidden methods
from ._cucim import (CuImage, DLDataType, DLDataTypeCode, __version__, cache,
                     filesystem, io)

__all__ = ['cli', 'CuImage', 'DLDataType', 'DLDataTypeCode', 'filesystem',
           'io', 'cache', 'converter', '__version__']


from ._cucim import _get_plugin_root  # isort:skip
from ._cucim import _set_plugin_root  # isort:skip
# Set plugin root path
_set_plugin_root(os.path.dirname(os.path.realpath(__file__)))
