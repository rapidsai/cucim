# Apache License, Version 2.0
# Copyright 2020 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import glob
import re
import sys
from os.path import join
from unittest.mock import patch

import auditwheel.elfutils
from auditwheel.main import main
from auditwheel.wheeltools import InWheelCtx

# How auditwheel repair works?
#
# From https://github.com/pypa/auditwheel/blob/3.2.0/auditwheel/wheel_abi.py#L38
# 1) Find a Python extension libraries(.so)
#      if so ==> A, else ==> B
# 2) `needed_libs` <== external libraries needed by A and B
# 3) From b in B,
#      if b is not in `needed_libs`, b is added to A
# 4) Only external libraries in A are patched to use locally copied .so files
#    - external libraries that exists under wheel path are discarded
#      - https://github.com/pypa/auditwheel/blob/3.2.0/auditwheel/policy/external_references.py#L61
#    - https://github.com/pypa/auditwheel/blob/3.2.0/auditwheel/repair.py#L62
#
# With current implementation,
# - `cucim/_cucim.cpython-XX-x86_64-linux-gnu.so` files are in A by 1)
# - `cucim/libcucim.so.??` is in B by 1)
# - `cucim/libcucim.so.??` and `libcudart.so.11.0` are in `needed_libs` by 2)
# - `cucim/cucim.kit.cuslide@??.??.??.so` is in A by 3)
#
# And only libz and libcudart are considered as external libraries.

# To work with cuCIM, we need to
# 1) make `cucim/libcucim.so.??` as Python extension library
#   - Patch elf_is_python_extension : https://github.com/pypa/auditwheel/blob/3.2.0/auditwheel/elfutils.py#L81
# 2) control how to copy external libraries
#   - Patch copylib: https://github.com/pypa/auditwheel/blob/3.2.0/auditwheel/repair.py#L108
#   - Need for libnvjpeg library
# 3) preprocess wheel metadata
#   - patch InWheelCtx.__enter__ : https://github.com/pypa/auditwheel/blob/3.2.0/auditwheel/wheeltools.py#L158
#   - `Root-Is-Purelib: true` -> `Root-Is-Purelib: false` from WHEEL file


# Parameters
PYTHON_EXTENSION_LIBRARIES = [r"cucim/libcucim\.so\.\d{1,2}"]

# 1) auditwheel.elfutils.elf_is_python_extension replacement
orig_elf_is_python_extension = auditwheel.elfutils.elf_is_python_extension


@functools.wraps(orig_elf_is_python_extension)
def elf_is_python_extension(fn, elf):
    if any(map(lambda x: re.fullmatch(x, fn), PYTHON_EXTENSION_LIBRARIES)):
        print("[cuCIM] Consider {} as a python extension.".format(fn))
        return True, 3
    return orig_elf_is_python_extension(fn, elf)


# 3) auditwheel.wheeltools.InWheelCtx.__enter__ replacement
orig_inwheelctx_enter = InWheelCtx.__enter__


@functools.wraps(orig_inwheelctx_enter)
def inwheelctx_enter(self):
    rtn = orig_inwheelctx_enter(self)

    # `self.path` is a path that extracted files from the wheel file exists

    # base_dir = glob.glob(join(self.path, '*.dist-info'))
    # wheel_path = join(base_dir[0], 'WHEEL')
    # with open(wheel_path, 'r') as f:
    #     wheel_text = f.read()

    # wheel_text = wheel_text.replace('Root-Is-Purelib: true', 'Root-Is-Purelib: false')

    # with open(wheel_path, 'w') as f:
    #     f.write(wheel_text)

    return rtn


# # sys.argv replacement
# testargs = ["auditwheel_repair.py", "repair", "--plat", "manylinux2014_x86_64", "-w", "wherehouse", "cuclara_image-0.1.1-py3-none-manylinux2014_x86_64.whl"]
# with patch.object(sys, 'argv', testargs):

if __name__ == "__main__":
    # Patch
    with patch.object(
        auditwheel.elfutils, "elf_is_python_extension", elf_is_python_extension
    ):
        with patch.object(InWheelCtx, "__enter__", inwheelctx_enter):
            main()
