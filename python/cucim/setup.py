# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from setuptools import setup
from setuptools.dist import Distribution as _Distribution


# As we vendor a prebuilt extension module, make sure the distribution is
# treated as impure so the wheel gets a platform tag.
class Distribution(_Distribution):
    def has_ext_modules(self):
        return True


setup(
    distclass=Distribution,
    options={"bdist_wheel": {"py_limited_api": "cp312"}},
)
