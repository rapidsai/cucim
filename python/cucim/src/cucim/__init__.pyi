#
# SPDX-FileCopyrightText: Copyright (c) 2020-2021, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

from ._version import __git_commit__, __version__

submodules = []

try:
    import cupy

    _is_cupy_available = True
    submodules += ["core", "skimage"]
    del cupy
except ImportError:
    pass

try:
    from .clara import CuImage, cli  # noqa: F401

    _is_clara_available = True
    submodules += ["clara"]
except ImportError:
    pass

__all__ = submodules + [  # noqa: F822
    "__git_commit__",
    "__version__",
    "is_available",
]
