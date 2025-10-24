#
# SPDX-FileCopyrightText: Copyright (c) 2020-2021, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
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
_is_cupy_available = False
_is_clara_available = False

# Try to import cupy first.
# If cucim.clara package is imported first, you may see the following error when running on CUDA 10.x (#44)
#   python3: Relink `/usr/lib/x86_64-linux-gnu/libnccl.so.2.8.3' with `/lib/x86_64-linux-gnu/librt.so.1' for IFUNC symbol `clock_gettime'
#   Segmentation fault
submodules = []
submod_attrs = {}

from ._version import __git_commit__, __version__

try:
    import cupy

    _is_cupy_available = True
    submodules += ["core", "skimage"]
except ImportError:
    pass

try:
    from .clara import CuImage, cli

    _is_clara_available = True
    submodules += ["clara"]
    submod_attrs["clara"] = ["CuImage", "cli"]
except ImportError:
    pass

import lazy_loader as _lazy

__getattr__, __lazy_dir__, _ = _lazy.attach_stub(__name__, __file__)


def __dir__():
    return __lazy_dir__() + ["__git_commit__", "__version__", "is_available"]


def is_available(module_name: str = "") -> bool:
    """Check if a specific module is available.

    If module_name is not specified, returns True if all of the modules are
    available.

    Parameters
    ----------
    module_name : str
        Name of the module to check. (e.g. "skimage", "core", and "clara")

    Returns
    -------
    bool
        True if the module is available, False otherwise.

    """
    if module_name in ("skimage", "core"):
        return _is_cupy_available
    elif module_name == "clara":
        return _is_clara_available
    else:
        return _is_cupy_available and _is_clara_available
