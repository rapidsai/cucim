#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
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

if _is_cupy_available:
    # If CuPy is available AND we aren't on a system CTK install, then
    # the end-user will have to have used the `ctk` extra to install `cupy-cuda13x` (or similar)
    # `cuda.pathfinder` will also be available
    # Use it to pre-load `cusolver` to get around an upstream issue in CuPy:
    # https://github.com/cupy/cupy/issues/10095
    #
    # If CuPy is available and we ARE on a system CTK install, then a user might have not installed the `ctk` extra
    # and so `cuda.pathfinder` might not be available. In this scenario, we don't NEED `cuda.pathfinder` since on CTK installs
    # CuPy can load `libcusolver.so` without issue.
    #
    # If CuPy is available, on a system CTK install, AND user installed the
    # `ctk` extra, all we do is pre-load the DSO that would've been loaded
    # anyway
    try:
        from cuda.pathfinder import (
            DynamicLibNotFoundError,
            load_nvidia_dynamic_lib,
        )

        load_nvidia_dynamic_lib("cusolver")
    except (ImportError, DynamicLibNotFoundError):
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
