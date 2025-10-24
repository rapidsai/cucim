# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
This module will hold copies of any upstream CuPy code that is needed, but has
not yet been merged to CuPy master.
"""

from cucim.skimage._vendored._pearsonr import pearsonr
from cucim.skimage._vendored.pad import pad
from cucim.skimage._vendored.signaltools import *  # noqa
