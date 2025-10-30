# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

from .version_requirements import is_installed

has_mpl = is_installed("matplotlib", ">=3.3")
if has_mpl:
    try:
        # will fail with
        #    ImportError: Failed to import any qt binding
        # if only matplotlib-base is installed
        from matplotlib import pyplot  # noqa
    except ImportError:
        has_mpl = False
