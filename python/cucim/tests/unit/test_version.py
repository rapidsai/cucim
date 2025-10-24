#
# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cucim


def test_version_constants_are_populated():
    # __git_commit__ will only be non-empty in a built distribution
    assert isinstance(cucim.__git_commit__, str)

    # __version__ should always be non-empty
    assert isinstance(cucim.__version__, str)
    assert len(cucim.__version__) > 0
