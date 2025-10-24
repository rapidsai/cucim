#
# SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from ...util.io import open_image_cucim


def test_load_non_existing_image():
    with pytest.raises(ValueError, match=r"Cannot open .*"):
        _ = open_image_cucim("/tmp/non_existing_image.tif")
