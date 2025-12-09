#
# SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
from unittest.mock import patch


def test_is_available():
    with patch("cucim._is_cupy_available", False):
        with patch("cucim._is_clara_available", False):
            import cucim

            assert cucim.is_available() is False
            assert cucim.is_available("skimage") is False
            assert cucim.is_available("clara") is False
            assert cucim.is_available("unknown") is False
        with patch("cucim._is_clara_available", True):
            import cucim

            assert cucim.is_available() is False
            assert cucim.is_available("skimage") is False
            assert cucim.is_available("clara") is True
            assert cucim.is_available("unknown") is False

    with patch("cucim._is_cupy_available", True):
        with patch("cucim._is_clara_available", False):
            import cucim

            assert cucim.is_available() is False
            assert cucim.is_available("skimage") is True
            assert cucim.is_available("clara") is False
            assert cucim.is_available("unknown") is False
        with patch("cucim._is_clara_available", True):
            import cucim

            assert cucim.is_available() is True
            assert cucim.is_available("skimage") is True
            assert cucim.is_available("clara") is True
            assert cucim.is_available("unknown") is True
