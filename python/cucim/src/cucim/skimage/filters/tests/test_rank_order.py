# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import cupy as cp
import skimage.data
import skimage.filters

from cucim.skimage.filters import rank_order

img = cp.asarray(skimage.data.camera())


def test_rank_order():
    expected, ov_expected = skimage.filters.rank_order(img.get())
    r, ov = rank_order(img)
    cp.testing.assert_allclose(r, expected)
    cp.testing.assert_allclose(ov, ov_expected)
