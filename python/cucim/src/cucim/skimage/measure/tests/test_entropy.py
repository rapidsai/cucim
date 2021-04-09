import cupy as cp
import numpy as np
from numpy.testing import assert_almost_equal

from cucim.skimage.measure import shannon_entropy


def test_shannon_ones():
    img = cp.ones((10, 10))
    res = shannon_entropy(img, base=np.e)
    assert_almost_equal(float(res), 0.0)


def test_shannon_all_unique():
    img = cp.arange(64)
    res = shannon_entropy(img, base=2)
    assert_almost_equal(float(res), np.log(64) / np.log(2))
