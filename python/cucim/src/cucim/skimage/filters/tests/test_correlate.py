import cupy as cp
import pytest
from cupy.testing import assert_array_equal
from cupyx.scipy import ndimage as ndi

from cucim.skimage._shared import testing
from cucim.skimage.filters import correlate_sparse


def test_correlate_sparse_valid_mode():
    image = cp.array([[0, 0, 1, 3, 5],
                      [0, 1, 4, 3, 4],
                      [1, 2, 5, 4, 1],
                      [2, 4, 5, 2, 1],
                      [4, 5, 1, 0, 0]], dtype=float)

    kernel = cp.array([0, 1, 2, 4, 8, 16, 32, 64, 128]).reshape((3, 3))

    cs_output = correlate_sparse(image, kernel, mode="valid")
    ndi_output = ndi.correlate(image, kernel, mode='wrap')
    ndi_output = ndi_output[1:4, 1:4]

    assert_array_equal(cs_output, ndi_output)


@pytest.mark.parametrize("mode", ["nearest", "reflect", "mirror"])
def test_correlate_sparse(mode):
    image = cp.array([[0, 0, 1, 3, 5],
                      [0, 1, 4, 3, 4],
                      [1, 2, 5, 4, 1],
                      [2, 4, 5, 2, 1],
                      [4, 5, 1, 0, 0]], dtype=float)

    kernel = cp.array([0, 1, 2, 4, 8, 16, 32, 64, 128]).reshape((3, 3))

    cs_output = correlate_sparse(image, kernel, mode=mode)
    ndi_output = ndi.correlate(image, kernel, mode=mode)
    assert_array_equal(cs_output, ndi_output)


@pytest.mark.parametrize("mode", ["nearest", "reflect", "mirror"])
def test_correlate_sparse_invalid_kernel(mode):
    image = cp.array([[0, 0, 1, 3, 5],
                      [0, 1, 4, 3, 4],
                      [1, 2, 5, 4, 1],
                      [2, 4, 5, 2, 1],
                      [4, 5, 1, 0, 0]], dtype=float)
    # invalid kernel size
    invalid_kernel = cp.array([0, 1, 2, 4]).reshape((2, 2))
    with testing.raises(ValueError):
        correlate_sparse(image, invalid_kernel, mode=mode)
