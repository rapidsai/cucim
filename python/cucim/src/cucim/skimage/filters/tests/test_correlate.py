import cupy as cp
import pytest
from cupy.testing import assert_array_equal
from cupyx.scipy import ndimage as ndi

from cucim.skimage._shared.utils import _supported_float_type
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
@pytest.mark.parametrize(
    "dtype", [cp.uint16, cp.int32, cp.float16, cp.float32, cp.float64]
)
def test_correlate_sparse(mode, dtype):
    image = cp.array([[0, 0, 1, 3, 5],
                      [0, 1, 4, 3, 4],
                      [1, 2, 5, 4, 1],
                      [2, 4, 5, 2, 1],
                      [4, 5, 1, 0, 0]], dtype=dtype)

    kernel = cp.array([0, 1, 2, 4, 8, 16, 32, 64, 128]).reshape((3, 3))

    cs_output = correlate_sparse(image, kernel, mode=mode)
    assert cs_output.dtype == _supported_float_type(image.dtype)
    ndi_output = ndi.correlate(
        image.astype(float, copy=False), kernel, mode=mode
    )
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
    with pytest.raises(ValueError):
        correlate_sparse(image, invalid_kernel, mode=mode)
