import cupy as cp
import numpy as np
import pytest

from cucim.skimage.filters import unsharp_mask


@pytest.mark.parametrize("shape,multichannel",
                         [((29,), False),
                          ((40, 4), True),
                          ((32, 32), False),
                          ((29, 31, 3), True),
                          ((13, 17, 4, 8), False)])
@pytest.mark.parametrize("dtype", [np.uint8, np.int8,
                                   np.uint16, np.int16,
                                   np.uint32, np.int32,
                                   np.uint64, np.int64,
                                   np.float16, np.float32, np.float64])
@pytest.mark.parametrize("radius", [0, 0.1, 2.0])
@pytest.mark.parametrize("amount", [0.0, 0.5, 2.0, -1.0])
@pytest.mark.parametrize("offset", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("preserve", [False, True])
def test_unsharp_masking_output_type_and_shape(
        radius, amount, shape, multichannel, dtype, offset, preserve):
    array = cp.random.random(shape)
    array = ((array + offset) * 128).astype(dtype)
    if (preserve is False) and (dtype in [np.float32, np.float64]):
        array /= max(cp.abs(array).max(), 1.0)
    output = unsharp_mask(array, radius, amount, multichannel, preserve)
    assert output.dtype in [np.float16, np.float32, np.float64]
    if np.dtype(dtype).kind == 'f':
        assert output.dtype == dtype
    assert output.shape == shape


@pytest.mark.parametrize("shape,multichannel",
                         [((32, 32), False),
                          ((15, 15, 2), True),
                          ((17, 19, 3), True)])
@pytest.mark.parametrize("radius", [(0.0, 0.0), (1.0, 1.0), (2.0, 1.5)])
@pytest.mark.parametrize("preserve", [False, True])
def test_unsharp_masking_with_different_radii(radius, shape,
                                              multichannel, preserve):
    amount = 1.0
    dtype = np.float64
    array = (cp.random.random(shape) * 96).astype(dtype)
    if preserve is False:
        array /= max(cp.abs(array).max(), 1.0)
    output = unsharp_mask(array, radius, amount, multichannel, preserve)
    assert output.dtype in [np.float32, np.float64]
    assert output.shape == shape


@pytest.mark.parametrize("shape,multichannel",
                         [((16, 16), False),
                          ((15, 15, 2), True),
                          ((13, 17, 3), True)])
@pytest.mark.parametrize("offset", [-5, 0, 5])
@pytest.mark.parametrize("preserve", [False, True])
def test_unsharp_masking_with_different_ranges(shape, offset,
                                               multichannel, preserve):
    radius = 2.0
    amount = 1.0
    dtype = np.int16
    array = (cp.random.random(shape) * 5 + offset).astype(dtype)
    negative = cp.any(array < 0)
    output = unsharp_mask(array, radius, amount, multichannel, preserve)
    if preserve is False:
        assert cp.any(output <= 1)
        assert cp.any(output >= -1)
        if negative is False:
            assert cp.any(output >= 0)
    assert output.dtype in [np.float32, np.float64]
    assert output.shape == shape
