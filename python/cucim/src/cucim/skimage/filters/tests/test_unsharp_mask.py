import cupy as cp
import pytest
from skimage._shared._warnings import expected_warnings

from cucim.skimage._shared.utils import _supported_float_type
from cucim.skimage.filters import unsharp_mask


@pytest.mark.parametrize("shape,multichannel",
                         [((29,), False),
                          ((40, 4), True),
                          ((32, 32), False),
                          ((29, 31, 3), True),
                          ((13, 17, 4, 8), False)])
@pytest.mark.parametrize("dtype", [cp.uint8, cp.int8,
                                   cp.uint16, cp.int16,
                                   cp.uint32, cp.int32,
                                   cp.uint64, cp.int64,
                                   cp.float16, cp.float32, cp.float64])
@pytest.mark.parametrize("radius", [0, 0.1, 2.0])
@pytest.mark.parametrize("amount", [0.0, 0.5, 2.0, -1.0])
@pytest.mark.parametrize("offset", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("preserve", [False, True])
def test_unsharp_masking_output_type_and_shape(
        radius, amount, shape, multichannel, dtype, offset, preserve):
    array = cp.random.random(shape)
    array = ((array + offset) * 128).astype(dtype)
    if (preserve is False) and (dtype in [cp.float32, cp.float64]):
        array /= max(cp.abs(array).max(), 1.0)
    channel_axis = -1 if multichannel else None
    output = unsharp_mask(array, radius, amount, preserve_range=preserve,
                          channel_axis=channel_axis)
    assert output.dtype in [cp.float32, cp.float64]
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
    dtype = cp.float64
    array = (cp.random.random(shape) * 96).astype(dtype)
    if preserve is False:
        array /= max(cp.abs(array).max(), 1.0)
    channel_axis = -1 if multichannel else None
    output = unsharp_mask(array, radius, amount, preserve_range=preserve,
                          channel_axis=channel_axis)
    assert output.dtype in [cp.float32, cp.float64]
    assert output.shape == shape


@pytest.mark.parametrize("shape, channel_axis",
                         [((16, 16), None),
                          ((15, 15, 2), -1),
                          ((13, 17, 3), -1),
                          ((2, 15, 15), 0),
                          ((3, 13, 17), 0)])
@pytest.mark.parametrize("offset", [-5, 0, 5])
@pytest.mark.parametrize("preserve", [False, True])
def test_unsharp_masking_with_different_ranges(shape, offset, channel_axis,
                                               preserve):
    radius = 2.0
    amount = 1.0
    dtype = cp.int16
    array = (cp.random.random(shape) * 5 + offset).astype(dtype)
    negative = cp.any(array < 0)
    output = unsharp_mask(array, radius, amount, preserve_range=preserve,
                          channel_axis=channel_axis)
    if preserve is False:
        assert cp.any(output <= 1)
        assert cp.any(output >= -1)
        if negative is False:
            assert cp.any(output >= 0)
    assert output.dtype in [cp.float32, cp.float64]
    assert output.shape == shape


@pytest.mark.parametrize("shape, multichannel",
                         [((16, 16), False),
                          ((15, 15, 2), True),
                          ((13, 17, 3), True)])
@pytest.mark.parametrize("offset", [-5, 0, 5])
@pytest.mark.parametrize("preserve", [False, True])
def test_unsharp_masking_with_different_ranges_dep(shape, offset,
                                                   multichannel, preserve):
    radius = 2.0
    amount = 1.0
    dtype = cp.int16
    array = (cp.random.random(shape) * 5 + offset).astype(dtype)
    negative = cp.any(array < 0)
    with expected_warnings(["`multichannel` is a deprecated argument"]):
        output = unsharp_mask(array, radius, amount, multichannel=multichannel,
                              preserve_range=preserve)
    if preserve is False:
        assert cp.any(output <= 1)
        assert cp.any(output >= -1)
        if negative is False:
            assert cp.any(output >= 0)
    assert output.dtype in [cp.float32, cp.float64]
    assert output.shape == shape

    # providing multichannel positionally also raises a warning
    with expected_warnings(["Providing the `multichannel`"]):
        output = unsharp_mask(array, radius, amount, multichannel, preserve)


@pytest.mark.parametrize("shape,channel_axis",
                         [((16, 16), None),
                          ((15, 15, 2), -1),
                          ((13, 17, 3), -1)])
@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("dtype",
                         [cp.uint8, cp.float16, cp.float32, cp.float64])
def test_unsharp_masking_dtypes(shape, channel_axis, preserve, dtype):
    radius = 2.0
    amount = 1.0
    array = (cp.random.random(shape) * 10).astype(dtype, copy=False)
    negative = cp.any(array < 0)
    output = unsharp_mask(array, radius, amount, preserve_range=preserve,
                          channel_axis=channel_axis)
    if preserve is False:
        assert cp.any(output <= 1)
        assert cp.any(output >= -1)
        if negative is False:
            assert cp.any(output >= 0)
    assert output.dtype == _supported_float_type(dtype)
    assert output.shape == shape
