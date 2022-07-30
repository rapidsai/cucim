import cupy as cp
import pytest

from cucim.skimage.filters._separable_conv_shmem import convolve1d


def _compare_implementations(
    shape, kernel_size, axis, dtype, mode, cval=0.0, origin=0,
    output_dtype=None, kernel_dtype=None, output_preallocated=False
):
    dtype = cp.dtype(dtype)
    if kernel_dtype is None:
        kernel_dtype = dtype
    rng = cp.random.default_rng(123)
    rtol = atol = 1e-5
    if dtype.kind == 'b':
        image = rng.integers(0, 1, shape, dtype=cp.uint8).astype(bool)
        kernel = rng.integers(0, 1, (kernel_size,), dtype=cp.uint8).astype(bool)
    elif dtype.kind in 'iu':
        image = rng.integers(0, 128, shape, dtype=dtype)
        kernel = rng.integers(0, 128, (kernel_size,), dtype=dtype)
    elif dtype.kind in 'c':
        real_dtype = cp.asarray([], dtype=dtype).real.dtype
        image = rng.standard_normal(shape, dtype=real_dtype)
        kernel = rng.standard_normal((kernel_size,), dtype=real_dtype)
        image = image + 1j * rng.standard_normal(shape, dtype=real_dtype)
        kernel = kernel + 1j * rng.standard_normal(
            (kernel_size,), dtype=real_dtype
        )
        if real_dtype == cp.float64:
            rtol = atol = 1e-10
    else:
        if dtype == cp.float16:
            image = rng.standard_normal(shape).astype(dtype)
            kernel = rng.standard_normal((kernel_size,)).astype(dtype)
            rtol = atol = 1e-3
        else:
            image = rng.standard_normal(shape, dtype=dtype)
            kernel = rng.standard_normal((kernel_size,), dtype=dtype)
            if dtype == cp.float64:
                rtol = atol = 1e-10
    kwargs = dict(axis=axis, mode=mode, cval=cval, origin=origin)
    if output_dtype is not None:
        output_dtype = cp.dtype(output_dtype)
    if output_preallocated:
        if output_dtype is None:
            output_dtype = image.dtype
        output1 = cp.empty(image.shape, dtype=output_dtype)
        output2 = cp.empty(image.shape, dtype=output_dtype)
        convolve1d(image, kernel, output=output1, algorithm='elementwise')
        convolve1d(
            image, kernel, output=output2, algorithm='shared_memory'
        )
        cp.testing.assert_allclose(output1, output2, rtol=rtol, atol=atol)
        return
    output1 = convolve1d(
        image, kernel, output=output_dtype, algorithm='elementwise'
    )
    output2 = convolve1d(
        image, kernel, output=output_dtype, algorithm='shared_memory'
    )
    cp.testing.assert_allclose(output1, output2, rtol=rtol, atol=atol)
    return


@pytest.mark.parametrize('shape', ((64, 57),))
@pytest.mark.parametrize('axis', (0, 1))
@pytest.mark.parametrize('origin', ('min', 0, 'max'))
@pytest.mark.parametrize(
    'kernel_size', tuple(range(1, 16)) + (30, 31, 63, 64, 125, 257)
)
def test_separable_kernel_sizes_and_origins(shape, axis, origin, kernel_size):
    if origin == 'min':
        origin = -kernel_size // 2
    elif origin == 'max':
        origin = kernel_size // 2
        if kernel_size % 2 == 0:
            origin -= 1
    _compare_implementations(
        shape,
        kernel_size=kernel_size,
        axis=axis,
        dtype=cp.float32,
        mode='nearest',
        origin=origin
    )


# test large and small sizes, highly anisotropic sizes
@pytest.mark.parametrize('shape', ((4000, 2000), (1, 1), (5, 500), (1500, 5)))
@pytest.mark.parametrize('axis', (-1, -2))
@pytest.mark.parametrize('kernel_size', (1, 7, 8))
@pytest.mark.parametrize(
    'mode', ('nearest', 'constant', 'reflect', 'wrap', 'mirror')
)
def test_separable_image_shapes_and_modes(shape, axis, kernel_size, mode):
    _compare_implementations(
        shape,
        kernel_size=kernel_size,
        axis=axis,
        dtype=cp.float32,
        mode=mode,
        origin=0,
    )


image_dtypes_tested = (
    cp.float16, cp.float32, cp.float64, cp.complex64, cp.complex128, bool,
    cp.int8, cp.uint8, cp.int16, cp.uint16, cp.int32, cp.uint32, cp.int64,
    cp.uint64,
)
@pytest.mark.parametrize('axis', (0, 1))
@pytest.mark.parametrize('image_dtype', image_dtypes_tested)
@pytest.mark.parametrize(
    'kernel_dtype', (None, cp.float16, cp.uint8, cp.complex64)
)
def test_separable_image_and_kernel_dtypes(axis, image_dtype, kernel_dtype):
    """Test many kernel and image dtype combinations"""
    _compare_implementations(
        (64, 32),
        kernel_size=5,
        axis=axis,
        dtype=image_dtype,
        mode='nearest',
        origin=0,
        kernel_dtype=kernel_dtype,
    )


image_dtypes_tested = (
    cp.float16, cp.float32, cp.float64, cp.complex64, cp.complex128, bool,
    cp.int8, cp.uint8, cp.int16, cp.uint16, cp.int32, cp.uint32, cp.int64,
    cp.uint64,
)
@pytest.mark.parametrize('axis', (0, 1))
@pytest.mark.parametrize('image_dtype', image_dtypes_tested)
@pytest.mark.parametrize(
    'output_dtype', (None, cp.float32, cp.int32, cp.complex64)
)
@pytest.mark.parametrize('output_preallocated', (False, True))
def test_separable_input_and_output_dtypes(
    axis, image_dtype, output_dtype, output_preallocated
):
    """Test many kernel and image dtype combinations"""
    if cp.dtype(image_dtype).kind == 'c' and output_dtype is not None:
        if not cp.dtype(output_dtype).kind == 'c':
            pytest.skip(reason='cannot cast complex values to real')
    _compare_implementations(
        (64, 32),
        kernel_size=5,
        axis=axis,
        dtype=image_dtype,
        mode='nearest',
        origin=0,
        kernel_dtype=None,
        output_dtype=output_dtype,
        output_preallocated=output_preallocated,
    )

# TODO: test correlate mode
# TODO: add separable min and max as well
# TODO: extend to nd
# TODO: extend to multi-channel data
