import cupy as cp
import numpy as np
import pytest
from cupy.testing import assert_allclose, assert_array_equal

from cucim.skimage.transform import integral_image, integrate

cp.random.seed(0)
x = (cp.random.rand(50, 50) * 255).astype(np.uint8)
s = integral_image(x)


@pytest.mark.parametrize(
    'dtype', [cp.float16, cp.float32, cp.float64, cp.uint8, cp.int32]
)
@pytest.mark.parametrize('dtype_as_kwarg', [False, True])
def test_integral_image_validity(dtype, dtype_as_kwarg):
    rstate = np.random.default_rng(1234)
    dtype_kwarg = dtype if dtype_as_kwarg else None
    y = cp.asarray((rstate.random((20, 20)) * 255).astype(dtype))
    out = integral_image(y, dtype=dtype_kwarg)
    if y.dtype.kind == 'f':
        if dtype_as_kwarg:
            assert out.dtype == dtype
            rtol = 1e-3 if dtype == np.float16 else 1e-7
            assert_allclose(out[-1, -1], y.sum(dtype=np.float64), rtol=rtol)
        else:
            assert out.dtype == np.float64
            assert_allclose(out[-1, -1], y.sum(dtype=np.float64))
    else:
        assert out.dtype.kind == y.dtype.kind
        if not (dtype_as_kwarg and dtype == np.uint8):
            # omit check for dtype=uint8 case as it will overflow
            assert out[-1, -1] == y.sum()


def test_basic():
    assert_array_equal(x[12:24, 10:20].sum(), integrate(s, (12, 10), (23, 19)))
    assert_array_equal(x[:20, :20].sum(), integrate(s, (0, 0), (19, 19)))
    assert_array_equal(x[:20, 10:20].sum(), integrate(s, (0, 10), (19, 19)))
    assert_array_equal(x[10:20, :20].sum(), integrate(s, (10, 0), (19, 19)))


def test_single():
    assert_array_equal(x[0, 0], integrate(s, (0, 0), (0, 0)))
    assert_array_equal(x[10, 10], integrate(s, (10, 10), (10, 10)))


def test_vectorized_integrate():
    r0 = np.array([12, 0, 0, 10, 0, 10, 30])
    c0 = np.array([10, 0, 10, 0, 0, 10, 31])
    r1 = np.array([23, 19, 19, 19, 0, 10, 49])
    c1 = np.array([19, 19, 19, 19, 0, 10, 49])
    # fmt: off
    x_cpu = cp.asnumpy(x)
    expected = np.array([x_cpu[12:24, 10:20].sum(),
                         x_cpu[:20, :20].sum(),
                         x_cpu[:20, 10:20].sum(),
                         x_cpu[10:20, :20].sum(),
                         x_cpu[0, 0],
                         x_cpu[10, 10],
                         x_cpu[30:, 31:].sum()])
    # fmt: on
    start_pts = [(r0[i], c0[i]) for i in range(len(r0))]
    end_pts = [(r1[i], c1[i]) for i in range(len(r0))]
    assert_array_equal(expected, integrate(s, start_pts, end_pts))
