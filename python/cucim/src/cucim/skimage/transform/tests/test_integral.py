import cupy as cp
import numpy as np
from cupy.testing import assert_array_equal

from cucim.skimage.transform import integral_image, integrate

cp.random.seed(0)
x = (cp.random.rand(50, 50) * 255).astype(np.uint8)
s = integral_image(x)


def test_validity():
    y = cp.arange(12).reshape((4, 3))

    y = (cp.random.rand(50, 50) * 255).astype(np.uint8)
    assert_array_equal(integral_image(y)[-1, -1], y.sum())


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
