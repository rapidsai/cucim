import cupy as cp
import numpy as np
import pytest

from cucim.skimage._shared.utils import _supported_float_type
from cucim.skimage.registration import optical_flow_tvl1
from cucim.skimage.transform import warp


def _sin_flow_gen(image0, max_motion=4.5, npics=5):
    """Generate a synthetic ground truth optical flow with a sinusoid as
      first component.

    Parameters
    ----------
    image0: ndarray
        The base image to be warped.
    max_motion: float
        Maximum flow magnitude.
    npics: int
        Number of sinusoid pics.

    Returns
    -------
    flow, image1 : ndarray
        The synthetic ground truth optical flow with a sinusoid as
        first component and the corresponding warped image.

    """
    grid = cp.meshgrid(*[cp.arange(n) for n in image0.shape], indexing="ij")
    grid = cp.stack(grid)
    # TODO: make upstream scikit-image PR changing gt_flow dtype to float
    gt_flow = cp.zeros_like(grid, dtype=float)
    gt_flow[0, ...] = max_motion * cp.sin(
        grid[0] / grid[0].max() * npics * np.pi
    )
    image1 = warp(image0, grid - gt_flow, mode="edge")
    return gt_flow, image1


@pytest.mark.parametrize("dtype", [cp.float16, cp.float32, cp.float64])
def test_2d_motion(dtype):
    # Generate synthetic data
    rnd = cp.random.RandomState(0)
    image0 = cp.array(rnd.normal(size=(256, 256)).astype(dtype))
    gt_flow, image1 = _sin_flow_gen(image0)
    image1 = image1.astype(dtype, copy=False)
    float_dtype = _supported_float_type(dtype)
    # Estimate the flow
    flow = optical_flow_tvl1(image0, image1, attachment=5, dtype=float_dtype)
    assert flow.dtype == float_dtype
    # Assert that the average absolute error is less then half a pixel
    assert abs(flow - gt_flow).mean() < 0.5

    if dtype != float_dtype:
        with pytest.raises(ValueError):
            optical_flow_tvl1(image0, image1, attachment=5, dtype=dtype)


@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
def test_3d_motion(dtype):
    # Generate synthetic data
    rnd = np.random.RandomState(0)
    image0 = cp.array(rnd.normal(size=(100, 100, 100))).astype(dtype)
    gt_flow, image1 = _sin_flow_gen(image0)
    image1 = image1.astype(dtype, copy=False)
    # Estimate the flow
    # TODO: note: when changing _sin_flow_gen to use a float deformation field
    #             had to increase attachment here from 5 to pass the tolerance.
    flow = optical_flow_tvl1(image0, image1, attachment=10, dtype=dtype)
    assert flow.dtype == dtype
    # Assert that the average absolute error is less then half a pixel
    assert abs(flow - gt_flow).mean() < 0.5


def test_no_motion_2d():
    rnd = np.random.default_rng(0)
    img = cp.array(rnd.normal(size=(256, 256)))

    flow = optical_flow_tvl1(img, img)

    assert cp.all(flow == 0)


def test_no_motion_3d():
    rnd = np.random.default_rng(0)
    img = cp.array(rnd.normal(size=(64, 64, 64)))

    flow = optical_flow_tvl1(img, img)

    assert cp.all(flow == 0)


def test_optical_flow_dtype():
    # Generate synthetic data
    rnd = np.random.default_rng(0)
    image0 = cp.array(rnd.normal(size=(256, 256)))
    gt_flow, image1 = _sin_flow_gen(image0)
    # Estimate the flow at double precision
    flow_f64 = optical_flow_tvl1(image0, image1, attachment=5, dtype=np.float64)

    assert flow_f64.dtype == np.float64

    # Estimate the flow at single precision
    flow_f32 = optical_flow_tvl1(image0, image1, attachment=5, dtype=np.float32)

    assert flow_f32.dtype == np.float32

    # Assert that floating point precision does not affect the quality
    # of the estimated flow

    assert cp.abs(flow_f64 - flow_f32).mean() < 1e-3


def test_incompatible_shapes():
    rnd = np.random.default_rng(0)
    I0 = cp.array(rnd.normal(size=(256, 256)))
    I1 = cp.array(rnd.normal(size=(128, 256)))
    with pytest.raises(ValueError):
        u, v = optical_flow_tvl1(I0, I1)


def test_wrong_dtype():
    rnd = np.random.default_rng(0)
    img = cp.array(rnd.normal(size=(256, 256)))
    with pytest.raises(ValueError):
        u, v = optical_flow_tvl1(img, img, dtype=np.int64)
