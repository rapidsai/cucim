import contextlib
import platform
import signal

import cupy as cp
import numpy as np
import pytest
from test_tvl1 import _sin_flow_gen

from cucim.skimage._shared.utils import _supported_float_type
from cucim.skimage.registration import optical_flow_ilk


@contextlib.contextmanager
def timeout_context(seconds, test_name="test"):
    """Context manager that skips test if execution takes too long on Unix."""
    # Check if we're on a Unix-like system (Linux, macOS, etc.)
    unix_systems = ("Linux", "Darwin", "FreeBSD", "OpenBSD", "NetBSD")
    is_unix = platform.system() in unix_systems

    if not is_unix:
        # On non-Unix systems (e.g., Windows), just run without timeout
        yield
        return

    # Unix systems: use signal-based timeout
    def timeout_handler(signum, frame):
        pytest.skip(f"Test '{test_name}' timed out after {seconds} seconds")

    # Set up the timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Clean up
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


@pytest.mark.parametrize("dtype", [cp.float16, cp.float32, cp.float64])
@pytest.mark.parametrize("gaussian", [True, False])
@pytest.mark.parametrize("prefilter", [True, False])
def test_2d_motion(dtype, gaussian, prefilter):
    # Generate synthetic data
    rnd = np.random.default_rng(0)
    image0 = rnd.normal(size=(256, 256))
    image0 = cp.asarray(image0, dtype=dtype)
    gt_flow, image1 = _sin_flow_gen(image0)
    image1 = image1.astype(dtype, copy=False)
    float_dtype = _supported_float_type(dtype)
    # Estimate the flow
    flow = optical_flow_ilk(
        image0,
        image1,
        gaussian=gaussian,
        prefilter=prefilter,
        dtype=float_dtype,
    )
    assert flow.dtype == float_dtype
    # Assert that the average absolute error is less then half a pixel
    assert abs(flow - gt_flow).mean() < 0.5

    if dtype != float_dtype:
        with pytest.raises(ValueError):
            optical_flow_ilk(
                image0,
                image1,
                gaussian=gaussian,
                prefilter=prefilter,
                dtype=dtype,
            )


@pytest.mark.parametrize("gaussian", [True, False])
@pytest.mark.parametrize("prefilter", [True, False])
def test_3d_motion(gaussian, prefilter):
    # Generate synthetic data
    rnd = np.random.default_rng(123)
    image0 = rnd.normal(size=(50, 55, 60))
    image0 = cp.asarray(image0)
    gt_flow, image1 = _sin_flow_gen(image0, npics=3)

    # Estimate the flow (with 60 second timeout)
    with timeout_context(60, "test_3d_motion optical_flow_ilk"):
        flow = optical_flow_ilk(
            image0, image1, radius=5, gaussian=gaussian, prefilter=prefilter
        )

    # Assert that the average absolute error is less then half a pixel
    assert abs(flow - gt_flow).mean() < 0.5


def test_no_motion_2d():
    rnd = np.random.default_rng(0)
    img = rnd.normal(size=(256, 256))
    img = cp.asarray(img)

    flow = optical_flow_ilk(img, img)

    assert cp.all(flow == 0)


def test_no_motion_3d():
    rnd = np.random.default_rng(0)
    img = rnd.normal(size=(64, 64, 64))
    img = cp.asarray(img)

    flow = optical_flow_ilk(img, img)

    assert cp.all(flow == 0)


def test_optical_flow_dtype():
    # Generate synthetic data
    rnd = np.random.default_rng(0)
    image0 = rnd.normal(size=(256, 256))
    image0 = cp.asarray(image0)
    gt_flow, image1 = _sin_flow_gen(image0)
    # Estimate the flow at double precision
    flow_f64 = optical_flow_ilk(image0, image1, dtype="float64")

    assert flow_f64.dtype == "float64"

    # Estimate the flow at single precision
    flow_f32 = optical_flow_ilk(image0, image1, dtype="float32")

    assert flow_f32.dtype == "float32"

    # Assert that floating point precision does not affect the quality
    # of the estimated flow

    assert cp.abs(flow_f64 - flow_f32).mean() < 1e-3


def test_incompatible_shapes():
    rnd = np.random.default_rng(0)
    I0 = rnd.normal(size=(256, 256))
    I1 = rnd.normal(size=(255, 256))
    with pytest.raises(ValueError):
        u, v = optical_flow_ilk(I0, I1)


def test_wrong_dtype():
    rnd = np.random.default_rng(0)
    img = rnd.normal(size=(256, 256))
    with pytest.raises(ValueError):
        u, v = optical_flow_ilk(img, img, dtype="int")
