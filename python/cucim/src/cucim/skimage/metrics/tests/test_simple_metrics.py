# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import math

import cupy as cp
import numpy as np
import pytest
from skimage import data

from cucim.skimage._shared._warnings import expected_warnings
from cucim.skimage.metrics import (
    mean_squared_error,
    normalized_mutual_information,
    normalized_root_mse,
    peak_signal_noise_ratio,
)

np.random.seed(
    5
)  # need exact NumPy seed here. (Don't use CuPy as it won't be identical)
cam = cp.asarray(data.camera())
sigma = 20.0
noise = cp.asarray(sigma * np.random.randn(*cam.shape))
cam_noisy = cp.clip(cam + noise, 0, 255)
cam_noisy = cam_noisy.astype(cam.dtype)

assert_equal = cp.testing.assert_array_equal
assert_almost_equal = cp.testing.assert_array_almost_equal


@pytest.mark.parametrize("dtype", [cp.uint8, cp.float32, cp.float64])
def test_PSNR_vs_IPOL(dtype):
    """Tests vs. imdiff result from the following IPOL article and code:
    https://www.ipol.im/pub/art/2011/g_lmii/.

    Values for current data.camera() calculated by Gregory Lee on Sep, 2020.
    Available at:
    https://github.com/scikit-image/scikit-image/pull/4913#issuecomment-700653165
    """
    p_IPOL = 22.409353363576034
    p = peak_signal_noise_ratio(
        cam.astype(dtype), cam_noisy.astype(dtype), data_range=255
    )
    # internally, mean_square_error always sets dtype=cp.float64 for accuracy
    assert p.dtype == cp.float64
    assert_almost_equal(p, p_IPOL, decimal=4)


@pytest.mark.parametrize("dtype", [cp.float16, cp.float32, cp.float64])
def test_PSNR_float(dtype):
    p_uint8 = peak_signal_noise_ratio(cam, cam_noisy)
    camf = (cam / 255.0).astype(dtype, copy=False)
    camf_noisy = (cam_noisy / 255.0).astype(dtype, copy=False)
    p_float64 = peak_signal_noise_ratio(camf, camf_noisy, data_range=1)
    assert p_float64.dtype == cp.float64
    decimal = 3 if dtype == cp.float16 else 5
    assert_almost_equal(p_uint8, p_float64, decimal=decimal)

    # mixed precision inputs
    p_mixed = peak_signal_noise_ratio(
        cam / 255.0, (cam_noisy / 255.0).astype(cp.float32), data_range=1
    )

    assert_almost_equal(p_mixed, p_float64, decimal=decimal)

    # mismatched dtype results in a warning if data_range is unspecified
    with expected_warnings(["Inputs have mismatched dtype"]):
        p_mixed = peak_signal_noise_ratio(
            cam / 255.0, (cam_noisy / 255.0).astype(cp.float32)
        )
    assert_almost_equal(p_mixed, p_float64, decimal=decimal)


def test_PSNR_errors():
    # shape mismatch
    with pytest.raises(ValueError):
        peak_signal_noise_ratio(cam, cam[:-1, :])


@pytest.mark.parametrize("dtype", [cp.float16, cp.float32, cp.float64])
def test_NRMSE(dtype):
    x = cp.ones(4, dtype=dtype)
    y = cp.asarray([0.0, 2.0, 2.0, 2.0], dtype=dtype)
    nrmse = normalized_root_mse(y, x, normalization="mean")
    assert nrmse.dtype == cp.float64
    assert_almost_equal(nrmse, 1 / cp.mean(y, dtype=cp.float64))
    assert_almost_equal(
        normalized_root_mse(y, x, normalization="euclidean"), 1 / math.sqrt(3)
    )
    assert_almost_equal(
        normalized_root_mse(y, x, normalization="min-max"),
        1 / (y.max() - y.min()),
    )

    # mixed precision inputs are allowed
    assert_almost_equal(
        normalized_root_mse(y, x.astype(cp.float32), normalization="min-max"),
        1 / (y.max() - y.min()),
    )


def test_NRMSE_no_int_overflow():
    camf = cam.astype(cp.float32)
    cam_noisyf = cam_noisy.astype(cp.float32)
    assert_almost_equal(
        mean_squared_error(cam, cam_noisy), mean_squared_error(camf, cam_noisyf)
    )
    assert_almost_equal(
        normalized_root_mse(cam, cam_noisy),
        normalized_root_mse(camf, cam_noisyf),
    )


def test_NRMSE_errors():
    x = cp.ones(4)
    # shape mismatch
    with pytest.raises(ValueError):
        normalized_root_mse(x[:-1], x)
    # invalid normalization name
    with pytest.raises(ValueError):
        normalized_root_mse(x, x, normalization="foo")


def test_nmi():
    assert_almost_equal(float(normalized_mutual_information(cam, cam)), 2)
    assert normalized_mutual_information(
        cam, cam_noisy
    ) < normalized_mutual_information(cam, cam)


def test_nmi_different_sizes():
    assert float(normalized_mutual_information(cam[:, :400], cam[:400, :])) > 1


@pytest.mark.parametrize("dtype", [cp.float16, cp.float32, cp.float64])
def test_nmi_random(dtype):
    rng = cp.random.default_rng()
    random1 = rng.random((100, 100)).astype(dtype)
    random2 = rng.random((100, 100)).astype(dtype)
    nmi = normalized_mutual_information(random1, random2, bins=10)
    assert nmi.dtype == cp.float64
    assert_almost_equal(nmi, 1, decimal=2)


def test_nmi_random_3d():
    random1, random2 = cp.random.random((2, 10, 100, 100))
    assert_almost_equal(
        float(normalized_mutual_information(random1, random2, bins=10)),
        1,
        decimal=2,
    )


def _previous_minmax_nrmse(image_true, image_test):
    """Reference the separate reductions and image temporaries used previously."""
    from cucim.skimage._shared.utils import _supported_float_type

    dtype = _supported_float_type([image_true.dtype, image_test.dtype])
    image_true, image_test = (
        x.astype(dtype, copy=False) for x in (image_true, image_test)
    )
    return cp.sqrt(mean_squared_error(image_true, image_test)) / (
        image_true.max() - image_true.min()
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("shape", [(7, 9), (257, 513), (3, 32771)])
@pytest.mark.parametrize(
    "layout", ["contiguous", "different_strides", "reverse", "slice"]
)
@pytest.mark.filterwarnings(
    "ignore:cupyx.allow_synchronize will be removed:UserWarning"
)
def test_NRMSE_minmax_combined(dtype, shape, layout):
    from cupyx import allow_synchronize

    rng = cp.random.default_rng(0)
    image_true = rng.random(shape, dtype=dtype) * 2 - 1
    image_test = rng.random(shape, dtype=dtype) * 2 - 1
    if layout == "different_strides":
        image_test = cp.asfortranarray(image_test)
    elif layout == "reverse":
        image_true = image_true[::-1, ::-1]
    elif layout == "slice":
        image_true = image_true[:, ::2]
        image_test = image_test[:, ::2]
    expected = _previous_minmax_nrmse(image_true, image_test)
    with allow_synchronize(False):
        actual = normalized_root_mse(
            image_true, image_test, normalization="min-max"
        )
    assert actual.dtype == cp.float64
    assert actual.shape == ()
    cp.testing.assert_allclose(actual, expected, rtol=1e-12)


@pytest.mark.parametrize(
    "dtype", ["uint8", "int16", "float16", "float32", "float64"]
)
@pytest.mark.parametrize("test_dtype", ["float32", "float64"])
def test_NRMSE_minmax_dtype_promotion(dtype, test_dtype):
    image_true = cp.arange(99, dtype=dtype).reshape(9, 11)
    image_test = (image_true + 0.125).astype(test_dtype)
    expected = _previous_minmax_nrmse(image_true, image_test)
    actual = normalized_root_mse(
        image_true, image_test, normalization="MIN-MAX"
    )
    assert actual.dtype == cp.float64
    cp.testing.assert_allclose(actual, expected, rtol=1e-12)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "case", ["true_nan", "test_nan", "all_nan", "infinity", "constant", "zero"]
)
def test_NRMSE_minmax_nonfinite_and_constant(dtype, case):
    image_true = cp.linspace(-1, 1, 131073, dtype=dtype)
    image_test = image_true + 0.25
    if case == "true_nan":
        image_true[4096] = cp.nan
    elif case == "test_nan":
        image_test[-1] = cp.nan
    elif case == "all_nan":
        image_true[:] = cp.nan
    elif case == "infinity":
        image_true[0] = cp.inf
    elif case == "constant":
        image_true[:] = 1
        image_test[:] = 2
    else:
        image_true[:] = image_test[:] = 0
    expected = _previous_minmax_nrmse(image_true, image_test)
    actual = normalized_root_mse(
        image_true, image_test, normalization="min-max"
    )
    cp.testing.assert_allclose(actual, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize("values", [[0, 1e20], [-2e38, 2e38], [0, 1e-30]])
def test_NRMSE_minmax_float32_arithmetic(values):
    # Squaring and range subtraction must retain float32 overflow/underflow,
    # even though the squared-error sum is accumulated in float64.
    image_true = cp.asarray(values, dtype=cp.float32)
    image_test = cp.zeros_like(image_true)
    expected = _previous_minmax_nrmse(image_true, image_test)
    actual = normalized_root_mse(
        image_true, image_test, normalization="min-max"
    )
    cp.testing.assert_allclose(actual, expected, rtol=1e-12, equal_nan=True)


def test_NRMSE_minmax_empty():
    with pytest.raises(ValueError):
        normalized_root_mse(cp.empty(0), cp.empty(0), normalization="min-max")


@pytest.mark.parametrize(
    "dtype,values,data_range",
    [
        ("bool", [False, True, False], 1),
        ("uint8", [0, 127, 255], 255),
        ("int8", [-128, 0, 127], 255),
        ("int8", [0, 1, 127], 127),
        ("float32", [-0.5, 0.25, 1], 2),
        ("float32", [0, 0.25, 1], 1),
        ("float64", [-0.5, 0.25, 1], 2),
        ("float64", [0, 0.25, 1], 1),
    ],
)
@pytest.mark.parametrize("transpose", [False, True])
def test_PSNR_auto_range_extrema(dtype, values, data_range, transpose):
    image_true = cp.resize(cp.asarray(values, dtype=dtype), (257, 513))
    if transpose:
        image_true = image_true.T
    image_test = cp.zeros_like(image_true)
    expected = peak_signal_noise_ratio(
        image_true, image_test, data_range=data_range
    )
    actual = peak_signal_noise_ratio(image_true, image_test)
    assert actual.dtype == cp.float64
    cp.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "values", [[-1.1, 0], [0, 1.1], [0, np.inf], [-np.inf, 0]]
)
def test_PSNR_auto_range_rejects_invalid_extrema(values):
    image_true = cp.asarray(values, dtype=cp.float32)
    with pytest.raises(ValueError, match="outside the range expected"):
        peak_signal_noise_ratio(image_true, cp.zeros_like(image_true))


def test_PSNR_auto_range_nan_propagates():
    # NaNs keep propagating even if the other values would fail validation.
    image_true = cp.asarray([np.nan, -2, 2], dtype=cp.float32)
    actual = peak_signal_noise_ratio(image_true, cp.zeros_like(image_true))
    assert bool(cp.isnan(actual))


def test_PSNR_auto_range_float32_negative_subnormal():
    tiny = np.nextafter(np.float32(0), np.float32(1))
    image_true = cp.asarray([0, -tiny, 0.5], dtype=cp.float32)
    image_test = cp.zeros_like(image_true)
    expected = peak_signal_noise_ratio(image_true, image_test, data_range=2)
    cp.testing.assert_array_equal(
        peak_signal_noise_ratio(image_true, image_test), expected
    )
