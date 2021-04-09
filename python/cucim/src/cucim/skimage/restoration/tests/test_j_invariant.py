import cupy as cp
import numpy as np
import pytest
from skimage.data import camera, chelsea
# from cucim.skimage.restoration import denoise_wavelet
from skimage.restoration import denoise_wavelet

from cucim.skimage.data import binary_blobs
from cucim.skimage.metrics import mean_squared_error as mse
from cucim.skimage.restoration import calibrate_denoiser, denoise_tv_chambolle
from cucim.skimage.restoration.j_invariant import _invariant_denoise
from cucim.skimage.util import img_as_float, random_noise

test_img = img_as_float(cp.asarray(camera()))
test_img_color = img_as_float(cp.asarray(chelsea()))
test_img_3d = img_as_float(binary_blobs(64, n_dim=3)) / 2
noisy_img = random_noise(test_img, mode="gaussian", var=0.01)
noisy_img_color = random_noise(test_img_color, mode="gaussian", var=0.01)
noisy_img_3d = random_noise(test_img_3d, mode="gaussian", var=0.1)


# TODO: replace with CuPy version once completed
def _denoise_wavelet(image, rescale_sigma=True, **kwargs):
    return cp.asarray(
        denoise_wavelet(
            cp.asnumpy(image), rescale_sigma=rescale_sigma, **kwargs
        )
    )


def test_invariant_denoise():
    # denoised_img = _invariant_denoise(noisy_img, _denoise_wavelet)
    denoised_img = _invariant_denoise(noisy_img, denoise_tv_chambolle)

    denoised_mse = mse(denoised_img, test_img)
    original_mse = mse(noisy_img, test_img)
    assert denoised_mse < original_mse


@pytest.mark.parametrize('dtype', [cp.float32, cp.float64])
def test_invariant_denoise_color(dtype):
    denoised_img_color = _invariant_denoise(
        noisy_img_color.astype(dtype),
        _denoise_wavelet,
        denoiser_kwargs=dict(multichannel=True),
    )
    assert denoised_img_color.dtype == dtype

    denoised_mse = mse(denoised_img_color, test_img_color)
    original_mse = mse(noisy_img_color, test_img_color)
    assert denoised_mse < original_mse


def test_invariant_denoise_3d():
    denoised_img_3d = _invariant_denoise(noisy_img_3d, _denoise_wavelet)

    denoised_mse = mse(denoised_img_3d, test_img_3d)
    original_mse = mse(noisy_img_3d, test_img_3d)
    assert denoised_mse < original_mse


def test_calibrate_denoiser_extra_output():
    parameter_ranges = {'sigma': np.linspace(0.1, 1, 5) / 2}
    _, (parameters_tested, losses) = calibrate_denoiser(
        noisy_img,
        _denoise_wavelet,
        denoise_parameters=parameter_ranges,
        extra_output=True
    )

    all_denoised = [_invariant_denoise(noisy_img, _denoise_wavelet,
                                       denoiser_kwargs=denoiser_kwargs)
                    for denoiser_kwargs in parameters_tested]

    ground_truth_losses = [float(mse(img, test_img)) for img in all_denoised]
    assert np.argmin(losses) == np.argmin(ground_truth_losses)


def test_calibrate_denoiser():
    parameter_ranges = {'sigma': np.linspace(0.1, 1, 5) / 2}

    denoiser = calibrate_denoiser(
        noisy_img, _denoise_wavelet, denoise_parameters=parameter_ranges
    )

    denoised_mse = mse(denoiser(noisy_img), test_img)
    original_mse = mse(noisy_img, test_img)
    assert denoised_mse < original_mse


def test_calibrate_denoiser_tv():
    parameter_ranges = {"weight": np.linspace(0.01, 0.4, 10)}

    denoiser = calibrate_denoiser(
        noisy_img, denoise_tv_chambolle, denoise_parameters=parameter_ranges
    )

    denoised_mse = mse(denoiser(noisy_img), test_img)
    original_mse = mse(noisy_img, test_img)
    assert denoised_mse < original_mse


def test_input_image_not_modified():
    input_image = noisy_img.copy()

    parameter_ranges = {'sigma': np.random.random(5) / 2}
    calibrate_denoiser(
        input_image, _denoise_wavelet, denoise_parameters=parameter_ranges
    )

    assert cp.all(noisy_img == input_image)
