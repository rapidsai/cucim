import platform

import cupy as cp
import numpy as np
import pytest
from cupy.testing import assert_array_equal
from skimage import data

from cucim.skimage import color, morphology
from cucim.skimage.util import img_as_bool

img = color.rgb2gray(cp.asarray(data.astronaut()))
bw_img = img > 100 / 255.

# TODO: Some tests fail unexpectedly on ARM.
ON_AARCH64 = platform.machine() == "aarch64"
ON_AARCH64_REASON = "TODO: Test fails unexpectedly on ARM."


@pytest.mark.xfail(ON_AARCH64, reason=ON_AARCH64_REASON)
def test_non_square_image():
    isotropic_res = morphology.isotropic_erosion(bw_img[:100, :200], 3)
    binary_res = img_as_bool(morphology.binary_erosion(
        bw_img[:100, :200], morphology.disk(3)))
    assert_array_equal(isotropic_res, binary_res)


@pytest.mark.xfail(ON_AARCH64, reason=ON_AARCH64_REASON)
def test_isotropic_erosion():
    isotropic_res = morphology.isotropic_erosion(bw_img, 3)
    binary_res = img_as_bool(
        morphology.binary_erosion(bw_img, morphology.disk(3))
    )
    assert_array_equal(isotropic_res, binary_res)


def _disk_with_spacing(
    radius, dtype=cp.uint8, *, strict_radius=True, spacing=None
):
    # Identical to morphology.disk, but with a spacing parameter and without
    # decomposition. This is different from morphology.ellipse which produces a
    # slightly different footprint.
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)

    if spacing is not None:
        X *= spacing[1]
        Y *= spacing[0]

    if not strict_radius:
        radius += 0.5
    return cp.asarray((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)


@pytest.mark.xfail(ON_AARCH64, reason=ON_AARCH64_REASON)
def test_isotropic_erosion_spacing():
    isotropic_res = morphology.isotropic_dilation(bw_img, 6, spacing=(1, 2))
    binary_res = img_as_bool(
        morphology.binary_dilation(
            bw_img, _disk_with_spacing(6, spacing=(1, 2))
        )
    )
    assert_array_equal(isotropic_res, binary_res)


@pytest.mark.xfail(ON_AARCH64, reason=ON_AARCH64_REASON)
def test_isotropic_dilation():
    isotropic_res = morphology.isotropic_dilation(bw_img, 3)
    binary_res = img_as_bool(
        morphology.binary_dilation(
            bw_img, morphology.disk(3)))
    assert_array_equal(isotropic_res, binary_res)


@pytest.mark.xfail(ON_AARCH64, reason=ON_AARCH64_REASON)
def test_isotropic_closing():
    isotropic_res = morphology.isotropic_closing(bw_img, 3)
    binary_res = img_as_bool(
        morphology.binary_closing(bw_img, morphology.disk(3))
    )
    assert_array_equal(isotropic_res, binary_res)


@pytest.mark.xfail(ON_AARCH64, reason=ON_AARCH64_REASON)
def test_isotropic_opening():
    isotropic_res = morphology.isotropic_opening(bw_img, 3)
    binary_res = img_as_bool(
        morphology.binary_opening(bw_img, morphology.disk(3))
    )
    assert_array_equal(isotropic_res, binary_res)


@pytest.mark.xfail(ON_AARCH64, reason=ON_AARCH64_REASON)
def test_footprint_overflow():
    img = cp.zeros((20, 20), dtype=bool)
    img[2:19, 2:19] = True
    isotropic_res = morphology.isotropic_erosion(img, 9)
    binary_res = img_as_bool(
        morphology.binary_erosion(img, morphology.disk(9))
    )
    assert_array_equal(isotropic_res, binary_res)


@pytest.mark.xfail(ON_AARCH64, reason=ON_AARCH64_REASON)
@pytest.mark.parametrize('out_dtype', [bool, cp.uint8, cp.int32])
def test_out_argument(out_dtype):
    for func in (morphology.isotropic_erosion, morphology.isotropic_dilation,
                 morphology.isotropic_opening, morphology.isotropic_closing):
        radius = 3
        img = cp.ones((10, 10), dtype=bool)
        img[2:5, 2:5] = 0
        out = cp.zeros_like(img, dtype=out_dtype)
        out_saved = out.copy()
        if out_dtype not in [bool, cp.uint8]:
            with pytest.raises(ValueError):
                func(img, radius, out=out)
        else:
            func(img, radius, out=out)
            assert cp.any(out != out_saved)
            assert_array_equal(out, func(img, radius))
