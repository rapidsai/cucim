# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import pytest
from cupy.testing import assert_allclose, assert_array_equal
from skimage.data import camera

from cucim.skimage.util import img_as_float, random_noise
from cucim.skimage.util.noise import _normal

camerad = cp.asarray(camera())


def test_set_seed():
    seed = 42
    cam = cp.asarray(camerad)
    test = random_noise(cam, rng=seed)
    assert_array_equal(test, random_noise(cam, rng=seed))


def test_salt():
    amount = 0.15
    cam = img_as_float(camerad)
    cam_noisy = random_noise(cam, rng=42, mode="salt", amount=amount)
    saltmask = cam != cam_noisy

    # Ensure all changes are to 1.0
    assert_allclose(cam_noisy[saltmask], cp.ones(int(saltmask.sum())))

    # Ensure approximately correct amount of noise was added
    proportion = float(saltmask.sum()) / (cam.shape[0] * cam.shape[1])
    tolerance = 1e-2
    assert abs(amount - proportion) <= tolerance


def test_salt_p1():
    image = cp.random.rand(2, 3)
    noisy = random_noise(image, mode="salt", amount=1)
    assert_array_equal(noisy, [[1, 1, 1], [1, 1, 1]])


def test_singleton_dim():
    """Ensure images where size of a given dimension is 1 work correctly."""
    image = cp.random.rand(1, 20)
    noisy = random_noise(image, mode="salt", amount=0.1, rng=42)
    assert cp.sum(noisy == 1) == 3  # GRL: modified to match value for CuPy


def test_pepper():
    seed = 42
    cam = img_as_float(camerad)
    data_signed = cam * 2.0 - 1.0  # Same image, on range [-1, 1]

    amount = 0.15
    cam_noisy = random_noise(cam, rng=seed, mode="pepper", amount=amount)
    peppermask = cam != cam_noisy

    # Ensure all changes are to 1.0
    assert_allclose(cam_noisy[peppermask], cp.zeros(int(peppermask.sum())))

    # Ensure approximately correct amount of noise was added
    proportion = float(peppermask.sum()) / (cam.shape[0] * cam.shape[1])
    tolerance = 1e-2
    assert abs(amount - proportion) <= tolerance

    # Check to make sure pepper gets added properly to signed images
    orig_zeros = (data_signed == -1).sum()
    cam_noisy_signed = random_noise(
        data_signed, rng=seed, mode="pepper", amount=0.15
    )

    proportion = float((cam_noisy_signed == -1).sum() - orig_zeros) / (
        cam.shape[0] * cam.shape[1]
    )
    assert abs(amount - proportion) <= tolerance


def test_salt_and_pepper():
    seed = 42
    cam = img_as_float(camerad)
    cam_noisy = random_noise(
        cam, rng=seed, mode="s&p", amount=0.15, salt_vs_pepper=0.25
    )
    saltmask = cp.logical_and(cam != cam_noisy, cam_noisy == 1.0)
    peppermask = cp.logical_and(cam != cam_noisy, cam_noisy == 0.0)

    # Ensure all changes are to 0. or 1.
    assert_allclose(cam_noisy[saltmask], cp.ones(int(saltmask.sum())))
    assert_allclose(cam_noisy[peppermask], cp.zeros(int(peppermask.sum())))

    # Ensure approximately correct amount of noise was added
    proportion = float(saltmask.sum() + peppermask.sum()) / (
        cam.shape[0] * cam.shape[1]
    )
    assert 0.11 < proportion <= 0.18

    # Verify the relative amount of salt vs. pepper is close to expected
    assert 0.18 < saltmask.sum() / peppermask.sum() < 0.35


def test_gaussian():
    seed = 42
    data = cp.zeros((128, 128)) + 0.5
    data_gaussian = random_noise(data, rng=seed, var=0.01)
    assert 0.008 < data_gaussian.var() < 0.012

    data_gaussian = random_noise(data, rng=seed, mean=0.3, var=0.015)
    assert 0.28 < data_gaussian.mean() - 0.5 < 0.32
    assert 0.012 < data_gaussian.var() < 0.018


def test_localvar():
    seed = 42
    data = cp.zeros((128, 128)) + 0.5
    local_vars = cp.zeros((128, 128)) + 0.001
    local_vars[:64, 64:] = 0.1
    local_vars[64:, :64] = 0.25
    local_vars[64:, 64:] = 0.45

    data_gaussian = random_noise(
        data, mode="localvar", rng=seed, local_vars=local_vars, clip=False
    )
    assert 0.0 < data_gaussian[:64, :64].var() < 0.002
    assert 0.095 < data_gaussian[:64, 64:].var() < 0.105
    assert 0.245 < data_gaussian[64:, :64].var() < 0.255
    assert 0.445 < data_gaussian[64:, 64:].var() < 0.455

    # Ensure local variance bounds checking works properly
    bad_local_vars = cp.zeros_like(data)
    with pytest.raises(ValueError):
        random_noise(data, mode="localvar", rng=seed, local_vars=bad_local_vars)

    bad_local_vars += 0.1
    bad_local_vars[0, 0] = -1
    with pytest.raises(ValueError):
        random_noise(data, mode="localvar", rng=seed, local_vars=bad_local_vars)


def test_speckle():
    seed = 42
    data = cp.zeros((128, 128)) + 0.1

    rng = cp.random.default_rng(seed)
    noise = _normal(rng, 0.1, 0.02**0.5, (128, 128))
    expected = cp.clip(data + data * noise, 0, 1)

    data_speckle = random_noise(
        data, mode="speckle", rng=seed, mean=0.1, var=0.02
    )
    assert_allclose(expected, data_speckle)


def test_poisson():
    seed = 42
    data = camerad  # 512x512 grayscale uint8
    cam_noisy = random_noise(data, mode="poisson", rng=seed)
    cam_noisy2 = random_noise(data, mode="poisson", rng=seed, clip=False)

    rng = cp.random.default_rng(seed)
    expected = rng.poisson(img_as_float(data) * 256) / 256.0
    assert_allclose(cam_noisy, cp.clip(expected, 0.0, 1.0))
    assert_allclose(cam_noisy2, expected)


def test_clip_poisson():
    seed = 42
    data = camerad  # 512x512 grayscale uint8
    data_signed = img_as_float(data) * 2.0 - 1.0  # Same image, on range [-1, 1]

    # Signed and unsigned, clipped
    cam_poisson = random_noise(data, mode="poisson", rng=seed, clip=True)
    cam_poisson2 = random_noise(
        data_signed, mode="poisson", rng=seed, clip=True
    )
    assert (cam_poisson.max() == 1.0) and (cam_poisson.min() == 0.0)
    assert (cam_poisson2.max() == 1.0) and (cam_poisson2.min() == -1.0)

    # Signed and unsigned, unclipped
    cam_poisson = random_noise(data, mode="poisson", rng=seed, clip=False)
    cam_poisson2 = random_noise(
        data_signed, mode="poisson", rng=seed, clip=False
    )
    assert (cam_poisson.max() > 1.15) and (cam_poisson.min() == 0.0)
    assert (cam_poisson2.max() > 1.3) and (cam_poisson2.min() == -1.0)


def test_clip_gaussian():
    seed = 42
    data = camerad  # 512x512 grayscale uint8
    data_signed = img_as_float(data) * 2.0 - 1.0  # Same image, on range [-1, 1]

    # Signed and unsigned, clipped
    cam_gauss = random_noise(data, mode="gaussian", rng=seed, clip=True)
    cam_gauss2 = random_noise(data_signed, mode="gaussian", rng=seed, clip=True)
    assert (cam_gauss.max() == 1.0) and (cam_gauss.min() == 0.0)
    assert (cam_gauss2.max() == 1.0) and (cam_gauss2.min() == -1.0)

    # Signed and unsigned, unclipped
    cam_gauss = random_noise(data, mode="gaussian", rng=seed, clip=False)
    cam_gauss2 = random_noise(
        data_signed, mode="gaussian", rng=seed, clip=False
    )
    assert (cam_gauss.max() > 1.22) and (cam_gauss.min() < -0.33)
    assert (cam_gauss2.max() > 1.219) and (cam_gauss2.min() < -1.219)


def test_clip_speckle():
    seed = 42
    data = camerad  # 512x512 grayscale uint8
    data_signed = img_as_float(data) * 2.0 - 1.0  # Same image, on range [-1, 1]

    # Signed and unsigned, clipped
    cam_speckle = random_noise(data, mode="speckle", rng=seed, clip=True)
    cam_speckle_sig = random_noise(
        data_signed, mode="speckle", rng=seed, clip=True
    )
    assert (cam_speckle.max() == 1.0) and (cam_speckle.min() == 0.0)
    assert (cam_speckle_sig.max() == 1.0) and (cam_speckle_sig.min() == -1.0)

    # Signed and unsigned, unclipped
    cam_speckle = random_noise(data, mode="speckle", rng=seed, clip=False)
    cam_speckle_sig = random_noise(
        data_signed, mode="speckle", rng=seed, clip=False
    )
    assert (cam_speckle.max() > 1.219) and (cam_speckle.min() == 0.0)
    assert (cam_speckle_sig.max() > 1.219) and (cam_speckle_sig.min() < -1.219)


def test_bad_mode():
    data = cp.zeros((64, 64))
    with pytest.raises(KeyError):
        random_noise(data, "perlin")
