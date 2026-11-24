# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools

import cupy
import numpy as np
import pytest
from PIL import Image, ImageEnhance

import cucim.core.operations.color as ccl
from cucim.core.operations.color.batched_photometric import _gaussian_weights


def make_images(count, height=37, width=29, seed=0):
    """Noise, gradients, color blocks, and gray images (hue edge cases)."""
    rng = np.random.default_rng(seed)
    images = np.empty((count, height, width, 3), dtype=np.uint8)
    y, x = np.mgrid[:height, :width]
    for n in range(count):
        kind = n % 4
        if kind == 0:
            images[n] = rng.integers(0, 256, (height, width, 3))
        elif kind == 1:
            images[n, ..., 0] = x * 255 // (width - 1)
            images[n, ..., 1] = y * 255 // (height - 1)
            images[n, ..., 2] = (x + y) * 255 // (width + height - 2)
        elif kind == 2:
            colors = rng.integers(0, 256, (4, 3))
            block = (y >= height // 2) * 2 + (x >= width // 2)
            images[n] = colors[block]
        else:
            images[n] = (x * 255 // (width - 1))[..., np.newaxis]
            images[n, : height // 3] = 255 * (n % 8 == 3)
    return images


def pil_hue(image, factor):
    # torchvision.transforms.functional.adjust_hue on a PIL image
    h, s, v = image.convert("HSV").split()
    shift = np.uint8(int(factor * 255) % 256)
    h = Image.fromarray(np.asarray(h) + shift, "L")
    return Image.merge("HSV", (h, s, v)).convert("RGB")


def pil_jitter(image, order, factors):
    pil_image = Image.fromarray(image)
    for op in order:
        if op < 0:
            break
        factor = float(factors[op])
        if op == 0:
            pil_image = ImageEnhance.Brightness(pil_image).enhance(factor)
        elif op == 1:
            pil_image = ImageEnhance.Contrast(pil_image).enhance(factor)
        elif op == 2:
            pil_image = ImageEnhance.Color(pil_image).enhance(factor)
        else:
            pil_image = pil_hue(pil_image, factor)
    return np.asarray(pil_image)


def blur_reference(image, sigma, kernel_size):
    """The kernels' separable float32 blur, summed in the same order."""
    weights = _gaussian_weights(np.float32([sigma]), kernel_size)[0]
    radius = kernel_size // 2
    height, width = image.shape[:2]
    padded = np.pad(
        image.astype(np.float32), ((0, 0), (radius, radius), (0, 0)), "reflect"
    )
    rows = np.zeros(image.shape, np.float32)
    for k in range(kernel_size):
        rows = rows + padded[:, k : k + width] * weights[k]
    padded = np.pad(rows, ((radius, radius), (0, 0), (0, 0)), "reflect")
    columns = np.zeros(image.shape, np.float32)
    for k in range(kernel_size):
        columns = columns + padded[k : k + height] * weights[k]
    return np.clip(np.rint(columns), 0, 255).astype(np.uint8)


def random_jitter(count, seed):
    rng = np.random.default_rng(seed)
    factors = np.stack(
        [
            rng.uniform(0.6, 1.4, count),
            rng.uniform(0.6, 1.4, count),
            rng.uniform(0.8, 1.2, count),
            rng.uniform(-0.1, 0.1, count),
        ],
        axis=1,
    ).astype(np.float32)
    return factors, rng


JITTER_NAMES = ("brightness", "contrast", "saturation", "hue")


def jitter_kwargs(factors):
    return dict(zip(JITTER_NAMES, factors.T))


@pytest.mark.parametrize(
    "op, factor",
    [(op, f) for op in range(3) for f in (0.0, 0.3, 1.0, 1.7)]
    + [(3, f) for f in (-0.5, -0.07, 0.0, 0.25, 0.5)],
)
def test_single_operation_matches_pillow(op, factor):
    images = make_images(8)
    factors = np.float32([1, 1, 1, 0])
    factors[op] = factor
    out = ccl.batched_color_jitter(
        cupy.asarray(images), **{JITTER_NAMES[op]: factors[op]}
    )
    expected = [pil_jitter(image, [op], factors) for image in images]
    np.testing.assert_array_equal(out.get(), np.stack(expected))


def test_all_orders_match_pillow():
    orders = np.array(list(itertools.permutations(range(4))), np.int32)
    count = 2 * len(orders)
    images = make_images(count, seed=1)
    factors, _ = random_jitter(count, seed=2)
    order = np.concatenate([orders, orders])
    out = ccl.batched_color_jitter(
        cupy.asarray(images), **jitter_kwargs(factors), order=order
    )
    expected = [
        pil_jitter(image, o, f) for image, o, f in zip(images, order, factors)
    ]
    np.testing.assert_array_equal(out.get(), np.stack(expected))


def test_partial_orders_and_skipped_images():
    count = 12
    images = make_images(count, seed=3)
    factors, rng = random_jitter(count, seed=4)
    order = np.full((count, 4), -1, np.int32)
    for n in range(count):
        length = n % 5
        order[n, :length] = rng.permutation(4)[:length]
    out = ccl.batched_color_jitter(
        cupy.asarray(images), **jitter_kwargs(factors), order=order
    )
    expected = [
        pil_jitter(image, o, f) for image, o, f in zip(images, order, factors)
    ]
    np.testing.assert_array_equal(out.get(), np.stack(expected))
    unchanged = order[:, 0] < 0
    np.testing.assert_array_equal(out.get()[unchanged], images[unchanged])


def test_layouts_and_inputs_agree():
    images = make_images(6, seed=5)
    factors, rng = random_jitter(6, seed=6)
    order = np.stack([rng.permutation(4) for _ in range(6)])
    kwargs = dict(jitter_kwargs(factors), order=order)
    expected = ccl.batched_color_jitter(cupy.asarray(images), **kwargs).get()

    channels_first = np.ascontiguousarray(images.transpose(0, 3, 1, 2))
    out = ccl.batched_color_jitter(channels_first, channel_axis=1, **kwargs)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out.transpose(0, 2, 3, 1), expected)

    pieces = [
        cupy.asarray(images[:2]),
        cupy.asarray(images[2]),
        cupy.asarray(images[3:]),
    ]
    out = ccl.batched_color_jitter(pieces, **kwargs)
    np.testing.assert_array_equal(out.get(), expected)

    strided = cupy.asarray(images.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
    out = ccl.batched_color_jitter(strided, **kwargs)
    np.testing.assert_array_equal(out.get(), expected)


def photometric_reference(
    images,
    order,
    factors,
    flip,
    gray,
    sigmas,
    solarize,
    threshold,
    kernel_size,
    mean,
    std,
):
    out = []
    for n, image in enumerate(images):
        if flip[n]:
            image = image[:, ::-1]
        image = pil_jitter(np.ascontiguousarray(image), order[n], factors[n])
        if gray[n]:
            luma = np.asarray(Image.fromarray(image).convert("L"))
            image = np.repeat(luma[..., np.newaxis], 3, axis=2)
        if sigmas[n] > 0:
            image = blur_reference(image, sigmas[n], kernel_size)
        if solarize[n]:
            image = np.where(image >= threshold, 255 - image, image)
        scaled = image.astype(np.float32) / np.float32(255)
        out.append(((scaled - mean) / std).transpose(2, 0, 1))
    return np.stack(out)


@pytest.mark.parametrize("kernel_size", [3, 9])
def test_photometric_augment_matches_reference(kernel_size):
    count = 32
    images = make_images(count, seed=7)
    factors, rng = random_jitter(count, seed=8)
    order = np.stack([rng.permutation(4) for _ in range(count)])
    order[rng.random(count) < 0.2] = -1
    flip = rng.random(count) < 0.5
    gray = rng.random(count) < 0.3
    sigmas = np.where(
        rng.random(count) < 0.5, rng.uniform(0.1, 2.0, count), 0
    ).astype(np.float32)
    solarize = rng.random(count) < 0.3
    mean = np.float32([0.485, 0.456, 0.406])
    std = np.float32([0.229, 0.224, 0.225])
    out = ccl.batched_photometric_augment(
        cupy.asarray(images),
        **jitter_kwargs(factors),
        order=order,
        horizontal_flip=flip,
        grayscale=gray,
        blur_sigma=sigmas,
        blur_kernel_size=kernel_size,
        solarize=solarize,
        solarize_threshold=100,
        mean=mean,
        std=std,
    )
    expected = photometric_reference(
        images,
        order,
        factors,
        flip,
        gray,
        sigmas,
        solarize,
        100,
        kernel_size,
        mean.reshape(1, 1, 3),
        std.reshape(1, 1, 3),
    )
    np.testing.assert_array_equal(out.get(), expected)


def test_blur_close_to_torchvision():
    torch = pytest.importorskip("torch")  # noqa: F841
    tf = pytest.importorskip("torchvision.transforms.functional")
    count = 16
    images = make_images(count, seed=9)
    sigmas = np.random.default_rng(10).uniform(0.1, 2.0, count)
    sigmas = sigmas.astype(np.float32)
    out = ccl.batched_photometric_augment(
        cupy.asarray(images), blur_sigma=sigmas
    )
    out = np.rint(out.get() * 255).astype(np.int16).transpose(0, 2, 3, 1)
    expected = np.stack(
        [
            np.asarray(
                tf.gaussian_blur(Image.fromarray(image), [9, 9], [float(s)] * 2)
            )
            for image, s in zip(images, sigmas)
        ]
    ).astype(np.int16)
    assert np.abs(out - expected).max() <= 1
    assert np.mean(out == expected) > 0.999


def test_nondefault_stream():
    images = cupy.asarray(make_images(8, seed=11))
    kwargs = dict(contrast=1.3, hue=-0.05, blur_sigma=1.5, solarize=True)
    expected = ccl.batched_photometric_augment(images, **kwargs)
    with cupy.cuda.Stream(non_blocking=True) as stream:
        out = ccl.batched_photometric_augment(images, **kwargs)
        stream.synchronize()
    cupy.testing.assert_array_equal(out, expected)


def test_empty_batch():
    images = cupy.empty((0, 8, 8, 3), cupy.uint8)
    assert ccl.batched_color_jitter(images, contrast=1.2).shape == (0, 8, 8, 3)
    out = ccl.batched_photometric_augment(images, contrast=1.2, blur_sigma=1)
    assert out.shape == (0, 3, 8, 8)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        (dict(hue=0.6), ValueError),
        (dict(brightness=-0.1), ValueError),
        (dict(brightness=[1.0, 1.1]), ValueError),
        (dict(brightness=1.1, order=[0, 0]), ValueError),
        (dict(brightness=1.1, contrast=1.1, order=[-1, 0]), ValueError),
        (dict(brightness=1.1, order=[1]), ValueError),
        (dict(brightness=1.1, order=[4]), ValueError),
        (dict(brightness=1.1, order=[0.0]), TypeError),
        (dict(channel_axis=0), ValueError),
    ],
)
def test_jitter_rejects_invalid_parameters(kwargs, error):
    images = cupy.zeros((3, 8, 8, 3), cupy.uint8)
    with pytest.raises(error):
        ccl.batched_color_jitter(images, **kwargs)


@pytest.mark.parametrize(
    "images, error",
    [
        (cupy.zeros((2, 8, 8, 3), cupy.float32), TypeError),
        (cupy.zeros((2, 8, 8, 4), cupy.uint8), ValueError),
        (cupy.zeros((8, 8, 3), cupy.uint8), ValueError),
        (
            [
                cupy.zeros((8, 8, 3), cupy.uint8),
                cupy.zeros((8, 9, 3), cupy.uint8),
            ],
            ValueError,
        ),
        ([], TypeError),
        ("images", TypeError),
    ],
)
def test_rejects_invalid_images(images, error):
    with pytest.raises(error):
        ccl.batched_color_jitter(images)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(blur_sigma=1.0, blur_kernel_size=4),
        dict(blur_sigma=1.0, blur_kernel_size=17),
        dict(blur_sigma=-1.0),
        dict(mean=(0, 0)),
        dict(std=(1, 0, 1)),
        dict(grayscale=[True, False]),
    ],
)
def test_photometric_augment_rejects_invalid_parameters(kwargs):
    images = cupy.zeros((3, 8, 8, 3), cupy.uint8)
    with pytest.raises(ValueError):
        ccl.batched_photometric_augment(images, **kwargs)
