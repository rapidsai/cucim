# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import cupy
import numpy as np

from .kernel.batched_photometric_source import batched_photometric_code

_NUM_PARAMS = 9
_BLOCK_SIZE = 256
_NULL = np.uint64(0)
_JITTER_NAMES = ("brightness", "contrast", "saturation", "hue")


@cupy.memoize(for_each_device=True)
def _get_kernel(name):
    module = cupy.RawModule(
        code=batched_photometric_code,
        options=("--std=c++11", "--fmad=false"),
    )
    return module.get_function(name)


def _launch(name, total, args):
    if total == 0:
        return
    blocks = (total + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    _get_kernel(name)((blocks,), (_BLOCK_SIZE,), args)


def _host_array(value, dtype):
    if isinstance(value, cupy.ndarray):
        value = value.get()
    return np.asarray(value, dtype=dtype)


def _per_image(value, count, name, dtype):
    array = _host_array(value, dtype)
    if array.ndim == 0:
        return np.full(count, array, dtype=dtype)
    if array.shape != (count,):
        raise ValueError(
            f"`{name}` must be a scalar or have shape ({count},), "
            f"got shape {array.shape}"
        )
    return array


class _ImageBatch:
    """Device pointers to N equally shaped uint8 RGB images."""

    def __init__(self, images, channel_axis):
        if channel_axis in (-1, 3):
            channels_last = True
        elif channel_axis == 1:
            channels_last = False
        else:
            raise ValueError(
                f"channel_axis must be -1, 3 or 1, got {channel_axis}"
            )
        self.to_numpy = isinstance(images, np.ndarray)
        if isinstance(images, (np.ndarray, cupy.ndarray)):
            if images.ndim != 4:
                raise ValueError(
                    "Expected a 4-dimensional image batch, got "
                    f"{images.ndim} dimensions"
                )
            groups = [images]
        elif isinstance(images, Sequence) and len(images) > 0:
            groups = list(images)
        else:
            raise TypeError(
                "images must be a cupy.ndarray, numpy.ndarray or a non-empty "
                "sequence of them"
            )

        # Keep the contiguous arrays referenced until the kernels are queued.
        self.arrays = []
        for group in groups:
            if isinstance(group, np.ndarray):
                group = cupy.asarray(group)
            elif not isinstance(group, cupy.ndarray):
                raise TypeError(
                    "images must be cupy.ndarray or numpy.ndarray, got "
                    f"{type(group).__name__}"
                )
            if group.dtype != cupy.uint8:
                raise TypeError(f"images must be uint8, got {group.dtype}")
            if group.ndim == 3:
                group = group[cupy.newaxis]
            if group.ndim != 4:
                raise ValueError(
                    "Expected images of 3 (one image) or 4 (a batch) "
                    f"dimensions, got {group.ndim}"
                )
            self.arrays.append(cupy.ascontiguousarray(group))

        self.image_shape = self.arrays[0].shape[1:]
        if any(a.shape[1:] != self.image_shape for a in self.arrays):
            raise ValueError("All images must have the same shape")
        if channels_last:
            self.height, self.width, channels = self.image_shape
        else:
            channels, self.height, self.width = self.image_shape
        if channels != 3:
            raise ValueError(f"Expected 3 color channels, got {channels}")
        self.count = sum(len(a) for a in self.arrays)
        self.area = self.height * self.width
        if channels_last:
            self.pixel_stride, self.channel_stride = 3, 1
        else:
            self.pixel_stride, self.channel_stride = 1, self.area
        image_bytes = np.uint64(3 * self.area)
        pointers = [
            np.uint64(a.data.ptr)
            + np.arange(len(a), dtype=np.uint64) * image_bytes
            for a in self.arrays
        ]
        self.pointers = cupy.asarray(np.concatenate(pointers))


def _jitter_params(count, brightness, contrast, saturation, hue, order):
    """Pack the jitter order, factors and hue shift into the kernel layout."""
    values = (brightness, contrast, saturation, hue)
    params = np.full((count, _NUM_PARAMS), -1, dtype=np.int32)
    params[:, 4:7] = 0
    params[:, 8] = 0
    factors = np.zeros((count, 4), dtype=np.float32)
    factors[:, :3] = 1
    enabled = []
    for op, (name, value) in enumerate(zip(_JITTER_NAMES, values)):
        if value is None:
            continue
        factor = _per_image(value, count, name, np.float64)
        if not np.all(np.isfinite(factor)):
            raise ValueError(f"`{name}` factors must be finite")
        if op == 3:
            if np.any(np.abs(factor) > 0.5):
                raise ValueError("`hue` factors must lie in [-0.5, 0.5]")
            # torchvision converts hue_factor * 255 to uint8 in float64.
            params[:, 8] = np.trunc(factor * 255).astype(np.int64) % 256
        elif np.any(factor < 0):
            raise ValueError(f"`{name}` factors must be non-negative")
        factors[:, op] = factor
        enabled.append(op)

    if order is None:
        params[:, : len(enabled)] = enabled
        return params, factors

    order = _host_array(order, None)
    if order.size == 0:
        return params, factors
    if not np.issubdtype(order.dtype, np.integer):
        raise TypeError(f"`order` must be integer, got {order.dtype}")
    if order.ndim == 1:
        order = np.broadcast_to(order, (count, order.shape[0]))
    if order.ndim != 2 or order.shape[0] != count or order.shape[1] > 4:
        raise ValueError(
            f"`order` must have shape (k,) or ({count}, k) with k <= 4, got "
            f"{order.shape}"
        )
    params[:, : order.shape[1]] = order
    sequence = params[:, :4]
    if np.any((sequence < -1) | (sequence > 3)):
        raise ValueError("`order` entries must be -1 or in [0, 3]")
    present = sequence >= 0
    if np.any(present[:, 1:] & ~present[:, :-1]):
        raise ValueError("`order` rows must not continue after a -1")
    for op, (name, value) in enumerate(zip(_JITTER_NAMES, values)):
        uses = (sequence == op).sum(axis=1)
        if np.any(uses > 1):
            raise ValueError(f"`order` repeats {name} within a row")
        if value is None and np.any(uses):
            raise ValueError(f"`order` uses {name}, but `{name}` is None")
    return params, factors


def _apply_prefix(batch, params, factors):
    """Runs the operations up to contrast; returns them with contrast means."""
    pixels = batch.count * batch.area
    uses_contrast = bool(np.any(params[:, :4] == 1))
    params_gpu = cupy.asarray(params)
    factors_gpu = cupy.asarray(factors)
    prefix = cupy.empty((batch.count, batch.height, batch.width, 3), np.uint8)
    gray = (
        cupy.empty((batch.count, batch.height, batch.width), np.uint8)
        if uses_contrast
        else None
    )
    _launch(
        "photometric_prefix",
        pixels,
        (
            batch.pointers,
            np.int64(batch.pixel_stride),
            np.int64(batch.channel_stride),
            params_gpu,
            factors_gpu,
            prefix,
            _NULL if gray is None else gray,
            np.int32(batch.height),
            np.int32(batch.width),
            np.int64(pixels),
        ),
    )
    means = _NULL
    if uses_contrast:
        # Pillow: int(mean + 0.5) with the mean in double precision.
        sums = gray.sum(axis=(1, 2), dtype=cupy.uint64)
        means = cupy.floor(sums / batch.area + 0.5).astype(cupy.float32)
    return prefix, means, params_gpu, factors_gpu


def batched_color_jitter(
    images,
    brightness=None,
    contrast=None,
    saturation=None,
    hue=None,
    order=None,
    *,
    channel_axis=-1,
):
    """Applies color jitter with separate factors and order for each image.

    Unlike :func:`color_jitter`, which draws one set of factors for the whole
    batch and launches a kernel per operation, every image here has its own
    factors and operation order, and the batch is processed in two passes.
    The arithmetic follows torchvision's ``ColorJitter`` on PIL images,
    including Pillow's rounding to uint8 after each operation, so results
    match it exactly.

    Parameters
    ----------
    images : cupy.ndarray, numpy.ndarray or sequence of them
        uint8 RGB images. An array must be a batch of shape (N, H, W, 3), or
        (N, 3, H, W) with ``channel_axis=1``. A sequence may mix single images
        and batches of the same image shape; it is processed without
        concatenating the inputs.
    brightness, contrast, saturation : float or array-like of shape (N,), optional
        Non-negative factors, applied as given: 1 leaves the image unchanged
        and 0 gives black, a uniform gray, or grayscale, respectively. ``None``
        disables the operation.
    hue : float or array-like of shape (N,), optional
        Hue shift in [-0.5, 0.5], as a fraction of the hue circle. ``None``
        disables the operation.
    order : array-like of int, shape (k,) or (N, k), optional
        Operations to apply, in order: 0 brightness, 1 contrast,
        2 saturation, 3 hue. Rows may end early with -1 padding; an image whose
        row is all -1 is returned unchanged. Defaults to the enabled operations
        in the order above, for every image.
    channel_axis : {-1, 1}, optional
        Position of the channel axis in the batch.

    Returns
    -------
    out : cupy.ndarray or numpy.ndarray
        uint8 batch with the layout of the input. A numpy.ndarray is returned
        only when ``images`` is one.

    Notes
    -----
    Factors are not sampled here. To reproduce ``ColorJitter``, draw each
    image's factors and a random permutation for ``order`` on the host.
    The kernels run on the current CuPy stream.

    Examples
    --------
    >>> import cupy as cp
    >>> import numpy as np
    >>> import cucim.core.operations.color as ccl
    >>> images = cp.random.randint(0, 256, (8, 64, 64, 3), dtype=cp.uint8)
    >>> rng = np.random.default_rng(0)
    >>> out = ccl.batched_color_jitter(
    ...     images,
    ...     brightness=rng.uniform(0.6, 1.4, 8),
    ...     contrast=rng.uniform(0.6, 1.4, 8),
    ...     order=[1, 0],
    ... )
    """
    batch = _ImageBatch(images, channel_axis)
    params, factors = _jitter_params(
        batch.count, brightness, contrast, saturation, hue, order
    )
    prefix, means, params_gpu, factors_gpu = _apply_prefix(
        batch, params, factors
    )
    out = cupy.empty((batch.count,) + batch.image_shape, dtype=cupy.uint8)
    pixels = batch.count * batch.area
    _launch(
        "photometric_suffix",
        pixels,
        (
            prefix,
            means,
            params_gpu,
            factors_gpu,
            out,
            np.int64(batch.pixel_stride),
            np.int64(batch.channel_stride),
            _NULL,
            _NULL,
            _NULL,
            np.int32(0),
            np.int64(batch.area),
            np.int64(pixels),
        ),
    )
    return cupy.asnumpy(out) if batch.to_numpy else out


def _gaussian_weights(sigmas, kernel_size):
    """torchvision's float32 1-D Gaussian kernel, one row per sigma."""
    half = (kernel_size - 1) * 0.5
    x = np.linspace(-half, half, kernel_size, dtype=np.float32)
    pdf = np.exp(np.float32(-0.5) * (x / sigmas[:, np.newaxis]) ** 2)
    return pdf / pdf.sum(axis=1, keepdims=True)


def batched_photometric_augment(
    images,
    brightness=None,
    contrast=None,
    saturation=None,
    hue=None,
    order=None,
    *,
    horizontal_flip=False,
    grayscale=False,
    blur_sigma=None,
    blur_kernel_size=9,
    solarize=False,
    solarize_threshold=128,
    mean=(0.0, 0.0, 0.0),
    std=(1.0, 1.0, 1.0),
    channel_axis=-1,
):
    """Fused per-image photometric augmentation and normalization.

    Applies, in this order and with separate parameters for each image:
    horizontal flip, color jitter (as in :func:`batched_color_jitter`),
    grayscale, Gaussian blur and solarization; then scales to [0, 1],
    normalizes with ``mean`` and ``std`` and writes a float32 (N, 3, H, W)
    batch. This is the photometric part of the SimCLR/DINO-style training
    augmentation, matching the sequence of torchvision's
    ``RandomHorizontalFlip``, ``ColorJitter``, ``RandomGrayscale``,
    ``GaussianBlur``, ``RandomSolarize``, ``ToTensor`` and ``Normalize`` on
    PIL images. The batch takes at most five kernel launches, independent of
    N.

    Parameters
    ----------
    images : cupy.ndarray, numpy.ndarray or sequence of them
        uint8 RGB images; see :func:`batched_color_jitter`.
    brightness, contrast, saturation, hue, order : optional
        Color jitter parameters; see :func:`batched_color_jitter`.
    horizontal_flip, grayscale, solarize : bool or array-like of shape (N,)
        Whether to apply each operation to each image. Grayscale keeps three
        identical channels.
    blur_sigma : float or array-like of shape (N,), optional
        Gaussian standard deviation for each image; 0 leaves that image
        unblurred. ``None`` disables blurring for the batch.
    blur_kernel_size : int, optional
        Odd size of the Gaussian kernel, which must be smaller than twice the
        image height and width. Borders are reflected.
    solarize_threshold : int, optional
        Values at or above the threshold ``v`` become ``255 - v``.
    mean, std : sequence of 3 floats, optional
        Per-channel normalization applied after scaling to [0, 1].
    channel_axis : {-1, 1}, optional
        Position of the channel axis in the input batch.

    Returns
    -------
    out : cupy.ndarray or numpy.ndarray
        float32 batch of shape (N, 3, H, W). A numpy.ndarray is returned only
        when ``images`` is one.

    Notes
    -----
    The blur is two separable float32 passes rather than torchvision's 2-D
    convolution, so a blurred value can occasionally differ from torchvision
    by one intensity level. All other operations match it exactly.
    Parameters are not sampled here; draw them per image on the host. The
    kernels run on the current CuPy stream.

    Examples
    --------
    >>> import cupy as cp
    >>> import numpy as np
    >>> import cucim.core.operations.color as ccl
    >>> n = 16
    >>> images = cp.random.randint(0, 256, (n, 112, 112, 3), dtype=cp.uint8)
    >>> rng = np.random.default_rng(0)
    >>> jitter = rng.random(n) < 0.8
    >>> order = np.where(
    ...     jitter[:, None], [rng.permutation(4) for _ in range(n)], -1
    ... )
    >>> out = ccl.batched_photometric_augment(
    ...     images,
    ...     brightness=rng.uniform(0.6, 1.4, n),
    ...     contrast=rng.uniform(0.6, 1.4, n),
    ...     saturation=rng.uniform(0.8, 1.2, n),
    ...     hue=rng.uniform(-0.1, 0.1, n),
    ...     order=order,
    ...     horizontal_flip=rng.random(n) < 0.5,
    ...     grayscale=rng.random(n) < 0.2,
    ...     blur_sigma=np.where(rng.random(n) < 0.5, rng.uniform(0.1, 2, n), 0),
    ...     mean=(0.485, 0.456, 0.406),
    ...     std=(0.229, 0.224, 0.225),
    ... )
    """
    batch = _ImageBatch(images, channel_axis)
    count = batch.count
    params, factors = _jitter_params(
        count, brightness, contrast, saturation, hue, order
    )
    params[:, 4] = _per_image(horizontal_flip, count, "horizontal_flip", bool)
    params[:, 5] = _per_image(grayscale, count, "grayscale", bool)
    params[:, 6] = _per_image(solarize, count, "solarize", bool)

    mean_std = np.concatenate(
        [_host_array(mean, np.float32), _host_array(std, np.float32)]
    )
    if mean_std.shape != (6,):
        raise ValueError("`mean` and `std` must each have 3 values")
    if np.any(mean_std[3:] == 0):
        raise ValueError("`std` must be nonzero")

    blur_images = np.empty(0, dtype=np.int32)
    if blur_sigma is not None:
        sigmas = _per_image(blur_sigma, count, "blur_sigma", np.float32)
        if not np.all(np.isfinite(sigmas)) or np.any(sigmas < 0):
            raise ValueError("`blur_sigma` values must be finite and >= 0")
        blur_images = np.flatnonzero(sigmas > 0).astype(np.int32)
    if len(blur_images):
        if (
            not isinstance(blur_kernel_size, (int, np.integer))
            or blur_kernel_size <= 0
            or blur_kernel_size % 2 == 0
        ):
            raise ValueError(
                "`blur_kernel_size` must be a positive odd integer, got "
                f"{blur_kernel_size}"
            )
        radius = blur_kernel_size // 2
        if radius >= min(batch.height, batch.width):
            raise ValueError(
                f"`blur_kernel_size` {blur_kernel_size} is too large for "
                f"{batch.height}x{batch.width} images"
            )
        params[blur_images, 7] = np.arange(len(blur_images))

    prefix, means, params_gpu, factors_gpu = _apply_prefix(
        batch, params, factors
    )
    mean_std = cupy.asarray(mean_std)
    final = cupy.empty((count, 3, batch.height, batch.width), cupy.float32)
    blur_count = len(blur_images)
    blur_shape = (blur_count, batch.height, batch.width, 3)
    blur_in = cupy.empty(blur_shape, cupy.uint8) if blur_count else _NULL
    threshold = np.int32(solarize_threshold)
    pixels = count * batch.area
    _launch(
        "photometric_suffix",
        pixels,
        (
            prefix,
            means,
            params_gpu,
            factors_gpu,
            _NULL,
            np.int64(0),
            np.int64(0),
            blur_in,
            final,
            mean_std,
            threshold,
            np.int64(batch.area),
            np.int64(pixels),
        ),
    )
    if blur_count:
        weights = cupy.asarray(
            _gaussian_weights(sigmas[blur_images], blur_kernel_size)
        )
        rows = cupy.empty(blur_shape, cupy.float32)
        total = blur_count * batch.area * 3
        _launch(
            "photometric_blur_rows",
            total,
            (
                blur_in,
                rows,
                weights,
                np.int32(radius),
                np.int32(batch.height),
                np.int32(batch.width),
                np.int64(total),
            ),
        )
        _launch(
            "photometric_blur_columns",
            total,
            (
                rows,
                final,
                cupy.asarray(blur_images),
                params_gpu,
                weights,
                mean_std,
                threshold,
                np.int32(radius),
                np.int32(batch.height),
                np.int32(batch.width),
                np.int64(total),
            ),
        )
    return cupy.asnumpy(final) if batch.to_numpy else final
