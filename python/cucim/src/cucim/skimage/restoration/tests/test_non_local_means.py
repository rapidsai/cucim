# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuCIM-specific tests of denoise_nl_means.

These compare directly against scikit-image's CPU implementation and check
the internal kernels, dtype handling and the documented deviations from
scikit-image.
"""

import itertools
import os

import cupy as cp
import cupyx.scipy.ndimage as ndi
import numpy as np
import pytest
import skimage.restoration

from cucim.skimage import restoration
from cucim.skimage.restoration import _nl_means_denoising as nlm
from cucim.skimage.restoration._nl_means_kernels import _preamble_template

# Expected agreement with scikit-image. In float32, scikit-image's fast mode
# computes in float64 while cuCIM computes in float32.
atol = {np.float32: 2e-5, np.float64: 1e-7}

# Schraudolph exp approximation constants used by scikit-image
EXP_A = 1512775.3951951856938
EXP_BC = 1072632447


def _smooth_noisy(shape, n_channels=None, sigma=0.1, seed=0):
    """Smooth random image in [0, 1] plus Gaussian noise (as NumPy array).

    Gives a mix of similar and dissimilar patches, so the weights are neither
    all zero nor all one.
    """
    rng = cp.random.default_rng(seed)
    full_shape = shape if n_channels is None else shape + (n_channels,)
    smooth_sigma = [2.5] * len(shape) + ([] if n_channels is None else [0])
    img = ndi.gaussian_filter(rng.uniform(0, 1, full_shape), smooth_sigma)
    img = (img - img.min()) / (img.max() - img.min())
    img += sigma * rng.standard_normal(full_shape)
    return cp.asnumpy(img)


def _fast_exp_ref(y):
    """NumPy version of scikit-image's _fast_exp (float64 result)."""
    hi = np.trunc(EXP_A * np.asarray(y, np.float64)).astype(np.int64) + EXP_BC
    return ((hi & 0xFFFFFFFF).astype(np.uint64) << np.uint64(32)).view(
        np.float64
    )


def _compare(image, dtype, channel_axis=None, **kwargs):
    image = image.astype(dtype)
    with np.errstate(invalid="ignore"):
        expected = skimage.restoration.denoise_nl_means(
            image, channel_axis=channel_axis, **kwargs
        )
    result = restoration.denoise_nl_means(
        cp.asarray(image), channel_axis=channel_axis, **kwargs
    )
    assert result.dtype == expected.dtype
    cp.testing.assert_allclose(result, expected, rtol=0, atol=atol[dtype])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("fast_mode", [True, False])
@pytest.mark.parametrize("sigma", [0.0, 0.1])
@pytest.mark.parametrize(
    "shape, n_channels, patch_size, patch_distance",
    [
        ((37, 41), None, 7, 5),
        ((37, 41), None, 3, 4),
        ((23, 19), 3, 4, 4),  # even patch size and RGB
        ((5, 6), 2, 3, 7),  # patch_distance larger than the image
        ((13, 11, 10), None, 5, 2),
        ((9, 8, 7), None, 3, 3),
    ],
)
def test_nl_means_vs_skimage(
    shape, n_channels, patch_size, patch_distance, sigma, fast_mode, dtype
):
    image = _smooth_noisy(shape, n_channels)
    _compare(
        image,
        dtype,
        channel_axis=None if n_channels is None else -1,
        patch_size=patch_size,
        patch_distance=patch_distance,
        h=0.08 if fast_mode else 0.05,
        sigma=sigma,
        fast_mode=fast_mode,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sigma", [0.0, 0.1])
@pytest.mark.parametrize(
    "shape, n_channels, patch_size, patch_distance",
    [
        ((9, 8, 7), 2, 3, 2),
        ((9, 8, 7), 3, 5, 1),
        ((6, 7, 5, 6), None, 3, 1),
        ((5, 6, 5, 4), 2, 3, 1),
        ((7, 6, 5, 6), None, 5, 1),
    ],
)
def test_nl_means_vs_skimage_fast_only(
    shape, n_channels, patch_size, patch_distance, sigma, dtype
):
    # 3D multichannel and 4D are only supported in fast mode
    image = _smooth_noisy(shape, n_channels)
    _compare(
        image,
        dtype,
        channel_axis=None if n_channels is None else -1,
        patch_size=patch_size,
        patch_distance=patch_distance,
        h=0.08,
        sigma=sigma,
        fast_mode=True,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nl_means_slow_mode_early_exit(dtype):
    # With sigma > 0 the slow-mode partial distances are not monotonic, so
    # scikit-image's per-row early exit differs from a cutoff applied to the
    # final distance (by ~4e-3 here). This checks the early exit is matched.
    image = _smooth_noisy((40, 37), seed=1)
    _compare(
        image,
        dtype,
        patch_size=7,
        patch_distance=5,
        h=0.06,
        sigma=0.1,
        fast_mode=False,
    )


@pytest.mark.parametrize("channel_axis", [0, 1, -1])
@pytest.mark.parametrize("fast_mode", [True, False])
def test_nl_means_channel_axis(channel_axis, fast_mode):
    image = _smooth_noisy((21, 17), n_channels=3)
    image = np.moveaxis(image, -1, channel_axis)
    _compare(
        image,
        np.float64,
        channel_axis=channel_axis,
        patch_size=5,
        patch_distance=3,
        h=0.08,
        fast_mode=fast_mode,
    )


@pytest.mark.parametrize("fast_mode", [True, False])
def test_nl_means_many_channels(fast_mode):
    image = _smooth_noisy((19, 23), n_channels=12)
    _compare(
        image,
        np.float32,
        channel_axis=-1,
        patch_size=5,
        patch_distance=3,
        h=0.1,
        fast_mode=fast_mode,
    )


@pytest.mark.parametrize("patch_size", [0, 1, 2])
@pytest.mark.parametrize("fast_mode", [True, False])
def test_nl_means_small_patch_size(patch_size, fast_mode):
    # patch_size 0 and 1 give patch size 1: scikit-image ends up averaging
    # uniformly over the search window
    image = _smooth_noisy((20, 30))
    _compare(
        image,
        np.float64,
        patch_size=patch_size,
        patch_distance=2,
        h=0.1,
        fast_mode=fast_mode,
    )


@pytest.mark.parametrize("fast_mode", [True, False])
def test_nl_means_zero_patch_distance(fast_mode):
    image = cp.asarray(_smooth_noisy((20, 30)))
    result = restoration.denoise_nl_means(image, 7, 0, 0.1, fast_mode=fast_mode)
    cp.testing.assert_allclose(result, image, rtol=1e-12)


def test_nl_means_non_contiguous():
    image = _smooth_noisy((40, 30), n_channels=3)
    view = image[::2, ::-1]
    expected = skimage.restoration.denoise_nl_means(
        view, 5, 3, 0.08, channel_axis=-1
    )
    result = restoration.denoise_nl_means(
        cp.asarray(image)[::2, ::-1], 5, 3, 0.08, channel_axis=-1
    )
    cp.testing.assert_allclose(result, expected, rtol=0, atol=1e-7)


@pytest.mark.parametrize(
    "dtype, expected",
    [
        (bool, np.float32),
        (np.int8, np.float32),
        (np.uint8, np.float32),
        (np.int16, np.float32),
        (np.uint16, np.float32),
        (np.float16, np.float32),
        (np.float32, np.float32),
        (np.int32, np.float64),
        (np.uint32, np.float64),
        (np.int64, np.float64),
        (np.float64, np.float64),
    ],
)
@pytest.mark.parametrize("preserve_range", [False, True])
def test_nl_means_output_dtype(dtype, expected, preserve_range):
    image = cp.zeros((16, 16), dtype=dtype)
    result = restoration.denoise_nl_means(
        image, 3, 2, 0.1, preserve_range=preserve_range
    )
    assert result.dtype == expected
    for compute_dtype in (cp.float32, cp.float64):
        result = restoration.denoise_nl_means(
            image,
            3,
            2,
            0.1,
            preserve_range=preserve_range,
            compute_dtype=compute_dtype,
        )
        assert result.dtype == expected


@pytest.mark.parametrize("preserve_range", [False, True])
def test_nl_means_uint8_vs_skimage(preserve_range):
    image = (_smooth_noisy((32, 30)).clip(0, 1) * 255).astype(np.uint8)
    h = 20.0 if preserve_range else 0.08
    expected = skimage.restoration.denoise_nl_means(
        image, 5, 3, h, preserve_range=preserve_range
    )
    result = restoration.denoise_nl_means(
        cp.asarray(image), 5, 3, h, preserve_range=preserve_range
    )
    # scikit-image returns float64; cuCIM computes and returns float32
    assert result.dtype == cp.float32
    scale = 255.0 if preserve_range else 1.0
    cp.testing.assert_allclose(result, expected, rtol=0, atol=2e-5 * scale)


@pytest.mark.parametrize("fast_mode", [True, False])
def test_nl_means_compute_dtype(fast_mode):
    image = _smooth_noisy((40, 37))
    kwargs = dict(patch_size=5, patch_distance=4, h=0.08, fast_mode=fast_mode)
    expected = skimage.restoration.denoise_nl_means(image, **kwargs)
    result32 = restoration.denoise_nl_means(
        cp.asarray(image), compute_dtype=cp.float32, **kwargs
    )
    assert result32.dtype == cp.float64
    cp.testing.assert_allclose(result32, expected, rtol=0, atol=2e-5)

    # float32 input computed in float64 (as scikit-image's fast mode does)
    image32 = image.astype(np.float32)
    expected = skimage.restoration.denoise_nl_means(image32, **kwargs)
    result64 = restoration.denoise_nl_means(
        cp.asarray(image32), compute_dtype=np.float64, **kwargs
    )
    assert result64.dtype == cp.float32
    cp.testing.assert_allclose(result64, expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    "shape, channel_axis",
    [((1, 30), None), ((20, 1), None), ((20, 30, 1), -1), ((1, 20, 30), 0)],
)
def test_nl_means_shape_preserved(shape, channel_axis):
    # scikit-image squeezes singleton dimensions from the output
    image = cp.asarray(np.random.default_rng(0).uniform(size=shape))
    result = restoration.denoise_nl_means(
        image, 3, 2, 0.1, channel_axis=channel_axis
    )
    assert result.shape == shape


def test_nl_means_invalid_arguments():
    image = cp.zeros((10, 10))
    with pytest.raises(ValueError):
        restoration.denoise_nl_means(image, patch_size=-1)
    with pytest.raises(ValueError):
        restoration.denoise_nl_means(image, patch_distance=-1)
    with pytest.raises(TypeError):
        restoration.denoise_nl_means(image, patch_size=3.5)
    for h in [0, -0.1, float("nan")]:
        with pytest.raises(ValueError):
            restoration.denoise_nl_means(image, h=h)
    with pytest.raises(ValueError):
        restoration.denoise_nl_means(image, compute_dtype=cp.float16)
    with pytest.raises(ValueError):
        restoration.denoise_nl_means(image.astype(cp.complex64))


def test_nl_means_slow_mode_overflow_guard():
    # For row 4, the patch centered one row below matches except for its last
    # row, which differs strongly. scikit-image does not apply its early exit
    # after the last row and evaluates _fast_exp far outside its domain. The
    # resulting huge negative weight dominates, so scikit-image returns that
    # patch's center value (0.02) for row 4. cuCIM gives the patch zero
    # weight; all remaining contributing patches have center value 0.
    profile = np.array([0, 0, 0, 0, 0, 0.02, 1, 1, 1, 1.0])
    image = np.tile(profile[:, np.newaxis], (1, 9))
    result = restoration.denoise_nl_means(
        cp.asarray(image), 3, 3, 0.01, fast_mode=False
    )
    cp.testing.assert_allclose(result, image, rtol=0, atol=1e-12)


@pytest.mark.parametrize("fast_mode", [True, False])
@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize(
    "shape, n_channels, patch_size, patch_distance, tile",
    [
        ((37, 41), 1, 7, 4, None),
        ((37, 41), 1, 7, 4, (8, 32)),
        ((23, 19), 3, 5, 3, (8, 8)),
        ((37, 41), 1, 3, 3, None),
        ((13, 11, 10), 1, 5, 2, None),
        ((13, 11, 10), 2, 3, 2, (4, 4, 8)),
        ((7, 6, 5, 6), 1, 5, 1, None),
        ((7, 6, 5, 6), 2, 3, 1, (2, 2, 4, 4)),
    ],
)
def test_nl_means_tiled_vs_direct(
    shape, n_channels, patch_size, patch_distance, tile, dtype, fast_mode
):
    image = cp.asarray(_smooth_noisy(shape, n_channels), dtype=dtype)
    kwargs = dict(
        patch_size=patch_size,
        patch_distance=patch_distance,
        h=0.08,
        var=0.01,
        fast_mode=fast_mode,
    )
    tiled = nlm._nl_means_denoising(image, kernel="tiled", tile=tile, **kwargs)
    direct = nlm._nl_means_denoising(image, kernel="direct", **kwargs)
    tol = 1e-6 if dtype == cp.float32 else 1e-12
    cp.testing.assert_allclose(tiled, direct, rtol=0, atol=tol)


def test_nl_means_tile_shrinks_to_fit(monkeypatch):
    # with too little shared memory the tile is shrunk, and with even less
    # the direct kernel is used
    image = cp.asarray(_smooth_noisy((40, 37)))[..., cp.newaxis]
    kwargs = dict(
        patch_size=7, patch_distance=3, h=0.08, var=0.0, fast_mode=False
    )
    expected = nlm._nl_means_denoising(image, kernel="direct", **kwargs)

    monkeypatch.setattr(nlm, "_max_smem_per_block", lambda: 4 * 1024)
    tile, nbytes = nlm._select_tile(2, 7, 8)
    assert math_prod(tile) < math_prod(nlm._default_tiles[2])
    assert nbytes <= 4 * 1024
    result = nlm._nl_means_denoising(image, kernel="tiled", **kwargs)
    cp.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)

    monkeypatch.setattr(nlm, "_max_smem_per_block", lambda: 1024)
    assert nlm._select_tile(2, 7, 8) == (None, 0)
    result = nlm._nl_means_denoising(image, **kwargs)
    cp.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)
    with pytest.raises(ValueError):
        nlm._nl_means_denoising(image, kernel="tiled", **kwargs)


def math_prod(values):
    return int(np.prod(values))


@pytest.mark.parametrize("c_type", ["float", "double"])
def test_nl_means_fast_exp(c_type):
    dtype = np.float32 if c_type == "float" else np.float64
    code = _preamble_template.format(c_type=c_type, index_type="int")
    code += """
extern "C" __global__ void fast_exp_kernel(const T* y, T* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = nlm_fast_exp(y[i]);
    }
}
"""
    kernel = cp.RawKernel(code, "fast_exp_kernel")
    max_y = 87.0 if c_type == "float" else 708.0
    y = -np.concatenate(
        [np.linspace(0, 5, 200_001), np.linspace(5, max_y, 10_001)]
    ).astype(dtype)
    y_gpu = cp.asarray(y)
    out = cp.empty_like(y_gpu)
    kernel(((y.size + 255) // 256,), (256,), (y_gpu, out, np.int32(y.size)))
    expected = _fast_exp_ref(y.astype(np.float64)).astype(dtype)
    if c_type == "double":
        # bit-identical to scikit-image's _fast_exp
        cp.testing.assert_array_equal(out, expected)
    else:
        # within one 2**-20 quantization step of (float)_fast_exp((double)y)
        # for distances up to the cutoff
        n = 200_001
        cp.testing.assert_allclose(out[:n], expected[:n], rtol=2.0**-19)
        # beyond it (slow mode only), the float32 product loses a few steps
        cp.testing.assert_allclose(out[n:], expected[n:], rtol=1e-5)


def test_nl_means_kernel_sources_compile():
    # generated variants in 2D-4D compile and run (patch size 1 has a
    # single-tap slow mode and an empty fast-mode patch)
    n_channels = 2
    for ndim, fast_mode, patch_size in itertools.product(
        (2, 3, 4), (True, False), (1, 5)
    ):
        image = cp.ones((5,) * ndim + (n_channels,), dtype=cp.float32)
        for kernel in ("direct", "tiled"):
            if kernel == "tiled" and patch_size == 1 and fast_mode:
                continue
            out = nlm._nl_means_denoising(
                image, patch_size, 1, 0.1, 0.0, fast_mode, kernel=kernel
            )
            cp.testing.assert_allclose(out, image, rtol=1e-6)


@pytest.mark.parametrize("fast_mode", [True, False])
@pytest.mark.parametrize("kernel", ["tiled", "direct"])
@pytest.mark.parametrize("shift_chunks", [2, 4, 7, 100])
def test_nl_means_shift_chunks(shift_chunks, kernel, fast_mode):
    # splitting the shifts along axis 0 across blocks (as done automatically
    # for small images) gives the same result as a single chunk
    image = cp.asarray(_smooth_noisy((23, 19, 9), n_channels=2))
    kwargs = dict(
        patch_size=5,
        patch_distance=3,
        h=0.08,
        var=0.01,
        fast_mode=fast_mode,
        kernel=kernel,
    )
    expected = nlm._nl_means_denoising(image, shift_chunks=1, **kwargs)
    result = nlm._nl_means_denoising(image, shift_chunks=shift_chunks, **kwargs)
    cp.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("fast_mode", [True, False])
@pytest.mark.parametrize("shape", [(1, 30), (25, 1), (1, 1), (1, 8, 9)])
def test_nl_means_singleton_axes(shape, fast_mode):
    # reflect padding of size-1 axes matches scikit-image
    image = np.random.default_rng(0).uniform(size=shape)
    kwargs = dict(patch_size=3, patch_distance=2, h=0.2, fast_mode=fast_mode)
    expected = skimage.restoration.denoise_nl_means(image, **kwargs)
    result = restoration.denoise_nl_means(cp.asarray(image), **kwargs)
    # scikit-image squeezes singleton dimensions
    assert result.shape == shape
    cp.testing.assert_allclose(
        result.ravel(), np.ravel(expected), rtol=0, atol=1e-7
    )


@pytest.mark.skipif(
    os.environ.get("CUCIM_RUN_NL_MEANS_MEM_TEST", "0") != "1",
    reason="needs ~27 GB of GPU memory; set CUCIM_RUN_NL_MEANS_MEM_TEST=1",
)
def test_nl_means_64bit_indexing():
    # A 46341 x 46341 image has more than 2**31 elements, so the kernels use
    # 64-bit indexing. A minimal search window keeps each call under a second.
    # Each output pixel only depends on the input within
    # patch_size // 2 + patch_distance pixels, so small crops (processed with
    # 32-bit indexing) give the expected values, including past linear index
    # 2**31.
    free, _ = cp.cuda.Device().mem_info
    if free < 28e9:
        pytest.skip("needs ~27 GB of free GPU memory")
    n = 46341
    w = 40
    # column in the last row where the linear index reaches 2**31
    c_2_31 = 2**31 - (n - 1) * n
    corners = [
        (0, 0),
        (n // 2, n // 2),
        (n - w, c_2_31 - w // 2),
        (n - w, n - w),
    ]
    image = cp.random.default_rng(0).random((n, n), dtype=cp.float32)
    assert image.size > 2**31
    try:
        # tiled fast mode, tiled slow mode and direct (fast, patch_size 3)
        for patch_size, fast_mode in [(5, True), (3, False), (3, True)]:
            kwargs = dict(
                patch_size=patch_size,
                patch_distance=1,
                h=0.1,
                fast_mode=fast_mode,
            )
            result = restoration.denoise_nl_means(image, **kwargs)
            margin = patch_size // 2 + 1
            for r0, c0 in corners:
                lo_r, lo_c = max(r0 - margin, 0), max(c0 - margin, 0)
                crop = image[lo_r : r0 + w + margin, lo_c : c0 + w + margin]
                expected = restoration.denoise_nl_means(crop, **kwargs)
                expected = expected[r0 - lo_r :, c0 - lo_c :][:w, :w]
                cp.testing.assert_allclose(
                    result[r0 : r0 + w, c0 : c0 + w], expected, rtol=1e-6
                )
            del result
    finally:
        del image
        cp.get_default_memory_pool().free_all_blocks()
