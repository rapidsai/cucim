# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import cupy as cp
import numpy as np
import pytest
from cupy.testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
)
from skimage import data
from skimage.draw import disk
from skimage.filters._multiotsu import (
    _get_multiotsu_thresh_indices,
    _get_multiotsu_thresh_indices_lut,
)

# from cupyx.scipy import ndimage as ndi
from cucim.skimage import util
from cucim.skimage._shared._dependency_checks import has_mpl
from cucim.skimage._shared._warnings import expected_warnings
from cucim.skimage._shared.utils import _supported_float_type
from cucim.skimage.color import rgb2gray
from cucim.skimage.exposure import histogram
from cucim.skimage.filters.thresholding import (
    _cross_entropy,
    threshold_isodata,
    threshold_li,
    threshold_local,
    threshold_mean,
    threshold_minimum,
    threshold_multiotsu,
    threshold_niblack,
    threshold_otsu,
    threshold_sauvola,
    threshold_triangle,
    threshold_yen,
    try_all_threshold,
)

# transfer images to GPU
astronautd = cp.array(data.astronaut())
camerad = cp.array(data.camera())
celld = cp.array(data.cell())
coinsd = cp.array(data.coins())
moond = cp.array(data.moon())


class TestSimpleImage:
    def setup_method(self):
        # fmt: off
        self.image = cp.array([[0, 0, 1, 3, 5],
                               [0, 1, 4, 3, 4],
                               [1, 2, 5, 4, 1],
                               [2, 4, 5, 2, 1],
                               [4, 5, 1, 0, 0]], dtype=int)
        # fmt: on

    def test_minimum(self):
        with pytest.raises(RuntimeError):
            threshold_minimum(self.image)

    @pytest.mark.skipif(not has_mpl, reason="matplotlib not installed")
    def test_try_all_threshold(self):
        fig, ax = try_all_threshold(self.image)
        all_texts = [axis.texts for axis in ax if axis.texts != []]
        text_content = [text.get_text() for x in all_texts for text in x]
        assert "RuntimeError" in text_content

    def test_otsu(self):
        assert threshold_otsu(self.image) == 2

    def test_otsu_negative_int(self):
        image = self.image - 2
        assert threshold_otsu(image) == 0

    def test_otsu_float_image(self):
        image = self.image.astype(cp.float64)
        assert 2 <= threshold_otsu(image) < 3

    def test_li(self):
        assert 2 < threshold_li(self.image) < 3

    def test_li_negative_int(self):
        image = self.image - 2
        assert 0 < threshold_li(image) < 1

    def test_li_float_image(self):
        image = self.image.astype(float)
        assert 2 < threshold_li(image) < 3

    def test_li_constant_image(self):
        assert threshold_li(cp.ones((10, 10))) == 1.0

    def test_yen(self):
        assert threshold_yen(self.image) == 2

    def test_yen_negative_int(self):
        image = self.image - 2
        assert threshold_yen(image) == 0

    def test_yen_float_image(self):
        image = self.image.astype(cp.float64)
        assert 2 <= threshold_yen(image) < 3

    def test_yen_arange(self):
        image = cp.arange(256)
        assert threshold_yen(image) == 127

    def test_yen_binary(self):
        image = cp.zeros([2, 256], dtype=cp.uint8)
        image[0] = 255
        assert threshold_yen(image) < 1

    def test_yen_blank_zero(self):
        image = cp.zeros((5, 5), dtype=cp.uint8)
        assert threshold_yen(image) == 0

    def test_yen_blank_max(self):
        image = cp.empty((5, 5), dtype=cp.uint8)
        image.fill(255)
        assert threshold_yen(image) == 255

    def test_isodata(self):
        assert threshold_isodata(self.image) == 2
        assert_array_equal(threshold_isodata(self.image, return_all=True), [2])

    def test_isodata_blank_zero(self):
        image = cp.zeros((5, 5), dtype=cp.uint8)
        assert threshold_isodata(image) == 0
        assert_array_equal(threshold_isodata(image, return_all=True), [0])

    def test_isodata_linspace(self):
        image = cp.linspace(-127, 0, 256)
        assert -63.8 < threshold_isodata(image) < -63.6
        assert_array_almost_equal(
            threshold_isodata(image, return_all=True),
            [-63.74804688, -63.25195312],
        )

    def test_isodata_16bit(self):
        np.random.seed(0)
        imfloat = cp.array(np.random.rand(256, 256))
        assert 0.49 < threshold_isodata(imfloat, nbins=1024) < 0.51
        assert all(
            0.49 < threshold_isodata(imfloat, nbins=1024, return_all=True)
        )

    @pytest.mark.parametrize("ndim", [2, 3])
    def test_threshold_local_gaussian(self, ndim):
        # fmt: off
        ref = cp.array(
            [[False, False, False, False,  True],  # noqa
             [False, False,  True, False,  True],  # noqa
             [False, False,  True,  True, False],  # noqa
             [False,  True,  True, False, False],  # noqa
             [ True,  True, False, False, False]]  # noqa
        )
        if ndim == 2:
            image = self.image
            block_sizes = [3, (3,) * image.ndim]
        else:
            image = cp.stack((self.image, ) * 5, axis=-1)
            ref = cp.stack((ref, ) * 5, axis=-1)
            block_sizes = [3, (3,) * image.ndim,
                           (3,) * (image.ndim - 1) + (1,)]

        for block_size in block_sizes:
            out = threshold_local(image, block_size, method='gaussian',
                                  mode='reflect')
            assert_array_equal(ref, image > out)

        out = threshold_local(image, 3, method='gaussian', mode='reflect',
                              param=1 / 3)
        assert_array_equal(ref, image > out)

    @pytest.mark.parametrize("ndim", [2, 3])
    def test_threshold_local_mean(self, ndim):
        # fmt: off
        ref = cp.array(
            [[False, False, False, False,  True],  # noqa
             [False, False,  True, False,  True],  # noqa
             [False, False,  True,  True, False],  # noqa
             [False,  True,  True, False, False],  # noqa
             [ True,  True, False, False, False]]  # noqa
        )
        if ndim == 2:
            image = self.image
            block_sizes = [3, (3,) * image.ndim]
        else:
            image = cp.stack((self.image, ) * 5, axis=-1)
            ref = cp.stack((ref, ) * 5, axis=-1)
            # Given the same data at each z location, the following block sizes
            # will all give an equivalent result.
            block_sizes = [3, (3,) * image.ndim,
                           (3,) * (image.ndim - 1) + (1,)]
        for block_size in block_sizes:
            out = threshold_local(image, block_size, method='mean',
                                  mode='reflect')
            assert_array_equal(ref, image > out)

    @pytest.mark.parametrize("block_size", [(3,), (3, 3, 3)])
    def test_threshold_local_invalid_block_size(self, block_size):
        # len(block_size) != image.ndim
        with pytest.raises(ValueError):
            threshold_local(self.image, block_size, method="mean")

    @pytest.mark.parametrize("ndim", [2, 3])
    def test_threshold_local_median(self, ndim):
        ref = cp.array(
            [
                [False, False, False, False, True],  # noqa
                [False, False, True, False, False],  # noqa
                [False, False, True, False, False],  # noqa
                [False, False, True, True, False],  # noqa
                [False, True, False, False, False],
            ]  # noqa
        )
        if ndim == 2:
            image = self.image
        else:
            image = cp.stack((self.image,) * 5, axis=-1)
            ref = cp.stack((ref,) * 5, axis=-1)
        out = threshold_local(image, 3, method="median", mode="reflect")
        assert_array_equal(ref, image > out)

    def test_threshold_local_median_constant_mode(self):
        out = threshold_local(
            self.image, 3, method="median", mode="constant", cval=20
        )

        # fmt: off
        expected = cp.array(
            [[20.,  1.,  3.,  4., 20.],   # noqa
             [ 1.,  1.,  3.,  4.,  4.],   # noqa
             [ 2.,  2.,  4.,  4.,  4.],   # noqa
             [ 4.,  4.,  4.,  1.,  2.],   # noqa
             [20.,  5.,  5.,  2., 20.]])  # noqa
        # fmt: on
        assert_array_equal(expected, out)

    def test_threshold_niblack(self):
        # fmt: off
        ref = cp.array(
            [[False, False, False, True, True],  # noqa
             [False, True, True, True, True],    # noqa
             [False, True, True, True, False],   # noqa
             [False, True, True, True, True],    # noqa
             [True, True, False, False, False]]  # noqa
        )
        # fmt: on
        thres = threshold_niblack(self.image, window_size=3, k=0.5)
        out = self.image > thres
        assert_array_equal(ref, out)

    def test_threshold_sauvola(self):
        # fmt: off
        ref = cp.array(
            [[False, False, False, True, True],  # noqa
             [False, False, True, True, True],   # noqa
             [False, False, True, True, False],  # noqa
             [False, True, True, True, False],   # noqa
             [True, True, False, False, False]]  # noqa
        )
        # fmt: on
        thres = threshold_sauvola(self.image, window_size=3, k=0.2, r=128)
        out = self.image > thres
        assert_array_equal(ref, out)

    def test_threshold_niblack_iterable_window_size(self):
        # fmt: off
        ref = cp.array(
            [[False, False, False, True, True],  # noqa
             [False, False, True, True, True],   # noqa
             [False, True, True, True, False],   # noqa
             [False, True, True, True, False],   # noqa
             [True, True, False, False, False]]  # noqa
        )
        # fmt: on
        thres = threshold_niblack(self.image, window_size=[3, 5], k=0.5)
        out = self.image > thres
        assert_array_equal(ref, out)

    def test_threshold_sauvola_iterable_window_size(self):
        # fmt: off
        ref = cp.array(
            [[False, False, False, True, True],  # noqa
             [False, False, True, True, True],   # noqa
             [False, False, True, True, False],  # noqa
             [False, True, True, True, False],   # noqa
             [True, True, False, False, False]]  # noqa
        )
        # fmt: on
        thres = threshold_sauvola(self.image, window_size=(3, 5), k=0.2, r=128)
        out = self.image > thres
        assert_array_equal(ref, out)


def test_otsu_camera_image():
    camera = util.img_as_ubyte(camerad)
    assert 101 < threshold_otsu(camera) < 103


def test_otsu_camera_image_histogram():
    camera = util.img_as_ubyte(camerad)
    hist = histogram(camera.ravel(), 256, source_range="image")
    assert 101 < threshold_otsu(hist=hist) < 103


def test_otsu_camera_image_counts():
    camera = util.img_as_ubyte(camerad)
    counts, bin_centers = histogram(camera.ravel(), 256, source_range="image")
    assert 101 < threshold_otsu(hist=counts) < 103


def test_otsu_zero_count_histogram():
    """Issue #5497.

    As the histogram returned by np.bincount starts with zero,
    it resulted in NaN-related issues.
    """
    x = cp.array([1, 2])

    t1 = threshold_otsu(x)
    t2 = threshold_otsu(hist=cp.bincount(x))
    assert t1 == t2


def test_otsu_coins_image():
    coins = util.img_as_ubyte(coinsd)
    assert 106 < threshold_otsu(coins) < 108


def test_otsu_coins_image_as_float():
    coins = util.img_as_float(coinsd)
    assert 0.41 < threshold_otsu(coins) < 0.42


def test_otsu_astro_image():
    img = util.img_as_ubyte(astronautd)
    with expected_warnings(["grayscale"]):
        assert 109 < threshold_otsu(img) < 111


def test_otsu_one_color_image():
    img = cp.ones((10, 10), dtype=np.uint8)
    assert threshold_otsu(img) == 1


def test_otsu_one_color_image_3d():
    img = cp.ones((10, 10, 10), dtype=np.uint8)
    assert threshold_otsu(img) == 1


def test_li_camera_image():
    image = util.img_as_ubyte(camerad)
    threshold = threshold_li(image)
    ce_actual = _cross_entropy(image, threshold)
    assert 78 < threshold_li(image) < 79
    assert ce_actual < _cross_entropy(image, threshold + 1)
    assert ce_actual < _cross_entropy(image, threshold - 1)


def test_li_coins_image():
    image = util.img_as_ubyte(coinsd)
    threshold = threshold_li(image)
    ce_actual = _cross_entropy(image, threshold)
    assert 94 < threshold_li(image) < 95
    assert ce_actual < _cross_entropy(image, threshold + 1)
    # in the case of the coins image, the minimum cross-entropy is achieved one
    # threshold below that found by the iterative method. Not sure why that is
    # but `threshold_li` does find the stationary point of the function (ie the
    # tolerance can be reduced arbitrarily but the exact same threshold is
    # found), so my guess is some kind of histogram binning effect.
    assert ce_actual < _cross_entropy(image, threshold - 2)


def test_li_coins_image_as_float():
    coins = util.img_as_float(coinsd)
    assert 94 / 255 < threshold_li(coins) < 95 / 255


def test_li_astro_image():
    image = util.img_as_ubyte(astronautd)
    threshold = threshold_li(image)
    ce_actual = _cross_entropy(image, threshold)
    assert 64 < threshold < 65
    assert ce_actual < _cross_entropy(image, threshold + 1)
    assert ce_actual < _cross_entropy(image, threshold - 1)


def test_li_nan_image():
    image = cp.full((5, 5), cp.nan)
    assert cp.isnan(threshold_li(image))


def test_li_inf_image():
    image = cp.array([cp.inf, cp.nan])
    assert threshold_li(image) == cp.inf


def test_li_inf_minus_inf():
    image = cp.array([cp.inf, -cp.inf])
    assert threshold_li(image) == 0


def test_li_constant_image_with_nan():
    image = cp.asarray([8, 8, 8, 8, cp.nan])
    assert threshold_li(image) == 8


def test_li_arbitrary_start_point():
    cell = celld
    max_stationary_point = threshold_li(cell)
    low_stationary_point = threshold_li(
        cell, initial_guess=float(cp.percentile(cell, 5))
    )
    optimum = threshold_li(cell, initial_guess=float(cp.percentile(cell, 95)))
    assert 67 < max_stationary_point < 68
    assert 48 < low_stationary_point < 49
    assert 111 < optimum < 112


def test_li_negative_inital_guess():
    with pytest.raises(ValueError):
        threshold_li(coinsd, initial_guess=-5)


def test_li_pathological_arrays():
    # See https://github.com/scikit-image/scikit-image/issues/4140
    a = cp.array([0, 0, 1, 0, 0, 1, 0, 1])
    b = cp.array([0, 0, 0.1, 0, 0, 0.1, 0, 0.1])
    c = cp.array([0, 0, 0.1, 0, 0, 0.1, 0.01, 0.1])
    d = cp.array([0, 0, 1, 0, 0, 1, 0.5, 1])
    e = cp.array([1, 1])
    f = cp.asarray([1, 2])
    arrays = [a, b, c, d, e, f]
    with np.errstate(divide="ignore"):
        # ignoring "divide by zero encountered in log" error from np.log(0)
        thresholds = cp.array([float(threshold_li(arr)) for arr in arrays])
    assert cp.all(cp.isfinite(thresholds))


def test_yen_camera_image():
    camera = util.img_as_ubyte(camerad)
    assert 145 < threshold_yen(camera) < 147


def test_yen_camera_image_histogram():
    camera = util.img_as_ubyte(camerad)
    hist = histogram(camera.ravel(), 256, source_range="image")
    assert 145 < threshold_yen(hist=hist) < 147


def test_yen_camera_image_counts():
    camera = util.img_as_ubyte(camerad)
    counts, bin_centers = histogram(camera.ravel(), 256, source_range="image")
    assert 145 < threshold_yen(hist=counts) < 147


def test_yen_coins_image():
    coins = util.img_as_ubyte(coinsd)
    assert 109 < threshold_yen(coins) < 111


def test_yen_coins_image_as_float():
    coins = util.img_as_float(coinsd)
    assert 0.43 < threshold_yen(coins) < 0.44


def test_local_even_block_size_error():
    img = camerad
    with pytest.raises(ValueError):
        threshold_local(img, block_size=4)


def test_isodata_camera_image():
    camera = util.img_as_ubyte(camerad)

    threshold = threshold_isodata(camera)
    assert (
        np.floor(
            (
                camera[camera <= threshold].mean()
                + camera[camera > threshold].mean()
            )
            / 2.0
        )
        == threshold
    )
    assert threshold == 102

    assert_array_equal(threshold_isodata(camera, return_all=True), [102, 103])


def test_isodata_camera_image_histogram():
    camera = util.img_as_ubyte(camerad)
    hist = histogram(camera.ravel(), 256, source_range="image")
    threshold = threshold_isodata(hist=hist)
    assert threshold == 102


def test_isodata_camera_image_counts():
    camera = util.img_as_ubyte(camerad)
    counts, bin_centers = histogram(camera.ravel(), 256, source_range="image")
    threshold = threshold_isodata(hist=counts)
    assert threshold == 102


def test_isodata_coins_image():
    coins = util.img_as_ubyte(coinsd)

    threshold = threshold_isodata(coins)
    assert (
        np.floor(
            (coins[coins <= threshold].mean() + coins[coins > threshold].mean())
            / 2.0
        )
        == threshold
    )
    assert threshold == 107

    assert_array_equal(threshold_isodata(coins, return_all=True), [107])


def test_isodata_moon_image():
    moon = util.img_as_ubyte(moond)

    threshold = threshold_isodata(moon)
    assert (
        np.floor(
            (moon[moon <= threshold].mean() + moon[moon > threshold].mean())
            / 2.0
        )
        == threshold
    )
    assert threshold == 86

    thresholds = threshold_isodata(moon, return_all=True)
    for threshold in thresholds:
        assert (
            np.floor(
                (moon[moon <= threshold].mean() + moon[moon > threshold].mean())
                / 2.0
            )
            == threshold
        )
    assert_array_equal(thresholds, [86, 87, 88, 122, 123, 124, 139, 140])


def test_isodata_moon_image_negative_int():
    moon = util.img_as_ubyte(moond).astype(cp.int32)
    moon -= 100

    threshold = threshold_isodata(moon)
    assert (
        np.floor(
            (moon[moon <= threshold].mean() + moon[moon > threshold].mean())
            / 2.0
        )
        == threshold
    )
    assert threshold == -14

    thresholds = threshold_isodata(moon, return_all=True)
    for threshold in thresholds:
        assert (
            np.floor(
                (moon[moon <= threshold].mean() + moon[moon > threshold].mean())
                / 2.0
            )
            == threshold
        )
    assert_array_equal(thresholds, [-14, -13, -12, 22, 23, 24, 39, 40])


def test_isodata_moon_image_negative_float():
    moon = util.img_as_ubyte(moond).astype(cp.float64)
    moon -= 100

    assert -14 < threshold_isodata(moon) < -13

    thresholds = threshold_isodata(moon, return_all=True)
    # fmt: off
    assert_array_almost_equal(thresholds,
        [-13.83789062, -12.84179688, -11.84570312, 22.02148438,   # noqa
          23.01757812,  24.01367188,  38.95507812, 39.95117188])  # noqa
    # fmt: on


def test_threshold_minimum():
    camera = util.img_as_ubyte(camerad)

    threshold = threshold_minimum(camera)
    assert_array_equal(threshold, 85)

    astronaut = util.img_as_ubyte(astronautd)
    threshold = threshold_minimum(astronaut)
    assert_array_equal(threshold, 114)


def test_threshold_minimum_histogram():
    camera = util.img_as_ubyte(camerad)
    hist = histogram(camera.ravel(), 256, source_range="image")
    threshold = threshold_minimum(hist=hist)
    assert_array_equal(threshold, 85)


def test_threshold_minimum_counts():
    camera = util.img_as_ubyte(camerad)
    counts, bin_centers = histogram(camera.ravel(), 256, source_range="image")
    threshold = threshold_minimum(hist=counts)
    assert_array_equal(threshold, 85)


def test_threshold_minimum_synthetic():
    img = cp.arange(25 * 25, dtype=cp.uint8).reshape((25, 25))
    img[0:9, :] = 50
    img[14:25, :] = 250

    threshold = threshold_minimum(img)
    assert_array_equal(threshold, 95)


def test_threshold_minimum_failure():
    img = cp.zeros((16 * 16), dtype=cp.uint8)
    with pytest.raises(RuntimeError):
        threshold_minimum(img)


def test_mean():
    img = cp.zeros((2, 6))
    img[:, 2:4] = 1
    img[:, 4:] = 2
    assert threshold_mean(img) == 1.0


# also run cases with nbins > 100000 to also test CuPy-based code path.
@pytest.mark.parametrize("kwargs", [{}, {"nbins": 300000}])
@pytest.mark.parametrize("dtype", [cp.uint8, cp.int16, cp.float16, cp.float32])
def test_triangle_uniform_images(dtype, kwargs):
    if dtype == cp.float16:
        # Avoid potential NaN bin edges if nbins exceeds float16 range.
        # (NaN in bin edges can cause out of bounds CUDA memory access in
        # `cp.histogram`)
        if "nbins" in kwargs:
            kwargs["nbins"] = min(
                kwargs["nbins"], int(np.finfo(np.float16).max)
            )
    assert threshold_triangle(cp.zeros((10, 10), dtype=dtype), **kwargs) == 0
    assert threshold_triangle(cp.ones((10, 10), dtype=dtype), **kwargs) == 1
    assert threshold_triangle(cp.full((10, 10), 2, dtype=dtype), **kwargs) == 2


# also run cases with nbins > 100000 to also test CuPy-based code path.
@pytest.mark.parametrize("kwargs", [{}, {"nbins": 300000}])
@pytest.mark.parametrize(
    "data, expected_value",
    [
        (cp.invert(cp.array(data.text())), 151),
        (cp.array(data.text()), 104),
        (coinsd, 80),
        (cp.invert(coinsd), 175),
    ],
)
def test_triangle_uint_images(data, expected_value, kwargs):
    assert threshold_triangle(data, **kwargs) == expected_value


def test_triangle_float_images():
    text = cp.array(data.text())
    int_bins = int(text.max() - text.min() + 1)
    # Set nbins to match the uint case and threshold as float.
    assert (
        round(float(threshold_triangle(text.astype(float), nbins=int_bins)))
        == 104
    )
    # Check that rescaling image to floats in unit interval is equivalent.
    assert (
        round(float(threshold_triangle(text / 255.0, nbins=int_bins) * 255))
        == 104
    )
    # Repeat for inverted image.
    assert (
        round(
            float(
                threshold_triangle(
                    cp.invert(text).astype(float), nbins=int_bins
                )
            )
        )
        == 151
    )
    assert (
        round(
            float(
                threshold_triangle(cp.invert(text) / 255, nbins=int_bins) * 255
            )
        )
        == 151
    )


def test_triangle_flip():
    # Depending on the skewness, the algorithm flips the histogram.
    # We check that the flip doesn't affect too much the result.
    img = camerad
    inv_img = cp.invert(img)
    t = threshold_triangle(inv_img)
    t_inv_img = inv_img > t
    t_inv_inv_img = cp.invert(t_inv_img)

    t = threshold_triangle(img)
    t_img = img > t

    # Check that most of the pixels are identical
    # See numpy #7685 for a future cp.testing API
    unequal_pos = cp.where(t_img.ravel() != t_inv_inv_img.ravel())
    assert len(unequal_pos[0]) / t_img.size < 1e-2


# TODO: need generic_filter
# @pytest.mark.parametrize(
#     "window_size, mean_kernel",
#     [(11, cp.full((11,) * 2,  1 / 11 ** 2)),
#      ((11, 11), cp.full((11, 11), 1 / 11 ** 2)),
#      ((9, 13), cp.full((9, 13), 1 / math.prod((9, 13)))),
#      ((13, 9), cp.full((13, 9), 1 / math.prod((13, 9)))),
#      ((1, 9), cp.full((1, 9), 1 / math.prod((1, 9))))
#      ]
# )
# def test_mean_std_2d(window_size, mean_kernel):
#     image = cp.asarray(np.random.rand(256, 256))
#     m, s = _mean_std(image, w=window_size)
#     expected_m = ndi.convolve(image, mean_kernel, mode='mirror')
#     cp.testing.assert_allclose(m, expected_m)
#     expected_s = ndi.generic_filter(image, cp.std, size=window_size,
#                                     mode='mirror')
#     cp.testing.assert_allclose(s, expected_s)

# TODO: need generic_filter
# @pytest.mark.parametrize(
#     "window_size, mean_kernel", [
#         (5, cp.full((5,) * 3, 1 / 5) ** 3),
#         ((5, 5, 5), cp.full((5, 5, 5), 1 / 5 ** 3)),
#         ((1, 5, 5), cp.full((1, 5, 5), 1 / 5 ** 2)),
#         ((3, 5, 7), cp.full((3, 5, 7), 1 / math.prod((3, 5, 7))))]
# )
# def test_mean_std_3d(window_size, mean_kernel):
#     image = cp.asarray(np.random.rand(40, 40, 40))
#     m, s = _mean_std(image, w=window_size)
#     expected_m = ndi.convolve(image, mean_kernel, mode='mirror')
#     cp.testing.assert_allclose(m, expected_m)
#     expected_s = ndi.generic_filter(image, cp.std, size=window_size,
#                                     mode='mirror')
#     cp.testing.assert_allclose(s, expected_s)


@pytest.mark.parametrize(
    "threshold_func",
    [threshold_local, threshold_niblack, threshold_sauvola],
)
@pytest.mark.parametrize("dtype", [cp.uint8, cp.int16, cp.float16, cp.float32])
def test_variable_dtypes(threshold_func, dtype):
    r = 255 * cp.random.rand(32, 16)
    r = r.astype(dtype, copy=False)

    kwargs = {}
    if threshold_func is threshold_local:
        kwargs = dict(block_size=9)
    elif threshold_func is threshold_sauvola:
        kwargs = dict(r=128)

    # use double precision result as a reference
    expected = threshold_func(r.astype(float), **kwargs)

    out = threshold_func(r, **kwargs)
    assert out.dtype == _supported_float_type(dtype)
    assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_niblack_sauvola_pathological_image():
    # For certain values, floating point error can cause
    # E(X^2) - (E(X))^2 to be negative, and taking the square root of this
    # resulted in NaNs. Here we check that these are safely caught.
    # see https://github.com/scikit-image/scikit-image/issues/3007
    value = 0.03082192 + 2.19178082e-09
    src_img = cp.full((4, 4), value).astype(cp.float64)
    assert not cp.any(cp.isnan(threshold_niblack(src_img)))


def test_bimodal_multiotsu_hist():
    for name in ["camera", "moon", "coins", "text", "clock", "page"]:
        img = cp.array(getattr(data, name)())
        assert threshold_otsu(img) == threshold_multiotsu(img, 2)

    for name in ["chelsea", "coffee", "astronaut", "rocket"]:
        img = rgb2gray(cp.array(getattr(data, name)()))
        assert threshold_otsu(img) == threshold_multiotsu(img, 2)


def test_check_multiotsu_results():
    # fmt: off
    image = 0.25 * cp.array([[0, 1, 2, 3, 4],
                             [0, 1, 2, 3, 4],
                             [0, 1, 2, 3, 4],
                             [0, 1, 2, 3, 4]])
    # fmt: on
    for idx in range(3, 6):
        thr_multi = threshold_multiotsu(image, classes=idx)
        assert len(thr_multi) == idx - 1


def test_multiotsu_output():
    image = cp.zeros((100, 100), dtype="int")
    coords = [(25, 25), (50, 50), (75, 75)]
    values = [64, 128, 192]
    for coor, val in zip(coords, values):
        rr, cc = disk(coor, 20)
        rr, cc = cp.asarray(rr), cp.asarray(cc)
        image[rr, cc] = val
    thresholds = [0, 64, 128]
    assert_array_equal(thresholds, threshold_multiotsu(image, classes=4))


def test_multiotsu_astro_image():
    img = util.img_as_ubyte(astronautd)
    with expected_warnings(["grayscale"]):
        assert_array_almost_equal(threshold_multiotsu(img), [58, 149])


def test_multiotsu_more_classes_then_values():
    img = cp.ones((10, 10), dtype=cp.uint8)
    with pytest.raises(ValueError):
        threshold_multiotsu(img, classes=2)
    img[:, 3:] = 2
    with pytest.raises(ValueError):
        threshold_multiotsu(img, classes=3)
    img[:, 6:] = 3
    with pytest.raises(ValueError):
        threshold_multiotsu(img, classes=4)


@pytest.mark.skip("_get_multiotsu_thresh_indices functions not implemented yet")
def test_multiotsu_lut():
    for classes in [2, 3, 4]:
        for name in ["camera", "moon", "coins", "text", "clock", "page"]:
            img = cp.array(getattr(data, name)())
            prob, bin_centers = histogram(
                img.ravel(), nbins=256, source_range="image", normalize=True
            )
            prob = prob.astype("float32")

            result_lut = _get_multiotsu_thresh_indices_lut(prob, classes - 1)
            result = _get_multiotsu_thresh_indices(prob, classes - 1)

            assert_array_equal(result_lut, result)


def test_multiotsu_missing_img_and_hist():
    with pytest.raises(Exception):
        threshold_multiotsu()


def test_multiotsu_hist_parameter():
    for classes in [2, 3, 4]:
        for name in ["camera", "moon", "coins", "text", "clock", "page"]:
            img = cp.array(getattr(data, name)())
            sk_hist = histogram(img, nbins=256)
            #
            thresh_img = threshold_multiotsu(img, classes)
            thresh_sk_hist = threshold_multiotsu(classes=classes, hist=sk_hist)
            assert cp.allclose(thresh_img, thresh_sk_hist)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize(
    "name", ["otsu", "yen", "isodata", "triangle", "mean", "li"]
)
def test_global_thresholds_many_bins_match_cpu(name, dtype):
    # 8-bit images have 256 bins, 16-bit images up to 65536. The kernels
    # accumulate in float64, scikit-image in float32, so for many bins the
    # thresholds can differ where the criterion is flat near its maximum (by a
    # relative 1e-8 in the value of the criterion for the case below).
    from skimage import filters as sk_filters

    from cucim.skimage import filters as cu_filters

    scale = np.iinfo(dtype).max
    rng = np.random.default_rng(0)
    image = np.clip(
        data.camera().astype(np.float64) / 255 * scale
        + rng.normal(0, 0.01 * scale, data.camera().shape),
        0,
        scale,
    ).astype(dtype)
    expected = getattr(sk_filters, f"threshold_{name}")(image)
    result = getattr(cu_filters, f"threshold_{name}")(cp.asarray(image))
    atol = 0 if dtype == np.uint8 else 1e-3 * scale
    assert_allclose(result, expected, rtol=1e-6, atol=atol)


@pytest.mark.parametrize("classes, nbins", [(2, 40), (3, 40), (4, 30), (5, 20)])
def test_multiotsu_matches_exhaustive_search(classes, nbins):
    import itertools

    rng = np.random.default_rng(classes)
    for _ in range(10):
        counts = rng.integers(0, 1000, nbins)
        counts[0] += 1
        counts[-1] += 1
        c = counts.astype(np.float64)
        P = np.concatenate(([0.0], np.cumsum(c)))
        S = np.concatenate(([0.0], np.cumsum(c * np.arange(nbins))))

        def term(u, v):
            w = P[v] - P[u]
            return np.where(
                w > 0, (S[v] - S[u]) ** 2 / np.where(w > 0, w, 1), 0
            )

        cand = np.array(
            list(itertools.combinations(range(nbins - 1), classes - 1))
        )
        edges = np.concatenate(
            [
                np.zeros((len(cand), 1), int),
                cand + 1,
                np.full((len(cand), 1), nbins),
            ],
            axis=1,
        )
        value = sum(term(edges[:, k], edges[:, k + 1]) for k in range(classes))
        expected = cand[int(np.argmax(value))]  # first maximum
        result = threshold_multiotsu(
            classes=classes, hist=(cp.asarray(counts), cp.arange(nbins))
        )
        assert_array_equal(result, expected)


def _multiotsu_plain_dp(counts, classes):
    """Thresholds from the dynamic programming that searches every pair of
    thresholds in each stage (smallest end bin among equal maxima)."""
    n = counts.size
    c = counts.astype(np.float32).astype(np.float64)
    P = np.concatenate(([0.0], np.cumsum(c)))
    S = np.concatenate(([0.0], np.cumsum(c * np.arange(n))))

    def term(u, v):
        w = P[v] - P[u]
        return np.where(w > 0, (S[v] - S[u]) ** 2 / np.where(w > 0, w, 1), 0)

    G = term(np.arange(n), n)  # the last class starts at bin u
    backs = []
    for k in range(classes - 1, 1, -1):
        remaining = classes - k
        u_end = n - remaining
        G_out = np.zeros(n + 1)
        back = np.zeros(n + 1, dtype=int)
        for u in range(k - 1, u_end):
            v = np.arange(u + 1, u_end + 1)
            value = term(u, v) + G[v]
            i = int(np.argmax(value))
            G_out[u], back[u] = value[i], v[i]
        G = G_out
        backs.append((k, back))
    v = np.arange(1, n - (classes - 1) + 1)
    chain = {1: {0: int(v[int(np.argmax(term(0, v) + G[v]))])}}
    chain.update({k: b for k, b in backs})
    u, thresholds = 0, []
    for k in range(1, classes):
        v = int(chain[k][u])
        thresholds.append(v - 1)
        u = v
    return np.array(thresholds)


@pytest.mark.parametrize("classes", [3, 4, 6])
@pytest.mark.parametrize("kind", ["random", "ties", "sparse", "palindrome"])
def test_multiotsu_divide_and_conquer_matches_plain_dp(classes, kind):
    # Every stage is searched by divide and conquer, which relies on the best
    # previous threshold not decreasing with the end bin. Check that the result
    # is the one of the search over all pairs, also for exact ties.
    rng = np.random.default_rng(classes)
    nbins = 701  # not a power of two: several levels with unequal bounds
    for _ in range(3):
        if kind == "random":
            counts = rng.integers(0, 1000, nbins)
        elif kind == "ties":
            counts = rng.integers(0, 3, nbins)
        elif kind == "sparse":
            counts = (rng.random(nbins) < 0.1) * rng.integers(1, 50, nbins)
        else:
            half = rng.integers(0, 100, nbins // 2)
            counts = np.concatenate([half, [5], half[::-1]])
        counts[0] += 1
        counts[-1] += 1
        expected = _multiotsu_plain_dp(counts, classes)
        result = threshold_multiotsu(
            classes=classes,
            hist=(cp.asarray(counts), cp.arange(nbins)),
        )
        assert_array_equal(result, expected)


def test_multiotsu_many_bins():
    # 16-bit image: 65536 bins; two classes give Otsu's threshold
    rng = np.random.default_rng(0)
    image = np.concatenate(
        [rng.normal(15000, 2000, 6000), rng.normal(40000, 3000, 4000)]
    )
    image = cp.asarray(np.clip(image, 0, 65535).astype(np.uint16)).reshape(
        100, 100
    )
    assert threshold_multiotsu(image, 2) == threshold_otsu(image)
    thresholds = threshold_multiotsu(image, 3)
    assert thresholds.shape == (2,)
    assert thresholds[0] < thresholds[1]
    assert thresholds.dtype == image.dtype or thresholds.dtype.kind in "iuf"


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
@pytest.mark.parametrize("name", ["otsu", "triangle"])
def test_constant_image_returns_the_pixel_value(name, dtype):
    from cucim.skimage import filters

    image = cp.full((20, 30), 7, dtype=dtype)
    result = getattr(filters, f"threshold_{name}")(image)
    assert result == 7
    assert result.dtype == dtype

    # almost constant: one pixel is different
    image[3, 4] = 9
    assert 7 <= getattr(filters, f"threshold_{name}")(image) <= 9


@pytest.mark.parametrize("nbins", [16, 100, 256, 1000, 4200])
def test_threshold_minimum_matches_skimage_on_synthetic_histograms(nbins):
    # the GPU smoothing loop gives the same threshold, or the same error, as
    # scikit-image (4200 bins use the scikit-image function itself)
    from skimage.filters import threshold_minimum as sk_threshold_minimum

    rng = np.random.default_rng(nbins)
    x = np.arange(nbins)
    # (the scikit-image function is slow for many bins and iterations)
    for kind in range(1 if nbins > 4096 else 4):
        if kind in (1, 2) and nbins > 256:
            continue  # (many iterations: slow for scikit-image)
        if kind == 0:  # bimodal
            counts = np.exp(-0.5 * ((x - nbins * 0.25) / (nbins * 0.07)) ** 2)
            counts = counts * 1000 + 600 * np.exp(
                -0.5 * ((x - nbins * 0.7) / (nbins * 0.1)) ** 2
            )
        elif kind == 1:  # noisy: many maxima
            counts = rng.integers(0, 1000, nbins).astype(float)
        elif kind == 2:  # unimodal: no second maximum
            counts = 1000 * np.exp(
                -0.5 * ((x - nbins * 0.5) / (nbins * 0.1)) ** 2
            )
        else:  # three modes
            counts = sum(
                a * np.exp(-0.5 * ((x - nbins * m) / (nbins * 0.05)) ** 2)
                for a, m in ((900, 0.2), (700, 0.5), (800, 0.8))
            )
        counts = np.round(counts + rng.integers(0, 3, nbins)).astype(np.float32)
        counts[0] += 1
        counts[-1] += 1
        centers = x * 1.5 + 2
        try:
            expected = sk_threshold_minimum(hist=(counts, centers))
        except RuntimeError as e:
            with pytest.raises(RuntimeError, match=str(e)[:20]):
                threshold_minimum(
                    hist=(cp.asarray(counts), cp.asarray(centers))
                )
        else:
            result = threshold_minimum(
                hist=(cp.asarray(counts), cp.asarray(centers))
            )
            assert float(result) == float(expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16, np.int8])
@pytest.mark.parametrize("kind", ["camera", "bimodal", "uniform"])
def test_threshold_li_integer_images(dtype, kind):
    from skimage.filters import threshold_li as sk_threshold_li

    rng = np.random.default_rng(0)
    shape = data.camera().shape
    if kind == "camera":
        values = data.camera().astype(np.float64)
    elif kind == "bimodal":
        values = np.where(
            rng.random(shape) < 0.4,
            rng.normal(60, 10, shape),
            rng.normal(180, 15, shape),
        )
    else:
        values = rng.integers(0, 256, shape).astype(np.float64)
    info = np.iinfo(dtype)
    if np.dtype(dtype).itemsize == 2:
        values = values * 100
    if info.min < 0:  # center the values around zero
        values = values - (values.max() + values.min()) / 2
    image = np.clip(values, info.min, info.max).astype(dtype)
    assert image.min() < image.max()

    calls = []
    result = threshold_li(cp.asarray(image), iter_callback=calls.append)
    # scikit-image subtracts the minimum in the image dtype, which overflows
    # for signed types when the range is larger than the maximum of the type
    # (it returns nan or a wrong value): use a float copy there (with the
    # tolerance of the integer algorithm)
    if (
        np.dtype(dtype).kind == "i"
        and int(image.max()) - int(image.min()) > info.max
    ):
        expected = sk_threshold_li(image.astype(np.float64), tolerance=0.5)
    else:
        expected = sk_threshold_li(image)
    assert_allclose(result, expected, rtol=1e-9)
    assert len(calls) >= 2
    assert_allclose(calls[-1], result)


def test_threshold_li_integer_constant_image():
    image = cp.full((10, 12), 5, dtype=cp.uint8)
    assert threshold_li(image) == 5


@pytest.mark.parametrize("window_size", [3, 15, (5, 11)])
def test_niblack_sauvola_local_statistics(window_size):
    # The local mean and standard deviation are computed in float64 before they
    # are cast: scikit-image casts the mean to float32 first, which loses the
    # standard deviation of images with a large offset. Reference: box filters
    # of the image shifted by the offset, in float64 on the host.
    import scipy.ndimage as scipy_ndi

    from cucim.skimage.filters.thresholding import _mean_std

    image = (data.camera().astype(np.float32) / 255 + 1000).astype(np.float32)
    shifted = image.astype(np.float64) - 1000.0
    mean0 = scipy_ndi.uniform_filter(shifted, window_size, mode="mirror")
    mean_sq0 = scipy_ndi.uniform_filter(
        shifted * shifted, window_size, mode="mirror"
    )
    expected_std = np.sqrt(np.maximum(mean_sq0 - mean0 * mean0, 0))

    m, s = _mean_std(cp.asarray(image), window_size)
    assert m.dtype == s.dtype == np.float32
    assert_allclose(m, mean0 + 1000.0, rtol=1e-6)
    # (the variance is a difference of values of about 1e6, in float64)
    assert_allclose(s, expected_std, rtol=1e-4, atol=1e-4 * expected_std.max())
