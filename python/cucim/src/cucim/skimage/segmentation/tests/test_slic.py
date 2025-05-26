from itertools import product

import cupy as cp
import numpy as np
import pytest
from cupy.testing import assert_array_equal
from numpy.testing import assert_equal
from skimage import data

from cucim.skimage import filters, img_as_float
from cucim.skimage._shared.testing import expected_warnings
from cucim.skimage.segmentation import slic


def test_color_2d():
    rng = np.random.default_rng(0)
    img = np.zeros((20, 21, 3))
    img[:10, :10, 0] = 1
    img[10:, :10, 1] = 1
    img[10:, 10:, 2] = 1
    img += 0.01 * rng.normal(size=img.shape)
    img[img > 1] = 1
    img[img < 0] = 0
    img = cp.asarray(img)
    seg = slic(
        img, n_segments=4, sigma=0, enforce_connectivity=False, start_label=0
    )

    # we expect 4 segments
    assert_equal(len(np.unique(seg)), 4)
    assert_equal(seg.shape, img.shape[:-1])
    assert_array_equal(seg[:10, :10], 0)
    assert_array_equal(seg[10:, :10], 2)
    assert_array_equal(seg[:10, 10:], 1)
    assert_array_equal(seg[10:, 10:], 3)


def test_multichannel_2d():
    rng = np.random.default_rng(0)
    img = np.zeros((20, 20, 8))
    img[:10, :10, 0:2] = 1
    img[:10, 10:, 2:4] = 1
    img[10:, :10, 4:6] = 1
    img[10:, 10:, 6:8] = 1
    img += 0.01 * rng.normal(size=img.shape)
    img = np.clip(img, 0, 1, out=img)
    img = cp.asarray(img)
    seg = slic(img, n_segments=4, enforce_connectivity=False, start_label=0)

    # we expect 4 segments
    assert_equal(len(np.unique(seg)), 4)
    assert_equal(seg.shape, img.shape[:-1])
    assert_array_equal(seg[:10, :10], 0)
    assert_array_equal(seg[10:, :10], 2)
    assert_array_equal(seg[:10, 10:], 1)
    assert_array_equal(seg[10:, 10:], 3)


def test_gray_2d():
    rng = np.random.default_rng(0)
    img = np.zeros((20, 21))
    img[:10, :10] = 0.33
    img[10:, :10] = 0.67
    img[10:, 10:] = 1.00
    img += 0.0033 * rng.normal(size=img.shape)
    img[img > 1] = 1
    img[img < 0] = 0
    img = cp.asarray(img)
    seg = slic(
        img,
        sigma=0,
        n_segments=4,
        compactness=1,
        channel_axis=None,
        convert2lab=False,
        start_label=0,
    )

    assert_equal(len(np.unique(seg)), 4)
    assert_equal(seg.shape, img.shape)
    assert_array_equal(seg[:10, :10], 0)
    assert_array_equal(seg[10:, :10], 2)
    assert_array_equal(seg[:10, 10:], 1)
    assert_array_equal(seg[10:, 10:], 3)


def test_gray2d_default_channel_axis():
    img = np.zeros((20, 21))
    img[:10, :10] = 0.33
    img = cp.asarray(img)
    with pytest.raises(
        ValueError, match="channel_axis=-1 indicates multichannel"
    ):
        slic(img)
    slic(img, channel_axis=None)


def _check_segment_labels(seg1, seg2, allowed_mismatch_ratio=0.1):
    size = seg1.size
    ndiff = int(cp.sum(seg1 != seg2))
    assert (ndiff / size) < allowed_mismatch_ratio


def test_slic_consistency_across_image_magnitude():
    # verify that that images of various scales across integer and float dtypes
    # give the same segmentation result
    img_uint8 = cp.asarray(data.cat()[:256, :128])
    img_uint16 = 256 * img_uint8.astype(np.uint16)
    img_float32 = img_as_float(img_uint8)
    img_float32_norm = img_float32 / img_float32.max()
    img_float32_offset = img_float32 + 1000.0

    seg1 = slic(img_uint8)
    seg2 = slic(img_uint16)
    seg3 = slic(img_float32)
    seg4 = slic(img_float32_norm)
    seg5 = slic(img_float32_offset)

    if False:  # (TODO: fix this test case)
        assert_array_equal(seg1, seg2)
        assert_array_equal(seg1, seg3)
        assert_array_equal(seg4, seg5)
    else:
        # TODO: scikit-image does not need these tolerances
        sz = img_uint8.size
        assert int(cp.sum(seg1 != seg2)) < 0.001 * sz
        assert int(cp.sum(seg1 != seg3)) < 0.001 * sz
        assert int(cp.sum(seg4 != seg5)) < 0.02 * sz

    # Floating point cases can have mismatch due to floating point error
    # exact match was observed on x86_64, but mismatches seen no i686.
    # For now just verify that a similar number of superpixels are present in
    # each case.
    n_seg1 = seg1.max()
    n_seg4 = seg4.max()
    assert abs(n_seg1 - n_seg4) / n_seg1 < 0.5


def test_color_3d():
    rng = np.random.default_rng(0)
    img = np.zeros((20, 21, 22, 3))
    slices = []
    for dim_size in img.shape[:-1]:
        midpoint = dim_size // 2
        slices.append((slice(None, midpoint), slice(midpoint, None)))
    slices = list(product(*slices))
    colors = list(product(*(([0, 1],) * 3)))
    for s, c in zip(slices, colors):
        img[s] = c
    img += 0.01 * rng.normal(size=img.shape)
    img[img > 1] = 1
    img[img < 0] = 0
    img = cp.asarray(img)
    seg = slic(img, sigma=0, n_segments=8, start_label=0)

    assert_equal(len(np.unique(seg)), 8)
    for s, c in zip(slices, range(8)):
        assert_array_equal(seg[s], c)


def test_gray_3d():
    rng = np.random.default_rng(0)
    img = np.zeros((20, 21, 22))
    slices = []
    for dim_size in img.shape:
        midpoint = dim_size // 2
        slices.append((slice(None, midpoint), slice(midpoint, None)))
    slices = list(product(*slices))
    shades = np.arange(0, 1.000001, 1.0 / 7)
    for s, sh in zip(slices, shades):
        img[s] = sh
    img += 0.001 * rng.normal(size=img.shape)
    img[img > 1] = 1
    img[img < 0] = 0
    img = cp.asarray(img)
    seg = slic(
        img,
        sigma=0,
        n_segments=8,
        compactness=1,
        channel_axis=None,
        convert2lab=False,
        start_label=0,
    )

    assert_equal(len(np.unique(seg)), 8)
    for s, c in zip(slices, range(8)):
        assert_array_equal(seg[s], c)


def test_list_sigma():
    rng = np.random.default_rng(0)
    img = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]], float)
    img += 0.1 * rng.normal(size=img.shape)
    img = cp.asarray(img)
    result_sigma = cp.asarray([[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]], int)
    with expected_warnings(
        ["Input image is 2D: sigma number of " "elements must be 2"]
    ):
        seg_sigma = slic(
            img,
            n_segments=2,
            sigma=[1, 50, 1],
            channel_axis=None,
            start_label=0,
        )
    assert_array_equal(seg_sigma, result_sigma)


def test_spacing():
    rng = np.random.default_rng(0)
    img = np.array([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], float)
    result_non_spaced = cp.asarray([[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]], int)
    result_spaced = cp.asarray([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], int)
    img += 0.1 * rng.normal(size=img.shape)
    img = cp.asarray(img)
    seg_non_spaced = slic(
        img,
        n_segments=2,
        sigma=0,
        channel_axis=None,
        compactness=1.0,
        start_label=0,
    )
    seg_spaced = slic(
        img,
        n_segments=2,
        sigma=0,
        spacing=[500, 1],
        compactness=1.0,
        channel_axis=None,
        start_label=0,
    )
    assert_array_equal(seg_non_spaced, result_non_spaced)
    assert_array_equal(seg_spaced, result_spaced)


def test_invalid_lab_conversion():
    img = cp.asarray([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], float) + 1
    with pytest.raises(ValueError):
        slic(img, channel_axis=-1, convert2lab=True, start_label=0)


def test_enforce_connectivity():
    img = cp.array(
        [[0, 0, 0, 1, 1, 1], [1, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 0]], float
    )

    segments_connected = slic(
        img,
        2,
        compactness=0.0001,
        enforce_connectivity=True,
        convert2lab=False,
        start_label=0,
        channel_axis=None,
    )
    segments_disconnected = slic(
        img,
        2,
        compactness=0.0001,
        enforce_connectivity=False,
        convert2lab=False,
        start_label=0,
        channel_axis=None,
    )

    # Make sure nothing fatal occurs (e.g. buffer overflow) at low values of
    # max_size_factor
    segments_connected_low_max = slic(
        img,
        2,
        compactness=0.0001,
        enforce_connectivity=True,
        convert2lab=False,
        max_size_factor=0.8,
        start_label=0,
        channel_axis=None,
    )

    result_connected = cp.array(
        [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]], float
    )

    result_disconnected = cp.array(
        [[0, 0, 0, 1, 1, 1], [1, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 0]], float
    )

    assert_array_equal(segments_connected, result_connected)
    assert_array_equal(segments_disconnected, result_disconnected)
    assert_array_equal(segments_connected_low_max, result_connected)


@pytest.mark.xfail(reason="slic_zero not implemented yet in cuCIM")
def test_slic_zero():
    # Same as test_color_2d but with slic_zero=True
    rng = np.random.default_rng(0)
    img = np.zeros((20, 21, 3))
    img[:10, :10, 0] = 1
    img[10:, :10, 1] = 1
    img[10:, 10:, 2] = 1
    img += 0.01 * rng.normal(size=img.shape)
    img[img > 1] = 1
    img[img < 0] = 0
    img = cp.asarray(img)
    seg = slic(img, n_segments=4, sigma=0, slic_zero=True, start_label=0)

    # we expect 4 segments
    assert_equal(len(cp.unique(seg)), 4)
    assert_equal(seg.shape, img.shape[:-1])
    assert_array_equal(seg[:10, :10], 0)
    assert_array_equal(seg[10:, :10], 2)
    assert_array_equal(seg[:10, 10:], 1)
    assert_array_equal(seg[10:, 10:], 3)


def test_more_segments_than_pixels():
    rng = np.random.default_rng(0)
    img = np.zeros((20, 21))
    img[:10, :10] = 0.33
    img[10:, :10] = 0.67
    img[10:, 10:] = 1.00
    img += 0.0033 * rng.normal(size=img.shape)
    img[img > 1] = 1
    img[img < 0] = 0
    img = cp.asarray(img)
    seg = slic(
        img,
        sigma=0,
        n_segments=500,
        compactness=1,
        channel_axis=None,
        convert2lab=False,
        start_label=0,
    )
    assert cp.all(seg.ravel() == cp.arange(seg.size)).get()


@pytest.mark.parametrize(
    "dtype", ["float16", "float32", "float64", "uint8", "int"]
)
def test_dtype_support(dtype):
    img = cp.random.rand(28, 28).astype(dtype)

    # Simply run the function to assert that it runs without error
    slic(img, start_label=1, channel_axis=None)


# TODO: fix and remove xfail
@pytest.mark.xfail(
    reason="bug fix needed in cuCIM. labels should not contain 0"
)
def test_start_label_fix():
    """Tests the fix for a bug producing a label < start_label (gh-6240).

    For the v0.19.1 release, the `img` and `slic` call as below result in two
    non-contiguous regions with value 0 despite `start_label=1`. We verify
    that the minimum label is now `start_label` as expected.
    """

    # generate bumpy data that gives unexpected label prior to bug fix
    rng = np.random.default_rng(9)
    img = rng.standard_normal((8, 13)) > 0

    img = cp.asarray(img)
    img = filters.gaussian(img, sigma=1)

    start_label = 1
    superp = slic(
        img,
        start_label=start_label,
        channel_axis=None,
        n_segments=6,
        compactness=0.01,
        enforce_connectivity=True,
        max_num_iter=10,
    )
    assert superp.min().get() == start_label


def test_raises_ValueError_if_input_has_NaN():
    img = np.zeros((4, 5), dtype=float)
    img[2, 3] = np.nan
    img = cp.asarray(img)
    with pytest.raises(ValueError):
        slic(img, channel_axis=None, check_finite_and_constant=True)

    # mask = ~np.isnan(img)
    # slic(img, mask=mask, channel_axis=None)


@pytest.mark.parametrize("inf", [-np.inf, np.inf])
def test_raises_ValueError_if_input_has_inf(inf):
    img = np.zeros((4, 5), dtype=float)
    img[2, 3] = inf
    with pytest.raises(ValueError):
        slic(img, channel_axis=None, check_finite_and_constant=True)

    # mask = np.isfinite(img)
    # slic(img, mask=mask, channel_axis=None)
