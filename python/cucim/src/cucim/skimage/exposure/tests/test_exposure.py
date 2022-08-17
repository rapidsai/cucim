import warnings

import cupy as cp
import numpy as np
import pytest
from cupy.testing import assert_array_almost_equal, assert_array_equal
from numpy.testing import assert_almost_equal
from skimage import data

from cucim.skimage import exposure, util
from cucim.skimage._shared._warnings import expected_warnings
from cucim.skimage._shared.utils import _supported_float_type
from cucim.skimage.color import rgb2gray
from cucim.skimage.exposure.exposure import intensity_range
from cucim.skimage.util.dtype import dtype_range

# Test integer histograms
# =======================


def test_wrong_source_range():
    im = cp.array([-1, 100], dtype=cp.int8)
    with pytest.raises(ValueError):
        frequencies, bin_centers = exposure.histogram(
            im, source_range="foobar"
        )


def test_negative_overflow():
    im = cp.array([-1, 100], dtype=cp.int8)
    frequencies, bin_centers = exposure.histogram(im)
    assert_array_equal(bin_centers, cp.arange(-1, 101))
    assert frequencies[0] == 1
    assert frequencies[-1] == 1
    assert_array_equal(frequencies[1:-1], 0)


def test_all_negative_image():
    im = cp.array([-100, -1], dtype=cp.int8)
    frequencies, bin_centers = exposure.histogram(im)
    assert_array_equal(bin_centers, cp.arange(-100, 0))
    assert frequencies[0] == 1
    assert frequencies[-1] == 1
    assert_array_equal(frequencies[1:-1], 0)


def test_int_range_image():
    im = cp.array([10, 100], dtype=cp.int8)
    frequencies, bin_centers = exposure.histogram(im)
    assert len(bin_centers) == len(frequencies)
    assert bin_centers[0] == 10
    assert bin_centers[-1] == 100


def test_multichannel_int_range_image():
    im = cp.array([[10, 5], [100, 102]], dtype=np.int8)
    frequencies, bin_centers = exposure.histogram(im, channel_axis=-1)
    for ch in range(im.shape[-1]):
        assert len(frequencies[ch]) == len(bin_centers)
    assert bin_centers[0] == 5
    assert bin_centers[-1] == 102


def test_peak_uint_range_dtype():
    im = cp.array([10, 100], dtype=cp.uint8)
    frequencies, bin_centers = exposure.histogram(im, source_range="dtype")
    assert_array_equal(bin_centers, cp.arange(0, 256))
    assert frequencies[10] == 1
    assert frequencies[100] == 1
    assert frequencies[101] == 0
    assert frequencies.shape == (256,)


def test_peak_int_range_dtype():
    im = cp.array([10, 100], dtype=cp.int8)
    frequencies, bin_centers = exposure.histogram(im, source_range="dtype")
    assert_array_equal(bin_centers, cp.arange(-128, 128))
    assert frequencies[128 + 10] == 1
    assert frequencies[128 + 100] == 1
    assert frequencies[128 + 101] == 0
    assert frequencies.shape == (256,)


def test_flat_uint_range_dtype():
    im = cp.linspace(0, 255, 256, dtype=cp.uint8)
    frequencies, bin_centers = exposure.histogram(im, source_range="dtype")
    assert_array_equal(bin_centers, cp.arange(0, 256))
    assert frequencies.shape == (256,)


def test_flat_int_range_dtype():
    im = cp.linspace(-128, 128, 256, dtype=cp.int8)
    frequencies, bin_centers = exposure.histogram(im, source_range="dtype")
    assert_array_equal(bin_centers, cp.arange(-128, 128))
    assert frequencies.shape == (256,)


@pytest.mark.parametrize('dtype', [cp.float16, cp.float32, cp.float64])
def test_peak_float_out_of_range_image(dtype):
    im = cp.array([10, 100], dtype=dtype)
    frequencies, bin_centers = exposure.histogram(im, nbins=90)
    # offset values by 0.5 for float...
    assert_array_equal(bin_centers, cp.arange(10, 100) + 0.5)


@pytest.mark.parametrize('dtype', [cp.float16, cp.float32, cp.float64])
def test_peak_float_out_of_range_dtype(dtype):
    im = cp.array([10, 100], dtype=dtype)
    nbins = 10
    frequencies, bin_centers = exposure.histogram(
        im, nbins=nbins, source_range='dtype'
    )
    assert bin_centers.dtype == dtype
    assert_almost_equal(cp.min(bin_centers).get(), -0.9, 3)
    assert_almost_equal(cp.max(bin_centers).get(), 0.9, 3)
    assert len(bin_centers) == 10


def test_normalize():
    im = cp.array([0, 255, 255], dtype=cp.uint8)
    frequencies, bin_centers = exposure.histogram(im, source_range='dtype',
                                                  normalize=False)

    expected = cp.zeros(256)
    expected[0] = 1
    expected[-1] = 2
    assert_array_equal(frequencies, expected)
    frequencies, bin_centers = exposure.histogram(im, source_range='dtype',
                                                  normalize=True)

    expected /= 3.0
    assert_array_equal(frequencies, expected)


# Test multichannel histograms
# ============================

@pytest.mark.parametrize('source_range', ['dtype', 'image'])
@pytest.mark.parametrize('dtype', [cp.uint8, cp.int16, cp.float64])
@pytest.mark.parametrize('channel_axis', [0, 1, -1])
def test_multichannel_hist_common_bins_uint8(dtype, source_range,
                                             channel_axis):
    """Check that all channels use the same binning."""
    # Construct multichannel image with uniform values within each channel,
    # but the full range of values across channels.
    shape = (5, 5)
    channel_size = shape[0] * shape[1]
    imin, imax = dtype_range[dtype]
    im = np.stack(
        (
            np.full(shape, imin, dtype=dtype),
            np.full(shape, imax, dtype=dtype),
        ),
        axis=channel_axis
    )
    im = cp.asarray(im)
    frequencies, bin_centers = exposure.histogram(
        im, source_range=source_range, channel_axis=channel_axis
    )
    if cp.issubdtype(dtype, cp.integer):
        assert_array_equal(bin_centers, np.arange(imin, imax + 1))
    assert frequencies[0][0] == channel_size
    assert frequencies[0][-1] == 0
    assert frequencies[1][0] == 0
    assert frequencies[1][-1] == channel_size


# Test histogram equalization
# ===========================

np.random.seed(0)

test_img_int = cp.array(data.camera())
# squeeze image intensities to lower image contrast
test_img = util.img_as_float(test_img_int)
test_img = exposure.rescale_intensity(test_img / 5.0 + 100)
test_img = cp.array(test_img)


def test_equalize_uint8_approx():
    """Check integer bins used for uint8 images."""
    img_eq0 = exposure.equalize_hist(test_img_int)
    img_eq1 = exposure.equalize_hist(test_img_int, nbins=3)
    cp.testing.assert_allclose(img_eq0, img_eq1)


def test_equalize_ubyte():
    img = util.img_as_ubyte(test_img)
    img_eq = exposure.equalize_hist(img)

    cdf, bin_edges = exposure.cumulative_distribution(img_eq)
    check_cdf_slope(cdf)


@pytest.mark.parametrize('dtype', [cp.float16, cp.float32, cp.float64])
def test_equalize_float(dtype):
    img = util.img_as_float(test_img).astype(dtype, copy=False)
    img_eq = exposure.equalize_hist(img)
    assert img_eq.dtype == _supported_float_type(dtype)

    cdf, bin_edges = exposure.cumulative_distribution(img_eq)
    check_cdf_slope(cdf)
    assert bin_edges.dtype == _supported_float_type(dtype)


def test_equalize_masked():
    img = util.img_as_float(test_img)
    mask = cp.zeros(test_img.shape)
    mask[100:400, 100:400] = 1
    img_mask_eq = exposure.equalize_hist(img, mask=mask)
    img_eq = exposure.equalize_hist(img)

    cdf, bin_edges = exposure.cumulative_distribution(img_mask_eq)
    check_cdf_slope(cdf)

    assert not (img_eq == img_mask_eq).all()


def check_cdf_slope(cdf):
    """Slope of cdf which should equal 1 for an equalized histogram."""
    norm_intensity = np.linspace(0, 1, len(cdf))
    slope, intercept = np.polyfit(norm_intensity, cp.asnumpy(cdf), 1)
    assert 0.9 < slope < 1.1


# Test intensity range
# ====================


@pytest.mark.parametrize("test_input,expected", [
    ('image', [0, 1]),
    ('dtype', [0, 255]),
    ((10, 20), [10, 20])
])
def test_intensity_range_uint8(test_input, expected):
    image = cp.array([0, 1], dtype=cp.uint8)
    out = intensity_range(image, range_values=test_input)
    assert_array_equal(out, cp.array(expected))


@pytest.mark.parametrize("test_input,expected", [
    ('image', [0.1, 0.2]),
    ('dtype', [-1, 1]),
    ((0.3, 0.4), [0.3, 0.4])
])
def test_intensity_range_float(test_input, expected):
    image = cp.array([0.1, 0.2], dtype=cp.float64)
    out = intensity_range(image, range_values=test_input)
    assert_array_equal(out, expected)


def test_intensity_range_clipped_float():
    image = cp.array([0.1, 0.2], dtype=cp.float64)
    out = intensity_range(image, range_values="dtype", clip_negative=True)
    assert_array_equal(out, (0, 1))


# Test rescale intensity
# ======================

uint10_max = 2**10 - 1
uint12_max = 2**12 - 1
uint14_max = 2**14 - 1
uint16_max = 2**16 - 1


def test_rescale_stretch():
    image = cp.array([51, 102, 153], dtype=cp.uint8)
    out = exposure.rescale_intensity(image)
    assert out.dtype == cp.uint8
    assert_array_almost_equal(out, [0, 127, 255])


def test_rescale_shrink():
    image = cp.array([51.0, 102.0, 153.0])
    out = exposure.rescale_intensity(image)
    assert_array_almost_equal(out, [0, 0.5, 1])


@pytest.mark.parametrize('dtype', [cp.float16, cp.float32, cp.float64])
def test_rescale_in_range(dtype):
    image = cp.array([51., 102., 153.], dtype=dtype)
    out = exposure.rescale_intensity(image, in_range=(0, 255))
    assert_array_almost_equal(out, [0.2, 0.4, 0.6], decimal=4)
    # with out_range='dtype', the output has the same dtype
    assert out.dtype == image.dtype


def test_rescale_in_range_clip():
    image = cp.array([51.0, 102.0, 153.0])
    out = exposure.rescale_intensity(image, in_range=(0, 102))
    assert_array_almost_equal(out, [0.5, 1, 1])


@pytest.mark.parametrize('dtype', [cp.int8, cp.int32, cp.float16, cp.float32,
                                   cp.float64])
def test_rescale_out_range(dtype):
    """Check that output range is correct.

    .. versionchanged:: 22.02.00
        float16 and float32 inputs now result in float32 output. Formerly they
        would give float64 outputs.
    """
    image = cp.array([-10, 0, 10], dtype=cp.int8)
    out = exposure.rescale_intensity(image, out_range=(0, 127))
    assert out.dtype == _supported_float_type(image.dtype)
    assert_array_almost_equal(out, [0, 63.5, 127])


def test_rescale_named_in_range():
    image = cp.array([0, uint10_max, uint10_max + 100], dtype=cp.uint16)
    out = exposure.rescale_intensity(image, in_range='uint10')
    assert_array_almost_equal(out, [0, uint16_max, uint16_max])


def test_rescale_named_out_range():
    image = cp.array([0, uint16_max], dtype=cp.uint16)
    out = exposure.rescale_intensity(image, out_range='uint10')
    assert_array_almost_equal(out, [0, uint10_max])


def test_rescale_uint12_limits():
    image = cp.array([0, uint16_max], dtype=cp.uint16)
    out = exposure.rescale_intensity(image, out_range='uint12')
    assert_array_almost_equal(out, [0, uint12_max])


def test_rescale_uint14_limits():
    image = cp.array([0, uint16_max], dtype=cp.uint16)
    out = exposure.rescale_intensity(image, out_range='uint14')
    assert_array_almost_equal(out, [0, uint14_max])


def test_rescale_all_zeros():
    image = cp.zeros((2, 2), dtype=cp.uint8)
    out = exposure.rescale_intensity(image)
    assert ~cp.isnan(out).all()
    assert_array_almost_equal(out, image)


def test_rescale_constant():
    image = cp.array([130, 130], dtype=cp.uint16)
    out = exposure.rescale_intensity(image, out_range=(0, 127))
    assert_array_almost_equal(out, [127, 127])


def test_rescale_same_values():
    image = cp.ones((2, 2))
    out = exposure.rescale_intensity(image)
    assert ~cp.isnan(out).all()
    assert_array_almost_equal(out, image)


@pytest.mark.parametrize(
    "in_range,out_range", [("image", "dtype"),
                           ("dtype", "image")]
)
def test_rescale_nan_warning(in_range, out_range):
    image = cp.arange(12, dtype=float).reshape(3, 4)
    image[1, 1] = cp.nan

    msg = (
        r"One or more intensity levels are NaN\."
        r" Rescaling will broadcast NaN to the full image\."
    )

    with expected_warnings([msg]):
        exposure.rescale_intensity(image, in_range, out_range)


@pytest.mark.parametrize(
    "out_range, out_dtype", [
        ('uint8', cp.uint8),
        ('uint10', cp.uint16),
        ('uint12', cp.uint16),
        ('uint16', cp.uint16),
        ('float', float),
    ]
)
def test_rescale_output_dtype(out_range, out_dtype):
    image = cp.array([-128, 0, 127], dtype=cp.int8)
    output_image = exposure.rescale_intensity(image, out_range=out_range)
    assert output_image.dtype == out_dtype


def test_rescale_no_overflow():
    image = cp.array([-128, 0, 127], dtype=cp.int8)
    output_image = exposure.rescale_intensity(image, out_range=cp.uint8)
    cp.testing.assert_array_equal(output_image, [0, 128, 255])
    assert output_image.dtype == cp.uint8


def test_rescale_float_output():
    image = cp.array([-128, 0, 127], dtype=cp.int8)
    output_image = exposure.rescale_intensity(image, out_range=(0, 255))
    cp.testing.assert_array_equal(output_image, [0, 128, 255])
    assert output_image.dtype == _supported_float_type(image.dtype)


def test_rescale_raises_on_incorrect_out_range():
    image = cp.array([-128, 0, 127], dtype=cp.int8)
    with pytest.raises(ValueError):
        _ = exposure.rescale_intensity(image, out_range="flat")


# Test adaptive histogram equalization
# ====================================


@pytest.mark.parametrize('dtype', [cp.float16, cp.float32, cp.float64])
def test_adapthist_grayscale(dtype):
    """Test a grayscale float image"""
    img = cp.array(data.astronaut())
    img = util.img_as_float(img).astype(dtype, copy=False)
    img = rgb2gray(img)
    img = cp.dstack((img, img, img))
    adapted = exposure.equalize_adapthist(img, kernel_size=(57, 51),
                                          clip_limit=0.01, nbins=128)
    assert img.shape == adapted.shape
    assert adapted.dtype == _supported_float_type(dtype)
    snr_decimal = 3 if dtype != cp.float16 else 2
    assert_almost_equal(float(peak_snr(img, adapted)), 100.140, snr_decimal)
    assert_almost_equal(float(norm_brightness_err(img, adapted)), 0.0529, 3)


def test_adapthist_color():
    """Test an RGB color uint16 image
    """
    img = util.img_as_uint(cp.array(data.astronaut()))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        hist, bin_centers = exposure.histogram(img)
        assert len(w) > 0
    adapted = exposure.equalize_adapthist(img, clip_limit=0.01)

    assert adapted.min() == 0
    assert adapted.max() == 1.0
    assert img.shape == adapted.shape
    full_scale = exposure.rescale_intensity(img)
    assert_almost_equal(float(peak_snr(full_scale, adapted)), 109.393, 1)
    assert_almost_equal(
        float(norm_brightness_err(full_scale, adapted)), 0.02, 2)
    return data, adapted


def test_adapthist_alpha():
    """Test an RGBA color image"""
    img = util.img_as_float(cp.array(data.astronaut()))
    alpha = cp.ones((img.shape[0], img.shape[1]), dtype=float)
    img = cp.dstack((img, alpha))
    adapted = exposure.equalize_adapthist(img)
    assert adapted.shape != img.shape
    img = img[:, :, :3]
    full_scale = exposure.rescale_intensity(img)
    assert img.shape == adapted.shape
    assert_almost_equal(float(peak_snr(full_scale, adapted)), 109.393, 2)
    assert_almost_equal(
        float(norm_brightness_err(full_scale, adapted)), 0.0248, 3
    )


def test_adapthist_grayscale_Nd():
    """
    Test for n-dimensional consistency with float images
    Note: Currently if img.ndim == 3, img.shape[2] > 4 must hold for the image
    not to be interpreted as a color image by @adapt_rgb
    """
    # take 2d image, subsample and stack it
    img = util.img_as_float(cp.array(data.astronaut()))
    img = rgb2gray(img)
    a = 15
    img2d = util.img_as_float(img[0:-1:a, 0:-1:a])
    img3d = cp.stack([img2d] * (img.shape[0] // a), axis=0)

    # apply CLAHE
    adapted2d = exposure.equalize_adapthist(img2d,
                                            kernel_size=5,
                                            clip_limit=0.05)
    adapted3d = exposure.equalize_adapthist(img3d,
                                            kernel_size=5,
                                            clip_limit=0.05)

    # check that dimensions of input and output match
    assert img2d.shape == adapted2d.shape
    assert img3d.shape == adapted3d.shape

    # check that the result from the stack of 2d images is similar
    # to the underlying 2d image
    assert cp.mean(cp.abs(adapted2d
                          - adapted3d[adapted3d.shape[0] // 2])) < 0.02


def test_adapthist_constant():
    """Test constant image, float and uint
    """
    img = cp.zeros((8, 8))
    img += 2
    img = img.astype(cp.uint16)
    adapted = exposure.equalize_adapthist(img, 3)
    assert cp.min(adapted) == cp.max(adapted)

    img = cp.zeros((8, 8))
    img += 0.1
    img = img.astype(cp.float64)
    adapted = exposure.equalize_adapthist(img, 3)
    assert cp.min(adapted) == cp.max(adapted)


def test_adapthist_borders():
    """Test border processing
    """
    img = rgb2gray(util.img_as_float(cp.array(data.astronaut())))

    # maximize difference between orig and processed img
    img /= 100.0
    img[img.shape[0] // 2, img.shape[1] // 2] = 1.0

    # check borders are processed for different kernel sizes
    border_index = -1
    for kernel_size in range(51, 71, 2):
        adapted = exposure.equalize_adapthist(img, kernel_size, clip_limit=0.5)
        # Check last columns are processed
        assert norm_brightness_err(adapted[:, border_index],
                                   img[:, border_index]) > 0.1
        # Check last rows are processed
        assert norm_brightness_err(adapted[border_index, :],
                                   img[border_index, :]) > 0.1


def test_adapthist_clip_limit():
    img_u = cp.array(data.moon())
    img_f = util.img_as_float(img_u)

    # uint8 input
    img_clahe0 = exposure.equalize_adapthist(img_u, clip_limit=0)
    img_clahe1 = exposure.equalize_adapthist(img_u, clip_limit=1)
    assert_array_equal(img_clahe0, img_clahe1)

    # float64 input
    img_clahe0 = exposure.equalize_adapthist(img_f, clip_limit=0)
    img_clahe1 = exposure.equalize_adapthist(img_f, clip_limit=1)
    assert_array_equal(img_clahe0, img_clahe1)


def peak_snr(img1, img2):
    """Peak signal to noise ratio of two images

    Parameters
    ----------
    img1 : array-like
    img2 : array-like

    Returns
    -------
    peak_snr : float
        Peak signal to noise ratio
    """
    if img1.ndim == 3:
        img1, img2 = rgb2gray(img1.copy()), rgb2gray(img2.copy())
    img1 = util.img_as_float(img1)
    img2 = util.img_as_float(img2)
    mse = 1.0 / img1.size * cp.square(img1 - img2).sum()
    _, max_ = dtype_range[img1.dtype.type]
    return 20 * cp.log(max_ / mse)


def norm_brightness_err(img1, img2):
    """Normalized Absolute Mean Brightness Error between two images

    Parameters
    ----------
    img1 : array-like
    img2 : array-like

    Returns
    -------
    norm_brightness_error : float
        Normalized absolute mean brightness error
    """
    if img1.ndim == 3:
        img1, img2 = rgb2gray(img1), rgb2gray(img2)
    ambe = cp.abs(img1.mean() - img2.mean())
    nbe = ambe / dtype_range[img1.dtype.type][1]
    return nbe


# Test Gamma Correction
# =====================


def test_adjust_gamma_1x1_shape():
    """Check that the shape is maintained"""
    img = cp.ones([1, 1])
    result = exposure.adjust_gamma(img, 1.5)
    assert img.shape == result.shape


def test_adjust_gamma_one():
    """Same image should be returned for gamma equal to one"""
    image = cp.random.uniform(0, 255, (8, 8))
    result = exposure.adjust_gamma(image, 1)
    assert_array_almost_equal(result, image)


@pytest.mark.parametrize('dtype', [cp.float16, cp.float32, cp.float64])
def test_adjust_gamma_zero(dtype):
    """White image should be returned for gamma equal to zero"""
    image = cp.random.uniform(0, 255, (8, 8)).astype(dtype, copy=False)
    result = exposure.adjust_gamma(image, 0)
    dtype = image.dtype.type
    assert_array_almost_equal(result, dtype_range[dtype][1])
    assert result.dtype == image.dtype


def test_adjust_gamma_less_one():
    """Verifying the output with expected results for gamma
    correction with gamma equal to half"""
    image = cp.arange(0, 255, 4, cp.uint8).reshape((8, 8))
    # fmt: off
    expected = cp.array([
        [  0,  31,  45,  55,  63,  71,  78,  84],  # noqa
        [ 90,  95, 100, 105, 110, 115, 119, 123],  # noqa
        [127, 131, 135, 139, 142, 146, 149, 153],
        [156, 159, 162, 165, 168, 171, 174, 177],
        [180, 183, 186, 188, 191, 194, 196, 199],
        [201, 204, 206, 209, 211, 214, 216, 218],
        [221, 223, 225, 228, 230, 232, 234, 236],
        [238, 241, 243, 245, 247, 249, 251, 253]], dtype=cp.uint8)
    # fmt: on

    result = exposure.adjust_gamma(image, 0.5)
    assert_array_equal(result, expected)


def test_adjust_gamma_greater_one():
    """Verifying the output with expected results for gamma
    correction with gamma equal to two"""
    image = cp.arange(0, 255, 4, cp.uint8).reshape((8, 8))
    # fmt: off
    expected = cp.array([
        [  0,   0,   0,   0,   1,   1,   2,   3],  # noqa
        [  4,   5,   6,   7,   9,  10,  12,  14],  # noqa
        [ 16,  18,  20,  22,  25,  27,  30,  33],  # noqa
        [ 36,  39,  42,  45,  49,  52,  56,  60],  # noqa
        [ 64,  68,  72,  76,  81,  85,  90,  95],  # noqa
        [100, 105, 110, 116, 121, 127, 132, 138],
        [144, 150, 156, 163, 169, 176, 182, 189],
        [196, 203, 211, 218, 225, 233, 241, 249]], dtype=cp.uint8)
    # fmt: on

    result = exposure.adjust_gamma(image, 2)
    assert_array_equal(result, expected)


def test_adjust_gamma_neggative():
    image = cp.arange(0, 255, 4, cp.uint8).reshape((8, 8))
    with pytest.raises(ValueError):
        exposure.adjust_gamma(image, -1)


def test_adjust_gamma_u8_overflow():
    img = 255 * cp.ones((2, 2), dtype=np.uint8)

    assert cp.all(exposure.adjust_gamma(img, gamma=1, gain=1.1) == 255)


# Test Logarithmic Correction
# ===========================

@pytest.mark.parametrize('dtype', [cp.float16, cp.float32, cp.float64])
def test_adjust_log_1x1_shape(dtype):
    """Check that the shape is maintained"""
    img = cp.ones([1, 1], dtype=dtype)
    result = exposure.adjust_log(img, 1)
    assert img.shape == result.shape
    assert result.dtype == dtype


def test_adjust_log():
    """Verifying the output with expected results for logarithmic
    correction with multiplier constant multiplier equal to unity"""
    image = cp.arange(0, 255, 4, cp.uint8).reshape((8, 8))
    # fmt: off
    expected = cp.array([
        [  0,   5,  11,  16,  22,  27,  33,  38],  # noqa
        [ 43,  48,  53,  58,  63,  68,  73,  77],  # noqa
        [ 82,  86,  91,  95, 100, 104, 109, 113],  # noqa
        [117, 121, 125, 129, 133, 137, 141, 145],
        [149, 153, 157, 160, 164, 168, 172, 175],
        [179, 182, 186, 189, 193, 196, 199, 203],
        [206, 209, 213, 216, 219, 222, 225, 228],
        [231, 234, 238, 241, 244, 246, 249, 252]], dtype=cp.uint8)
    # fmt: on

    result = exposure.adjust_log(image, 1)
    assert_array_equal(result, expected)


def test_adjust_inv_log():
    """Verifying the output with expected results for inverse logarithmic
    correction with multiplier constant multiplier equal to unity"""
    image = cp.arange(0, 255, 4, cp.uint8).reshape((8, 8))
    # fmt: off
    expected = cp.array([
        [  0,   2,   5,   8,  11,  14,  17,  20],  # noqa
        [ 23,  26,  29,  32,  35,  38,  41,  45],  # noqa
        [ 48,  51,  55,  58,  61,  65,  68,  72],  # noqa
        [ 76,  79,  83,  87,  90,  94,  98, 102],  # noqa
        [106, 110, 114, 118, 122, 126, 130, 134],
        [138, 143, 147, 151, 156, 160, 165, 170],
        [174, 179, 184, 188, 193, 198, 203, 208],
        [213, 218, 224, 229, 234, 239, 245, 250]], dtype=cp.uint8)
    # fmt: on

    result = exposure.adjust_log(image, 1, True)
    assert_array_equal(result, expected)


# Test Sigmoid Correction
# =======================

@pytest.mark.parametrize('dtype', [cp.float16, cp.float32, cp.float64])
def test_adjust_sigmoid_1x1_shape(dtype):
    """Check that the shape is maintained"""
    img = cp.ones([1, 1], dtype=dtype)
    result = exposure.adjust_sigmoid(img, 1, 5)
    assert img.shape == result.shape
    assert result.dtype == dtype


def test_adjust_sigmoid_cutoff_one():
    """Verifying the output with expected results for sigmoid correction
    with cutoff equal to one and gain of 5"""
    image = cp.arange(0, 255, 4, cp.uint8).reshape((8, 8))
    # fmt: off
    expected = cp.array([
        [  1,   1,   1,   2,   2,   2,   2,   2],  # noqa
        [  3,   3,   3,   4,   4,   4,   5,   5],  # noqa
        [  5,   6,   6,   7,   7,   8,   9,  10],  # noqa
        [ 10,  11,  12,  13,  14,  15,  16,  18],  # noqa
        [ 19,  20,  22,  24,  25,  27,  29,  32],  # noqa
        [ 34,  36,  39,  41,  44,  47,  50,  54],  # noqa
        [ 57,  61,  64,  68,  72,  76,  80,  85],  # noqa
        [ 89,  94,  99, 104, 108, 113, 118, 123]], dtype=cp.uint8)  # noqa
    # fmt: on

    result = exposure.adjust_sigmoid(image, 1, 5)
    assert_array_equal(result, expected)


def test_adjust_sigmoid_cutoff_zero():
    """Verifying the output with expected results for sigmoid correction
    with cutoff equal to zero and gain of 10"""
    image = cp.arange(0, 255, 4, cp.uint8).reshape((8, 8))
    # fmt: off
    expected = cp.array([
        [127, 137, 147, 156, 166, 175, 183, 191],
        [198, 205, 211, 216, 221, 225, 229, 232],
        [235, 238, 240, 242, 244, 245, 247, 248],
        [249, 250, 250, 251, 251, 252, 252, 253],
        [253, 253, 253, 253, 254, 254, 254, 254],
        [254, 254, 254, 254, 254, 254, 254, 254],
        [254, 254, 254, 254, 254, 254, 254, 254],
        [254, 254, 254, 254, 254, 254, 254, 254]], dtype=cp.uint8)
    # fmt: on

    result = exposure.adjust_sigmoid(image, 0, 10)
    assert_array_equal(result, expected)


def test_adjust_sigmoid_cutoff_half():
    """Verifying the output with expected results for sigmoid correction
    with cutoff equal to half and gain of 10"""
    image = cp.arange(0, 255, 4, cp.uint8).reshape((8, 8))
    # fmt: off
    expected = cp.array([
        [  1,   1,   2,   2,   3,   3,   4,   5],  # noqa
        [  5,   6,   7,   9,  10,  12,  14,  16],  # noqa
        [ 19,  22,  25,  29,  34,  39,  44,  50],  # noqa
        [ 57,  64,  72,  80,  89,  99, 108, 118],  # noqa
        [128, 138, 148, 158, 167, 176, 184, 192],
        [199, 205, 211, 217, 221, 226, 229, 233],
        [236, 238, 240, 242, 244, 246, 247, 248],
        [249, 250, 250, 251, 251, 252, 252, 253]], dtype=cp.uint8)
    # fmt: on
    result = exposure.adjust_sigmoid(image, 0.5, 10)
    assert_array_equal(result, expected)


def test_adjust_inv_sigmoid_cutoff_half():
    """Verifying the output with expected results for inverse sigmoid
    correction with cutoff equal to half and gain of 10"""
    image = cp.arange(0, 255, 4, cp.uint8).reshape((8, 8))
    # fmt: off
    expected = cp.array([
        [253, 253, 252, 252, 251, 251, 250, 249],
        [249, 248, 247, 245, 244, 242, 240, 238],
        [235, 232, 229, 225, 220, 215, 210, 204],
        [197, 190, 182, 174, 165, 155, 146, 136],
        [126, 116, 106,  96,  87,  78,  70,  62],  # noqa
        [ 55,  49,  43,  37,  33,  28,  25,  21],  # noqa
        [ 18,  16,  14,  12,  10,   8,   7,   6],  # noqa
        [  5,   4,   4,   3,   3,   2,   2,   1]], dtype=cp.uint8)  # noqa
    # fmt: on

    result = exposure.adjust_sigmoid(image, 0.5, 10, True)
    assert_array_equal(result, expected)


def test_is_low_contrast():
    image = cp.linspace(0, 0.04, 100)
    assert exposure.is_low_contrast(image)
    image[-1] = 1
    assert exposure.is_low_contrast(image)
    assert not exposure.is_low_contrast(image, upper_percentile=100)

    image = (image * 255).astype(cp.uint8)
    assert exposure.is_low_contrast(image)
    assert not exposure.is_low_contrast(image, upper_percentile=100)

    image = (image.astype(cp.uint16)) * 2 ** 8
    assert exposure.is_low_contrast(image)
    assert not exposure.is_low_contrast(image, upper_percentile=100)


def test_is_low_contrast_boolean():
    image = cp.zeros((8, 8), dtype=bool)
    assert exposure.is_low_contrast(image)

    image[:5] = 1
    assert not exposure.is_low_contrast(image)


# Test negative input
#####################

@pytest.mark.parametrize("exposure_func", [exposure.adjust_gamma,
                                           exposure.adjust_log,
                                           exposure.adjust_sigmoid])
def test_negative_input(exposure_func):
    image = cp.arange(-10, 245, 4).reshape((8, 8)).astype(cp.double)
    with pytest.raises(ValueError):
        exposure_func(image)


# Test Dask Compatibility
# =======================


# TODO: this Dask-based test case does not work (segfault!)
# @pytest.mark.xfail(True, reason="dask case not currently supported")
@pytest.mark.skip("dask case not currently supported")
def test_dask_histogram():
    pytest.importorskip('dask', reason="dask python library is not installed")
    import dask.array as da

    dask_array = da.from_array(cp.array([[0, 1], [1, 2]]), chunks=(1, 2))
    output_hist, output_bins = exposure.histogram(dask_array)
    expected_bins = [0, 1, 2]
    expected_hist = [1, 2, 1]
    assert cp.allclose(expected_bins, output_bins)
    assert cp.allclose(expected_hist, output_hist)
