import itertools

import cupy as cp
import numpy as np
import pytest
from cupy.testing import assert_array_almost_equal, assert_array_equal
from numpy.testing import assert_no_warnings

from cucim.skimage._shared.testing import expected_warnings
from cucim.skimage.color.colorlabel import hsv2rgb, label2rgb, rgb2hsv


def test_shape_mismatch():
    image = cp.ones((3, 3))
    label = cp.ones((2, 2))
    with pytest.raises(ValueError):
        label2rgb(image, label, bg_label=-1)


def test_wrong_kind():
    label = cp.ones((3, 3))
    # Must not raise an error.
    label2rgb(label, bg_label=-1)
    # kind='foo' is wrong.
    with pytest.raises(ValueError):
        label2rgb(label, kind="foo", bg_label=-1)


@pytest.mark.parametrize("channel_axis", [0, 1, -1])
def test_uint_image(channel_axis):
    img = cp.random.randint(0, 255, (10, 10), dtype=cp.uint8)
    labels = cp.zeros((10, 10), dtype=cp.int64)
    labels[1:3, 1:3] = 1
    labels[6:9, 6:9] = 2
    output = label2rgb(labels, image=img, bg_label=0,
                       channel_axis=channel_axis)
    # Make sure that the output is made of floats and in the correct range
    assert cp.issubdtype(output.dtype, cp.floating)
    assert output.max() <= 1

    # size 3 (RGB) along the specified channel_axis
    new_axis = channel_axis % output.ndim
    assert output.shape[new_axis] == 3


def test_rgb():
    image = cp.ones((1, 3))
    label = cp.arange(3).reshape(1, -1)
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    # Set alphas just in case the defaults change
    rgb = label2rgb(label, image=image, colors=colors, alpha=1,
                    image_alpha=1, bg_label=-1)
    assert_array_almost_equal(rgb, [colors])


def test_alpha():
    image = cp.random.uniform(size=(3, 3))
    label = cp.random.randint(0, 9, size=(3, 3))
    # If we set `alpha = 0`, then rgb should match image exactly.
    rgb = label2rgb(label, image=image, alpha=0, image_alpha=1,
                    bg_label=-1)
    assert_array_almost_equal(rgb[..., 0], image)
    assert_array_almost_equal(rgb[..., 1], image)
    assert_array_almost_equal(rgb[..., 2], image)


def test_no_input_image():
    label = cp.arange(3).reshape(1, -1)
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    rgb = label2rgb(label, colors=colors, bg_label=-1)
    assert_array_almost_equal(rgb, [colors])


def test_image_alpha():
    image = cp.random.uniform(size=(1, 3))
    label = cp.arange(3).reshape(1, -1)
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    # If we set `image_alpha = 0`, then rgb should match label colors exactly.
    rgb = label2rgb(label, image=image, colors=colors, alpha=1,
                    image_alpha=0, bg_label=-1)
    assert_array_almost_equal(rgb, [colors])


def test_color_names():
    image = cp.ones((1, 3))
    label = cp.arange(3).reshape(1, -1)
    cnames = ['red', 'lime', 'blue']
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    # Set alphas just in case the defaults change
    rgb = label2rgb(label, image=image, colors=cnames, alpha=1,
                    image_alpha=1, bg_label=-1)
    assert_array_almost_equal(rgb, [colors])


def test_bg_and_color_cycle():
    image = cp.zeros((1, 10))  # dummy image
    label = cp.arange(10).reshape(1, -1)
    colors = [(1, 0, 0), (0, 0, 1)]
    bg_color = (0, 0, 0)
    rgb = label2rgb(label, image=image, bg_label=0, bg_color=bg_color,
                    colors=colors, alpha=1)
    assert_array_almost_equal(rgb[0, 0], bg_color)
    for pixel, color in zip(rgb[0, 1:], itertools.cycle(colors)):
        assert_array_almost_equal(pixel, color)


def test_negative_labels():
    labels = cp.array([0, -1, -2, 0])
    rout = cp.array([(0., 0., 0.), (0., 0., 1.), (1., 0., 0.), (0., 0., 0.)])
    assert_array_almost_equal(
        rout, label2rgb(labels, bg_label=0, alpha=1, image_alpha=1))


def test_nonconsecutive():
    labels = cp.array([0, 2, 4, 0])
    colors = [(1, 0, 0), (0, 0, 1)]
    rout = cp.array([(1., 0., 0.), (0., 0., 1.), (1., 0., 0.), (1., 0., 0.)])
    assert_array_almost_equal(
        rout, label2rgb(labels, colors=colors, alpha=1,
                        image_alpha=1, bg_label=-1))


def test_label_consistency():
    """Assert that the same labels map to the same colors."""
    label_1 = cp.arange(5).reshape(1, -1)
    label_2 = cp.array([0, 1])
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1)]
    # Set alphas just in case the defaults change
    rgb_1 = label2rgb(label_1, colors=colors, bg_label=-1)
    rgb_2 = label2rgb(label_2, colors=colors, bg_label=-1)
    for label_id in label_2.ravel():
        assert_array_almost_equal(rgb_1[label_1 == label_id],
                                  rgb_2[label_2 == label_id])


def test_leave_labels_alone():
    labels = cp.array([-1, 0, 1])
    labels_saved = labels.copy()

    label2rgb(labels, bg_label=-1)
    label2rgb(labels, bg_label=1)
    assert_array_equal(labels, labels_saved)


# TODO: diagnose test error that occurs only with CUB enabled: CuPy bug?
@pytest.mark.parametrize("channel_axis", [0, 1, -1])
def test_avg(channel_axis):
    # label image
    # fmt: off
    label_field = cp.asarray([[1, 1, 1, 2],
                              [1, 2, 2, 2],
                              [3, 3, 4, 4]], dtype=np.uint8)

    # color image
    r = cp.asarray([[1., 1., 0., 0.],
                    [0., 0., 1., 1.],
                    [0., 0., 0., 0.]])
    g = cp.asarray([[0., 0., 0., 1.],
                    [1., 1., 1., 0.],
                    [0., 0., 0., 0.]])
    b = cp.asarray([[0., 0., 0., 1.],
                    [0., 1., 1., 1.],
                    [0., 0., 1., 1.]])
    image = cp.dstack((r, g, b))

    # reference label-colored image
    rout = cp.asarray([[0.5, 0.5, 0.5, 0.5],
                       [0.5, 0.5, 0.5, 0.5],
                       [0. , 0. , 0. , 0. ]])   # noqa
    gout = cp.asarray([[0.25, 0.25, 0.25, 0.75],
                       [0.25, 0.75, 0.75, 0.75],
                       [0. , 0. , 0. , 0.  ]])  # noqa
    bout = cp.asarray([[0. , 0. , 0. , 1. ],    # noqa
                       [0. , 1. , 1. , 1. ],    # noqa
                       [0.0, 0.0, 1.0, 1.0]])   # noqa
    expected_out = cp.dstack((rout, gout, bout))

    # test standard averaging
    _image = cp.moveaxis(image, source=-1, destination=channel_axis)
    out = label2rgb(label_field, _image, kind='avg', bg_label=-1,
                    channel_axis=channel_axis)
    out = cp.moveaxis(out, source=channel_axis, destination=-1)
    assert_array_equal(out, expected_out)

    # test averaging with custom background value
    out_bg = label2rgb(label_field, _image, bg_label=2, bg_color=(0, 0, 0),
                       kind='avg', channel_axis=channel_axis)
    out_bg = cp.moveaxis(out_bg, source=channel_axis, destination=-1)
    expected_out_bg = expected_out.copy()
    expected_out_bg[label_field == 2] = 0
    assert_array_equal(out_bg, expected_out_bg)

    # test default background color
    out_bg = label2rgb(label_field, _image, bg_label=2, kind='avg',
                       channel_axis=channel_axis)
    out_bg = cp.moveaxis(out_bg, source=channel_axis, destination=-1)
    assert_array_equal(out_bg, expected_out_bg)


def test_negative_intensity():
    labels = cp.arange(100).reshape(10, 10)
    image = cp.full((10, 10), -1, dtype="float64")
    with pytest.warns(UserWarning):
        label2rgb(labels, image, bg_label=-1)


def test_bg_color_rgb_string():
    img = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
    labels = np.zeros((10, 10), dtype=np.int64)
    labels[1:3, 1:3] = 1
    labels[6:9, 6:9] = 2
    img = cp.asarray(img)
    labels = cp.asarray(labels)
    output = label2rgb(labels, image=img, alpha=0.9, bg_label=0, bg_color='red')
    assert output[0, 0, 0] > 0.9  # red channel


def test_avg_with_2d_image():
    img = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
    labels = np.zeros((10, 10), dtype=np.int64)
    labels[1:3, 1:3] = 1
    labels[6:9, 6:9] = 2
    img = cp.asarray(img)
    labels = cp.asarray(labels)
    assert_no_warnings(label2rgb, labels, image=img, bg_label=0, kind='avg')


@pytest.mark.parametrize('image_type', ['rgb', 'gray', None])
def test_label2rgb_nd(image_type):
    # validate 1D and 3D cases by testing their output relative to the 2D case
    shape = (10, 10)
    if image_type == 'rgb':
        img = cp.random.randint(0, 255, shape + (3,), dtype=np.uint8)
    elif image_type == 'gray':
        img = cp.random.randint(0, 255, shape, dtype=np.uint8)
    else:
        img = None

    # add a couple of rectangular labels
    labels = cp.zeros(shape, dtype=np.int64)
    # Note: Have to choose labels here so that the 1D slice below also contains
    #       both label values. Otherwise the labeled colors will not match.
    labels[2:-2, 1:3] = 1
    labels[3:-3, 6:9] = 2

    # label in the 2D case (correct 2D output is tested in other funcitons)
    labeled_2d = label2rgb(labels, image=img, bg_label=0)

    # labeling a single line gives an equivalent result
    image_1d = img[5] if image_type is not None else None
    labeled_1d = label2rgb(labels[5], image=image_1d, bg_label=0)
    expected = labeled_2d[5]
    assert_array_equal(labeled_1d, expected)

    # Labeling a 3D stack of duplicates gives the same result in each plane
    image_3d = cp.stack((img, ) * 4) if image_type is not None else None
    labels_3d = cp.stack((labels,) * 4)
    labeled_3d = label2rgb(labels_3d, image=image_3d, bg_label=0)
    for labeled_plane in labeled_3d:
        assert_array_equal(labeled_plane, labeled_2d)


def test_label2rgb_shape_errors():
    img = cp.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    labels = cp.zeros((10, 10), dtype=np.int64)
    labels[2:5, 2:5] = 1

    # mismatched 2D shape
    with pytest.raises(ValueError):
        label2rgb(labels, img[1:])

    # too many axes in img
    with pytest.raises(ValueError):
        label2rgb(labels, img[..., np.newaxis])

    # too many channels along the last axis
    with pytest.raises(ValueError):
        label2rgb(labels, np.concatenate((img, img), axis=-1))


def test_overlay_full_saturation():
    rgb_img = cp.random.uniform(size=(10, 10, 3))
    labels = cp.ones((10, 10), dtype=np.int64)
    labels[5:, 5:] = 2
    labels[:3, :3] = 0
    alpha = 0.3
    rgb = label2rgb(labels, image=rgb_img, alpha=alpha,
                    bg_label=0, saturation=1)
    # check that rgb part of input image is preserved, where labels=0
    assert_array_almost_equal(rgb_img[:3, :3] * (1 - alpha), rgb[:3, :3])


def test_overlay_custom_saturation():
    rgb_img = cp.random.uniform(size=(10, 10, 3))
    labels = cp.ones((10, 10), dtype=np.int64)
    labels[5:, 5:] = 2
    labels[:3, :3] = 0
    alpha = 0.3
    saturation = 0.3
    rgb = label2rgb(labels, image=rgb_img, alpha=alpha,
                    bg_label=0, saturation=saturation)

    hsv = rgb2hsv(rgb_img)
    hsv[..., 1] *= saturation
    saturated_img = hsv2rgb(hsv)

    # check that rgb part of input image is saturated, where labels=0
    assert_array_almost_equal(saturated_img[:3, :3] * (1 - alpha), rgb[:3, :3])


def test_saturation_warning():
    rgb_img = cp.random.uniform(size=(10, 10, 3))
    labels = cp.ones((10, 10), dtype=np.int64)
    with expected_warnings(["saturation must be in range"]):
        label2rgb(labels, image=rgb_img,
                  bg_label=0, saturation=2)
    with expected_warnings(["saturation must be in range"]):
        label2rgb(labels, image=rgb_img,
                  bg_label=0, saturation=-1)
