import math

import cupy as cp
import pytest
from skimage.draw import disk
from skimage.draw.draw3d import ellipsoid

from cucim.skimage import feature
from cucim.skimage._vendored import pad
from cucim.skimage.feature import blob_dog, blob_doh, blob_log


@pytest.mark.parametrize(
    'dtype', [cp.uint8, cp.float16, cp.float32, cp.float64]
)
@pytest.mark.parametrize('threshold_type', ['absolute', 'relative'])
def test_blob_dog(dtype, threshold_type):
    img = cp.ones((512, 512), dtype=dtype)

    xs, ys = cp.asarray(disk((400, 130), 5))
    img[xs, ys] = 255

    xs, ys = cp.asarray(disk((100, 300), 25))
    img[xs, ys] = 255

    xs, ys = cp.asarray(disk((200, 350), 45))
    img[xs, ys] = 255

    if threshold_type == 'absolute':
        threshold = 2.0
        if img.dtype.kind != 'f':
            # account for internal scaling to [0, 1] by img_as_float
            threshold /= img.ptp()
        threshold_rel = None
    elif threshold_type == 'relative':
        threshold = None
        threshold_rel = 0.5

    blobs = blob_dog(
        img,
        min_sigma=4,
        max_sigma=50,
        threshold=threshold,
        threshold_rel=threshold_rel,
    )

    def radius(x):
        return math.sqrt(2) * x[2]

    s = sorted(blobs, key=radius)
    thresh = 5
    ratio_thresh = 0.25

    b = s[0]
    assert abs(b[0] - 400) <= thresh
    assert abs(b[1] - 130) <= thresh
    assert abs(radius(b) - 5) <= ratio_thresh * 5

    b = s[1]
    assert abs(b[0] - 100) <= thresh
    assert abs(b[1] - 300) <= thresh
    assert abs(radius(b) - 25) <= ratio_thresh * 25

    b = s[2]
    assert abs(b[0] - 200) <= thresh
    assert abs(b[1] - 350) <= thresh
    assert abs(radius(b) - 45) <= ratio_thresh * 45

    # Testing no peaks
    img_empty = cp.zeros((100, 100), dtype=dtype)
    assert blob_dog(img_empty).size == 0


@pytest.mark.parametrize(
    'dtype', [cp.uint8, cp.float16, cp.float32, cp.float64]
)
@pytest.mark.parametrize('threshold_type', ['absolute', 'relative'])
def test_blob_dog_3d(dtype, threshold_type):
    # Testing 3D
    r = 10
    pad_width = 10
    im3 = cp.asarray(ellipsoid(r, r, r))
    im3 = pad(im3, pad_width, mode='constant')

    if threshold_type == 'absolute':
        threshold = 0.001
        threshold_rel = 0
    elif threshold_type == 'relative':
        threshold = 0
        threshold_rel = 0.5

    blobs = blob_dog(
        im3,
        min_sigma=3,
        max_sigma=10,
        sigma_ratio=1.2,
        threshold=threshold,
        threshold_rel=threshold_rel,
    )
    b = blobs[0]

    assert b.shape == (4,)
    assert b[0] == r + pad_width + 1
    assert b[1] == r + pad_width + 1
    assert b[2] == r + pad_width + 1
    assert abs(math.sqrt(3) * b[3] - r) < 1.1


@pytest.mark.parametrize(
    'dtype', [cp.uint8, cp.float16, cp.float32, cp.float64]
)
@pytest.mark.parametrize('threshold_type', ['absolute', 'relative'])
def test_blob_dog_3d_anisotropic(dtype, threshold_type):
    # Testing 3D anisotropic
    r = 10
    pad_width = 10
    im3 = cp.asarray(ellipsoid(r / 2, r, r))
    im3 = pad(im3, pad_width, mode='constant')

    if threshold_type == 'absolute':
        threshold = 0.001
        threshold_rel = None
    elif threshold_type == 'relative':
        threshold = None
        threshold_rel = 0.5

    blobs = blob_dog(
        im3.astype(dtype, copy=False),
        min_sigma=[1.5, 3, 3],
        max_sigma=[5, 10, 10],
        sigma_ratio=1.2,
        threshold=threshold,
        threshold_rel=threshold_rel,
    )
    b = blobs[0]

    assert b.shape == (6,)
    assert b[0] == r / 2 + pad_width + 1
    assert b[1] == r + pad_width + 1
    assert b[2] == r + pad_width + 1
    assert abs(math.sqrt(3) * b[3] - r / 2) < 1.1
    assert abs(math.sqrt(3) * b[4] - r) < 1.1
    assert abs(math.sqrt(3) * b[5] - r) < 1.1


def test_blob_dog_excl_border():
    # Testing exclude border

    # image where blob is 5 px from borders, radius 5
    img = cp.ones((512, 512))
    xs, ys = cp.asarray(disk((5, 5), 5))
    img[xs, ys] = 255
    blobs = blob_dog(
        img,
        min_sigma=1.5,
        max_sigma=5,
        sigma_ratio=1.2,
    )
    assert blobs.shape[0] == 1
    b = blobs[0]
    assert b[0] == b[1] == 5, "blob should be 5 px from x and y borders"

    blobs = blob_dog(
        img,
        min_sigma=1.5,
        max_sigma=5,
        sigma_ratio=1.2,
        exclude_border=6,
    )
    msg = "zero blobs should be detected, as only blob is 5 px from border"
    assert blobs.shape[0] == 0, msg


@pytest.mark.parametrize('anisotropic', [False, True])
@pytest.mark.parametrize('ndim', [1, 2, 3, 4])
@pytest.mark.parametrize('function_name', ['blob_dog', 'blob_log'])
def test_nd_blob_no_peaks_shape(function_name, ndim, anisotropic):
    # uniform image so no blobs will be found
    z = cp.zeros((16,) * ndim, dtype=cp.float32)
    if anisotropic:
        max_sigma = 8 + cp.arange(ndim)
    else:
        max_sigma = 8
    blob_func = getattr(feature, function_name)
    blobs = blob_func(z, max_sigma=max_sigma)
    # z.ndim +  (z.ndim sigmas if anisotropic, only one sigma otherwise)
    expected_shape = 2 * z.ndim if anisotropic else z.ndim + 1
    assert blobs.shape == (0, expected_shape)


@pytest.mark.parametrize(
    'dtype', [cp.uint8, cp.float16, cp.float32, cp.float64]
)
@pytest.mark.parametrize('threshold_type', ['absolute', 'relative'])
def test_blob_log(dtype, threshold_type):
    img = cp.ones((256, 256), dtype=dtype)

    xs, ys = cp.asarray(disk((200, 65), 5))
    img[xs, ys] = 255

    xs, ys = cp.asarray(disk((80, 25), 15))
    img[xs, ys] = 255

    xs, ys = cp.asarray(disk((50, 150), 25))
    img[xs, ys] = 255

    xs, ys = cp.asarray(disk((100, 175), 30))
    img[xs, ys] = 255

    if threshold_type == 'absolute':
        threshold = 1
        if img.dtype.kind != 'f':
            # account for internal scaling to [0, 1] by img_as_float
            threshold /= img.ptp()
        threshold_rel = None
    elif threshold_type == 'relative':
        threshold = None
        threshold_rel = 0.5

    blobs = blob_log(img, min_sigma=5, max_sigma=20, threshold=threshold,
                     threshold_rel=threshold_rel)

    def radius(x):
        return math.sqrt(2) * x[2]

    s = sorted(blobs, key=radius)
    thresh = 3

    b = s[0]
    assert abs(b[0] - 200) <= thresh
    assert abs(b[1] - 65) <= thresh
    assert abs(radius(b) - 5) <= thresh

    b = s[1]
    assert abs(b[0] - 80) <= thresh
    assert abs(b[1] - 25) <= thresh
    assert abs(radius(b) - 15) <= thresh

    b = s[2]
    assert abs(b[0] - 50) <= thresh
    assert abs(b[1] - 150) <= thresh
    assert abs(radius(b) - 25) <= thresh

    b = s[3]
    assert abs(b[0] - 100) <= thresh
    assert abs(b[1] - 175) <= thresh
    assert abs(radius(b) - 30) <= thresh

    # Testing log scale
    blobs = blob_log(
        img,
        min_sigma=5,
        max_sigma=20,
        threshold=threshold,
        threshold_rel=threshold_rel,
        log_scale=True)

    b = s[0]
    assert abs(b[0] - 200) <= thresh
    assert abs(b[1] - 65) <= thresh
    assert abs(radius(b) - 5) <= thresh

    b = s[1]
    assert abs(b[0] - 80) <= thresh
    assert abs(b[1] - 25) <= thresh
    assert abs(radius(b) - 15) <= thresh

    b = s[2]
    assert abs(b[0] - 50) <= thresh
    assert abs(b[1] - 150) <= thresh
    assert abs(radius(b) - 25) <= thresh

    b = s[3]
    assert abs(b[0] - 100) <= thresh
    assert abs(b[1] - 175) <= thresh
    assert abs(radius(b) - 30) <= thresh

    # Testing no peaks
    img_empty = cp.zeros((100, 100))
    assert blob_log(img_empty).size == 0


def test_blob_log_no_warnings():
    img = cp.ones((11, 11))

    xs, ys = cp.asarray(disk((5, 5), 2))
    img[xs, ys] = 255

    xs, ys = cp.asarray(disk((7, 6), 2))
    img[xs, ys] = 255

    blob_log(img, max_sigma=20, num_sigma=10, threshold=.1)


def test_blob_log_3d():
    # Testing 3D
    r = 6
    pad_width = 10
    im3 = cp.asarray(ellipsoid(r, r, r))
    im3 = pad(im3, pad_width, mode='constant')

    blobs = blob_log(im3, min_sigma=3, max_sigma=10)
    b = blobs[0]

    assert b.shape == (4,)
    assert b[0] == r + pad_width + 1
    assert b[1] == r + pad_width + 1
    assert b[2] == r + pad_width + 1
    assert abs(math.sqrt(3) * b[3] - r) < 1


def test_blob_log_3d_anisotropic():
    # Testing 3D anisotropic
    r = 6
    pad_width = 10
    im3 = cp.asarray(ellipsoid(r / 2, r, r))
    im3 = pad(im3, pad_width, mode='constant')

    blobs = blob_log(
        im3,
        min_sigma=[1, 2, 2],
        max_sigma=[5, 10, 10],
    )

    b = blobs[0]
    assert b.shape == (6,)
    assert b[0] == r / 2 + pad_width + 1
    assert b[1] == r + pad_width + 1
    assert b[2] == r + pad_width + 1
    assert abs(math.sqrt(3) * b[3] - r / 2) < 1
    assert abs(math.sqrt(3) * b[4] - r) < 1
    assert abs(math.sqrt(3) * b[5] - r) < 1


def test_blob_log_exclude_border():
    # image where blob is 5 px from borders, radius 5
    img = cp.ones((512, 512))
    xs, ys = cp.asarray(disk((5, 5), 5))
    img[xs, ys] = 255

    blobs = blob_log(
        img,
        min_sigma=1.5,
        max_sigma=5,
    )
    assert blobs.shape[0] == 1
    b = blobs[0]
    assert b[0] == b[1] == 5, "blob should be 5 px from x and y borders"

    blobs = blob_log(
        img,
        min_sigma=1.5,
        max_sigma=5,
        exclude_border=6,
    )
    msg = "zero blobs should be detected, as only blob is 5 px from border"
    assert blobs.shape[0] == 0, msg


@pytest.mark.parametrize("dtype", [cp.uint8, cp.float16, cp.float32])
@pytest.mark.parametrize('threshold_type', ['absolute', 'relative'])
def test_blob_doh(dtype, threshold_type):
    img = cp.ones((512, 512), dtype=dtype)

    xs, ys = cp.asarray(disk((400, 130), 20))
    img[xs, ys] = 255

    xs, ys = cp.asarray(disk((460, 50), 30))
    img[xs, ys] = 255

    xs, ys = cp.asarray(disk((100, 300), 40))
    img[xs, ys] = 255

    xs, ys = cp.asarray(disk((200, 350), 50))
    img[xs, ys] = 255

    if threshold_type == 'absolute':
        # Note: have to either scale up threshold or rescale the image to the
        #       range [0, 1] internally.
        threshold = 0.05
        if img.dtype.kind == 'f':
            # account for lack of internal scaling to [0, 1] by img_as_float
            ptp = img.ptp()
            threshold *= ptp ** 2
        threshold_rel = None
    elif threshold_type == 'relative':
        threshold = None
        threshold_rel = 0.5

    blobs = blob_doh(
        img,
        min_sigma=1,
        max_sigma=60,
        num_sigma=10,
        threshold=threshold,
        threshold_rel=threshold_rel)

    def radius(x):
        return x[2]

    s = sorted(blobs, key=radius)
    thresh = 4

    b = s[0]
    assert abs(b[0] - 400) <= thresh
    assert abs(b[1] - 130) <= thresh
    assert abs(radius(b) - 20) <= thresh

    b = s[1]
    assert abs(b[0] - 460) <= thresh
    assert abs(b[1] - 50) <= thresh
    assert abs(radius(b) - 30) <= thresh

    b = s[2]
    assert abs(b[0] - 100) <= thresh
    assert abs(b[1] - 300) <= thresh
    assert abs(radius(b) - 40) <= thresh

    b = s[3]
    assert abs(b[0] - 200) <= thresh
    assert abs(b[1] - 350) <= thresh
    assert abs(radius(b) - 50) <= thresh


def test_blob_doh_log_scale():
    img = cp.ones((512, 512), dtype=cp.uint8)

    xs, ys = cp.asarray(disk((400, 130), 20))
    img[xs, ys] = 255

    xs, ys = cp.asarray(disk((460, 50), 30))
    img[xs, ys] = 255

    xs, ys = cp.asarray(disk((100, 300), 40))
    img[xs, ys] = 255

    xs, ys = cp.asarray(disk((200, 350), 50))
    img[xs, ys] = 255

    blobs = blob_doh(
        img,
        min_sigma=1,
        max_sigma=60,
        num_sigma=10,
        log_scale=True,
        threshold=.05)

    def radius(x):
        return x[2]

    s = sorted(blobs, key=radius)
    thresh = 10

    b = s[0]
    assert abs(b[0] - 400) <= thresh
    assert abs(b[1] - 130) <= thresh
    assert abs(radius(b) - 20) <= thresh

    b = s[2]
    assert abs(b[0] - 460) <= thresh
    assert abs(b[1] - 50) <= thresh
    assert abs(radius(b) - 30) <= thresh

    b = s[1]
    assert abs(b[0] - 100) <= thresh
    assert abs(b[1] - 300) <= thresh
    assert abs(radius(b) - 40) <= thresh

    b = s[3]
    assert abs(b[0] - 200) <= thresh
    assert abs(b[1] - 350) <= thresh
    assert abs(radius(b) - 50) <= thresh


def test_blob_doh_no_peaks():
    # Testing no peaks
    img_empty = cp.zeros((100, 100))
    assert blob_doh(img_empty).size == 0


def test_blob_doh_overlap():
    img = cp.ones((256, 256), dtype=cp.uint8)

    xs, ys = cp.asarray(disk((100, 100), 20))
    img[xs, ys] = 255

    xs, ys = cp.asarray(disk((120, 100), 30))
    img[xs, ys] = 255

    blobs = blob_doh(
        img,
        min_sigma=1,
        max_sigma=60,
        num_sigma=10,
        threshold=.05
    )

    assert len(blobs) == 1


def test_blob_log_overlap_3d():
    r1, r2 = 7, 6
    pad1, pad2 = 11, 12
    blob1 = cp.asarray(ellipsoid(r1, r1, r1))
    blob1 = pad(blob1, pad1, mode='constant')
    blob2 = cp.asarray(ellipsoid(r2, r2, r2))
    blob2 = pad(blob2, [(pad2, pad2), (pad2 - 9, pad2 + 9), (pad2, pad2)],
                mode='constant')
    im3 = cp.logical_or(blob1, blob2)

    blobs = blob_log(im3, min_sigma=2, max_sigma=10, overlap=0.1)
    assert len(blobs) == 1


def test_blob_log_anisotropic():
    image = cp.zeros((50, 50))
    image[20, 10:20] = 1
    isotropic_blobs = blob_log(image, min_sigma=0.5, max_sigma=2, num_sigma=3)
    assert len(isotropic_blobs) > 1  # many small blobs found in line
    ani_blobs = blob_log(image, min_sigma=[0.5, 5], max_sigma=[2, 20],
                         num_sigma=3)  # 10x anisotropy, line is 1x10
    assert len(ani_blobs) == 1  # single anisotropic blob found


def test_no_blob():
    im = cp.zeros((10, 10))
    blobs = blob_log(im, min_sigma=2, max_sigma=5, num_sigma=4)
    assert len(blobs) == 0
