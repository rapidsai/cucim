import math
import warnings

import cupy as cp
import pytest
from cupy.testing import (
    assert_allclose,
    assert_array_equal,
)
from scipy.ndimage import find_objects as cpu_find_objects
from skimage import measure as measure_cpu

from cucim.skimage import data, measure
from cucim.skimage._vendored import ndimage as ndi
from cucim.skimage.measure._regionprops_gpu import (
    equivalent_diameter_area,
    regionprops_area,
    regionprops_area_bbox,
    regionprops_bbox_coords,
    regionprops_coords,
    regionprops_extent,
    regionprops_image,
    regionprops_num_pixels,
)


def get_labels_nd(
    shape,
    blob_size_fraction=0.05,
    volume_fraction=0.25,
    rng=5,
    insert_holes=False,
    dilate_blobs=False,
):
    ndim = len(shape)
    blobs_kwargs = dict(
        blob_size_fraction=blob_size_fraction,
        volume_fraction=volume_fraction,
        rng=rng,
    )
    blobs = data.binary_blobs(max(shape), n_dim=ndim, **blobs_kwargs)
    # crop to rectangular
    blobs = blobs[tuple(slice(s) for s in shape)]

    if dilate_blobs:
        blobs = ndi.binary_dilation(blobs, 3)

    if insert_holes:
        blobs2_kwargs = dict(
            blob_size_fraction=blob_size_fraction / 5,
            volume_fraction=0.1,
            rng=rng,
        )
        # create smaller blobs and invert them to create a holes mask to apply
        # to the original blobs
        temp = data.binary_blobs(max(shape), n_dim=ndim, **blobs2_kwargs)
        temp = temp[tuple(slice(s) for s in shape)]
        mask = cp.logical_and(blobs > 0, temp == 0)
        blobs = blobs * mask

    # binary blobs only creates square outputs
    labels = measure.label(blobs)
    # print(f"# labels generated = {labels.max()}")
    return labels


def get_intensity_image(shape, dtype=cp.float32, seed=5, num_channels=None):
    npixels = math.prod(shape)
    rng = cp.random.default_rng(seed)
    dtype = cp.dtype(dtype)
    if dtype.kind == "f":
        img = cp.arange(npixels, dtype=cp.float32) - npixels // 2
        img = img.reshape(shape)
        if dtype == cp.float16:
            temp = 100 * rng.standard_normal(img.shape, dtype=cp.float32)
            img += temp.astype(cp.float16)
        else:
            img += 100 * rng.standard_normal(img.shape, dtype=dtype)
    else:
        iinfo = cp.iinfo(dtype)
        imax = min(16384, iinfo.max)
        imin = max(0, iinfo.min)
        img = rng.integers(imin, imax, shape)

    if num_channels and num_channels > 1:
        # generate slightly shifted versions for the additional channels
        img = cp.stack((img,) * num_channels, axis=-1)
        for c in range(1, num_channels):
            img[..., c] = cp.roll(img[..., c], shift=c, axis=0)
    return img


@pytest.mark.parametrize("precompute_max", [False, True])
@pytest.mark.parametrize("ndim", [2, 3])
def test_num_pixels(precompute_max, ndim):
    shape = (256, 512) if ndim == 2 else (15, 63, 37)
    labels = get_labels_nd(shape)

    max_label = int(cp.max(labels)) if precompute_max else None
    num_pixels = regionprops_num_pixels(labels, max_label=max_label)
    expected = measure_cpu.regionprops_table(
        cp.asnumpy(labels), properties=["num_pixels"]
    )
    assert_allclose(num_pixels, expected["num_pixels"])


@pytest.mark.parametrize("precompute_max", [False, True])
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("area_dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("spacing", [None, (0.5, 0.35, 0.75)])
def test_area(precompute_max, ndim, area_dtype, spacing):
    shape = (256, 512) if ndim == 2 else (45, 63, 37)
    labels = get_labels_nd(shape)
    # discard any extra dimensions from spacing
    if spacing is not None:
        spacing = spacing[:ndim]

    max_label = int(cp.max(labels)) if precompute_max else None
    area = regionprops_area(
        labels, spacing=spacing, max_label=max_label, dtype=area_dtype
    )
    expected = measure_cpu.regionprops_table(
        cp.asnumpy(labels),
        spacing=spacing,
        properties=["area", "equivalent_diameter_area"],
    )
    assert_allclose(area, expected["area"])

    ed = equivalent_diameter_area(area, ndim)
    assert_allclose(
        ed, expected["equivalent_diameter_area"], rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("area_dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("spacing", [None, (0.5, 0.35, 0.75)])
def test_extent(ndim, area_dtype, spacing):
    shape = (512, 512) if ndim == 2 else (64, 64, 64)
    labels = get_labels_nd(shape)
    # discard any extra dimensions from spacing
    if spacing is not None:
        spacing = spacing[:ndim]

    # compute area
    max_label = int(cp.max(labels))
    area = regionprops_area(
        labels, spacing=spacing, max_label=max_label, dtype=area_dtype
    )

    # compute bounding-box area
    bbox, slices = regionprops_bbox_coords(
        labels,
        max_label=max_label,
        return_slices=True,
    )
    area_bbox = regionprops_area_bbox(
        bbox, area_dtype=cp.float32, spacing=spacing
    )

    # compute extents from these
    extent = regionprops_extent(area=area, area_bbox=area_bbox)

    # compare to expected result
    expected = measure_cpu.regionprops_table(
        cp.asnumpy(labels), spacing=spacing, properties=["extent"]
    )
    assert_allclose(extent, expected["extent"], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("precompute_max", [False, True])
@pytest.mark.parametrize("dtype", [cp.uint32, cp.int64])
@pytest.mark.parametrize("return_slices", [False, True])
@pytest.mark.parametrize("ndim", [2, 3])
def test_bbox_coords_and_area(precompute_max, ndim, dtype, return_slices):
    shape = (1024, 512) if ndim == 2 else (80, 64, 48)
    labels = get_labels_nd(shape)

    max_label = int(cp.max(labels)) if precompute_max else None
    bbox, slices = regionprops_bbox_coords(
        labels,
        max_label=max_label,
        return_slices=return_slices,
    )
    assert bbox.dtype == cp.uint32
    if not return_slices:
        slices is None
    else:
        expected_slices = cpu_find_objects(cp.asnumpy(labels))
        assert slices == expected_slices

    spacing = (0.35, 0.75, 0.5)[:ndim]
    expected_bbox = measure_cpu.regionprops_table(
        cp.asnumpy(labels), spacing=spacing, properties=["bbox", "area_bbox"]
    )
    if ndim == 2:
        # TODO make ordering of bbox consistent with regionprops bbox?
        assert_allclose(bbox[:, 0], expected_bbox["bbox-0"])
        assert_allclose(bbox[:, 1], expected_bbox["bbox-1"])
        assert_allclose(bbox[:, 2], expected_bbox["bbox-2"])
        assert_allclose(bbox[:, 3], expected_bbox["bbox-3"])
    elif ndim == 3:
        assert_allclose(bbox[:, 0], expected_bbox["bbox-0"])
        assert_allclose(bbox[:, 1], expected_bbox["bbox-1"])
        assert_allclose(bbox[:, 2], expected_bbox["bbox-2"])
        assert_allclose(bbox[:, 3], expected_bbox["bbox-3"])
        assert_allclose(bbox[:, 4], expected_bbox["bbox-4"])
        assert_allclose(bbox[:, 5], expected_bbox["bbox-5"])

    # compute area_bbox from bbox array
    area_bbox = regionprops_area_bbox(
        bbox, area_dtype=cp.float32, spacing=spacing
    )
    assert_allclose(area_bbox, expected_bbox["area_bbox"], rtol=1e-5)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("num_channels", [1, 3])
@pytest.mark.parametrize(
    "blob_kwargs", [{}, dict(blob_size_fraction=0.12, volume_fraction=0.3)]
)
def test_image(ndim, num_channels, blob_kwargs):
    shape = (256, 512) if ndim == 2 else (64, 64, 80)

    labels = get_labels_nd(shape, **blob_kwargs)
    intensity_image = get_intensity_image(
        shape, dtype=cp.uint16, num_channels=num_channels
    )
    max_label = int(cp.max(labels))
    images, intensity_images, images_convex = regionprops_image(
        labels,
        max_label=max_label,
        intensity_image=intensity_image,
        compute_convex=True,
    )
    assert len(images) == max_label
    assert len(intensity_images) == max_label
    assert len(images_convex) == max_label

    # suppress any QHull warnings coming from the scikit-image implementation
    warnings.filterwarnings(
        "ignore",
        message="Failed to get convex hull image",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="divide by zero",
        category=RuntimeWarning,
    )
    expected = measure_cpu.regionprops_table(
        cp.asnumpy(labels),
        intensity_image=cp.asnumpy(intensity_image),
        properties=["image", "image_intensity", "image_convex"],
    )
    warnings.resetwarnings()

    for n in range(max_label):
        assert_array_equal(images[n], expected["image"][n])
        assert_array_equal(intensity_images[n], expected["image_intensity"][n])
        # Note if 3d blobs are size 1 on one of the axes, it can cause QHull to
        # fail and return a zeros convex image for that label. This has been
        # resolved for cuCIM, but not yet for scikit-image.
        # The test case with blob_kwargs != {} was chosen as a known good
        # setting where such an edge case does NOT occur.
        if blob_kwargs:
            assert_array_equal(images_convex[n], expected["image_convex"][n])
        else:
            # Can't compare to scikit-image in this case
            # Just make sure the convex size is not smaller than the original
            assert (images_convex[n].sum()) >= (images[n].sum())


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("spacing", [None, (1, 1, 1), (1.5, 0.5, 0.76)])
def test_coords(ndim, spacing):
    shape = (256, 512) if ndim == 2 else (35, 63, 37)
    if spacing is not None:
        spacing = spacing[:ndim]
    labels = get_labels_nd(shape)
    max_label = int(cp.max(labels))
    coords, coords_scaled = regionprops_coords(
        labels,
        max_label=max_label,
        spacing=spacing,
        compute_coords=True,
        compute_coords_scaled=True,
    )
    assert len(coords) == max_label
    assert len(coords_scaled) == max_label

    expected = measure_cpu.regionprops_table(
        cp.asnumpy(labels),
        spacing=spacing,
        properties=["coords", "coords_scaled"],
    )
    for n in range(max_label):
        # cast to Python int to match dtype from CPU case
        assert_array_equal(coords[n].astype(int), expected["coords"][n])

        assert_allclose(
            coords_scaled[n], expected["coords_scaled"][n], rtol=1e-5
        )
