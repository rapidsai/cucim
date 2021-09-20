import cucim.core.operations.expose.transform as expose

color_jitter = expose.color_jitter
image_flip = expose.image_flip
image_rotate_90 = expose.image_rotate_90
scale_intensity = expose.scale_intensity_range
zoom = expose.zoom
rand_zoom = expose.rand_zoom
rand_image_flip = expose.rand_image_flip
rand_image_rotate_90 = expose.rand_image_rotate_90


def test_exposed_transforms():
    assert color_jitter is not None
    assert image_flip is not None
    assert image_rotate_90 is not None
    assert scale_intensity is not None
    assert zoom is not None
    assert rand_zoom is not None
    assert rand_image_flip is not None
    assert rand_image_rotate_90 is not None
