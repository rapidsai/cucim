from cucim.core.operations.expose.transform import (color_jitter, image_flip,
                                                    image_rotate_90,
                                                    normalize_data,
                                                    rand_color_jitter,
                                                    rand_image_flip,
                                                    rand_image_rotate_90,
                                                    rand_zoom,
                                                    scale_intensity_range, zoom)


def test_exposed_transforms():
    assert color_jitter is not None
    assert rand_color_jitter is not None
    assert image_flip is not None
    assert image_rotate_90 is not None
    assert scale_intensity_range is not None
    assert normalize_data is not None
    assert zoom is not None
    assert rand_zoom is not None
    assert rand_image_flip is not None
    assert rand_image_rotate_90 is not None
