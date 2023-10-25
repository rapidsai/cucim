import cupy as cp

import cucim
import cucim.skimage


if __name__ == "__main__":
    # verify that all top-level modules are available
    assert cucim.is_available('clara')
    assert cucim.is_available('core')
    assert cucim.is_available('skimage')

    # generate a synthetic image and apply a filter
    img = cucim.skimage.data.binary_blobs(length=512, n_dim=2)
    assert isinstance(img, cp.ndarray)
    assert img.dtype.kind == 'b'
    assert img.shape == (512, 512)

    eroded = cucim.skimage.morphology.binary_erosion(
        img, cp.ones((3, 3), dtype=bool)
    )
