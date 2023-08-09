"""GPU Image Processing for Python

This module is a CuPy based implementation of a subset of scikit-image.

It is a collection of algorithms for image processing and computer vision.

The main package only provides a few utilities for converting between image
data types; for most features, you need to import one of the following
subpackages:

Subpackages
-----------
color
    Color space conversion.
data
    Test images and example data.
exposure
    Image intensity adjustment, e.g., histogram equalization, etc.
feature
    Feature detection and extraction, e.g., texture analysis corners, etc.
filters
    Sharpening, edge finding, rank filters, thresholding, etc.
measure
    Measurement of image properties, e.g., region properties and contours.
metrics
    Metrics corresponding to images, e.g. distance metrics, similarity, etc.
morphology
    Morphological operations, e.g., opening or skeletonization.
restoration
    Restoration algorithms, e.g., deconvolution algorithms, denoising, etc.
segmentation
    Partitioning an image into multiple regions.
transform
    Geometric and other transforms, e.g., rotation or the Radon transform.
util
    Generic utilities.

Utility Functions
-----------------
img_as_float
    Convert an image to floating point format, with values in [0, 1].
    Is similar to `img_as_float64`, but will not convert lower-precision
    floating point arrays to `float64`.
img_as_float32
    Convert an image to single-precision (32-bit) floating point format,
    with values in [0, 1].
img_as_float64
    Convert an image to double-precision (64-bit) floating point format,
    with values in [0, 1].
img_as_uint
    Convert an image to unsigned integer format, with values in [0, 65535].
img_as_int
    Convert an image to signed integer format, with values in [-32768, 32767].
img_as_ubyte
    Convert an image to unsigned byte format, with values in [0, 255].
img_as_bool
    Convert an image to boolean format, with values either True or False.
dtype_limits
    Return intensity limits, i.e. (min, max) tuple, of the image's dtype.

"""

import lazy_loader as lazy

__getattr__, __lazy_dir__, _ = lazy.attach_stub(__name__, __file__)


def __dir__():
    return __lazy_dir__()


# Legacy imports into the root namespace; not advertised in __all__
from .util.dtype import (dtype_limits, img_as_bool, img_as_float,
                         img_as_float32, img_as_float64, img_as_int,
                         img_as_ubyte, img_as_uint)
from .util.lookfor import lookfor
