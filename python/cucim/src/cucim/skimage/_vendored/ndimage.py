# locally defined filters that are more efficient than in CuPy

from cupyx.scipy.ndimage import distance_transform_edt  # NOQA
from cupyx.scipy.ndimage import fourier_ellipsoid  # NOQA
from cupyx.scipy.ndimage import fourier_gaussian  # NOQA
from cupyx.scipy.ndimage import fourier_shift  # NOQA
from cupyx.scipy.ndimage import fourier_uniform  # NOQA
from cupyx.scipy.ndimage import generic_filter  # NOQA
from cupyx.scipy.ndimage import generic_filter1d  # NOQA
from cupyx.scipy.ndimage import label  # NOQA
from cupyx.scipy.ndimage import value_indices  # NOQA

from cucim.skimage._vendored._ndimage_filters import convolve  # NOQA
from cucim.skimage._vendored._ndimage_filters import convolve1d  # NOQA
from cucim.skimage._vendored._ndimage_filters import correlate  # NOQA
from cucim.skimage._vendored._ndimage_filters import correlate1d  # NOQA
from cucim.skimage._vendored._ndimage_filters import gaussian_filter  # NOQA
from cucim.skimage._vendored._ndimage_filters import gaussian_filter1d  # NOQA
from cucim.skimage._vendored._ndimage_filters import gaussian_laplace  # NOQA
from cucim.skimage._vendored._ndimage_filters import generic_laplace  # NOQA
from cucim.skimage._vendored._ndimage_filters import laplace  # NOQA
from cucim.skimage._vendored._ndimage_filters import maximum_filter  # NOQA
from cucim.skimage._vendored._ndimage_filters import maximum_filter1d  # NOQA
from cucim.skimage._vendored._ndimage_filters import median_filter  # NOQA
from cucim.skimage._vendored._ndimage_filters import minimum_filter  # NOQA
from cucim.skimage._vendored._ndimage_filters import minimum_filter1d  # NOQA
from cucim.skimage._vendored._ndimage_filters import percentile_filter  # NOQA
from cucim.skimage._vendored._ndimage_filters import prewitt  # NOQA
from cucim.skimage._vendored._ndimage_filters import rank_filter  # NOQA
from cucim.skimage._vendored._ndimage_filters import sobel  # NOQA
from cucim.skimage._vendored._ndimage_filters import uniform_filter  # NOQA
from cucim.skimage._vendored._ndimage_filters import uniform_filter1d  # NOQA
from cucim.skimage._vendored._ndimage_filters import (  # NOQA
    gaussian_gradient_magnitude,
    generic_gradient_magnitude,
)

# interpolation
from cucim.skimage._vendored._ndimage_interpolation import rotate  # NOQA
from cucim.skimage._vendored._ndimage_interpolation import shift  # NOQA
from cucim.skimage._vendored._ndimage_interpolation import spline_filter  # NOQA
from cucim.skimage._vendored._ndimage_interpolation import zoom  # NOQA
from cucim.skimage._vendored._ndimage_interpolation import (  # NOQA
    affine_transform,
    map_coordinates,
    spline_filter1d,
)

# morphology
from cucim.skimage._vendored._ndimage_morphology import binary_closing  # NOQA
from cucim.skimage._vendored._ndimage_morphology import binary_dilation  # NOQA
from cucim.skimage._vendored._ndimage_morphology import binary_erosion  # NOQA
from cucim.skimage._vendored._ndimage_morphology import binary_opening  # NOQA
from cucim.skimage._vendored._ndimage_morphology import black_tophat  # NOQA
from cucim.skimage._vendored._ndimage_morphology import grey_closing  # NOQA
from cucim.skimage._vendored._ndimage_morphology import grey_dilation  # NOQA
from cucim.skimage._vendored._ndimage_morphology import grey_erosion  # NOQA
from cucim.skimage._vendored._ndimage_morphology import grey_opening  # NOQA
from cucim.skimage._vendored._ndimage_morphology import white_tophat  # NOQA
from cucim.skimage._vendored._ndimage_morphology import (  # NOQA
    binary_fill_holes,
    binary_hit_or_miss,
    binary_propagation,
    generate_binary_structure,
    iterate_structure,
    morphological_gradient,
    morphological_laplace,
)

# Import the rest of the cupyx.scipy.ndimage API here


try:
    # `sum_labels` is only available as `sum` in older releases of CuPy
    from cupyx.scipy.ndimage import sum_labels  # NOQA
except ImportError:
    from cupyx.scipy.ndimage import sum as sum_labels  # NOQA
try:
    from cupyx.scipy.ndimage import sum  # NOQA
except ImportError:
    # `sum` is deprecated and may be removed in a future version of CuPy
    pass

from cupyx.scipy.ndimage import center_of_mass  # NOQA
from cupyx.scipy.ndimage import extrema  # NOQA
from cupyx.scipy.ndimage import histogram  # NOQA
from cupyx.scipy.ndimage import labeled_comprehension  # NOQA
from cupyx.scipy.ndimage import maximum  # NOQA
from cupyx.scipy.ndimage import maximum_position  # NOQA
from cupyx.scipy.ndimage import mean  # NOQA
from cupyx.scipy.ndimage import median  # NOQA
from cupyx.scipy.ndimage import minimum  # NOQA
from cupyx.scipy.ndimage import minimum_position  # NOQA
from cupyx.scipy.ndimage import standard_deviation  # NOQA
from cupyx.scipy.ndimage import variance  # NOQA

try:
    # only available in CuPy >=13.4
    from cupyx.scipy.ndimage import find_objects
except ImportError:
    # vendored implementation (without internal Cython helper function)
    from cucim.skimage._vendored._ndimage_measurements import find_objects
