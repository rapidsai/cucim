from ._gabor import gabor, gabor_kernel
from ._gaussian import _guess_spatial_dimensions  # noqa
from ._gaussian import difference_of_gaussians, gaussian
# from . import rank
from ._median import median
from ._rank_order import rank_order
from ._sparse import correlate_sparse
from ._unsharp_mask import unsharp_mask
from ._window import window
from .edges import (farid, farid_h, farid_v, laplace, prewitt,  # noqa
                    prewitt_h, prewitt_v, roberts, roberts_neg_diag,
                    roberts_pos_diag, scharr, scharr_h, scharr_v, sobel,
                    sobel_h, sobel_v)
from .lpi_filter import LPIFilter2D, inverse, wiener
from .ridges import frangi, hessian, meijering, sato
from .thresholding import (apply_hysteresis_threshold, threshold_isodata,
                           threshold_li, threshold_local, threshold_mean,
                           threshold_minimum, threshold_multiotsu,
                           threshold_niblack, threshold_otsu,
                           threshold_sauvola, threshold_triangle,
                           threshold_yen, try_all_threshold)

__all__ = [
    "inverse",
    "correlate_sparse",
    "wiener",
    "LPIFilter2D",
    "gaussian",
    "difference_of_gaussians",
    "median",
    "sobel",
    "sobel_h",
    "sobel_v",
    "scharr",
    "scharr_h",
    "scharr_v",
    "prewitt",
    "prewitt_h",
    "prewitt_v",
    "roberts",
    "roberts_pos_diag",
    "roberts_neg_diag",
    "laplace",
    "rank_order",
    "gabor_kernel",
    "gabor",
    "try_all_threshold",
    "meijering",
    "sato",
    "frangi",
    "hessian",
    "threshold_otsu",
    "threshold_yen",
    "threshold_isodata",
    "threshold_li",
    "threshold_local",
    "threshold_minimum",
    "threshold_mean",
    "threshold_niblack",
    "threshold_sauvola",
    "threshold_triangle",
    "threshold_multiotsu",
    "apply_hysteresis_threshold",
    # "rank",
    "unsharp_mask",
    "window",
]
