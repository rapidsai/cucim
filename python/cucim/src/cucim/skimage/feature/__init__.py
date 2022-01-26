from .._shared.utils import deprecated
from ._basic_features import multiscale_basic_features
from ._canny import canny
from ._daisy import daisy
from .corner import (corner_foerstner, corner_harris, corner_kitchen_rosenfeld,
                     corner_peaks, corner_shi_tomasi, hessian_matrix,
                     hessian_matrix_det, hessian_matrix_eigvals, shape_index,
                     structure_tensor, structure_tensor_eigenvalues)
from .peak import peak_local_max
from .template import match_template


@deprecated(
    alt_func="cucim.skimage.registration.phase_cross_correlation",
    removed_version="0.19",
)
def masked_register_translation(
    src_image, target_image, src_mask, target_mask=None, overlap_ratio=0.3
):
    from ..registration import phase_cross_correlation

    return phase_cross_correlation(
        src_image,
        target_image,
        reference_mask=src_mask,
        moving_mask=target_mask,
        overlap_ratio=overlap_ratio,
    )


@deprecated(
    alt_func="cucim.skimage.registration.phase_cross_correlation",
    removed_version="0.19",
)
def register_translation(
    src_image, target_image, upsample_factor=1, space="real", return_error=True
):
    from ..registration._phase_cross_correlation import \
        phase_cross_correlation as func
    return func(src_image, target_image, upsample_factor, space, return_error)


__all__ = ['canny',
           'daisy',
           'multiscale_basic_features',
           'peak_local_max',
           'structure_tensor',
           'structure_tensor_eigenvalues',
           'structure_tensor_eigvals',
           'hessian_matrix',
           'hessian_matrix_det',
           'hessian_matrix_eigvals',
           'shape_index',
           'corner_kitchen_rosenfeld',
           'corner_harris',
           'corner_shi_tomasi',
           'corner_foerstner',
           # 'corner_subpix',
           'corner_peaks',
           # 'corner_moravec',
           # 'corner_fast',
           # 'corner_orientations',
           'match_template',
           'register_translation',
           'masked_register_translation']
