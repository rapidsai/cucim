from .._shared.utils import deprecated
from ._basic_features import multiscale_basic_features
from ._canny import canny
from ._daisy import daisy
from .blob import blob_dog, blob_doh, blob_log
from .corner import (corner_foerstner, corner_harris, corner_kitchen_rosenfeld,
                     corner_peaks, corner_shi_tomasi, hessian_matrix,
                     hessian_matrix_det, hessian_matrix_eigvals, shape_index,
                     structure_tensor, structure_tensor_eigenvalues)
from .match import match_descriptors
from .peak import peak_local_max
from .template import match_template

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
           'match_descriptors',
           'blob_dog',
           'blob_log',
           'blob_doh']
