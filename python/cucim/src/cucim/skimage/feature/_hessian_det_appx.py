import math
import os

import cupy as cp

hessian_file = "_hessian_det_appx.cu"


def _dtype_to_cuda_float_type(dtype):
    """Maps a float data type from cupy to cuda.
    Returns a cuda (c++) data type.
    Parameters
    ----------
    dtype : cupy dtype
        A cupy dtype from type float.
    Returns
    -------
    cpp_float_type : cuda (c++) data type
        Supported cuda data type
    """
    cpp_float_types = {
        cp.float32: 'float',
        cp.float64: 'double',
    }
    dtype = cp.dtype(dtype)
    if dtype.type not in cpp_float_types:
        raise ValueError(f"unrecognized dtype: {dtype.type}")
    return cpp_float_types[dtype.type]


def _get_hessian_det_appx_kernel(dtype) -> cp.RawModule:
    """Loads all kernels in cuda/_hessian_det_appx.cu.
    Returns a cupy RawModule.
    Parameters
    ----------
    dtype : cupy dtype
        Only the cupy dtypes float32 and float64 are supported.
    Returns
    -------
    out : RawModule
        A cupy RawModule containing the __global__ functions _hessian_matrix_det.
    """
    global hessian_file

    image_t = _dtype_to_cuda_float_type(dtype)

    _preamble = f"""
#define IMAGE_T {image_t}
        """

    kernel_directory = os.path.join(os.path.normpath(os.path.dirname(__file__)), 'cuda')
    with open(os.path.join(kernel_directory, hessian_file), 'rt') as f:
        _code = '\n'.join(f.readlines())

    return cp.RawModule(code=_preamble + _code,
                        options=('--std=c++11',),
                        name_expressions=["_hessian_matrix_det"])


def _hessian_matrix_det(img: cp.ndarray, sigma) -> cp.ndarray:
    """Compute the approximate Hessian Determinant over a 2D image.
    This method uses box filters over integral images to compute the
    approximate Hessian Determinant as described in [1]_.
    Parameters
    ----------
    img : array
        The integral image over which to compute Hessian Determinant.
    sigma : float
        Standard deviation used for the Gaussian kernel, used for the Hessian
        matrix
    Returns
    -------
    out : array
        The array of the Determinant of Hessians.
    References
    ----------
    .. [1] Herbert Bay, Andreas Ess, Tinne Tuytelaars, Luc Van Gool,
           "SURF: Speeded Up Robust Features"
           ftp://ftp.vision.ee.ethz.ch/publications/articles/eth_biwi_00517.pdf
    Notes
    -----
    The running time of this method only depends on size of the image. It is
    independent of `sigma` as one would expect. The downside is that the
    result for `sigma` less than `3` is not accurate, i.e., not similar to
    the result obtained if someone computed the Hessian and took its
    determinant.
    """

    _hessian_det_appx_kernel = _get_hessian_det_appx_kernel(img.dtype).get_function("_hessian_matrix_det")

    out = cp.empty_like(img, dtype=img.dtype)

    block_size = 64
    grid_size = int(math.ceil(img.size / block_size))
    _hessian_det_appx_kernel((grid_size,), (block_size,),
                             (img.ravel(), img.shape[0], img.shape[1], float(sigma), out))

    return out
