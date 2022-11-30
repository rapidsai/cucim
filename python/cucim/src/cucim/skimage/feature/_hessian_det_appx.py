import math
import os

import cupy as cp


def _dtype_to_cuda_float_type(dtype):
    """Maps a float data type from cupy to cuda.

    Returns a cuda (c++) data type.

    Parameters
    ----------
    dtype : cupy dtype
        A cupy dtype from type float.

    Returns
    -------
    cpp_float_type : str
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


@cp.memoize(for_each_device=True)
def _get_hessian_det_appx_kernel(dtype, large_int) -> cp.RawModule:
    """Loads all kernels in cuda/_hessian_det_appx.cu.
    Returns a cupy RawModule.

    Parameters
    ----------
    dtype : cupy dtype
        Only the cupy dtypes float32 and float64 are supported.

    Returns
    -------
    out : RawModule
        A cupy RawModule containing the __global__ functions
        `_hessian_matrix_det`.
    """
    image_t = _dtype_to_cuda_float_type(dtype)

    int_t = 'long long' if large_int else 'int'

    _preamble = f"""
#define IMAGE_T {image_t}
#define INT_T {int_t}
        """

    kernel_directory = os.path.join(
        os.path.normpath(os.path.dirname(__file__)), 'cuda')
    cu_file = os.path.join(kernel_directory, "_hessian_det_appx.cu")
    with open(cu_file, 'rt') as f:
        _code = f.read()

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
    rawmodule = _get_hessian_det_appx_kernel(img.dtype, max(img.shape) > 2**31)
    _hessian_det_appx_kernel = rawmodule.get_function("_hessian_matrix_det")

    out = cp.empty_like(img, dtype=img.dtype)

    block_size = 64
    grid_size = int(math.ceil(img.size / block_size))
    _hessian_det_appx_kernel(
        (grid_size,),
        (block_size,),
        (img.ravel(), img.shape[0], img.shape[1], float(sigma), out)
    )
    return out
