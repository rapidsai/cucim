/***********************************************************************

_hessian_det_appx.cu has the following scikit-image (v0.19.3) functions as cuda functions:
    - _clip
    - _integ
    - _hessian_matrix_det

Helpful example to get into ElementWiseKernel:
    - https://github.com/yuyu2172/cupy/blob/a651541668c8dd024973526dbd5ed672cc482260/examples/gemm/README.md
***********************************************************************/


/* @func _clip
    Clip coordinate between low and high values.
    This method was created so that `hessian_det_appx` does not have to make
    a Python call.
    Parameters
    ----------
    x : int
        Coordinate to be clipped.
    low : int
        The lower bound.
    high : int
        The higher bound.
    Returns
    -------
    x : int
        `x` clipped between `high` and `low`.
    """
    assert 0 <= low <= high

    if x > high:
        return high
    elif x < low:
        return low
    else:
        return x
*/
__device__
inline int _clip(const int x, const int low, const int high)
{
//     assert((int)((0 <= low) && (low <= high));

    if (x > high)
    {
        return high;
    }
    else if (x < low)
    {
        return low;
    }
    else
    {
        return x;
    }
}


/*@func _integ
    Integrate over the 2D integral image in the given window.
    This method was created so that `hessian_det_appx` does not have to make
    a Python call.
    Parameters
    ----------
    img : array
        The integral image over which to integrate.
    r : int
        The row number of the top left corner.
    c : int
        The column number of the top left corner.
    rl : int
        The number of rows over which to integrate.
    cl : int
        The number of columns over which to integrate.
    Returns
    -------
    ans : double
        The integral over the given window.
*/
__device__
inline IMAGE_T _integ(const IMAGE_T * img,
                     const INT_T img_rows,
                     const INT_T img_cols,
                     int r,
                     int c,
                     const int rl,
                     const int cl)
{

    r = _clip(r, 0, img_rows - 1);
    c = _clip(c, 0, img_cols - 1);

    const int r2 = _clip(r + rl, 0, img_rows - 1);
    const int c2 = _clip(c + cl, 0, img_cols - 1);

    IMAGE_T ans = img[r * img_cols + c] + img[r2 * img_cols + c2] - img[r * img_cols + c2] - img[r2 * img_cols + c];

    return max(0., ans);
}


/*@func _hessian_matrix_det
    Compute the approximate Hessian Determinant over a 2D image.
    This method uses box filters over integral images to compute the
    approximate Hessian Determinant as described in [1]_.
    Parameters
    ----------
    img : array
        The integral image over which to compute Hessian Determinant.
    sigma : double
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
*/
extern "C" __global__
void _hessian_matrix_det(
        const IMAGE_T* img,
        const INT_T img_rows,
        const INT_T img_cols,
        const double sigma,
        IMAGE_T* out)
{
    // Some note her: Input type from cupy must be a cp.float64, which maps to cuda's double
    // If someone wants to fasten this function with cp.float32, this function must be re-written with cuda's float

    // *************************************************************************
    // This function is derived from Scikit-Image _hessian_matrix_det (v0.19.3):
    // *************************************************************************

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= img_rows*img_cols)
    {
        return;
    }

    const int r = (int) floorf(tid / img_cols);
    const int c = (int) tid % img_cols;

    int size = (int)(3.0 * sigma);

    const int s2 = floorf((size - 1) / 2);
    const int s3 = floorf(size / 3);
    const int l = floorf(size / 3);
    const int w = size;
    const int b = floorf((size - 1) / 2);

    const IMAGE_T w_i = 1.0 / size / size;

    if ((size % 2) == 0)
    {
        size += 1;
    }

    const IMAGE_T tl = _integ(img, img_rows, img_cols, r - s3, c - s3, s3, s3);  // top left
    const IMAGE_T br = _integ(img, img_rows, img_cols, r + 1, c + 1, s3, s3);  // bottom right
    const IMAGE_T bl = _integ(img, img_rows, img_cols, r - s3, c + 1, s3, s3);  // bottom left
    const IMAGE_T tr = _integ(img, img_rows, img_cols, r + 1, c - s3, s3, s3);  // top right

    IMAGE_T dxy = bl + tr - tl - br;
    dxy = -dxy * w_i;

    IMAGE_T mid = _integ(img, img_rows, img_cols, r - s3 + 1, c - s2, 2 * s3 - 1, w);  // middle box
    IMAGE_T side = _integ(img, img_rows, img_cols, r - s3 + 1, c - (int)floorf(s3 / 2), 2 * s3 - 1, s3);  // sides

    IMAGE_T dxx = mid - 3 * side;
    dxx = -dxx * w_i;

    mid = _integ(img, img_rows, img_cols, r - s2, c - s3 + 1, w, 2 * s3 - 1);
    side = _integ(img, img_rows, img_cols, r - (int)floorf(s3 / 2), c - s3 + 1, s3, 2 * s3 - 1);

    IMAGE_T dyy = mid - 3 * side;
    dyy = -dyy * w_i;

    out[tid] = (dxx * dyy - 0.81 * (dxy * dxy));
}
