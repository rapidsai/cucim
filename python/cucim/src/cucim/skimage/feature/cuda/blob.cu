/***********************************************************************

blob.cu has the following scikit-image functions as cuda functions:
    - _compute_disk_overlap
    - _compute_sphere_overlap
    - _blob_overlap
    - _prune_blobs

Helpful example to get into ElementWiseKernel:
    - https://github.com/yuyu2172/cupy/blob/a651541668c8dd024973526dbd5ed672cc482260/examples/gemm/README.md
***********************************************************************/


/*@func _compute_disk_overlap
    Compute fraction of surface overlap between two disks of radii
    ``r1`` and ``r2``, with centers separated by a distance ``d``.

    Parameters
    ----------
    d : float
        Distance between centers.
    r1 : float
        Radius of the first disk.
    r2 : float
        Radius of the second disk.

    Returns
    -------
    fraction: float
        Fraction of area of the overlap between the two disks.
*/
__device__
inline BLOB_T _compute_disk_overlap(const BLOB_T d, const BLOB_T r1, const BLOB_T r2)
{

    const BLOB_T pi = 3.141592653589793;
    BLOB_T ratio1 = 0.0;
    BLOB_T ratio2 = 0.0;
    BLOB_T r = 0.0;


    ratio1 = (d*d + r1*r1 - r2*r2) / (2 * d * r1);
    if (ratio1 > 1.0)
    {
        ratio1 = 1.0;
    }
    else if (ratio1 < -1.0)
    {
        ratio1 = -1.0;
    }

    ratio2 = (d*d + r2*r2 - r1*r1) / (2 * d * r2);
    if (ratio2 > 1.0)
    {
        ratio2 = 1.0;
    }
    else if (ratio2 < -1.0)
    {
        ratio2 = -1.0;
    }

    BLOB_T a = -d + r2 + r1;
    BLOB_T b = d - r2 + r1;
    BLOB_T c = d + r2 - r1;
    BLOB_T _d = d + r2 + r1;

    r = min(r1, r2);
    return (r1*r1 * acos(ratio1) + r2*r2 * acos(ratio2) - 0.5 * sqrt(abs(a * b * c * _d))) / (pi * r*r);
}


/*@func _compute_sphere_overlap
    Compute volume overlap fraction between two spheres of radii
    ``r1`` and ``r2``, with centers separated by a distance ``d``.

    Parameters
    ----------
    d : float
        Distance between centers.
    r1 : float
        Radius of the first sphere.
    r2 : float
        Radius of the second sphere.

    Returns
    -------
    fraction: float
        Fraction of volume of the overlap between the two spheres.

    Notes
    -----
    See for example http://mathworld.wolfram.com/Sphere-SphereIntersection.html
    for more details.
*/
__device__
inline BLOB_T _compute_sphere_overlap(const BLOB_T d, const BLOB_T r1, const BLOB_T r2) {

    const BLOB_T pi = 3.141592653589793;
    const BLOB_T r = min(r1, r2);
    const BLOB_T vol = (pi / (12 * d) * (r1 + r2 - d) * (r1 + r2 - d) * \
                      (d*d + 2 * d * (r1 + r2) - 3 * (r1*r1 + r2*r2) + 6 * r1 * r2));
    return vol / (4.0/3.0 * pi * r*r*r);
}


/*@func _blob_overlap
    Finds the overlapping area fraction between two blobs.

    Returns a float representing fraction of overlapped area. Note that 0.0
    is *always* returned for dimension greater than 3.

    Parameters
    ----------
    blob1 : sequence of arrays
        A sequence of ``(row, col, sigma)`` or ``(pln, row, col, sigma)``,
        where ``row, col`` (or ``(pln, row, col)``) are coordinates
        of blob and ``sigma`` is the standard deviation of the Gaussian kernel
        which detected the blob.
    blob2 : sequence of arrays
        A sequence of ``(row, col, sigma)`` or ``(pln, row, col, sigma)``,
        where ``row, col`` (or ``(pln, row, col)``) are coordinates
        of blob and ``sigma`` is the standard deviation of the Gaussian kernel
        which detected the blob.
    sigma_dim : int, optional
        The dimensionality of the sigma value. Can be 1 or the same as the
        dimensionality of the blob space (2 or 3).

    Returns
    -------
    f : float
        Fraction of overlapped area (or volume in 3D).

*/
__device__
inline BLOB_T _blob_overlap(BLOB_T* blob1, BLOB_T* blob2, const INT_T sigma_dim, const INT_T n) {

    const INT_T ndim = n - sigma_dim;
    BLOB_T pos1 = 0.0;
    BLOB_T pos2 = 0.0;
    BLOB_T _sum = 0.0;
    BLOB_T r1 = 0.0;
    BLOB_T r2 = 0.0;
    BLOB_T d = 0.0;

    //printf("blob1: [%3.1f %3.1f %3.1f]\n", blob1[0], blob1[1], blob1[2]);
    //printf("blob2: [%3.1f %3.1f %3.1f]\n", blob2[0], blob2[1], blob2[2]);

    if (ndim > 3)
    {
        return 0.0;
    }
    const BLOB_T root_ndim = sqrt((BLOB_T)ndim);

    if (blob1[n-1]== 0.0  && blob2[n-1]== 0.0)
    {
        return 0.0;
    }
    else if (blob1[n-1] > blob2[n-1])
    {
        r1 = 1.0;
        r2 = blob2[n-1] / blob1[n-1];
        for (INT_T _i=0; _i < ndim; _i++)
        {
            pos1 = blob1[_i] / (blob1[min(n-sigma_dim+_i, n-1)] * root_ndim);
            pos2 = blob2[_i] / (blob1[min(n-sigma_dim+_i, n-1)] * root_ndim);
            _sum += ((pos2 - pos1) * (pos2 - pos1));
        }
    }
    else
    {

        r1 = blob1[n-1] / blob2[n-1];
        r2 = 1.0;
        for (INT_T _i=0; _i < ndim; _i++)
        {
            pos1 = blob1[_i] / (blob2[min(n-sigma_dim+_i, n-1)] * root_ndim);
            pos2 = blob2[_i] / (blob2[min(n-sigma_dim+_i, n-1)] * root_ndim);
            _sum += ((pos2 - pos1) * (pos2 - pos1));
        }
    }

    d = sqrt(_sum);

    if (d > (r1 + r2)) // centers farther than sum of radii, so no overlap
    {
        return 0.0;
    }

    // one blob is inside the other
    if (d <= abs(r1 - r2))
    {
        return 1.0;
    }
    if (ndim == 2)
    {
        return _compute_disk_overlap(d, r1, r2);
    }
    else // ndim=3 http://mathworld.wolfram.com/Sphere-SphereIntersection.html
    {
        return _compute_sphere_overlap(d, r1, r2);
    }
}


extern "C" __global__
void _prune_blobs(
        BLOB_T * blobs_array,
        const INT_T n_rows,
        const INT_T n_cols,
        const double overlap,
        INT_T sigma_dim)
{

    // *************************************************************************
    // This function is derived from Scikit-Image _prune_blobs (v0.11.x):
    // *************************************************************************

    INT_T tid = blockDim.x * blockIdx.x + threadIdx.x;
    BLOB_T *blob1;
    BLOB_T *blob2;

    if (tid >= n_rows)
    {
        return;  // all done
    }

    for(INT_T k=0; k<n_rows; k++)
    {
        // blob[tid] --> blob1
        // blob[k  ] --> blob2

        if (tid >= k)
        {
            // skip calculation, it is done already
            continue;
        }

        blob1 = &blobs_array[(INT_T)tid*n_cols];
        blob2 = &blobs_array[(INT_T)k*n_cols];

        if (_blob_overlap(blob1, blob2, sigma_dim, n_cols) > overlap)
        {
            // note: this test works even in the anisotropic case because all sigmas increase together.
            if (blob1[n_cols-1] > blob2[n_cols-1])
            {
                blob2[n_cols-1] = 0.0;
            }
            else
            {
                blob1[n_cols-1] = 0.0;
            }
        }
    }
}
