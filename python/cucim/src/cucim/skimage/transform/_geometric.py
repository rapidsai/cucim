# TODO: not yet converted for GPU use

import math
import textwrap

import cupy as cp
import numpy as np
from scipy import spatial

from .._shared.utils import get_bound_method_class, safe_as_int

_sin, _cos = math.sin, math.cos


def _affine_matrix_from_vector(v):
    """Affine matrix from linearized (d, d + 1) matrix entries."""
    nparam = v.size
    # solve for d in: d * (d + 1) = nparam
    d = (1 + math.sqrt(1 + 4 * nparam)) / 2 - 1
    dimensionality = round(d)  # round to prevent approx errors
    if d != dimensionality:
        raise ValueError('Invalid number of elements for '
                         f'linearized matrix: {nparam}')
    matrix = np.eye(dimensionality + 1)
    matrix[:-1, :] = np.reshape(v, (dimensionality, dimensionality + 1))
    return matrix


def _center_and_normalize_points(points):
    """Center and normalize image points.

    The points are transformed in a two-step procedure that is expressed
    as a transformation matrix. The matrix of the resulting points is usually
    better conditioned than the matrix of the original points.

    Center the image points, such that the new coordinate system has its
    origin at the centroid of the image points.

    Normalize the image points, such that the mean distance from the points
    to the origin of the coordinate system is sqrt(D).

    If the points are all identical, the returned values will contain nan.

    Parameters
    ----------
    points : (N, D) ndarray
        The coordinates of the image points.

    Returns
    -------
    matrix : (D+1, D+1) ndarray
        The transformation matrix to obtain the new points.
    new_points : (N, D) ndarray
        The transformed image points.
    has_nan : bool
        Indicates if all points were identical causing rms=0.

    References
    ----------
    .. [1] Hartley, Richard I. "In defense of the eight-point algorithm."
           Pattern Analysis and Machine Intelligence, IEEE Transactions on 19.6
           (1997): 580-593.

    """
    # TODO: grlee77: exclude numpy arrays?
    xp = cp.get_array_module(points)
    n, d = points.shape

    centroid = xp.mean(points, axis=0)

    diff = points - centroid
    rms = math.sqrt(xp.sum(diff * diff) / points.shape[0])

    # if all the points are the same, the transformation matrix cannot be
    # created. We return an equivalent matrix with np.nans as sentinel values.
    # This obviates the need for try/except blocks in functions calling this
    # one, and those are only needed when actual 0 is reached, rather than some
    # small value; ie, we don't need to worry about numerical stability here,
    # only actual 0.
    if rms == 0:
        return (xp.full((d + 1, d + 1), xp.nan),
                xp.full_like(points, xp.nan),
                True)

    norm_factor = math.sqrt(d) / rms

    matrix = xp.concatenate(
        (
            norm_factor
            * xp.concatenate((xp.eye(d), -centroid[:, xp.newaxis]), axis=1),
            xp.asarray([[0] * d + [1]]),
        ),
        axis=0,
    )

    points_h = xp.concatenate([points.T, xp.ones((1, n))], axis=0)
    new_points_h = (matrix @ points_h).T

    new_points = new_points_h[:, :d]
    new_points /= new_points_h[:, d:]

    return matrix, new_points, False


def _umeyama(src, dst, estimate_scale):
    """Estimate N-D similarity transformation with or without scaling.

    Parameters
    ----------
    src : (M, N) ndarray
        Source coordinates.
    dst : (M, N) ndarray
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.

    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.

    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`

    """

    num = src.shape[0]
    dim = src.shape[1]

    # TODO: grlee77: exclude numpy arrays?
    xp = cp.get_array_module(src)

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = dst_demean.T @ src_demean / num

    # Eq. (39).
    d = xp.ones((dim,), dtype=xp.float64)
    if xp.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = xp.eye(dim + 1, dtype=xp.float64)

    U, S, V = xp.linalg.svd(A)

    # Eq. (40) and (43).
    rank = xp.linalg.matrix_rank(A)
    if rank == 0:
        return xp.nan * T
    elif rank == dim - 1:
        if xp.linalg.det(U) * xp.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ xp.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ xp.diag(d) @ V

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale

    return T


class GeometricTransform:
    """Base class for geometric transformations.

    """
    def __call__(self, coords):
        """Apply forward transformation.

        Parameters
        ----------
        coords : (N, 2) ndarray
            Source coordinates.

        Returns
        -------
        coords : (N, 2) ndarray
            Destination coordinates.

        """
        raise NotImplementedError()

    def inverse(self, coords):
        """Apply inverse transformation.

        Parameters
        ----------
        coords : (N, 2) ndarray
            Destination coordinates.

        Returns
        -------
        coords : (N, 2) ndarray
            Source coordinates.

        """
        raise NotImplementedError()

    def residuals(self, src, dst):
        """Determine residuals of transformed destination coordinates.

        For each transformed source coordinate the euclidean distance to the
        respective destination coordinate is determined.

        Parameters
        ----------
        src : (N, 2) ndarray
            Source coordinates.
        dst : (N, 2) ndarray
            Destination coordinates.

        Returns
        -------
        residuals : (N, ) ndarray
            Residual for coordinate.

        """
        xp = cp.get_array_module(src)
        return xp.sqrt(xp.sum((self(src) - dst) ** 2, axis=1))

    def __add__(self, other):
        """Combine this transformation with another.

        """
        raise NotImplementedError()


class FundamentalMatrixTransform(GeometricTransform):
    """Fundamental matrix transformation.

    The fundamental matrix relates corresponding points between a pair of
    uncalibrated images. The matrix transforms homogeneous image points in one
    image to epipolar lines in the other image.

    The fundamental matrix is only defined for a pair of moving images. In the
    case of pure rotation or planar scenes, the homography describes the
    geometric relation between two images (`ProjectiveTransform`). If the
    intrinsic calibration of the images is known, the essential matrix describes
    the metric relation between the two images (`EssentialMatrixTransform`).

    References
    ----------
    .. [1] Hartley, Richard, and Andrew Zisserman. Multiple view geometry in
           computer vision. Cambridge university press, 2003.

    Parameters
    ----------
    matrix : (3, 3) ndarray, optional
        Fundamental matrix.

    Attributes
    ----------
    params : (3, 3) ndarray
        Fundamental matrix.

    """

    # CuPy Backend: if matrix is None cannot infer array module from it
    #               added explicit xp module argument for now
    def __init__(self, matrix=None, *, dimensionality=2, xp=cp):
        if matrix is None:
            # default to an identity transform
            matrix = xp.eye(dimensionality + 1)
        else:
            dimensionality = matrix.shape[0] - 1
        if matrix.shape != (dimensionality + 1, dimensionality + 1):
            raise ValueError("Invalid shape of transformation matrix")
        self.params = matrix
        if dimensionality != 2:
            raise NotImplementedError(
                f"{self.__class__} is only implemented for 2D coordinates "
                "(i.e. 3D transformation matrices)."
            )

    def __call__(self, coords):
        """Apply forward transformation.

        Parameters
        ----------
        coords : (N, 2) ndarray
            Source coordinates.

        Returns
        -------
        coords : (N, 3) ndarray
            Epipolar lines in the destination image.

        """
        xp = cp.get_array_module(coords)
        coords_homogeneous = xp.column_stack([coords, xp.ones(coords.shape[0])])
        return coords_homogeneous @ self.params.T

    def inverse(self, coords):
        """Apply inverse transformation.

        Parameters
        ----------
        coords : (N, 2) ndarray
            Destination coordinates.

        Returns
        -------
        coords : (N, 3) ndarray
            Epipolar lines in the source image.

        """
        xp = cp.get_array_module(coords)
        coords_homogeneous = xp.column_stack([coords, xp.ones(coords.shape[0])])
        return coords_homogeneous @ self.params

    def _setup_constraint_matrix(self, src, dst):
        """Setup and solve the homogeneous epipolar constraint matrix::

            dst' * F * src = 0.

        Parameters
        ----------
        src : (N, 2) ndarray
            Source coordinates.
        dst : (N, 2) ndarray
            Destination coordinates.

        Returns
        -------
        F_normalized : (3, 3) ndarray
            The normalized solution to the homogeneous system. If the system
            is not well-conditioned, this matrix contains NaNs.
        src_matrix : (3, 3) ndarray
            The transformation matrix to obtain the normalized source
            coordinates.
        dst_matrix : (3, 3) ndarray
            The transformation matrix to obtain the normalized destination
            coordinates.

        """
        if src.shape != dst.shape:
            raise ValueError("src and dst shapes must be identical.")
        if src.shape[0] < 8:
            raise ValueError("src.shape[0] must be equal or larger than 8.")
        xp = cp.get_array_module(src)

        # Center and normalize image points for better numerical stability.
        try:
            src_matrix, src, has_nan1 = _center_and_normalize_points(src)
            dst_matrix, dst, has_nan2 = _center_and_normalize_points(dst)
        except ZeroDivisionError:
            self.params = xp.full((3, 3), xp.nan)
            return 3 * [xp.full((3, 3), xp.nan)]

        if has_nan1 or has_nan2:
            self.params = xp.full((3, 3), xp.nan)
            return 3 * [xp.full((3, 3), xp.nan)]

        # Setup homogeneous linear equation as dst' * F * src = 0.
        A = xp.ones((src.shape[0], 9))
        A[:, :2] = src
        A[:, :3] *= dst[:, 0, xp.newaxis]
        A[:, 3:5] = src
        A[:, 3:6] *= dst[:, 1, xp.newaxis]
        A[:, 6:8] = src

        # Solve for the nullspace of the constraint matrix.
        _, _, V = xp.linalg.svd(A)
        F_normalized = V[-1, :].reshape(3, 3)

        return F_normalized, src_matrix, dst_matrix

    def estimate(self, src, dst):
        """Estimate fundamental matrix using 8-point algorithm.

        The 8-point algorithm requires at least 8 corresponding point pairs for
        a well-conditioned solution, otherwise the over-determined solution is
        estimated.

        Parameters
        ----------
        src : (N, 2) ndarray
            Source coordinates.
        dst : (N, 2) ndarray
            Destination coordinates.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.

        """
        xp = cp.get_array_module(src)

        F_normalized, src_matrix, dst_matrix = self._setup_constraint_matrix(
            src, dst
        )

        # Enforcing the internal constraint that two singular values must be
        # non-zero and one must be zero.
        U, S, V = xp.linalg.svd(F_normalized)
        S[2] = 0
        F = U @ xp.diag(S) @ V

        self.params = dst_matrix.T @ F @ src_matrix

        return True

    def residuals(self, src, dst):
        """Compute the Sampson distance.

        The Sampson distance is the first approximation to the geometric error.

        Parameters
        ----------
        src : (N, 2) ndarray
            Source coordinates.
        dst : (N, 2) ndarray
            Destination coordinates.

        Returns
        -------
        residuals : (N, ) ndarray
            Sampson distance.

        """
        xp = cp.get_array_module(src)
        src_homogeneous = xp.column_stack([src, xp.ones(src.shape[0])])
        dst_homogeneous = xp.column_stack([dst, xp.ones(dst.shape[0])])

        F_src = self.params @ src_homogeneous.T
        Ft_dst = self.params.T @ dst_homogeneous.T

        dst_F_src = xp.sum(dst_homogeneous * F_src.T, axis=1)

        return xp.abs(dst_F_src) / xp.sqrt(
            F_src[0] ** 2 + F_src[1] ** 2 + Ft_dst[0] ** 2 + Ft_dst[1] ** 2
        )


class EssentialMatrixTransform(FundamentalMatrixTransform):
    """Essential matrix transformation.

    The essential matrix relates corresponding points between a pair of
    calibrated images. The matrix transforms normalized, homogeneous image
    points in one image to epipolar lines in the other image.

    The essential matrix is only defined for a pair of moving images capturing a
    non-planar scene. In the case of pure rotation or planar scenes, the
    homography describes the geometric relation between two images
    (`ProjectiveTransform`). If the intrinsic calibration of the images is
    unknown, the fundamental matrix describes the projective relation between
    the two images (`FundamentalMatrixTransform`).

    References
    ----------
    .. [1] Hartley, Richard, and Andrew Zisserman. Multiple view geometry in
           computer vision. Cambridge university press, 2003.

    Parameters
    ----------
    rotation : (3, 3) ndarray, optional
        Rotation matrix of the relative camera motion.
    translation : (3, 1) ndarray, optional
        Translation vector of the relative camera motion. The vector must
        have unit length.
    matrix : (3, 3) ndarray, optional
        Essential matrix.

    Attributes
    ----------
    params : (3, 3) ndarray
        Essential matrix.

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage import transform
    >>>
    >>> tform_matrix = transform.EssentialMatrixTransform(
    ...     rotation=cp.eye(3), translation=cp.array([0, 0, 1])
    ... )
    >>> tform_matrix.params
    array([[ 0., -1.,  0.],
           [ 1.,  0.,  0.],
           [ 0.,  0.,  0.]])
    >>> src = cp.array([[ 1.839035, 1.924743],
    ...                 [ 0.543582, 0.375221],
    ...                 [ 0.47324 , 0.142522],
    ...                 [ 0.96491 , 0.598376],
    ...                 [ 0.102388, 0.140092],
    ...                 [15.994343, 9.622164],
    ...                 [ 0.285901, 0.430055],
    ...                 [ 0.09115 , 0.254594]])
    >>> dst = cp.array([[1.002114, 1.129644],
    ...                 [1.521742, 1.846002],
    ...                 [1.084332, 0.275134],
    ...                 [0.293328, 0.588992],
    ...                 [0.839509, 0.08729 ],
    ...                 [1.779735, 1.116857],
    ...                 [0.878616, 0.602447],
    ...                 [0.642616, 1.028681]])
    >>> tform_matrix.estimate(src, dst)
    True
    >>> tform_matrix.residuals(src, dst)
    array([0.42455187, 0.01460448, 0.13847034, 0.12140951, 0.27759346,
           0.32453118, 0.00210776, 0.26512283])
    """

    # CuPy Backend: if matrix is None cannot infer array module from it
    #               added explicit xp module argument for now
    def __init__(
        self,
        rotation=None,
        translation=None,
        matrix=None,
        *,
        dimensionality=2,
        xp=cp,
    ):
        super().__init__(matrix=matrix, dimensionality=dimensionality)
        if rotation is not None:
            if translation is None:
                raise ValueError("Both rotation and translation required")
            if rotation.shape != (3, 3):
                raise ValueError("Invalid shape of rotation matrix")
            if abs(xp.linalg.det(rotation) - 1) > 1e-6:
                raise ValueError("Rotation matrix must have unit determinant")
            if translation.size != 3:
                raise ValueError("Invalid shape of translation vector")
            if abs(xp.linalg.norm(translation) - 1) > 1e-6:
                raise ValueError("Translation vector must have unit length")
            # Matrix representation of the cross product for t.
            if isinstance(translation, cp.ndarray):
                translation = cp.asnumpy(translation)
            # CuPy Backend: TODO: always keep t_x, rotation, etc. on host?
            # fmt: off
            t_x = xp.array([0, -translation[2], translation[1],
                            translation[2], 0, -translation[0],
                            -translation[1], translation[0], 0]).reshape(3, 3)
            # fmt: on
            self.params = t_x @ rotation
        elif matrix is not None:
            if matrix.shape != (3, 3):
                raise ValueError("Invalid shape of transformation matrix")
            self.params = matrix
        else:
            # default to an identity transform
            self.params = xp.eye(3)

    def estimate(self, src, dst):
        """Estimate essential matrix using 8-point algorithm.

        The 8-point algorithm requires at least 8 corresponding point pairs for
        a well-conditioned solution, otherwise the over-determined solution is
        estimated.

        Parameters
        ----------
        src : (N, 2) ndarray
            Source coordinates.
        dst : (N, 2) ndarray
            Destination coordinates.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.

        """

        xp = cp.get_array_module(src)
        E_normalized, src_matrix, dst_matrix = self._setup_constraint_matrix(
            src, dst
        )

        # Enforcing the internal constraint that two singular values must be
        # equal and one must be zero.
        U, S, V = xp.linalg.svd(E_normalized)
        S[0] = (S[0] + S[1]) / 2.0
        S[1] = S[0]
        S[2] = 0
        E = U @ xp.diag(S) @ V

        self.params = dst_matrix.T @ E @ src_matrix

        return True


class ProjectiveTransform(GeometricTransform):
    r"""Projective transformation.

    Apply a projective transformation (homography) on coordinates.

    For each homogeneous coordinate :math:`\mathbf{x} = [x, y, 1]^T`, its
    target position is calculated by multiplying with the given matrix,
    :math:`H`, to give :math:`H \mathbf{x}`::

      [[a0 a1 a2]
       [b0 b1 b2]
       [c0 c1 1 ]].

    E.g., to rotate by theta degrees clockwise, the matrix should be::

      [[cos(theta) -sin(theta) 0]
       [sin(theta)  cos(theta) 0]
       [0            0         1]]

    or, to translate x by 10 and y by 20::

      [[1 0 10]
       [0 1 20]
       [0 0 1 ]].

    Parameters
    ----------
    matrix : (D+1, D+1) ndarray, optional
        Homogeneous transformation matrix.
    dimensionality : int, optional
        The number of dimensions of the transform. This is ignored if
        ``matrix`` is not None.

    Attributes
    ----------
    params : (D+1, D+1) ndarray
        Homogeneous transformation matrix.

    """

    def __init__(self, matrix=None, *, dimensionality=2, xp=cp):
        if matrix is None:
            # default to an identity transform
            matrix = xp.eye(dimensionality + 1)
        else:
            dimensionality = matrix.shape[0] - 1
            if matrix.shape != (dimensionality + 1, dimensionality + 1):
                raise ValueError("invalid shape of transformation matrix")
        self.params = matrix
        self._coeffs = range(matrix.size - 1)

    @property
    def _inv_matrix(self):
        xp = cp.get_array_module(self.params)
        return xp.linalg.inv(self.params)

    def _apply_mat(self, coords, matrix):
        xp = cp.get_array_module(coords)
        ndim = matrix.shape[0] - 1
        coords = xp.array(coords, copy=False, ndmin=2)

        src = xp.concatenate([coords, xp.ones((coords.shape[0], 1))], axis=1)
        dst = src @ matrix.T

        # below, we will divide by the last dimension of the homogeneous
        # coordinate matrix. In order to avoid division by zero,
        # we replace exact zeros in this column with a very small number.
        if xp is np:
            dst[dst[:, ndim] == 0, ndim] = np.finfo(float).eps
        else:
            # indexing as above not supported by CuPy
            tmp = dst[:, ndim]
            idx = cp.where(tmp == 0)  # synchronize
            tmp[idx] = np.finfo(float).eps
            dst[:, ndim] = tmp

        # rescale to homogeneous coordinates
        dst[:, :ndim] /= dst[:, ndim:ndim + 1]

        return dst[:, :ndim]

    def __array__(self, dtype=None):
        """Return the transform parameters as an array.

         Note, __array__ is not currently supported by CuPy
        """
        if dtype is None:
            return self.params
        else:
            return self.params.astype(dtype)

    def __call__(self, coords):
        """Apply forward transformation.

        Parameters
        ----------
        coords : (N, D) ndarray
            Source coordinates.

        Returns
        -------
        coords_out : (N, D) ndarray
            Destination coordinates.

        """
        return self._apply_mat(coords, self.params)

    def inverse(self, coords):
        """Apply inverse transformation.

        Parameters
        ----------
        coords : (N, D) ndarray
            Destination coordinates.

        Returns
        -------
        coords_out : (N, D) ndarray
            Source coordinates.

        """
        return self._apply_mat(coords, self._inv_matrix)

    def estimate(self, src, dst, weights=None):
        """Estimate the transformation from a set of corresponding points.

        You can determine the over-, well- and under-determined parameters
        with the total least-squares method.

        Number of source and destination coordinates must match.

        The transformation is defined as::

            X = (a0*x + a1*y + a2) / (c0*x + c1*y + 1)
            Y = (b0*x + b1*y + b2) / (c0*x + c1*y + 1)

        These equations can be transformed to the following form::

            0 = a0*x + a1*y + a2 - c0*x*X - c1*y*X - X
            0 = b0*x + b1*y + b2 - c0*x*Y - c1*y*Y - Y

        which exist for each set of corresponding points, so we have a set of
        N * 2 equations. The coefficients appear linearly so we can write
        A x = 0, where::

            A   = [[x y 1 0 0 0 -x*X -y*X -X]
                   [0 0 0 x y 1 -x*Y -y*Y -Y]
                    ...
                    ...
                  ]
            x.T = [a0 a1 a2 b0 b1 b2 c0 c1 c3]

        In case of total least-squares the solution of this homogeneous system
        of equations is the right singular vector of A which corresponds to the
        smallest singular value normed by the coefficient c3.

        Weights can be applied to each pair of corresponding points to
        indicate, particularly in an overdetermined system, if point pairs have
        higher or lower confidence or uncertainties associated with them. From
        the matrix treatment of least squares problems, these weight values are
        normalised, square-rooted, then built into a diagonal matrix, by which
        A is multiplied.

        In case of the affine transformation the coefficients c0 and c1 are 0.
        Thus the system of equations is::

            A   = [[x y 1 0 0 0 -X]
                   [0 0 0 x y 1 -Y]
                    ...
                    ...
                  ]
            x.T = [a0 a1 a2 b0 b1 b2 c3]

        Parameters
        ----------
        src : (N, 2) ndarray
            Source coordinates.
        dst : (N, 2) ndarray
            Destination coordinates.
        weights : (N,) ndarray, optional
            Relative weight values for each pair of points.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.

        """

        xp = cp.get_array_module(src)
        n, d = src.shape
        src_matrix, src, has_nan1 = _center_and_normalize_points(src)
        dst_matrix, dst, has_nan2 = _center_and_normalize_points(dst)
        if has_nan1 or has_nan2:
            self.params = xp.full((d + 1, d + 1), xp.nan)
            return False
        # params: a0, a1, a2, b0, b1, b2, c0, c1
        A = xp.zeros((n * d, (d + 1) ** 2))
        for ddim in range(d):
            A[
                ddim * n:(ddim + 1) * n, ddim * (d + 1):ddim * (d + 1) + d
            ] = src
            A[ddim * n:(ddim + 1) * n, ddim * (d + 1) + d] = 1
            A[ddim * n:(ddim + 1) * n, -d - 1:-1] = src
            A[ddim * n:(ddim + 1) * n, -1] = -1
            A[ddim * n:(ddim + 1) * n, -d - 1:] *= -dst[:, ddim:(ddim + 1)]

        # Select relevant columns, depending on params
        A = A[:, list(self._coeffs) + [-1]]

        # Get the vectors that correspond to singular values, also applying
        # the weighting if provided
        if weights is None:
            _, _, V = xp.linalg.svd(A)
        else:
            W = xp.diag(xp.tile(xp.sqrt(weights / xp.max(weights)), d))
            _, _, V = xp.linalg.svd(W @ A)

        # if the last element of the vector corresponding to the smallest
        # singular value is close to zero, this implies a degenerate case
        # because it is a rank-defective transform, which would map points
        # to a line rather than a plane.
        if xp.isclose(V[-1, -1], 0):
            self.params = xp.full((d + 1, d + 1), xp.nan)
            return False

        H = np.zeros(
            (d + 1, d + 1)
        )  # np here because .flat not implemented in CuPy
        # solution is right singular vector that corresponds to smallest
        # singular value
        try:
            H.flat[list(self._coeffs.get()) + [-1]] = -V[-1, :-1] / V[-1, -1]
        except AttributeError:
            try:
                V = V.get()
            except AttributeError:
                pass
            H.flat[list(self._coeffs) + [-1]] = -V[-1, :-1] / V[-1, -1]
        H[d, d] = 1
        H = xp.asarray(H)

        # De-center and de-normalize
        H = xp.linalg.inv(dst_matrix) @ H @ src_matrix

        # Small errors can creep in if points are not exact, causing the last
        # element of H to deviate from unity. Correct for that here.
        H /= H[-1, -1]

        self.params = H

        return True

    def __add__(self, other):
        """Combine this transformation with another."""
        if isinstance(other, ProjectiveTransform):
            # combination of the same types result in a transformation of this
            # type again, otherwise use general projective transformation
            if type(self) == type(other):
                tform = self.__class__
            else:
                tform = ProjectiveTransform
            return tform(other.params @ self.params)
        elif (
            hasattr(other, "__name__")
            and other.__name__ == "inverse"
            and hasattr(get_bound_method_class(other), "_inv_matrix")
        ):
            return ProjectiveTransform(other.__self__._inv_matrix @ self.params)
        else:
            raise TypeError(
                "Cannot combine transformations of differing types."
            )

    def __nice__(self):
        """common 'paramstr' used by __str__ and __repr__"""
        npstring = np.array2string(cp.asnumpy(self.params), separator=', ')
        paramstr = 'matrix=\n' + textwrap.indent(npstring, '    ')
        return paramstr

    def __repr__(self):
        """Add standard repr formatting around a __nice__ string"""
        paramstr = self.__nice__()
        classname = self.__class__.__name__
        classstr = classname
        return '<{}({}) at {}>'.format(classstr, paramstr, hex(id(self)))

    def __str__(self):
        """Add standard str formatting around a __nice__ string"""
        paramstr = self.__nice__()
        classname = self.__class__.__name__
        classstr = classname
        return '<{}({})>'.format(classstr, paramstr)

    @property
    def dimensionality(self):
        """The dimensionality of the transformation."""
        return self.params.shape[0] - 1


class AffineTransform(ProjectiveTransform):
    """Affine transformation.

    Has the following form::

        X = a0 * x + a1 * y + a2
          =   sx * x * [cos(rotation) + tan(shear_y) * sin(rotation)]
            - sy * y * [tan(shear_x) * cos(rotation) + sin(rotation)]
            + translation_x

        Y = b0 * x + b1 * y + b2
          =   sx * x * [sin(rotation) - tan(shear_y) * cos(rotation)]
            - sy * y * [tan(shear_x) * sin(rotation) - cos(rotation)]
            + translation_y

    where ``sx`` and ``sy`` are scale factors in the x and y directions.

    This is equivalent to applying the operations in the following order:

    1. Scale
    2. Shear
    3. Rotate
    4. Translate

    The homogeneous transformation matrix is::

        [[a0  a1  a2]
         [b0  b1  b2]
         [0   0    1]]

    In 2D, the transformation parameters can be given as the homogeneous
    transformation matrix, above, or as the implicit parameters, scale,
    rotation, shear, and translation in x (a2) and y (b2). For 3D and higher,
    only the matrix form is allowed.

    In narrower transforms, such as the Euclidean (only rotation and
    translation) or Similarity (rotation, translation, and a global scale
    factor) transforms, it is possible to specify 3D transforms using implicit
    parameters also.

    Parameters
    ----------
    matrix : (D+1, D+1) ndarray, optional
        Homogeneous transformation matrix. If this matrix is provided, it is an
        error to provide any of scale, rotation, shear, or translation.
    scale : {s as float or (sx, sy) as ndarray, list or tuple}, optional
        Scale factor(s). If a single value, it will be assigned to both
        sx and sy. Only available for 2D.

        .. versionadded:: 0.17
           Added support for supplying a single scalar value.
    rotation : float, optional
        Rotation angle, clockwise, as radians. Only available for 2D.
    shear : float or 2-tuple of float, optional
        The x and y shear angles, clockwise, by which these axes are
        rotated around the origin [2].
        If a single value is given, take that to be the x shear angle, with
        the y angle remaining 0. Only available in 2D.
    translation : (tx, ty) as ndarray, list or tuple, optional
        Translation parameters. Only available for 2D.
    dimensionality : int, optional
        The dimensionality of the transform. This is not used if any other
        parameters are provided.

    Attributes
    ----------
    params : (D+1, D+1) ndarray
        Homogeneous transformation matrix.

    Raises
    ------
    ValueError
        If both ``matrix`` and any of the other parameters are provided.

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage import transform
    >>> from skimage import data
    >>> img = cp.array(data.astronaut())

    Define source and destination points:

    >>> src = cp.array([[150, 150],
    ...                 [250, 100],
    ...                 [150, 200]])
    >>> dst = cp.array([[200, 200],
    ...                 [300, 150],
    ...                 [150, 400]])

    Estimate the transformation matrix:

    >>> tform = transform.AffineTransform()
    >>> tform.estimate(src, dst)
    True

    Apply the transformation:

    >>> warped = transform.warp(img, inverse_map=tform.inverse)

    References
    ----------
    .. [1] Wikipedia, "Affine transformation",
           https://en.wikipedia.org/wiki/Affine_transformation#Image_transformation
    .. [2] Wikipedia, "Shear mapping",
           https://en.wikipedia.org/wiki/Shear_mapping
    """

    def __init__(self, matrix=None, scale=None, rotation=None, shear=None,
                 translation=None, *, dimensionality=2, xp=cp):
        params = any(param is not None
                     for param in (scale, rotation, shear, translation))

        # these parameters get overwritten if a higher-D matrix is given
        self._coeffs = range(dimensionality * (dimensionality + 1))

        if params and matrix is not None:
            raise ValueError("You cannot specify the transformation matrix and"
                             " the implicit parameters at the same time.")

        if params and dimensionality > 2:
            raise ValueError("Parameter input is only supported in 2D.")
        elif matrix is not None:
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Invalid shape of transformation matrix.")
            else:
                dimensionality = matrix.shape[0] - 1
                nparam = dimensionality * (dimensionality + 1)
            self._coeffs = range(nparam)
            self.params = matrix
        elif params:  # note: 2D only
            if scale is None:
                scale = (1, 1)
            if rotation is None:
                rotation = 0
            if shear is None:
                shear = 0
            if translation is None:
                translation = (0, 0)

            if np.isscalar(scale):
                sx = sy = scale
            else:
                sx, sy = scale

            if np.isscalar(shear):
                shear_x, shear_y = (shear, 0)
            else:
                shear_x, shear_y = shear

            a0 = sx * (
                math.cos(rotation) + math.tan(shear_y) * math.sin(rotation)
            )
            a1 = -sy * (
                math.tan(shear_x) * math.cos(rotation) + math.sin(rotation)
            )
            a2 = translation[0]

            b0 = sx * (
                math.sin(rotation) - math.tan(shear_y) * math.cos(rotation)
            )
            b1 = -sy * (
                math.tan(shear_x) * math.sin(rotation) - math.cos(rotation)
            )
            b2 = translation[1]
            self.params = xp.array([[a0, a1, a2], [b0, b1, b2], [0, 0, 1]])
        else:
            # default to an identity transform
            self.params = xp.eye(dimensionality + 1)

    @property
    def scale(self):
        xp = cp.get_array_module(self.params)
        if self.dimensionality != 2:
            return xp.sqrt(xp.sum(self.params * self.params, axis=0))[
                : self.dimensionality
            ]
        else:
            ss = xp.sum(self.params * self.params, axis=0)
            ss[1] = ss[1] / (math.tan(self.shear) ** 2 + 1)
            return xp.sqrt(ss)[:self.dimensionality]

    @property
    def rotation(self):
        if self.dimensionality != 2:
            raise NotImplementedError(
                "The rotation property is only implemented for 2D transforms."
            )
        return math.atan2(self.params[1, 0], self.params[0, 0])

    @property
    def shear(self):
        if self.dimensionality != 2:
            raise NotImplementedError(
                "The shear property is only implemented for 2D transforms."
            )
        beta = math.atan2(-self.params[0, 1], self.params[1, 1])
        return beta - self.rotation

    @property
    def translation(self):
        return self.params[0:self.dimensionality, self.dimensionality]


# CuPy Backend: TODO: PiecewiseAffineTransform is inefficient currently
#                     (It only operates via transfer to/from CPU).
class PiecewiseAffineTransform(GeometricTransform):
    """Piecewise affine transformation.

    Control points are used to define the mapping. The transform is based on
    a Delaunay triangulation of the points to form a mesh. Each triangle is
    used to find a local affine transform.

    Attributes
    ----------
    affines : list of AffineTransform objects
        Affine transformations for each triangle in the mesh.
    inverse_affines : list of AffineTransform objects
        Inverse affine transformations for each triangle in the mesh.

    """

    def __init__(self):
        self._tesselation = None
        self._inverse_tesselation = None
        self.affines = None
        self.inverse_affines = None

    def estimate(self, src, dst):
        """Estimate the transformation from a set of corresponding points.

        Number of source and destination coordinates must match.

        Parameters
        ----------
        src : (N, D) ndarray
            Source coordinates.
        dst : (N, D) ndarray
            Destination coordinates.

        Returns
        -------
        success : bool
            True, if all pieces of the model are successfully estimated.

        """

        ndim = src.shape[1]
        # forward piecewise affine
        # triangulate input positions into mesh
        xp = cp.get_array_module(src)
        # TODO: update if spatial.Delaunay become available in CuPy
        self._tesselation = spatial.Delaunay(cp.asnumpy(src))

        ok = True

        # find affine mapping from source positions to destination
        self.affines = []

        try:
            tesselation_simplices = self._tesselation.simplices
        except AttributeError:
            # vertices is deprecated and scheduled for removal in SciPy 1.11
            tesselation_simplices = self._tesselation.vertices
        for tri in xp.asarray(tesselation_simplices):
            affine = AffineTransform(dimensionality=ndim)
            ok &= affine.estimate(src[tri, :], dst[tri, :])
            self.affines.append(affine)

        # inverse piecewise affine
        # triangulate input positions into mesh
        # TODO: update if spatial.Delaunay become available in CuPy
        self._inverse_tesselation = spatial.Delaunay(cp.asnumpy(dst))
        # find affine mapping from source positions to destination
        self.inverse_affines = []
        try:
            inv_tesselation_simplices = self._inverse_tesselation.simplices
        except AttributeError:
            # vertices is deprecated and scheduled for removal in SciPy 1.11
            inv_tesselation_simplices = self._inverse_tesselation.vertices
        for tri in xp.asarray(inv_tesselation_simplices):
            affine = AffineTransform(dimensionality=ndim)
            ok &= affine.estimate(dst[tri, :], src[tri, :])
            self.inverse_affines.append(affine)

        return ok

    def __call__(self, coords):
        """Apply forward transformation.

        Coordinates outside of the mesh will be set to `- 1`.

        Parameters
        ----------
        coords : (N, D) ndarray
            Source coordinates.

        Returns
        -------
        coords : (N, D) ndarray
            Transformed coordinates.

        """

        xp = cp.get_array_module(coords)
        out = xp.empty_like(coords, xp.float64)

        # determine triangle index for each coordinate
        # coords must be on host for calls to _tesselation methods
        simplex = self._tesselation.find_simplex(cp.asnumpy(coords))
        simplex = xp.asarray(simplex)

        # coordinates outside of mesh
        out[simplex == -1, :] = -1

        try:
            tesselation_simplices = self._tesselation.simplices
        except AttributeError:
            # vertices is deprecated and scheduled for removal in SciPy 1.11
            tesselation_simplices = self._tesselation.vertices
        for index in range(len(tesselation_simplices)):
            # affine transform for triangle
            affine = self.affines[index]
            # all coordinates within triangle
            index_mask = simplex == index

            out[index_mask, :] = affine(coords[index_mask, :])

        return out

    def inverse(self, coords):
        """Apply inverse transformation.

        Coordinates outside of the mesh will be set to `- 1`.

        Parameters
        ----------
        coords : (N, D) ndarray
            Source coordinates.

        Returns
        -------
        coords : (N, D) ndarray
            Transformed coordinates.

        """

        xp = cp.get_array_module(coords)
        out = xp.empty_like(coords, xp.float64)

        # determine triangle index for each coordinate
        # coords must be on host for calls to _tesselation methods
        simplex = self._inverse_tesselation.find_simplex(cp.asnumpy(coords))
        simplex = xp.asarray(simplex)

        # coordinates outside of mesh
        out[simplex == -1, :] = -1

        try:
            inv_tesselation_simplices = self._inverse_tesselation.simplices
        except AttributeError:
            # vertices is deprecated and scheduled for removal in SciPy 1.11
            inv_tesselation_simplices = self._inverse_tesselation.vertices
        for index in range(len(inv_tesselation_simplices)):
            # affine transform for triangle
            affine = self.inverse_affines[index]
            # all coordinates within triangle
            index_mask = simplex == index

            out[index_mask, :] = affine(coords[index_mask, :])

        return out


def _euler_rotation(axis, angle):
    """Produce a single-axis Euler rotation matrix.

    Parameters
    ----------
    axis : int in {0, 1, 2}
        The axis of rotation.
    angle : float
        The angle of rotation in radians.

    Returns
    -------
    Ri : ndarray of float, shape (3, 3)
        The rotation matrix along axis `axis`.
    """
    i = axis
    s = (-1) ** i * _sin(angle)
    c = _cos(angle)
    R2 = np.array([[c, -s],  # noqa
                   [s, c]])  # noqa
    Ri = np.eye(3)
    # We need the axes other than the rotation axis, in the right order:
    # 0 -> (1, 2); 1 -> (0, 2); 2 -> (0, 1).
    axes = sorted({0, 1, 2} - {axis})
    # We then embed the 2-axis rotation matrix into the full matrix.
    # (1, 2) -> R[1:3:1, 1:3:1] = R2, (0, 2) -> R[0:3:2, 0:3:2] = R2, etc.
    sl = slice(axes[0], axes[1] + 1, axes[1] - axes[0])
    Ri[sl, sl] = R2
    return Ri


def _euler_rotation_matrix(angles, axes=None):
    """Produce an Euler rotation matrix from the given angles.

    The matrix will have dimension equal to the number of angles given.

    Parameters
    ----------
    angles : ndarray of float, shape (3,)
        The transformation angles in radians.
    axes : list of int
        The axes about which to produce the rotation. Defaults to 0, 1, 2.

    Returns
    -------
    R : ndarray of float, shape (3, 3)
        The Euler rotation matrix.
    """
    if axes is None:
        axes = range(3)
    dim = len(angles)
    R = np.eye(dim)
    for i, angle in zip(axes, angles):
        R = R @ _euler_rotation(i, angle)
    return R


class EuclideanTransform(ProjectiveTransform):
    """Euclidean transformation, also known as a rigid transform.

    Has the following form::

        X = a0 * x - b0 * y + a1 =
          = x * cos(rotation) - y * sin(rotation) + a1

        Y = b0 * x + a0 * y + b1 =
          = x * sin(rotation) + y * cos(rotation) + b1

    where the homogeneous transformation matrix is::

        [[a0  b0  a1]
         [b0  a0  b1]
         [0   0    1]]

    The Euclidean transformation is a rigid transformation with rotation and
    translation parameters. The similarity transformation extends the Euclidean
    transformation with a single scaling factor.

    Parameters
    ----------
    matrix : (D+1, D+1) ndarray, optional
        Homogeneous transformation matrix.
    rotation : float or sequence of float, optional
        Rotation angle, clockwise, as radians. If given as
        a vector, it is interpreted as Euler rotation angles [1]_. Only 2D
        (single rotation) and 3D (Euler rotations) values are supported. For
        higher dimensions, you must provide or estimate the transformation
        matrix.
    translation : sequence of float, length D, optional
        Translation parameters for each axis.
    dimensionality : int, optional
        The dimensionality of the transform.

    Attributes
    ----------
    params : (D+1, D+1) ndarray
        Homogeneous transformation matrix.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
    """

    def __init__(self, matrix=None, rotation=None, translation=None, *,
                 dimensionality=2, xp=cp,):
        params_given = rotation is not None or translation is not None

        if params_given and matrix is not None:
            raise ValueError("You cannot specify the transformation matrix and"
                             " the implicit parameters at the same time.")
        elif matrix is not None:
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Invalid shape of transformation matrix.")
            self.params = matrix
        elif params_given:
            if rotation is None:
                dimensionality = len(translation)
                if dimensionality == 2:
                    rotation = 0
                elif dimensionality == 3:
                    rotation = np.zeros(3)
                else:
                    raise ValueError(
                        "Parameters cannot be specified for dimension "
                        f"{dimensionality} transforms"
                    )
            else:
                if not np.isscalar(rotation) and len(rotation) != 3:
                    raise ValueError(
                        "Parameters cannot be specified for dimension "
                        f"{dimensionality} transforms"
                    )
            if translation is None:
                translation = (0,) * dimensionality

            if dimensionality == 2:
                # fmt: off
                self.params = xp.array([
                    [math.cos(rotation), - math.sin(rotation), 0],  # NOQA
                    [math.sin(rotation),   math.cos(rotation), 0],  # NOQA
                    [                 0,                    0, 1],  # NOQA
                ])
                # fmt: on

            elif dimensionality == 3:
                self.params = xp.eye(dimensionality + 1)
                self.params[:dimensionality, :dimensionality] = xp.asarray(
                    _euler_rotation_matrix(rotation)
                )
            self.params[0:dimensionality, dimensionality] = xp.asarray(
                translation
            )
        else:
            # default to an identity transform
            self.params = xp.eye(dimensionality + 1)

    def estimate(self, src, dst):
        """Estimate the transformation from a set of corresponding points.

        You can determine the over-, well- and under-determined parameters
        with the total least-squares method.

        Number of source and destination coordinates must match.

        Parameters
        ----------
        src : (N, D) ndarray
            Source coordinates.
        dst : (N, D) ndarray
            Destination coordinates.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.

        """

        self.params = _umeyama(src, dst, False)

        # _umeyama will return nan if the problem is not well-conditioned.
        xp = cp.get_array_module(self.params)
        return not xp.any(xp.isnan(self.params))

    @property
    def rotation(self):
        if self.dimensionality == 2:
            return math.atan2(self.params[1, 0], self.params[1, 1])
        elif self.dimensionality == 3:
            # Returning 3D Euler rotation matrix
            return self.params[:3, :3]
        else:
            raise NotImplementedError(
                'Rotation only implemented for 2D and 3D transforms.'
            )

    @property
    def translation(self):
        return self.params[0:self.dimensionality, self.dimensionality]


class SimilarityTransform(EuclideanTransform):
    """Similarity transformation.

    Has the following form in 2D::

        X = a0 * x - b0 * y + a1 =
          = s * x * cos(rotation) - s * y * sin(rotation) + a1

        Y = b0 * x + a0 * y + b1 =
          = s * x * sin(rotation) + s * y * cos(rotation) + b1

    where ``s`` is a scale factor and the homogeneous transformation matrix is::

        [[a0  b0  a1]
         [b0  a0  b1]
         [0   0    1]]

    The similarity transformation extends the Euclidean transformation with a
    single scaling factor in addition to the rotation and translation
    parameters.

    Parameters
    ----------
    matrix : (dim+1, dim+1) ndarray, optional
        Homogeneous transformation matrix.
    scale : float, optional
        Scale factor. Implemented only for 2D and 3D.
    rotation : float, optional
        Rotation angle, clockwise, as radians.
        Implemented only for 2D and 3D. For 3D, this is given in XZX Euler
        angles.
    translation : (dim,) ndarray-like, optional
        x, y[, z] translation parameters. Implemented only for 2D and 3D.

    Attributes
    ----------
    params : (dim+1, dim+1) ndarray
        Homogeneous transformation matrix.

    """

    def __init__(self, matrix=None, scale=None, rotation=None,
                 translation=None, *, dimensionality=2, xp=cp):
        self.params = None
        params = any(param is not None
                     for param in (scale, rotation, translation))

        if params and matrix is not None:
            raise ValueError("You cannot specify the transformation matrix and"
                             " the implicit parameters at the same time.")
        elif matrix is not None:
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Invalid shape of transformation matrix.")
            else:
                self.params = matrix
                dimensionality = matrix.shape[0] - 1
        if params:
            if dimensionality not in (2, 3):
                raise ValueError('Parameters only supported for 2D and 3D.')
            matrix = xp.eye(dimensionality + 1, dtype=float)
            if scale is None:
                scale = 1
            if rotation is None:
                rotation = 0 if dimensionality == 2 else (0, 0, 0)
            if translation is None:
                translation = (0,) * dimensionality
            if dimensionality == 2:
                c, s = _cos(rotation), _sin(rotation)
                matrix[:2, :2] = xp.array([[c, -s], [s, c]], dtype=float)
            else:  # 3D rotation
                matrix[:3, :3] = xp.asarray(_euler_rotation_matrix(rotation))

            matrix[:dimensionality, :dimensionality] *= scale
            matrix[:dimensionality, dimensionality] = xp.asarray(translation)
            self.params = matrix
        elif self.params is None:
            # default to an identity transform
            self.params = xp.eye(dimensionality + 1)

    def estimate(self, src, dst):
        """Estimate the transformation from a set of corresponding points.

        You can determine the over-, well- and under-determined parameters
        with the total least-squares method.

        Number of source and destination coordinates must match.

        Parameters
        ----------
        src : (N, 2) ndarray
            Source coordinates.
        dst : (N, 2) ndarray
            Destination coordinates.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.

        """

        self.params = _umeyama(src, dst, estimate_scale=True)

        # _umeyama will return nan if the problem is not well-conditioned.
        xp = cp.get_array_module(self.params)
        return not xp.any(xp.isnan(self.params))

    @property
    def scale(self):
        # det = scale**(# of dimensions), therefore scale = det**(1/2)
        if self.dimensionality == 2:
            return math.sqrt(np.linalg.det(cp.asnumpy(self.params)))
        elif self.dimensionality == 3:
            return math.pow(np.linalg.det(cp.asnumpy(self.params)), 1 / 3)
        else:
            raise NotImplementedError(
                'Scale is only implemented for 2D and 3D.')


class PolynomialTransform(GeometricTransform):
    """2D polynomial transformation.

    Has the following form::

        X = sum[j=0:order]( sum[i=0:j]( a_ji * x**(j - i) * y**i ))
        Y = sum[j=0:order]( sum[i=0:j]( b_ji * x**(j - i) * y**i ))

    Parameters
    ----------
    params : (2, N) ndarray, optional
        Polynomial coefficients where `N * 2 = (order + 1) * (order + 2)`. So,
        a_ji is defined in `params[0, :]` and b_ji in `params[1, :]`.

    Attributes
    ----------
    params : (2, N) ndarray
        Polynomial coefficients where `N * 2 = (order + 1) * (order + 2)`. So,
        a_ji is defined in `params[0, :]` and b_ji in `params[1, :]`.

    """

    def __init__(self, params=None, *, dimensionality=2, xp=cp):
        if dimensionality != 2:
            raise NotImplementedError(
                "Polynomial transforms are only implemented for 2D."
            )
        if params is None:
            # default to transformation which preserves original coordinates
            params = xp.asarray(np.array([[0, 1, 0], [0, 0, 1]]))
        if params.shape[0] != 2:
            raise ValueError("invalid shape of transformation parameters")
        self.params = params

    def estimate(self, src, dst, order=2, weights=None):
        """Estimate the transformation from a set of corresponding points.

        You can determine the over-, well- and under-determined parameters
        with the total least-squares method.

        Number of source and destination coordinates must match.

        The transformation is defined as::

            X = sum[j=0:order]( sum[i=0:j]( a_ji * x**(j - i) * y**i ))
            Y = sum[j=0:order]( sum[i=0:j]( b_ji * x**(j - i) * y**i ))

        These equations can be transformed to the following form::

            0 = sum[j=0:order]( sum[i=0:j]( a_ji * x**(j - i) * y**i )) - X
            0 = sum[j=0:order]( sum[i=0:j]( b_ji * x**(j - i) * y**i )) - Y

        which exist for each set of corresponding points, so we have a set of
        N * 2 equations. The coefficients appear linearly so we can write
        A x = 0, where::

            A   = [[1 x y x**2 x*y y**2 ... 0 ...             0 -X]
                   [0 ...                 0 1 x y x**2 x*y y**2 -Y]
                    ...
                    ...
                  ]
            x.T = [a00 a10 a11 a20 a21 a22 ... ann
                   b00 b10 b11 b20 b21 b22 ... bnn c3]

        In case of total least-squares the solution of this homogeneous system
        of equations is the right singular vector of A which corresponds to the
        smallest singular value normed by the coefficient c3.

        Weights can be applied to each pair of corresponding points to
        indicate, particularly in an overdetermined system, if point pairs have
        higher or lower confidence or uncertainties associated with them. From
        the matrix treatment of least squares problems, these weight values are
        normalised, square-rooted, then built into a diagonal matrix, by which
        A is multiplied.

        Parameters
        ----------
        src : (N, 2) ndarray
            Source coordinates.
        dst : (N, 2) ndarray
            Destination coordinates.
        order : int, optional
            Polynomial order (number of coefficients is order + 1).
        weights : (N,) ndarray, optional
            Relative weight values for each pair of points.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.

        """
        xp = cp.get_array_module(src)
        xs = src[:, 0]
        ys = src[:, 1]
        xd = dst[:, 0]
        yd = dst[:, 1]
        rows = src.shape[0]

        # number of unknown polynomial coefficients
        order = safe_as_int(order)
        u = (order + 1) * (order + 2)

        A = xp.zeros((rows * 2, u + 1))
        pidx = 0
        for j in range(order + 1):
            for i in range(j + 1):
                A[:rows, pidx] = xs ** (j - i) * ys ** i
                A[rows:, pidx + u // 2] = xs ** (j - i) * ys ** i
                pidx += 1

        A[:rows, -1] = xd
        A[rows:, -1] = yd

        # Get the vectors that correspond to singular values, also applying
        # the weighting if provided
        if weights is None:
            _, _, V = xp.linalg.svd(A)
        else:
            W = xp.diag(xp.tile(xp.sqrt(weights / xp.max(weights)), 2))
            _, _, V = xp.linalg.svd(W @ A)

        # solution is right singular vector that corresponds to smallest
        # singular value
        params = -V[-1, :-1] / V[-1, -1]

        self.params = params.reshape((2, u // 2))

        return True

    def __call__(self, coords):
        """Apply forward transformation.

        Parameters
        ----------
        coords : (N, 2) ndarray
            source coordinates

        Returns
        -------
        coords : (N, 2) ndarray
            Transformed coordinates.

        """
        x = coords[:, 0]
        y = coords[:, 1]
        xp = cp.get_array_module(coords)
        u = len(self.params.ravel())
        # number of coefficients -> u = (order + 1) * (order + 2)
        order = int((-3 + math.sqrt(9 - 4 * (2 - u))) / 2)
        dst = xp.zeros(coords.shape)

        pidx = 0
        for j in range(order + 1):
            for i in range(j + 1):
                dst[:, 0] += self.params[0, pidx] * x ** (j - i) * y ** i
                dst[:, 1] += self.params[1, pidx] * x ** (j - i) * y ** i
                pidx += 1

        return dst

    def inverse(self, coords):
        raise Exception(
            'There is no explicit way to do the inverse polynomial '
            'transformation. Instead, estimate the inverse transformation '
            'parameters by exchanging source and destination coordinates,'
            'then apply the forward transformation.')


TRANSFORMS = {
    'euclidean': EuclideanTransform,
    'similarity': SimilarityTransform,
    'affine': AffineTransform,
    'piecewise-affine': PiecewiseAffineTransform,
    'projective': ProjectiveTransform,
    'fundamental': FundamentalMatrixTransform,
    'essential': EssentialMatrixTransform,
    'polynomial': PolynomialTransform,
}


def estimate_transform(ttype, src, dst, *args, **kwargs):
    """Estimate 2D geometric transformation parameters.

    You can determine the over-, well- and under-determined parameters
    with the total least-squares method.

    Number of source and destination coordinates must match.

    Parameters
    ----------
    ttype : {'euclidean', similarity', 'affine', 'piecewise-affine', \
             'projective', 'polynomial'}
        Type of transform.
    kwargs : ndarray or int
        Function parameters (src, dst, n, angle)::

            NAME / TTYPE        FUNCTION PARAMETERS
            'euclidean'         `src, `dst`
            'similarity'        `src, `dst`
            'affine'            `src, `dst`
            'piecewise-affine'  `src, `dst`
            'projective'        `src, `dst`
            'polynomial'        `src, `dst`, `order` (polynomial order,
                                                      default order is 2)

        Also see examples below.

    Returns
    -------
    tform : :class:`GeometricTransform`
        Transform object containing the transformation parameters and providing
        access to forward and inverse transformation functions.

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage import transform

    >>> # estimate transformation parameters
    >>> src = cp.array([0, 0, 10, 10]).reshape((2, 2))
    >>> dst = cp.array([12, 14, 1, -20]).reshape((2, 2))

    >>> tform = transform.estimate_transform('similarity', src, dst)

    >>> cp.allclose(tform.inverse(tform(src)), src)
    array(True)

    >>> # warp image using the estimated transformation
    >>> from skimage import data
    >>> image = cp.array(data.camera())

    >>> transform.warp(image, inverse_map=tform.inverse) # doctest: +SKIP

    >>> # create transformation with explicit parameters
    >>> tform2 = transform.SimilarityTransform(scale=1.1, rotation=1,
    ...     translation=(10, 20))

    >>> # unite transformations, applied in order from left to right
    >>> tform3 = tform + tform2
    >>> cp.allclose(tform3(src), tform2(tform(src)))
    array(True)

    """
    ttype = ttype.lower()
    if ttype not in TRANSFORMS:
        raise ValueError('the transformation type \'%s\' is not'
                         'implemented' % ttype)

    tform = TRANSFORMS[ttype](dimensionality=src.shape[1])
    tform.estimate(src, dst, *args, **kwargs)

    return tform


def matrix_transform(coords, matrix):
    """Apply 2D matrix transform.

    Parameters
    ----------
    coords : (N, 2) ndarray
        x, y coordinates to transform
    matrix : (3, 3) ndarray
        Homogeneous transformation matrix.

    Returns
    -------
    coords : (N, 2) ndarray
        Transformed coordinates.

    """
    return ProjectiveTransform(matrix)(coords)
