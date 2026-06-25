# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import math
import textwrap
import warnings

import cupy as cp
import numpy as np
from scipy import spatial

from .._shared.compat import NP_COPY_IF_NEEDED
from .._shared.utils import (
    FailedEstimation,
    _deprecate_estimate,
    _deprecate_inherited_estimate,
    _update_from_estimate_docstring,
    get_bound_method_class,
    safe_as_int,
)

_sin, _cos = math.sin, math.cos


def _affine_matrix_from_vector(v):
    """Affine matrix from linearized (d, d + 1) matrix entries."""
    nparam = v.size
    # solve for d in: d * (d + 1) = nparam
    d = (1 + math.sqrt(1 + 4 * nparam)) / 2 - 1
    dimensionality = round(d)  # round to prevent approx errors
    if d != dimensionality:
        raise ValueError(
            f"Invalid number of elements for linearized matrix: {nparam}"
        )
    matrix = np.eye(dimensionality + 1)
    matrix[:-1, :] = np.reshape(v, (dimensionality, dimensionality + 1))
    return matrix


def _calc_center_normalize(points, scaling="rms"):
    """Calculate transformation matrix to center and normalize points.

    Points are centered at their centroid, then scaled according to `scaling`.

    Parameters
    ----------
    points : (N, D) ndarray
        The coordinates of the image points.
    scaling : {'rms', 'mrs', 'raw'}, optional
        Scaling mode. 'raw' performs no centering/scaling.

    Returns
    -------
    matrix : (D+1, D+1) ndarray
        The homogeneous transformation matrix.

    References
    ----------
    .. [1] Hartley, Richard I. "In defense of the eight-point algorithm."
           Pattern Analysis and Machine Intelligence, IEEE Transactions on 19.6
           (1997): 580-593.

    """
    xp = cp.get_array_module(points)
    n, d = points.shape
    scaling = scaling.lower()

    if scaling == "raw":
        return xp.eye(d + 1)

    centroid = xp.mean(points, axis=0)
    diff = points - centroid
    if scaling == "rms":
        divisor = xp.sqrt(xp.mean(diff * diff))
    elif scaling == "mrs":
        divisor = xp.mean(xp.sqrt(xp.sum(diff * diff, axis=1))) / math.sqrt(d)
    else:
        raise ValueError(f'Unexpected "scaling" of "{scaling}"')

    # if all the points are the same, the transformation matrix cannot be
    # created. We return an equivalent matrix with np.nans as sentinel values.
    # This obviates the need for try/except blocks in functions calling this
    # one, and those are only needed when actual 0 is reached, rather than some
    # small value; ie, we don't need to worry about numerical stability here,
    # only actual 0.
    if divisor == 0:
        return xp.full((d + 1, d + 1), xp.nan)
    matrix = xp.eye(d + 1)
    matrix[:d, d] = -centroid
    matrix[:d, :] /= divisor
    return matrix


def _center_and_normalize_points(points, scaling="rms"):
    """Convenience function to compute and apply center/normalize."""
    xp = cp.get_array_module(points)
    matrix = _calc_center_normalize(points, scaling=scaling)
    if not xp.all(xp.isfinite(matrix)):
        return matrix + xp.nan, xp.full_like(points, xp.nan)
    return matrix, _apply_homogeneous(matrix, points)


def _append_homogeneous_dim(points):
    xp = cp.get_array_module(points)
    points = xp.asarray(points)
    return xp.concatenate(
        (points, xp.ones((len(points), 1), dtype=points.dtype)), axis=1
    )


def _apply_homogeneous(matrix, points):
    xp = cp.get_array_module(points)
    copy_arr = NP_COPY_IF_NEEDED if xp is np else False
    points = xp.array(points, copy=copy_arr, ndmin=2)
    points_h = _append_homogeneous_dim(points)
    new_points_h = points_h @ matrix.T
    divs = new_points_h[:, -1]
    eps = np.finfo(float).eps
    divs = xp.where(divs == 0, eps, divs)
    return new_points_h[:, :-1] / divs[:, None]


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

    # Eq. (40) and (43). Rank calculation from SVD.
    tol = S.max() * max(A.shape) * np.finfo(float).eps
    rank = xp.count_nonzero(S > tol)
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
    """Base class for geometric transformations."""

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
        residuals : (N,) ndarray
            Residual for coordinate.

        """
        xp = cp.get_array_module(src)
        return xp.sqrt(xp.sum((self(src) - dst) ** 2, axis=1))

    def __add__(self, other):
        """Combine this transformation with another."""
        raise NotImplementedError()

    @classmethod
    def identity(cls, dimensionality=None, xp=cp):
        d = 2 if dimensionality is None else dimensionality
        return cls(dimensionality=d, xp=xp)

    @classmethod
    def from_estimate(cls, src, dst, *args, **kwargs):
        """Estimate transform from matching source/destination coordinates."""
        return _from_estimate(cls, src, dst, *args, **kwargs)


def _from_estimate(cls, src, dst, *args, **kwargs):
    xp = cp.get_array_module(src)
    tf = cls.identity(src.shape[1], xp=xp)
    estimate_func = getattr(tf, "_estimate", None)
    if estimate_func is None:
        result = tf.estimate(src, dst, *args, **kwargs)
        msg = None if result else "Estimation failed"
    else:
        msg = estimate_func(src, dst, *args, **kwargs)
        # Backward compatibility if an _estimate still returns bool.
        if msg is True:
            msg = None
        elif msg is False:
            msg = "Estimation failed"
    return tf if msg is None else FailedEstimation(f"{cls.__name__}: {msg}")


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

    Examples
    --------
    >>> import numpy as np
    >>> import cucim.skimage as ski
    >>> tform_matrix = ski.transform.FundamentalMatrixTransform()

    Define source and destination points:

    >>> src = np.array([1.839035, 1.924743,
    ...                 0.543582, 0.375221,
    ...                 0.473240, 0.142522,
    ...                 0.964910, 0.598376,
    ...                 0.102388, 0.140092,
    ...                15.994343, 9.622164,
    ...                 0.285901, 0.430055,
    ...                 0.091150, 0.254594]).reshape(-1, 2)
    >>> dst = np.array([1.002114, 1.129644,
    ...                 1.521742, 1.846002,
    ...                 1.084332, 0.275134,
    ...                 0.293328, 0.588992,
    ...                 0.839509, 0.087290,
    ...                 1.779735, 1.116857,
    ...                 0.878616, 0.602447,
    ...                 0.642616, 1.028681]).reshape(-1, 2)

    Estimate the transformation matrix:

    >>> tform_matrix.estimate(src, dst)
    True
    >>> tform_matrix.params
    array([[-0.21785884,  0.41928191, -0.03430748],
           [-0.07179414,  0.04516432,  0.02160726],
           [ 0.24806211, -0.42947814,  0.02210191]])

    Compute the Sampson distance:

    >>> tform_matrix.residuals(src, dst)
    array([0.0053886 , 0.00526101, 0.08689701, 0.01850534, 0.09418259,
           0.00185967, 0.06160489, 0.02655136])

    Apply inverse transformation:

    >>> tform_matrix.inverse(dst)
    array([[-0.0513591 ,  0.04170974,  0.01213043],
           [-0.21599496,  0.29193419,  0.00978184],
           [-0.0079222 ,  0.03758889, -0.00915389],
           [ 0.14187184, -0.27988959,  0.02476507],
           [ 0.05890075, -0.07354481, -0.00481342],
           [-0.21985267,  0.36717464, -0.01482408],
           [ 0.01339569, -0.03388123,  0.00497605],
           [ 0.03420927, -0.1135812 ,  0.02228236]])
    """

    # CuPy Backend: if matrix is None cannot infer array module from it
    #               added explicit xp module argument for now
    scaling = "rms"

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
        return _append_homogeneous_dim(coords) @ self.params.T

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
        return _append_homogeneous_dim(coords) @ self.params

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

        src_matrix = _calc_center_normalize(src, self.scaling)
        dst_matrix = _calc_center_normalize(dst, self.scaling)
        if xp.any(xp.isnan(src_matrix + dst_matrix)):
            self.params = xp.full((3, 3), xp.nan)
            return 3 * [xp.full((3, 3), xp.nan)]
        src_h = _append_homogeneous_dim(_apply_homogeneous(src_matrix, src))
        dst_h = _append_homogeneous_dim(_apply_homogeneous(dst_matrix, dst))

        # Setup homogeneous linear equation as dst' * F * src = 0.
        cols = [(d_v * s_v) for d_v in dst_h.T for s_v in src_h.T]
        A = xp.stack(cols, axis=1)

        # Solve for the nullspace of the constraint matrix.
        _, _, V = xp.linalg.svd(A)
        F_normalized = V[-1, :].reshape(3, 3)

        return F_normalized, src_matrix, dst_matrix

    @classmethod
    def from_estimate(cls, src, dst):
        """Estimate fundamental matrix using the 8-point algorithm."""
        return super().from_estimate(src, dst)

    def _estimate(self, src, dst):
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
        F_normalized, src_matrix, dst_matrix = self._setup_constraint_matrix(
            src, dst
        )
        xp = cp.get_array_module(F_normalized)
        if xp.any(xp.isnan(F_normalized + src_matrix + dst_matrix)):
            return "Scaling failed for input points"

        # Enforcing the internal constraint that two singular values must be
        # non-zero and one must be zero (rank 2).
        U, S, V = xp.linalg.svd(F_normalized)
        S[2] = 0
        F = U @ xp.diag(S) @ V

        self.params = dst_matrix.T @ F @ src_matrix

        return None

    @_deprecate_estimate
    def estimate(self, src, dst):
        """Backward-compatible estimate method."""
        return self._estimate(src, dst) is None

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
        residuals : (N,) ndarray
            Sampson distance.

        """
        xp = cp.get_array_module(src)
        src_homogeneous = _append_homogeneous_dim(src)
        dst_homogeneous = _append_homogeneous_dim(dst)

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

    _rot_det_tol = 1e-6
    _trans_len_tol = 1e-6

    def __init__(
        self,
        *,
        rotation=None,
        translation=None,
        matrix=None,
        dimensionality=2,
        xp=cp,
    ):
        n_rt_none = sum(p is None for p in (rotation, translation))
        if n_rt_none == 1:
            raise ValueError(
                "Both rotation and translation required when one is specified."
            )
        if n_rt_none == 0:
            if matrix is not None:
                raise ValueError(
                    "Do not specify rotation or translation when matrix is specified."
                )
            matrix = self._rt2matrix(rotation, translation, xp=xp)
        super().__init__(matrix=matrix, dimensionality=dimensionality, xp=xp)

    def _rt2matrix(self, rotation, translation, xp=cp):
        # Compute small matrix on CPU then transfer to GPU as needed.
        if isinstance(rotation, cp.ndarray):
            rotation = cp.asnumpy(rotation)
        else:
            rotation = np.asarray(rotation)
        if isinstance(translation, cp.ndarray):
            translation = cp.asnumpy(translation)
        else:
            translation = np.asarray(translation)
        if rotation.shape != (3, 3):
            raise ValueError("Invalid shape of rotation matrix")
        if abs(np.linalg.det(rotation) - 1) > self._rot_det_tol:
            raise ValueError("Rotation matrix must have unit determinant")
        if translation.size != 3:
            raise ValueError("Invalid shape of translation vector")
        if abs(np.linalg.norm(translation) - 1) > self._trans_len_tol:
            raise ValueError("Translation vector must have unit length")
        t0, t1, t2 = [float(translation[i]) for i in range(3)]
        t_x = np.array([[0, -t2, t1], [t2, 0, -t0], [-t1, t0, 0]], dtype=float)
        return xp.asarray(t_x @ rotation)

    @classmethod
    def from_estimate(cls, src, dst):
        """Estimate essential matrix using the 8-point algorithm."""
        return super().from_estimate(src, dst)

    def _estimate(self, src, dst):
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

        E_normalized, src_matrix, dst_matrix = self._setup_constraint_matrix(
            src, dst
        )
        xp = cp.get_array_module(E_normalized)
        if xp.any(xp.isnan(E_normalized + src_matrix + dst_matrix)):
            return "Scaling failed for input points"

        # Enforcing the internal constraint that two singular values must be
        # equal and one must be zero.
        U, S, V = xp.linalg.svd(E_normalized)
        S[0] = (S[0] + S[1]) / 2.0
        S[1] = S[0]
        S[2] = 0
        E = U @ xp.diag(S) @ V

        self.params = dst_matrix.T @ E @ src_matrix

        return None

    @_deprecate_estimate
    def estimate(self, src, dst):
        """Backward-compatible estimate method."""
        return self._estimate(src, dst) is None


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

    scaling = "rms"

    def __init__(self, matrix=None, *, dimensionality=2, xp=cp):
        if matrix is None:
            # default to an identity transform
            matrix = xp.eye(dimensionality + 1)
        else:
            dimensionality = matrix.shape[0] - 1
            if matrix.shape != (dimensionality + 1, dimensionality + 1):
                raise ValueError("invalid shape of transformation matrix")
        self.params = matrix

    @property
    def _coeff_inds(self):
        return range(self.params.size - 1)

    @property
    def _inv_matrix(self):
        xp = cp.get_array_module(self.params)
        return xp.linalg.inv(self.params)

    def __array__(self, dtype=None, copy=None):
        """Return the transform parameters as an array.

        Note, __array__ is not currently supported by CuPy
        """
        return self.params if dtype is None else self.params.astype(dtype)

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
        return _apply_homogeneous(self.params, coords)

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
        return _apply_homogeneous(self._inv_matrix, coords)

    @classmethod
    def from_estimate(cls, src, dst, weights=None):
        """Estimate projective transform from corresponding points."""
        return super().from_estimate(src, dst, weights)

    @_deprecate_estimate
    def estimate(self, src, dst, weights=None):
        return self._estimate(src, dst, weights) is None

    def _estimate(self, src, dst, weights=None):
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
        src_matrix, src = _center_and_normalize_points(
            src, scaling=self.scaling
        )
        dst_matrix, dst = _center_and_normalize_points(
            dst, scaling=self.scaling
        )
        if not xp.all(xp.isfinite(src_matrix + dst_matrix)):
            self.params = xp.full((d + 1, d + 1), xp.nan)
            return "Scaling generated NaN values"
        # params: a0, a1, a2, b0, b1, b2, c0, c1
        A = xp.zeros((n * d, (d + 1) ** 2))
        for ddim in range(d):
            A[
                ddim * n : (ddim + 1) * n, ddim * (d + 1) : ddim * (d + 1) + d
            ] = src
            A[ddim * n : (ddim + 1) * n, ddim * (d + 1) + d] = 1
            A[ddim * n : (ddim + 1) * n, -d - 1 : -1] = src
            A[ddim * n : (ddim + 1) * n, -1] = -1
            A[ddim * n : (ddim + 1) * n, -d - 1 :] *= -dst[:, ddim : (ddim + 1)]

        # Select relevant columns, depending on params
        A = A[:, list(self._coeff_inds) + [-1]]

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
            return "Right singular vector has 0 final element"

        H = np.zeros((d + 1, d + 1))
        # solution is right singular vector that corresponds to smallest
        # singular value
        V_cpu = cp.asnumpy(V) if isinstance(V, cp.ndarray) else V
        H.flat[list(self._coeff_inds) + [-1]] = -V_cpu[-1, :-1] / V_cpu[-1, -1]
        H[d, d] = 1
        H = xp.asarray(H)

        # De-center and de-normalize
        H = xp.linalg.inv(dst_matrix) @ H @ src_matrix

        # Small errors can creep in if points are not exact, causing the last
        # element of H to deviate from unity. Correct for that here.
        H /= H[-1, -1]

        self.params = H

        return None

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
        npstring = np.array2string(cp.asnumpy(self.params), separator=", ")
        paramstr = "matrix=\n" + textwrap.indent(npstring, "    ")
        return paramstr

    def __repr__(self):
        """Add standard repr formatting around a __nice__ string"""
        paramstr = self.__nice__()
        classname = self.__class__.__name__
        classstr = classname
        return f"<{classstr}({paramstr}) at {hex(id(self))}>"

    def __str__(self):
        """Add standard str formatting around a __nice__ string"""
        paramstr = self.__nice__()
        classname = self.__class__.__name__
        classstr = classname
        return f"<{classstr}({paramstr})>"

    @property
    def dimensionality(self):
        """The dimensionality of the transformation."""
        return self.params.shape[0] - 1


@_update_from_estimate_docstring
@_deprecate_inherited_estimate
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

    def __init__(
        self,
        matrix=None,
        *,
        scale=None,
        rotation=None,
        shear=None,
        translation=None,
        dimensionality=None,
        xp=cp,
    ):
        n_srst_none = sum(
            p is None for p in (scale, rotation, shear, translation)
        )
        if n_srst_none != 4:
            if matrix is not None:
                raise ValueError(
                    "Do not specify implicit parameters when matrix is specified."
                )
            if dimensionality is not None and dimensionality > 2:
                raise ValueError(
                    "Implicit parameters only valid for 2D transforms."
                )
            matrix = self._srst2matrix(
                scale, rotation, shear, translation, xp=xp
            )
        if dimensionality is None:
            dimensionality = 2 if matrix is None else matrix.shape[0] - 1
        super().__init__(matrix=matrix, dimensionality=dimensionality, xp=xp)

    @property
    def _coeff_inds(self):
        return range(self.dimensionality * (self.dimensionality + 1))

    def _srst2matrix(self, scale, rotation, shear, translation, xp=cp):
        scale = (1, 1) if scale is None else scale
        sx, sy = (scale, scale) if np.isscalar(scale) else scale
        rotation = 0 if rotation is None else rotation
        if not np.isscalar(rotation):
            raise ValueError("rotation must be scalar for 2D transforms.")
        shear = 0 if shear is None else shear
        shear_x, shear_y = (shear, 0) if np.isscalar(shear) else shear
        translation = (0, 0) if translation is None else translation
        if np.isscalar(translation):
            raise ValueError("translation must be length 2.")
        a2, b2 = translation

        a0 = sx * (math.cos(rotation) + math.tan(shear_y) * math.sin(rotation))
        a1 = -sy * (math.tan(shear_x) * math.cos(rotation) + math.sin(rotation))
        b0 = sx * (math.sin(rotation) - math.tan(shear_y) * math.cos(rotation))
        b1 = -sy * (math.tan(shear_x) * math.sin(rotation) - math.cos(rotation))
        return xp.array([[a0, a1, a2], [b0, b1, b2], [0, 0, 1]])

    @property
    def scale(self):
        xp = cp.get_array_module(self.params)
        if self.dimensionality != 2:
            return xp.sqrt(xp.sum(self.params * self.params, axis=0))[
                : self.dimensionality
            ]
        ss = xp.sum(self.params * self.params, axis=0)
        ss[1] = ss[1] / (math.tan(self.shear) ** 2 + 1)
        return xp.sqrt(ss)[: self.dimensionality]

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
        return self.params[0 : self.dimensionality, self.dimensionality]


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

    @classmethod
    def from_estimate(cls, src, dst):
        """Estimate piecewise affine transform from corresponding points."""
        return super().from_estimate(src, dst)

    @_deprecate_estimate
    def estimate(self, src, dst):
        return self._estimate(src, dst) is None

    def _estimate(self, src, dst):
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

        _, D = src.shape
        # forward piecewise affine
        # triangulate input positions into mesh
        xp = cp.get_array_module(src)
        # TODO: update if spatial.Delaunay become available in CuPy
        self._tesselation = spatial.Delaunay(cp.asnumpy(src))

        fail_matrix = xp.full((D + 1, D + 1), xp.nan)
        messages = []

        # find affine mapping from source positions to destination
        self.affines = []

        try:
            tesselation_simplices = self._tesselation.simplices
        except AttributeError:
            # vertices is deprecated and scheduled for removal in SciPy 1.11
            tesselation_simplices = self._tesselation.vertices
        for tri in xp.asarray(tesselation_simplices):
            affine = AffineTransform.from_estimate(src[tri, :], dst[tri, :])
            if not affine:
                messages.append(
                    f"Failure at forward simplex {len(self.affines)}: {affine}"
                )
                affine = AffineTransform(fail_matrix.copy())
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
            affine = AffineTransform.from_estimate(dst[tri, :], src[tri, :])
            if not affine:
                messages.append(
                    f"Failure at inverse simplex {len(self.inverse_affines)}: {affine}"
                )
                affine = AffineTransform(fail_matrix.copy())
            self.inverse_affines.append(affine)

        return "; ".join(messages) if messages else None

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

    @classmethod
    def identity(cls, dimensionality=None, xp=cp):
        return cls()


def _euler_rotation_matrix(angles, degrees=False):
    """Produce an Euler rotation matrix from the given intrinsic rotation angles
    for the axes x, y and z.

    Parameters
    ----------
    angles : array of float, shape (3,)
        The transformation angles in radians.
    degrees : bool, optional
        If True, then the given angles are assumed to be in degrees. Default is
        False.

    Returns
    -------
    R : array of float, shape (3, 3)
        The Euler rotation matrix.
    """
    return spatial.transform.Rotation.from_euler(
        "XYZ", angles=cp.asnumpy(angles), degrees=degrees
    ).as_matrix()


class EuclideanTransform(ProjectiveTransform):
    """Euclidean transformation, also known as a rigid transform.

    Has the following form::

        X = a0 * x - b0 * y + a1 =
          = x * cos(rotation) - y * sin(rotation) + a1

        Y = b0 * x + a0 * y + b1 =
          = x * sin(rotation) + y * cos(rotation) + b1

    where the homogeneous transformation matrix is::

        [[a0 -b0  a1]
         [b0  a0  b1]
         [0   0    1]]

    The Euclidean transformation is a rigid transformation with rotation and
    translation parameters. The similarity transformation extends the Euclidean
    transformation with a single scaling factor.

    In 2D and 3D, the transformation parameters may be provided either via
    `matrix`, the homogeneous transformation matrix, above, or via the
    implicit parameters `rotation` and/or `translation` (where `a1` is the
    translation along `x`, `b1` along `y`, etc.). Beyond 3D, if the
    transformation is only a translation, you may use the implicit parameter
    `translation`; otherwise, you must use `matrix`.

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
    translation : (x, y[, z, ...]) sequence of float, length D, optional
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

    _estimate_scale = False

    def __init__(
        self,
        matrix=None,
        *,
        rotation=None,
        translation=None,
        dimensionality=None,
        xp=cp,
    ):
        n_rt_none = sum(p is None for p in (rotation, translation))
        if n_rt_none != 2:
            if matrix is not None:
                raise ValueError(
                    "Do not specify implicit parameters when matrix is specified."
                )
            n_dims, chk_msg = self._rt2ndims_msg(rotation, translation)
            if chk_msg is not None:
                raise ValueError(chk_msg)
            if dimensionality is not None and dimensionality != n_dims:
                raise ValueError(
                    f"Dimensionality {dimensionality} does not match "
                    f"inferred transform dimensionality {n_dims}."
                )
            matrix = self._rt2matrix(rotation, translation, n_dims, xp=xp)
        if dimensionality is None:
            dimensionality = 2 if matrix is None else matrix.shape[0] - 1
        super().__init__(matrix=matrix, dimensionality=dimensionality, xp=xp)

    def _rt2ndims_msg(self, rotation, translation):
        if rotation is not None:
            n = 1 if np.isscalar(rotation) else len(rotation)
            msg = (
                "rotation must be scalar (2D) or length 3 (3D)"
                if n not in (1, 3)
                else None
            )
            return (2 if n == 1 else n), msg
        if translation is not None:
            return (2 if np.isscalar(translation) else len(translation)), None
        return None, None

    def _rt2matrix(self, rotation, translation, n_dims, xp=cp):
        if translation is None:
            translation = (0,) * n_dims
        if rotation is None:
            rotation = 0 if n_dims == 2 else np.zeros(3)
        matrix = xp.eye(n_dims + 1)
        if n_dims == 2:
            cos_r, sin_r = math.cos(rotation), math.sin(rotation)
            matrix[:2, :2] = xp.array([[cos_r, -sin_r], [sin_r, cos_r]])
        elif n_dims == 3:
            matrix[:3, :3] = xp.asarray(_euler_rotation_matrix(rotation))
        matrix[:n_dims, n_dims] = xp.asarray(translation)
        return matrix

    @_deprecate_estimate
    def estimate(self, src, dst):
        return self._estimate(src, dst) is None

    @classmethod
    def from_estimate(cls, src, dst):
        """Estimate Euclidean transform from corresponding points."""
        # Avoid ProjectiveTransform.from_estimate(weights=...) call path.
        return _from_estimate(cls, src, dst)

    def _estimate(self, src, dst):
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

        self.params = _umeyama(src, dst, self._estimate_scale)

        # _umeyama will return nan if the problem is not well-conditioned.
        xp = cp.get_array_module(self.params)
        return (
            "Poor conditioning for estimation"
            if xp.any(xp.isnan(self.params))
            else None
        )

    @property
    def rotation(self):
        if self.dimensionality == 2:
            return math.atan2(self.params[1, 0], self.params[1, 1])
        elif self.dimensionality == 3:
            # Returning 3D Euler rotation matrix
            return self.params[:3, :3]
        else:
            raise NotImplementedError(
                "Rotation only implemented for 2D and 3D transforms."
            )

    @property
    def translation(self):
        return self.params[0 : self.dimensionality, self.dimensionality]


@_update_from_estimate_docstring
@_deprecate_inherited_estimate
class SimilarityTransform(EuclideanTransform):
    """Similarity transformation.

    Has the following form in 2D::

        X = a0 * x - b0 * y + a1 =
          = s * x * cos(rotation) - s * y * sin(rotation) + a1

        Y = b0 * x + a0 * y + b1 =
          = s * x * sin(rotation) + s * y * cos(rotation) + b1

    where ``s`` is a scale factor and the homogeneous transformation matrix is::

        [[a0  -b0  a1]
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

    _estimate_scale = True

    def __init__(
        self,
        matrix=None,
        *,
        scale=None,
        rotation=None,
        translation=None,
        dimensionality=None,
        xp=cp,
    ):
        n_srt_none = sum(p is None for p in (scale, rotation, translation))
        if n_srt_none != 3:
            if matrix is not None:
                raise ValueError(
                    "Do not specify implicit parameters when matrix is specified."
                )
            self._check_scale(scale, (rotation, translation), dimensionality)
            if scale is not None and not np.isscalar(scale):
                n_dims, chk_msg = len(scale), None
            else:
                n_dims, chk_msg = self._rt2ndims_msg(rotation, translation)
            if chk_msg is not None:
                raise ValueError(chk_msg)
            n_dims = (
                n_dims
                if n_dims is not None
                else dimensionality
                if dimensionality is not None
                else 2
            )
            if dimensionality is not None and dimensionality != n_dims:
                raise ValueError(
                    f"Dimensionality {dimensionality} does not match "
                    f"inferred transform dimensionality {n_dims}."
                )
            matrix = self._rt2matrix(rotation, translation, n_dims, xp=xp)
            if scale not in (None, 1):
                matrix[:n_dims, :n_dims] *= xp.asarray(scale)
        if dimensionality is None:
            dimensionality = 2 if matrix is None else matrix.shape[0] - 1
        super().__init__(matrix=matrix, dimensionality=dimensionality, xp=xp)

    def _check_scale(self, scale, other_params, dimensionality):
        if (
            dimensionality in (None, 2)
            or scale is None
            or not np.isscalar(scale)
        ):
            return
        if all(p is None for p in other_params):
            warnings.warn(
                "Passing scalar `scale` with dimensionality > 2 and no other "
                "implicit parameters is deprecated.",
                FutureWarning,
                stacklevel=2,
            )

    @property
    def scale(self):
        # det = scale**(# of dimensions), therefore scale = det**(1/2)
        if self.dimensionality == 2:
            return math.sqrt(np.linalg.det(cp.asnumpy(self.params)))
        elif self.dimensionality == 3:
            return math.pow(np.linalg.det(cp.asnumpy(self.params)), 1 / 3)
        else:
            raise NotImplementedError(
                "Scale is only implemented for 2D and 3D."
            )


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

    def __init__(self, params=None, *, dimensionality=None, xp=cp):
        if dimensionality is None:
            dimensionality = 2
        elif dimensionality != 2:
            raise NotImplementedError(
                "Polynomial transforms are only implemented for 2D."
            )
        self.params = (
            xp.asarray([[0, 1, 0], [0, 0, 1]])
            if params is None
            else xp.asarray(params)
        )
        if self.params.shape == () or self.params.shape[0] != 2:
            raise ValueError("Transformation parameters must have shape (2, N)")

    @classmethod
    def from_estimate(cls, src, dst, order=2, weights=None):
        """Estimate polynomial transform from corresponding points."""
        return super().from_estimate(src, dst, order, weights)

    @_deprecate_estimate
    def estimate(self, src, dst, order=2, weights=None):
        return self._estimate(src, dst, order=order, weights=weights) is None

    def _estimate(self, src, dst, order=2, weights=None):
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
                A[:rows, pidx] = xs ** (j - i) * ys**i
                A[rows:, pidx + u // 2] = xs ** (j - i) * ys**i
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

        return None

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
                dst[:, 0] += self.params[0, pidx] * x ** (j - i) * y**i
                dst[:, 1] += self.params[1, pidx] * x ** (j - i) * y**i
                pidx += 1

        return dst

    def inverse(self, coords):
        raise Exception(
            "There is no explicit way to do the inverse polynomial "
            "transformation. Instead, estimate the inverse transformation "
            "parameters by exchanging source and destination coordinates,"
            "then apply the forward transformation."
        )

    @classmethod
    def identity(cls, dimensionality=None, xp=cp):
        return cls(params=None, dimensionality=dimensionality, xp=xp)


TRANSFORMS = {
    "euclidean": EuclideanTransform,
    "similarity": SimilarityTransform,
    "affine": AffineTransform,
    "piecewise-affine": PiecewiseAffineTransform,
    "projective": ProjectiveTransform,
    "fundamental": FundamentalMatrixTransform,
    "essential": EssentialMatrixTransform,
    "polynomial": PolynomialTransform,
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
        raise ValueError(f"the transformation type '{ttype}' is notimplemented")

    return TRANSFORMS[ttype].from_estimate(src, dst, *args, **kwargs)


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
