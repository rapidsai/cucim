import cupy as cp
import numpy as np
from cupyx import rsqrt  # reciprocal sqrt

from cucim.core.operations.morphology import distance_transform_edt

from .._shared.utils import _supported_float_type, deprecate_kwarg


@cp.fuse()
def _fused_curvature(phi, x_start, x_end, y_start, y_end, ul, ur, ll, lr):
    fy = (y_end - y_start) / 2.0
    fx = (x_end - x_start) / 2.0
    fyy = y_end + y_start - 2 * phi
    fxx = x_end + x_start - 2 * phi
    fxy = .25 * (lr + ul - ur - ll)
    grad2 = fx**2 + fy**2
    K = (fxx * fy**2 - 2 * fxy * fx * fy + fyy * fx**2)
    K /= (grad2 * cp.sqrt(grad2) + 1e-8)
    return K


def _cv_curvature(phi):
    """Returns the 'curvature' of a level set 'phi'.
    """
    P = cp.pad(phi, 1, mode='edge')
    y_start = P[:-2, 1:-1]
    y_end = P[2:, 1:-1]
    x_start = P[1:-1, :-2]
    x_end = P[1:-1, 2:]

    lower_right = P[2:, 2:]
    lower_left = P[2:, :-2]
    upper_right = P[:-2, 2:]
    upper_left = P[:-2, :-2]
    K = _fused_curvature(phi, x_start, x_end, y_start, y_end, upper_left,
                         upper_right, lower_left, lower_right)
    return K


@cp.fuse()
def _fused_variance_kernel1(eta, x_start, x_mid, x_end, y_start, y_mid, y_end):
    phixp = x_end - x_mid
    phixn = x_mid - x_start
    phix0 = x_end - x_start
    phix0 /= 2.0
    phixp *= phixp
    phixn *= phixn
    phix0 *= phix0

    phiyp = y_end - y_mid
    phiyn = y_mid - y_start
    phiy0 = y_end - y_start
    phiy0 /= 2.0
    phiyp *= phiyp
    phiyn *= phiyn
    phiy0 *= phiy0

    C1 = rsqrt(eta + phixp + phiy0)
    C2 = rsqrt(eta + phixn + phiy0)
    C3 = rsqrt(eta + phix0 + phiyp)
    C4 = rsqrt(eta + phix0 + phiyn)

    K = x_end * C1
    K += x_start * C2
    K += y_end * C3
    K += y_start * C4

    Csum = C1
    Csum += C2
    Csum += C3
    Csum += C4

    return K, Csum


@cp.fuse()
def _fused_hphi_hinv(phi):
    Hphi = (phi > 0).astype(phi.dtype)
    Hinv = 1.0 - Hphi
    return Hphi, Hinv


@cp.fuse()
def _fused_variance_kernel2(
    image, c1, c2, lam1, lam2, phi, K, dt, mu, delta_phi, Csum
):
    difference_term = image - c1
    difference_term *= difference_term
    difference_term *= -lam1

    term2 = image - c2
    term2 *= term2
    term2 *= lam2
    difference_term += term2

    new_phi = phi + (dt * delta_phi) * (mu * K + difference_term)
    out = new_phi / (1 + mu * dt * delta_phi * Csum)
    return out


def _cv_calculate_variation(image, phi, mu, lambda1, lambda2, dt):
    """Returns the variation of level set 'phi' based on algorithm parameters.
    """
    eta = 1e-16
    P = cp.pad(phi, 1, mode='edge')

    x_end = P[1:-1, 2:]
    x_mid = P[1:-1, 1:-1]
    x_start = P[1:-1, :-2]

    y_end = P[2:, 1:-1]
    y_mid = P[1:-1, 1:-1]
    y_start = P[:-2, 1:-1]

    K, Csum = _fused_variance_kernel1(
        eta, x_start, x_mid, x_end, y_start, y_mid, y_end
    )
    Hphi, Hinv = _fused_hphi_hinv(phi)
    c1, c2 = _cv_calculate_averages(image, Hphi, Hinv)
    delta_phi = _cv_delta(phi)
    out = _fused_variance_kernel2(
        image, c1, c2, lambda1, lambda2, phi, K, dt, mu, delta_phi, Csum
    )
    return out


@cp.fuse()
def _cv_heavyside(x, eps=1.):
    """Returns the result of a regularised heavyside function of the
    input value(s).
    """
    return 0.5 * (1. + (2. / cp.pi) * cp.arctan(x / eps))


@cp.fuse()
def _cv_delta(x, eps=1.):
    """Returns the result of a regularised dirac function of the
    input value(s).
    """
    return eps / (eps * eps + x * x)


@cp.fuse()
def _fused_inplace_eps_div(num, denom, eps):
    denom += eps
    num /= denom
    return


def _cv_calculate_averages(image, H, Hinv):
    """Returns the average values 'inside' and 'outside'.
    """
    Hsum = cp.sum(H)
    Hinvsum = cp.sum(Hinv)
    avg_inside = cp.sum(image * H)
    avg_oustide = cp.sum(image * Hinv)

    eps = 10 * cp.finfo(image.dtype).eps
    _fused_inplace_eps_div(avg_inside, Hsum, eps)
    _fused_inplace_eps_div(avg_oustide, Hinvsum, eps)
    return (avg_inside, avg_oustide)


@cp.fuse()
def _fused_difference_op1(image, c, h, lam):
    out = image - c
    out *= out
    out *= h
    out *= lam
    return out


def _cv_difference_from_average_term(image, Hphi, lambda_pos, lambda_neg):
    """Returns the 'energy' contribution due to the difference from
    the average value within a region at each point.
    """
    Hinv = 1. - Hphi
    (c1, c2) = _cv_calculate_averages(image, Hphi, Hinv)
    out = _fused_difference_op1(image, c1, Hphi, lambda_pos)
    out += _fused_difference_op1(image, c2, Hinv, lambda_neg)
    return out


def _cv_edge_length_term(phi, mu):
    """Returns the 'energy' contribution due to the length of the
    edge between regions at each point, multiplied by a factor 'mu'.
    """
    e = _cv_curvature(phi)
    e *= mu
    return e


def _cv_energy(image, phi, mu, lambda1, lambda2):
    """Returns the total 'energy' of the current level set function.
    """
    H = _cv_heavyside(phi)
    avgenergy = _cv_difference_from_average_term(image, H, lambda1, lambda2)
    lenenergy = _cv_edge_length_term(phi, mu)
    return cp.sum(avgenergy) + cp.sum(lenenergy)


def _cv_checkerboard(image_size, square_size, dtype=cp.float64):
    """Generates a checkerboard level set function.

    According to Pascal Getreuer, such a level set function has fast
    convergence.
    """
    yv = cp.arange(image_size[0], dtype=dtype)[:, np.newaxis]
    xv = cp.arange(image_size[1], dtype=dtype)[np.newaxis, :]
    sf = cp.pi / square_size
    xv *= sf
    yv *= sf
    cp.sin(xv, out=xv)
    cp.sin(yv, out=yv)
    return xv * yv


def _cv_large_disk(image_size):
    """Generates a disk level set function.

    The disk covers the whole image along its smallest dimension.
    """
    res = cp.ones(image_size, dtype=bool)
    centerY = int((image_size[0] - 1) / 2)
    centerX = int((image_size[1] - 1) / 2)
    res[centerY, centerX] = 0.
    radius = float(min(centerX, centerY))
    out = radius - distance_transform_edt(res)
    out /= radius
    return out


def _cv_small_disk(image_size):
    """Generates a disk level set function.

    The disk covers half of the image along its smallest dimension.
    """
    res = cp.ones(image_size, dtype=bool)
    centerY = int((image_size[0] - 1) / 2)
    centerX = int((image_size[1] - 1) / 2)
    res[centerY, centerX] = 0.
    radius = float(min(centerX, centerY)) / 2.0
    out = radius - distance_transform_edt(res)
    out /= radius * 3
    return out


def _cv_init_level_set(init_level_set, image_shape, dtype=cp.float64):
    """Generates an initial level set function conditional on input arguments.
    """
    if type(init_level_set) == str:
        if init_level_set == 'checkerboard':
            res = _cv_checkerboard(image_shape, 5, dtype)
        elif init_level_set == 'disk':
            res = _cv_large_disk(image_shape)
        elif init_level_set == 'small disk':
            res = _cv_small_disk(image_shape)
        else:
            raise ValueError("Incorrect name for starting level set preset.")
    else:
        res = init_level_set
    return res.astype(dtype, copy=False)


@deprecate_kwarg({'max_iter': 'max_num_iter'}, removed_version="1.0",
                 deprecated_version="0.19")
def chan_vese(image, mu=0.25, lambda1=1.0, lambda2=1.0, tol=1e-3,
              max_num_iter=500, dt=0.5, init_level_set='checkerboard',
              extended_output=False):
    """Chan-Vese segmentation algorithm.

    Active contour model by evolving a level set. Can be used to
    segment objects without clearly defined boundaries.

    Parameters
    ----------
    image : (M, N) ndarray
        Grayscale image to be segmented.
    mu : float, optional
        'edge length' weight parameter. Higher `mu` values will
        produce a 'round' edge, while values closer to zero will
        detect smaller objects.
    lambda1 : float, optional
        'difference from average' weight parameter for the output
        region with value 'True'. If it is lower than `lambda2`, this
        region will have a larger range of values than the other.
    lambda2 : float, optional
        'difference from average' weight parameter for the output
        region with value 'False'. If it is lower than `lambda1`, this
        region will have a larger range of values than the other.
    tol : float, positive, optional
        Level set variation tolerance between iterations. If the
        L2 norm difference between the level sets of successive
        iterations normalized by the area of the image is below this
        value, the algorithm will assume that the solution was
        reached.
    max_num_iter : uint, optional
        Maximum number of iterations allowed before the algorithm
        interrupts itself.
    dt : float, optional
        A multiplication factor applied at calculations for each step,
        serves to accelerate the algorithm. While higher values may
        speed up the algorithm, they may also lead to convergence
        problems.
    init_level_set : str or (M, N) ndarray, optional
        Defines the starting level set used by the algorithm.
        If a string is inputted, a level set that matches the image
        size will automatically be generated. Alternatively, it is
        possible to define a custom level set, which should be an
        array of float values, with the same shape as 'image'.
        Accepted string values are as follows.

        'checkerboard'
            the starting level set is defined as
            sin(x/5*pi)*sin(y/5*pi), where x and y are pixel
            coordinates. This level set has fast convergence, but may
            fail to detect implicit edges.
        'disk'
            the starting level set is defined as the opposite
            of the distance from the center of the image minus half of
            the minimum value between image width and image height.
            This is somewhat slower, but is more likely to properly
            detect implicit edges.
        'small disk'
            the starting level set is defined as the
            opposite of the distance from the center of the image
            minus a quarter of the minimum value between image width
            and image height.
    extended_output : bool, optional
        If set to True, the return value will be a tuple containing
        the three return values (see below). If set to False which
        is the default value, only the 'segmentation' array will be
        returned.

    Returns
    -------
    segmentation : (M, N) ndarray, bool
        Segmentation produced by the algorithm.
    phi : (M, N) ndarray of floats
        Final level set computed by the algorithm.
    energies : list of floats
        Shows the evolution of the 'energy' for each step of the
        algorithm. This should allow to check whether the algorithm
        converged.

    Notes
    -----
    The Chan-Vese Algorithm is designed to segment objects without
    clearly defined boundaries. This algorithm is based on level sets
    that are evolved iteratively to minimize an energy, which is
    defined by weighted values corresponding to the sum of differences
    intensity from the average value outside the segmented region, the
    sum of differences from the average value inside the segmented
    region, and a term which is dependent on the length of the
    boundary of the segmented region.

    This algorithm was first proposed by Tony Chan and Luminita Vese,
    in a publication entitled "An Active Contour Model Without Edges"
    [1]_.

    This implementation of the algorithm is somewhat simplified in the
    sense that the area factor 'nu' described in the original paper is
    not implemented, and is only suitable for grayscale images.

    Typical values for `lambda1` and `lambda2` are 1. If the
    'background' is very different from the segmented object in terms
    of distribution (for example, a uniform black image with figures
    of varying intensity), then these values should be different from
    each other.

    Typical values for mu are between 0 and 1, though higher values
    can be used when dealing with shapes with very ill-defined
    contours.

    The 'energy' which this algorithm tries to minimize is defined
    as the sum of the differences from the average within the region
    squared and weighed by the 'lambda' factors to which is added the
    length of the contour multiplied by the 'mu' factor.

    Supports 2D grayscale images only, and does not implement the area
    term described in the original article.

    References
    ----------
    .. [1] An Active Contour Model without Edges, Tony Chan and
           Luminita Vese, Scale-Space Theories in Computer Vision,
           1999, :DOI:`10.1007/3-540-48236-9_13`
    .. [2] Chan-Vese Segmentation, Pascal Getreuer Image Processing On
           Line, 2 (2012), pp. 214-224,
           :DOI:`10.5201/ipol.2012.g-cv`
    .. [3] The Chan-Vese Algorithm - Project Report, Rami Cohen, 2011
           :arXiv:`1107.2782`
    """
    if len(image.shape) != 2:
        raise ValueError("Input image should be a 2D array.")

    float_dtype = _supported_float_type(image.dtype)
    phi = _cv_init_level_set(init_level_set, image.shape, dtype=float_dtype)
    if type(phi) != cp.ndarray or phi.shape != image.shape:
        raise ValueError("The dimensions of initial level set do not "
                         "match the dimensions of image.")

    image = image.astype(float_dtype, copy=False)
    image = image - cp.min(image)
    if cp.max(image) != 0:
        image = image / cp.max(image)

    i = 0
    if extended_output:
        old_energy = _cv_energy(image, phi, mu, lambda1, lambda2)
        energies = []
    phivar = tol + 1

    while phivar > tol and i < max_num_iter:
        # Save old level set values
        oldphi = phi

        # Calculate new level set
        phi = _cv_calculate_variation(image, phi, mu, lambda1, lambda2, dt)
        phivar = phi - oldphi
        phivar *= phivar
        phivar = cp.sqrt(phivar.mean())

        if extended_output:
            # Extract energy
            new_energy = _cv_energy(image, phi, mu, lambda1, lambda2)

            # Could compare energy to the previous level set to see if
            # continuing is necessary

            # Save old energy values
            energies.append(old_energy)
            old_energy = new_energy
        i += 1

    segmentation = phi > 0

    if extended_output:
        return (segmentation, phi, energies)
    else:
        return segmentation
