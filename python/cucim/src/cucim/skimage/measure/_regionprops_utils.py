import math

import cupy as cp
import cupyx.scipy.ndimage as ndi
import numpy as np

# Don't allocate STREL_* on GPU as we don't know in advance which device
# fmt: off
STREL_4 = np.array([[0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0]], dtype=np.uint8)
STREL_8 = np.ones((3, 3), dtype=np.uint8)
# fmt: on

# Coefficients from
# Ohser J., Nagel W., Schladitz K. (2002) The Euler Number of Discretized Sets
# - On the Choice of Adjacency in Homogeneous Lattices.
# In: Mecke K., Stoyan D. (eds) Morphology of Condensed Matter. Lecture Notes
# in Physics, vol 600. Springer, Berlin, Heidelberg.
# The value of coefficients correspond to the contributions to the Euler number
# of specific voxel configurations, which are themselves encoded thanks to a
# LUT. Computing the Euler number from the addition of the contributions of
# local configurations is possible thanks to an integral geometry formula
# (see the paper by Ohser et al. for more details).
EULER_COEFS2D_4 = [0, 1, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0]
EULER_COEFS2D_8 = [0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, -1, 0]
# fmt: off
EULER_COEFS3D_26 = np.array([0, 1, 1, 0, 1, 0, -2, -1,
                            1, -2, 0, -1, 0, -1, -1, 0,
                            1, 0, -2, -1, -2, -1, -1, -2,
                            -6, -3, -3, -2, -3, -2, 0, -1,
                            1, -2, 0, -1, -6, -3, -3, -2,
                            -2, -1, -1, -2, -3, 0, -2, -1,
                            0, -1, -1, 0, -3, -2, 0, -1,
                            -3, 0, -2, -1, 0, 1, 1, 0,
                            1, -2, -6, -3, 0, -1, -3, -2,
                            -2, -1, -3, 0, -1, -2, -2, -1,
                            0, -1, -3, -2, -1, 0, 0, -1,
                            -3, 0, 0, 1, -2, -1, 1, 0,
                            -2, -1, -3, 0, -3, 0, 0, 1,
                            -1, 4, 0, 3, 0, 3, 1, 2,
                            -1, -2, -2, -1, -2, -1, 1,
                            0, 0, 3, 1, 2, 1, 2, 2, 1,
                            1, -6, -2, -3, -2, -3, -1, 0,
                            0, -3, -1, -2, -1, -2, -2, -1,
                            -2, -3, -1, 0, -1, 0, 4, 3,
                            -3, 0, 0, 1, 0, 1, 3, 2,
                            0, -3, -1, -2, -3, 0, 0, 1,
                            -1, 0, 0, -1, -2, 1, -1, 0,
                            -1, -2, -2, -1, 0, 1, 3, 2,
                            -2, 1, -1, 0, 1, 2, 2, 1,
                            0, -3, -3, 0, -1, -2, 0, 1,
                            -1, 0, -2, 1, 0, -1, -1, 0,
                            -1, -2, 0, 1, -2, -1, 3, 2,
                            -2, 1, 1, 2, -1, 0, 2, 1,
                            -1, 0, -2, 1, -2, 1, 1, 2,
                            -2, 3, -1, 2, -1, 2, 0, 1,
                            0, -1, -1, 0, -1, 0, 2, 1,
                            -1, 2, 0, 1, 0, 1, 1, 0, ])
# fmt: on


def euler_number(image, connectivity=None):
    """Calculate the Euler characteristic in binary image.

    For 2D objects, the Euler number is the number of objects minus the number
    of holes. For 3D objects, the Euler number is obtained as the number of
    objects plus the number of holes, minus the number of tunnels, or loops.

    Parameters
    ----------
    image: (N, M) ndarray or (N, M, D) ndarray.
        2D or 3D images.
        If image is not binary, all values strictly greater than zero
        are considered as the object.
    connectivity : int, optional
        Maximum number of orthogonal hops to consider a pixel/voxel
        as a neighbor.
        Accepted values are ranging from  1 to input.ndim. If ``None``, a full
        connectivity of ``input.ndim`` is used.
        4 or 8 neighborhoods are defined for 2D images (connectivity 1 and 2,
        respectively).
        6 or 26 neighborhoods are defined for 3D images, (connectivity 1 and 3,
        respectively). Connectivity 2 is not defined.

    Returns
    -------
    euler_number : int
        Euler characteristic of the set of all objects in the image.

    Notes
    -----
    The Euler characteristic is an integer number that describes the
    topology of the set of all objects in the input image. If object is
    4-connected, then background is 8-connected, and conversely.

    The computation of the Euler characteristic is based on an integral
    geometry formula in discretized space. In practice, a neighbourhood
    configuration is constructed, and a LUT is applied for each
    configuration. The coefficients used are the ones of Ohser et al.

    It can be useful to compute the Euler characteristic for several
    connectivities. A large relative difference between results
    for different connectivities suggests that the image resolution
    (with respect to the size of objects and holes) is too low.

    References
    ----------
    .. [1] S. Rivollier. Analyse d’image geometrique et morphometrique par
           diagrammes de forme et voisinages adaptatifs generaux. PhD thesis,
           2010. Ecole Nationale Superieure des Mines de Saint-Etienne.
           https://tel.archives-ouvertes.fr/tel-00560838
    .. [2] Ohser J., Nagel W., Schladitz K. (2002) The Euler Number of
           Discretized Sets - On the Choice of Adjacency in Homogeneous
           Lattices. In: Mecke K., Stoyan D. (eds) Morphology of Condensed
           Matter. Lecture Notes in Physics, vol 600. Springer, Berlin,
           Heidelberg.

    Examples
    --------
    >>> import cupy as cp
    >>> SAMPLE = cp.zeros((100,100,100))
    >>> SAMPLE[40:60, 40:60, 40:60] = 1
    >>> euler_number(SAMPLE) # doctest: +ELLIPSIS
    1...
    >>> SAMPLE[45:55,45:55,45:55] = 0;
    >>> euler_number(SAMPLE) # doctest: +ELLIPSIS
    2...
    >>> SAMPLE = cp.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    ...                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    ...                    [1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0],
    ...                    [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1],
    ...                    [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]])
    >>> euler_number(SAMPLE)  # doctest:
    array(0)
    >>> euler_number(SAMPLE, connectivity=1)  # doctest:
    array(2)
    """  # noqa

    # as image can be a label image, transform it to binary
    image = (image > 0).astype(int)
    image = cp.pad(image, pad_width=1, mode='constant')

    # check connectivity
    if connectivity is None:
        connectivity = image.ndim

    # config variable is an adjacency configuration. A coefficient given by
    # variable coefs is attributed to each configuration in order to get
    # the Euler characteristic.
    if image.ndim == 2:

        config = cp.array([[0, 0, 0], [0, 1, 4], [0, 2, 8]])
        if connectivity == 1:
            coefs = EULER_COEFS2D_4
        else:
            coefs = EULER_COEFS2D_8
        bins = 16
    else:  # 3D images
        if connectivity == 2:
            raise NotImplementedError(
                'For 3D images, Euler number is implemented '
                'for connectivities 1 and 3 only')

        # fmt: off
        config = cp.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 1, 4], [0, 2, 8]],
                           [[0, 0, 0], [0, 16, 64], [0, 32, 128]]])
        # fmt: on
        if connectivity == 1:
            coefs = EULER_COEFS3D_26[::-1]
        else:
            coefs = EULER_COEFS3D_26
        bins = 256

    # XF has values in the 0-255 range in 3D, and in the 0-15 range in 2D,
    # with one unique value for each binary configuration of the
    # 27-voxel cube in 3D / 8-pixel square in 2D, up to symmetries
    XF = ndi.convolve(image, config, mode='constant', cval=0)
    h = cp.bincount(XF.ravel(), minlength=bins)

    coefs = cp.asarray(coefs)
    if image.ndim == 2:
        return coefs @ h
    else:
        return int(0.125 * coefs @ h)


def perimeter(image, neighbourhood=4):
    """Calculate total perimeter of all objects in binary image.

    Parameters
    ----------
    image : (N, M) ndarray
        2D binary image.
    neighbourhood : 4 or 8, optional
        Neighborhood connectivity for border pixel determination. It is used to
        compute the contour. A higher neighbourhood widens the border on which
        the perimeter is computed.

    Returns
    -------
    perimeter : float
        Total perimeter of all objects in binary image.

    References
    ----------
    .. [1] K. Benkrid, D. Crookes. Design and FPGA Implementation of
           a Perimeter Estimator. The Queen's University of Belfast.
           http://www.cs.qub.ac.uk/~d.crookes/webpubs/papers/perimeter.doc

    Examples
    --------
    >>> import cupy as cp
    >>> from skimage import data
    >>> from cucim.skimage import util
    >>> from cucim.skimage.measure import label
    >>> # coins image (binary)
    >>> img_coins = cp.array(data.coins() > 110)
    >>> # total perimeter of all objects in the image
    >>> perimeter(img_coins, neighbourhood=4)  # doctest: +ELLIPSIS
    array(7796.86799644)
    >>> perimeter(img_coins, neighbourhood=8)  # doctest: +ELLIPSIS
    array(8806.26807333)

    """
    if image.ndim != 2:
        raise NotImplementedError('`perimeter` supports 2D images only')

    if neighbourhood == 4:
        strel = STREL_4
    else:
        strel = STREL_8
    strel = cp.asarray(strel)
    image = image.astype(cp.uint8)
    eroded_image = ndi.binary_erosion(image, strel, border_value=0)
    border_image = image - eroded_image

    perimeter_weights = cp.zeros(50, dtype=cp.double)
    perimeter_weights[[5, 7, 15, 17, 25, 27]] = 1
    perimeter_weights[[21, 33]] = math.sqrt(2)
    perimeter_weights[[13, 23]] = (1 + math.sqrt(2)) / 2

    perimeter_image = ndi.convolve(border_image, cp.array([[10, 2, 10],
                                                           [2, 1, 2],
                                                           [10, 2, 10]]),
                                   mode='constant', cval=0)

    # You can also write
    # return perimeter_weights[perimeter_image].sum()
    # but that was measured as taking much longer than bincount + cp.dot (5x
    # as much time)
    perimeter_histogram = cp.bincount(perimeter_image.ravel(), minlength=50)
    total_perimeter = perimeter_histogram @ perimeter_weights
    return total_perimeter


def perimeter_crofton(image, directions=4):
    """Calculate total Crofton perimeter of all objects in binary image.

    Parameters
    ----------
    image : (N, M) ndarray
        2D image. If image is not binary, all values strictly greater than zero
        are considered as the object.
    directions : 2 or 4, optional
        Number of directions used to approximate the Crofton perimeter. By
        default, 4 is used: it should be more accurate than 2.
        Computation time is the same in both cases.

    Returns
    -------
    perimeter : float
        Total perimeter of all objects in binary image.

    Notes
    -----
    This measure is based on Crofton formula [1], which is a measure from
    integral geometry. It is defined for general curve length evaluation via
    a double integral along all directions. In a discrete
    space, 2 or 4 directions give a quite good approximation, 4 being more
    accurate than 2 for more complex shapes.

    Similar to :func:`~.measure.perimeter`, this function returns an
    approximation of the perimeter in continuous space.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Crofton_formula
    .. [2] S. Rivollier. Analyse d’image geometrique et morphometrique par
           diagrammes de forme et voisinages adaptatifs generaux. PhD thesis,
           2010.
           Ecole Nationale Superieure des Mines de Saint-Etienne.
           https://tel.archives-ouvertes.fr/tel-00560838

    Examples
    --------
    >>> import cupy as cp
    >>> from cucim.skimage import util
    >>> from skimage import data
    >>> from skimage.measure import label
    >>> # coins image (binary)
    >>> img_coins = cp.array(data.coins() > 110)
    >>> # total perimeter of all objects in the image
    >>> perimeter_crofton(img_coins, directions=2)  # doctest: +ELLIPSIS
    array(8144.57895443)
    >>> perimeter_crofton(img_coins, directions=4)  # doctest: +ELLIPSIS
    array(7837.07740694)
    """
    if image.ndim != 2:
        raise NotImplementedError(
            '`perimeter_crofton` supports 2D images only')

    # as image could be a label image, transform it to binary image
    image = (image > 0).astype(cp.uint8)
    image = cp.pad(image, pad_width=1, mode="constant")
    XF = ndi.convolve(image, cp.array([[0, 0, 0], [0, 1, 4], [0, 2, 8]]),
                      mode='constant', cval=0)

    h = cp.bincount(XF.ravel(), minlength=16)

    # definition of the LUT
    # fmt: off
    if directions == 2:
        coefs = [0, np.pi / 2, 0, 0, 0, np.pi / 2, 0, 0,
                 np.pi / 2, np.pi, 0, 0, np.pi / 2, np.pi, 0, 0]
    else:
        sq2 = math.sqrt(2)
        coefs = [0, np.pi / 4 * (1 + 1 / sq2),
                 np.pi / (4 * sq2),
                 np.pi / (2 * sq2), 0,
                 np.pi / 4 * (1 + 1 / sq2),
                 0, np.pi / (4 * sq2), np.pi / 4, np.pi / 2,
                 np.pi / (4 * sq2), np.pi / (4 * sq2),
                 np.pi / 4, np.pi / 2, 0, 0]
    # fmt: on

    total_perimeter = cp.asarray(coefs) @ h
    return total_perimeter
