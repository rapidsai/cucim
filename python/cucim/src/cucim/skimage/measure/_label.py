import cupy as cp
import scipy.ndimage as cpu_ndi

from ._label_kernels import _label


def _get_structure(ndim, connectivity):
    if connectivity is None:
        # use the full connectivity by default
        connectivity = ndim
    if not 1 <= connectivity <= ndim:
        raise ValueError("Connectivity below 1 or above %d is illegal." % ndim)
    return cpu_ndi.generate_binary_structure(ndim, connectivity)


# TODO: currently uses int32 for the labels. should add int64 option as well
def label(input, background=None, return_num=False, connectivity=None):
    r"""Label connected regions of an integer array.

    Two pixels are connected when they are neighbors and have the same value.
    In 2D, they can be neighbors either in a 1- or 2-connected sense.
    The value refers to the maximum number of orthogonal hops to consider a
    pixel/voxel a neighbor::

      1-connectivity     2-connectivity     diagonal connection close-up

           [ ]           [ ]  [ ]  [ ]             [ ]
            |               \  |  /                 |  <- hop 2
      [ ]--[x]--[ ]      [ ]--[x]--[ ]        [x]--[ ]
            |               /  |  \             hop 1
           [ ]           [ ]  [ ]  [ ]

    Parameters
    ----------
    input : ndarray of dtype int
        Image to label.
    background : int, optional
        Consider all pixels with this value as background pixels, and label
        them as 0. By default, 0-valued pixels are considered as background
        pixels.
    return_num : bool, optional
        Whether to return the number of assigned labels.
    connectivity : int, optional
        Maximum number of orthogonal hops to consider a pixel/voxel
        as a neighbor.
        Accepted values are ranging from  1 to input.ndim. If ``None``, a full
        connectivity of ``input.ndim`` is used.

    Returns
    -------
    labels : ndarray of dtype int
        Labeled array, where all connected regions are assigned the
        same integer value.
    num : int, optional
        Number of labels, which equals the maximum label index and is only
        returned if return_num is `True`.

    See Also
    --------
    regionprops
    regionprops_table

    References
    ----------
    .. [1] Christophe Fiorio and Jens Gustedt, "Two linear time Union-Find
           strategies for image processing", Theoretical Computer Science
           154 (1996), pp. 165-181.
    .. [2] Kensheng Wu, Ekow Otoo and Arie Shoshani, "Optimizing connected
           component labeling algorithms", Paper LBNL-56864, 2005,
           Lawrence Berkeley National Laboratory (University of California),
           http://repositories.cdlib.org/lbnl/LBNL-56864

    Notes
    -----
    Currently the cucim implementation of this function always uses 32-bit
    integers for the label array. This is done for performance. In the future
    64-bit integer support may also be added for better skimage compatibility.

    Examples
    --------
    >>> import cupy as cp
    >>> x = cp.eye(3).astype(int)
    >>> print(x)
    [[1 0 0]
     [0 1 0]
     [0 0 1]]
    >>> print(label(x, connectivity=1))
    [[1 0 0]
     [0 2 0]
     [0 0 3]]
    >>> print(label(x, connectivity=2))
    [[1 0 0]
     [0 1 0]
     [0 0 1]]
    >>> print(label(x, background=-1))
    [[1 2 2]
     [2 1 2]
     [2 2 1]]
    >>> x = cp.asarray([[1, 0, 0],
    ...                 [1, 1, 5],
    ...                 [0, 0, 0]])
    >>> print(label(x))
    [[1 0 0]
     [1 1 2]
     [0 0 0]]
    """
    ndim = input.ndim
    structure = _get_structure(ndim, connectivity)
    if background is None:
        background = 0
    elif background != 0:
        # offset so that background becomes 0 as expected by _label below
        input = input - background

    if input.dtype.kind not in "bui":
        # skimage always copies the input into a np.intp dtype array so do the
        # same here for non-integer dtypes.
        input = input.astype(cp.intp)

    labels = cp.empty(input.shape, order="C", dtype=cp.int32)
    num = _label(input, structure, labels, greyscale_mode=True)

    if return_num:
        return labels, num
    return labels
