import cupy as cp
from cupyx.scipy.stats import entropy as scipy_entropy


def shannon_entropy(image, base=2):
    """Calculate the Shannon entropy of an image.

    The Shannon entropy is defined as S = -sum(pk * log(pk)),
    where pk are frequency/probability of pixels of value k.

    Parameters
    ----------
    image : (N, M) ndarray
        Grayscale input image.
    base : float, optional
        The logarithmic base to use.

    Returns
    -------
    entropy : 0-dimensional float cupy.ndarray

    Notes
    -----
    The returned value is measured in bits or shannon (Sh) for base=2, natural
    unit (nat) for base=np.e and hartley (Hart) for base=10.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Entropy_(information_theory) <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`_  # noqa
    .. [2] https://en.wiktionary.org/wiki/Shannon_entropy

    Examples
    --------
    >>> import cupy as cp
    >>> from skimage import data
    >>> from cucim.skimage.measure import shannon_entropy
    >>> shannon_entropy(cp.array(data.camera()))
    array(7.23169501)
    """

    _, counts = cp.unique(image, return_counts=True)
    return scipy_entropy(counts, base=base)
