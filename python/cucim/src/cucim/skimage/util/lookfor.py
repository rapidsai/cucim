import sys

from cucim.skimage._shared.utils import deprecate_func

try:
    from numpy import lookfor as np_lookfor
except ImportError:
    np_lookfor = None


@deprecate_func(
    deprecated_version="24.06",
    removed_version="24.12",
)
def lookfor(what):
    """Do a keyword search on scikit-image docstrings.

    Parameters
    ----------
    what : str
        Words to look for.

    Notes
    -----
    This untested search function is not currently working as expected will be
    removed as it is unneeded.

    Examples
    --------
    >>> import cucim.skimage
    >>> cucim.skimage.lookfor('median')  # doctest: +SKIP
    Search results for 'median'
    ---------------------------
    Nothing found.
    """
    if np_lookfor is None:
        raise RuntimeError(
            "lookfor unavailable (numpy.lookfor was removed in numpy 2.0)"
        )
    return np_lookfor(what, sys.modules[__name__.split(".")[0]])
