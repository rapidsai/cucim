import sys

import numpy as np


def lookfor(what):
    """Do a keyword search on scikit-image docstrings.

    Parameters
    ----------
    what : str
        Words to look for.

    Examples
    --------
    >>> import cucim.skimage
    >>> cucim.skimage.lookfor('median')  # doctest: +SKIP
    Search results for 'median'
    ---------------------------
    cucim.skimage.filters.median
        Return local median of an image.
    cucim.skimage.measure.block_reduce
        Downsample image by applying function `func` to local blocks.
    cucim.skimage.filters.threshold_local
        Compute a threshold mask image based on local pixel neighborhood.
    cucim.skimage.registration.optical_flow_ilk
        Coarse to fine optical flow estimator.
    cucim.skimage.registration.optical_flow_tvl1
        Coarse to fine optical flow estimator.
    """
    return np.lookfor(what, sys.modules[__name__.split('.')[0]])
