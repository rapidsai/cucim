"""Misc utility functions that are not from SciPy, NumPy or scikit-image.

"""
import math

import numpy

if hasattr(math, 'prod'):
    prod = math.prod  # available in Python 3.8+ only
else:
    prod = numpy.prod


def ndim(a):
    """
    Return the number of dimensions of an array.

    Parameters
    ----------
    a : array_like
        Input array.  If it is not already an ndarray, a conversion is
        attempted.

    Returns
    -------
    number_of_dimensions : int
        The number of dimensions in `a`.  Scalars are zero-dimensional.

    See Also
    --------
    ndarray.ndim : equivalent method
    shape : dimensions of array
    ndarray.shape : dimensions of array

    Examples
    --------
    >>> from cucim.numpy import ndim
    >>> ndim([[1,2,3],[4,5,6]])
    2
    >>> ndim(cupy.asarray([[1,2,3],[4,5,6]]))
    2
    >>> ndim(1)
    0

    """
    try:
        return a.ndim
    except AttributeError:
        return numpy.asarray(a).ndim
