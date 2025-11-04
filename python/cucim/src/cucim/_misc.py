# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Misc utility functions that are not from SciPy, NumPy or scikit-image.

"""
import numpy as np


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
        return np.asarray(a).ndim
