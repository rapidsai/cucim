# SPDX-FileCopyrightText: Copyright (c) 2003-2009 Massachusetts Institute of Technology
# SPDX-FileCopyrightText: Copyright (c) 2009-2011 Broad Institute
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND (GPL-2.0-only OR BSD-3-Clause)

"""
_rank_order.py - convert an image of any type to an image of ints whose
pixels have an identical rank order compared to the original image

Originally part of CellProfiler, code licensed under both GPL and BSD licenses.
Website: http://www.cellprofiler.org
Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.
Original author: Lee Kamentstky
"""
import cupy as cp


def rank_order(image):
    """Return an image of the same shape where each pixel is the
    index of the pixel value in the ascending order of the unique
    values of ``image``, aka the rank-order value.

    Parameters
    ----------
    image : cp.ndarray

    Returns
    -------
    labels : cp.ndarray of unsigned integers, of shape image.shape
        New array where each pixel has the rank-order value of the
        corresponding pixel in ``image``. Pixel values are between 0 and
        n - 1, where n is the number of distinct unique values in
        ``image``. The dtype of this array will be determined by
        ``cp.min_scalar_type(image.size)``.
    original_values : 1-D cp.ndarray
        Unique original values of ``image``. This will have the same dtype as
        ``image``.

    Examples
    --------
    >>> a = cp.asarray([[1, 4, 5], [4, 4, 1], [5, 1, 1]])
    >>> a
    array([[1, 4, 5],
           [4, 4, 1],
           [5, 1, 1]])
    >>> rank_order(a)
    (array([[0, 1, 2],
           [1, 1, 0],
           [2, 0, 0]], dtype=uint8), array([1, 4, 5]))
    >>> b = cp.asarray([-1., 2.5, 3.1, 2.5])
    >>> rank_order(b)
    (array([0, 1, 2, 1], dtype=uint8), array([-1. ,  2.5,  3.1]))
    """
    flat_image = image.ravel()
    unsigned_dtype = cp.min_scalar_type(flat_image.size)
    sort_order = flat_image.argsort().astype(unsigned_dtype, copy=False)
    flat_image = flat_image[sort_order]
    sort_rank = cp.zeros_like(sort_order)
    is_different = flat_image[:-1] != flat_image[1:]
    cp.cumsum(is_different, out=sort_rank[1:], dtype=sort_rank.dtype)
    original_values = cp.zeros((int(sort_rank[-1]) + 1,), image.dtype)
    original_values[0] = flat_image[0]
    original_values[1:] = flat_image[1:][is_different]
    int_image = cp.zeros_like(sort_order)
    int_image[sort_order] = sort_rank
    return (int_image.reshape(image.shape), original_values)
