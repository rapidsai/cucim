# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp

from .._shared.utils import _supported_float_type, check_nD
from ..util import img_as_float


def _prepare_grayscale_input_2D(image):
    image = cp.squeeze(image)
    check_nD(image, 2)
    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    return image.astype(float_dtype, copy=False)


def _prepare_grayscale_input_nD(image):
    image = cp.squeeze(image)
    check_nD(image, range(2, 6))
    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    return image.astype(float_dtype, copy=False)


def _mask_border_keypoints(image_shape, keypoints, distance):
    """Mask coordinates that are within certain distance from the image border.

    Parameters
    ----------
    image_shape : (2,) array_like
        Shape of the image as ``(rows, cols)``.
    keypoints : (N, 2) array
        Keypoint coordinates as ``(rows, cols)``.
    distance : int
        Image border distance.

    Returns
    -------
    mask : (N,) bool array
        Mask indicating if pixels are within the image (``True``) or in the
        border region of the image (``False``).

    """

    rows = image_shape[0]
    cols = image_shape[1]

    mask = (
        ((distance - 1) < keypoints[:, 0])
        & (keypoints[:, 0] < (rows - distance + 1))
        & ((distance - 1) < keypoints[:, 1])
        & (keypoints[:, 1] < (cols - distance + 1))
    )

    return mask
