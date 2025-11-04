# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import functools
import warnings

import cupy as cp

from .dtype import img_as_float


def _rename_image_params(func):
    wm_images = (
        "Since version 0.24, the two input images are named `image0` and "
        "`image1` (instead of `image1` and `image2`, respectively). Please use "
        "`image0, image1` to avoid this warning for now, and avoid an error "
        "from version 0.26 onwards."
    )

    wm_method = (
        "Starting in version 0.24, all arguments following `image0, image1` "
        "(including `method`) will be keyword-only. Please pass `method=` "
        "in the function call to avoid this warning for now, and avoid an "
        "error from version 0.26 onwards."
    )

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Turn all args into kwargs
        for i, (value, param) in enumerate(
            zip(args, ["image0", "image1", "method", "n_tiles"])
        ):
            if i >= 2:
                warnings.warn(wm_method, category=FutureWarning, stacklevel=2)
            if param in kwargs:
                raise ValueError(
                    f"{param} passed both as positional and keyword argument."
                )
            else:
                kwargs[param] = value
        args = tuple()

        # Account for `image2` if given
        if "image2" in kwargs.keys():
            warnings.warn(wm_images, category=FutureWarning, stacklevel=2)

            # Safely move `image2` to `image1` if that's empty
            if "image1" in kwargs.keys():
                # Safely move `image1` to `image0`
                if "image0" in kwargs.keys():
                    raise ValueError(
                        "Three input images given; please use only `image0` "
                        "and `image1`."
                    )
                kwargs["image0"] = kwargs.pop("image1")
            kwargs["image1"] = kwargs.pop("image2")

        return func(*args, **kwargs)

    return wrapper


@cp.memoize(for_each_device=True)
def _checkerboard_kernel():
    code = """
    int y_size = image0.shape()[1];
    int coord_x = i / y_size;
    int coord_y = i % y_size;
    int tile_x = coord_x / step_x;
    int tile_y = coord_y / step_y;
    if ((tile_x + tile_y) % 2) {
        out[i] = image1[i];
    } else {
        out[i] = image0[i];
    }
    """

    return cp.ElementwiseKernel(
        "raw X image0, raw X image1, int32 step_x, int32 step_y",
        "raw X out",
        code,
        name="cucim_compare_images_checkerboard",
    )


@cp.fuse()
def compare_absdiff(img1, img2):
    return cp.abs(img2 - img1)


@cp.fuse()
def compare_blend(img1, img2):
    return 0.5 * (img1 + img2)


@_rename_image_params
def compare_images(image0, image1, *, method="diff", n_tiles=(8, 8)):
    """
    Return an image showing the differences between two images.

    .. versionadded:: 0.16

    Parameters
    ----------
    image0, image1 : ndarray, shape (M, N)
        Images to process, must be of the same shape.

        .. versionchanged:: 0.24
            `image1` and `image2` were renamed into `image0` and `image1`
            respectively.
    method : string, optional
        Method used for the comparison.
        Valid values are {'diff', 'blend', 'checkerboard'}.
        Details are provided in the note section.

        .. versionchanged:: 0.24
            This parameter and following ones are keyword-only.
    n_tiles : tuple, optional
        Used only for the `checkerboard` method. Specifies the number
        of tiles (row, column) to divide the image.

    Returns
    -------
    comparison : ndarray, shape (M, N)
        Image showing the differences.

    Notes
    -----
    ``'diff'`` computes the absolute difference between the two images.
    ``'blend'`` computes the mean value.
    ``'checkerboard'`` makes tiles of dimension `n_tiles` that display
    alternatively the first and the second image. Note that images must be
    2-dimensional to be compared with the checkerboard method.
    """

    if image1.shape != image0.shape:
        raise ValueError("Images must have the same shape.")

    img1 = img_as_float(image0)
    img2 = img_as_float(image1)

    if method == "diff":
        comparison = compare_absdiff(img1, img2)
    elif method == "blend":
        comparison = compare_blend(img1, img2)
    elif method == "checkerboard":
        # indexing logic assumes C-contiguous memory layout
        if not img1.flags.c_contiguous:
            img1 = cp.ascontiguousarray(img1)
        if not img2.flags.c_contiguous:
            img2 = cp.ascontiguousarray(img2)

        if img1.ndim != 2:
            raise ValueError(
                "Images must be 2-dimensional to be compared with the "
                "checkerboard method."
            )
        shapex, shapey = img1.shape
        comparison = cp.empty_like(img1)
        stepx = shapex // n_tiles[0]
        stepy = shapey // n_tiles[1]
        if max(img1.shape) >= (1 << 31):
            raise ValueError(
                "axis dimensions exceeding 32-bit integer range are unsupported"
            )
        kern = _checkerboard_kernel()
        kern(img1, img2, stepx, stepy, comparison, size=comparison.size)
    else:
        raise ValueError(
            "Wrong value for `method`. "
            'Must be either "diff", "blend" or "checkerboard".'
        )
    return comparison
