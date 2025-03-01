import math

import cupy as cp
from packaging.version import parse

from cucim.skimage._vendored import ndimage as ndi
from cucim.skimage.util import map_array

CUPY_GTE_13_3_0 = parse(cp.__version__) >= parse("13.3.0")

# Need some default includes so uint32_t, uint64_t, etc. are defined

if CUPY_GTE_13_3_0:
    _includes = r"""
#include <cupy/cuda_workaround.h>  // provide std:: coverage
"""
else:
    _includes = r"""
#include <type_traits>  // let Jitify handle this
"""


def _get_count_dtype(label_image_size):
    """atomicAdd only supports int32, uint32, int64, uint64, float32, float64"""
    int32_count = label_image_size < 2**32
    count_dtype = cp.dtype(cp.uint32 if int32_count else cp.uint64)
    return count_dtype, int32_count


def _get_min_integer_dtype(max_size, signed=False):
    # negate to get a signed integer type, but need to also subtract 1, due
    # to asymmetric range on positive side, e.g. we want
    #    max_sz = 127 -> int8  (signed)   uint8 (unsigned)
    #    max_sz = 128 -> int16 (signed)   uint8 (unsigned)
    func = cp.min_scalar_type
    return func(-max_size - 1) if signed else func(max_size)


def _check_intensity_image_shape(label_image, intensity_image):
    ndim = label_image.ndim
    if intensity_image.shape[:ndim] != label_image.shape:
        raise ValueError(
            "Initial dimensions of `intensity_image` must match the shape of "
            "`label_image`. (`intensity_image` may have additional trailing "
            "channels/batch dimensions)"
        )

    num_channels = (
        math.prod(intensity_image.shape[ndim:])
        if intensity_image.ndim > ndim
        else 1
    )
    return num_channels


def _unravel_loop_index_declarations(var_name, ndim, uint_t="unsigned int"):
    if ndim == 1:
        code = f"""
        {uint_t} in_coord[1];"""
        return code

    code = f"""
        // variables for unraveling a linear index to a coordinate array
        {uint_t} in_coord[{ndim}];
        {uint_t} temp_floor;"""
    for d in range(ndim):
        code += f"""
        {uint_t} dim{d}_size = {var_name}.shape()[{d}];"""
    return code


def _unravel_loop_index(
    var_name,
    ndim,
    uint_t="unsigned int",
    raveled_index="i",
    omit_declarations=False,
):
    """
    declare a multi-index array in_coord and unravel the 1D index, i into it.
    This code assumes that the array is a C-ordered array.
    """
    code = (
        ""
        if omit_declarations
        else _unravel_loop_index_declarations(var_name, ndim, uint_t)
    )
    if ndim == 1:
        code = f"""
        in_coord[0] = {raveled_index};\n"""
        return code

    code += f"{uint_t} temp_idx = {raveled_index};"
    for d in range(ndim - 1, 0, -1):
        code += f"""
        temp_floor = temp_idx / dim{d}_size;
        in_coord[{d}] = temp_idx - temp_floor * dim{d}_size;
        temp_idx = temp_floor;"""
    code += """
        in_coord[0] = temp_idx;"""
    return code


def _reverse_label_values(label_image, max_label):
    """reverses the value of all labels (keeping background value=0 the same)"""
    dtype = label_image.dtype
    labs = cp.asarray(tuple(range(max_label + 1)), dtype=dtype)
    rev_labs = cp.asarray((0,) + tuple(range(max_label, 0, -1)), dtype=dtype)
    return map_array(label_image, labs, rev_labs)


def _find_close_labels(labels, binary_image, max_label):
    # check possibly too-close regions for which we may need to
    # manually recompute the regions perimeter in isolation
    labels_dilated2 = ndi.grey_dilation(labels, 5, mode="constant")
    labels2 = labels_dilated2 * binary_image
    rev_labels = _reverse_label_values(labels, max_label=max_label)
    rev_labels = ndi.grey_dilation(rev_labels, 5, mode="constant")
    rev_labels = rev_labels * binary_image
    labels3 = _reverse_label_values(rev_labels, max_label=max_label)
    diffs = cp.logical_or(labels != labels2, labels != labels3)
    labels_close = cp.asnumpy(cp.unique(labels[diffs]))
    return labels_close
