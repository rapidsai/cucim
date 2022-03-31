"""A vendored subset of cupyx.scipy.ndimage._filters"""

import cupy

from cucim.skimage._vendored import _ndimage_filters_core as _filters_core


@cupy.memoize(for_each_device=True)
def _get_correlate_kernel(mode, w_shape, int_type, offsets, cval):
    return _filters_core._generate_nd_kernel(
        'correlate',
        'W sum = (W)0;',
        'sum += cast<W>({value}) * wval;',
        'y = cast<Y>(sum);',
        mode, w_shape, int_type, offsets, cval, ctype='W')
