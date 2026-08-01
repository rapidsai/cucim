# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

"""Kernels for scikit-image label.

These are copied from CuPy, with modification to add a greyscale_mode
parameter as needed for scikit-image.

"""

import cupy
import numpy


def _label(x, structure, y, greyscale_mode=False):
    elems = numpy.where(structure != 0)
    vecs = [elems[dm] - 1 for dm in range(x.ndim)]
    offset = vecs[0]
    for dm in range(1, x.ndim):
        offset = offset * 3 + vecs[dm]
    indxs = numpy.where(offset < 0)[0]
    dirs = [[vecs[dm][dr] for dm in range(x.ndim)] for dr in indxs]
    dirs = cupy.array(dirs, dtype=numpy.int32)
    ndirs = indxs.shape[0]
    y_shape = cupy.array(y.shape, dtype=numpy.int32)
    count = cupy.zeros(2, dtype=y.dtype)
    _kernel_init()(x, y)
    if greyscale_mode:
        _kernel_connect(True)(x, y_shape, dirs, ndirs, x.ndim, y, size=y.size)
    else:
        _kernel_connect(False)(y_shape, dirs, ndirs, x.ndim, y, size=y.size)
    _kernel_count()(y, count, size=y.size)
    maxlabel = int(count[0])  # synchronize
    labels = cupy.empty(maxlabel, dtype=y.dtype)
    _kernel_labels()(y, count, labels, size=y.size)
    _kernel_finalize()(maxlabel, cupy.sort(labels), y, size=y.size)
    return maxlabel


"""
Elementwise kernels for use by label
"""


def _kernel_init():
    return cupy.ElementwiseKernel(
        "X x",
        "Y y",
        "if (x == 0) { y = -1; } else { y = i; }",
        "cucim_skimage_measure_label_init",
    )


def _kernel_connect(greyscale_mode=False, int_t="int"):
    """
    Notes
    -----
    dirs is a (n_neig//2, ndim) of relative offsets to the neighboring voxels.
    For example, for structure = np.ones((3, 3)):
        dirs = array([[-1, -1],
                      [-1,  0],
                      [-1,  1],
                      [ 0, -1]], dtype=int32)
    (Implementation assumes a centro-symmetric structure)
    ndirs = dirs.shape[0]

    In the dirs loop below, there is a loop over the ndim neighbors:
        Here, index j corresponds to the current pixel and k is the current
        neighbor location.
    """
    in_params = "raw int32 shape, raw int32 dirs, int32 ndirs, int32 ndim"
    if greyscale_mode:
        # greyscale mode -> different values receive different labels
        x_condition = "if (x[k] != x[j]) continue;"
        in_params = "raw X x, " + in_params
    else:
        # binary mode -> all non-background voxels treated the same
        x_condition = ""

    code = f"""
        using atomic_t = typename cupy::type_traits::conditional<
            cupy::type_traits::is_same<Y, long long>::value,
            unsigned long long,
            Y
        >::type;
        if (y[i] < 0) continue;
        for (int dr = 0; dr < ndirs; dr++) {{
            Y j = i;
            Y rest = j;
            Y stride = 1;
            Y k = 0;
            for (int dm = ndim-1; dm >= 0; dm--) {{
                Y pos = rest % shape[dm] + dirs[dm + dr * ndim];
                if (pos < 0 || pos >= shape[dm]) {{
                    k = -1;
                    break;
                }}
                k += pos * stride;
                rest /= shape[dm];
                stride *= shape[dm];
            }}
            if (k < 0) continue;
            if (y[k] < 0) continue;
            {x_condition}
            while (1) {{
                while (j != y[j]) {{ j = y[j]; }}
                while (k != y[k]) {{ k = y[k]; }}
                if (j == k) break;
                if (j < k) {{
                    Y old = atomicCAS((atomic_t*)&y[k], k, j);
                    if (old == k) break;
                    k = old;
                }}
                else {{
                    Y old = atomicCAS((atomic_t*)&y[j], j, k);
                    if (old == j) break;
                    j = old;
                }}
            }}
        }}
        """

    return cupy.ElementwiseKernel(
        in_params,
        "raw Y y",
        code,
        "cucim_skimage_measure_label_connect",
    )


def _kernel_count():
    return cupy.ElementwiseKernel(
        "",
        "raw Y y, raw Y count",
        """
        if (y[i] < 0) continue;
        Y j = i;
        while (j != y[j]) { j = y[j]; }
        if (j != i) y[i] = j;
        else atomicAdd(&count[0], 1);
        """,
        "cucim_skimage_measure_label_count",
    )


def _kernel_labels():
    return cupy.ElementwiseKernel(
        "",
        "raw Y y, raw Y count, raw Y labels",
        """
        if (y[i] != i) continue;
        Y j = atomicAdd(&count[1], 1);
        labels[j] = i;
        """,
        "cucim_skimage_measure_label_labels",
    )


def _kernel_finalize():
    return cupy.ElementwiseKernel(
        "Y maxlabel",
        "raw Y labels, raw Y y",
        """
        if (y[i] < 0) {
            y[i] = 0;
            continue;
        }
        Y yi = y[i];
        Y j_min = 0;
        Y j_max = maxlabel - 1;
        Y j = (j_min + j_max) / 2;
        while (j_min < j_max) {
            if (yi == labels[j]) break;
            if (yi < labels[j]) j_max = j - 1;
            else j_min = j + 1;
            j = (j_min + j_max) / 2;
        }
        y[i] = j + 1;
        """,
        "cucim_skimage_measure_label_finalize",
    )
