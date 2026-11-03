# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

"""Kernels for scikit-image label.

These are copied from CuPy, with modification to add a greyscale_mode
parameter as needed for scikit-image.

"""

import functools

import cupy
import numpy


def _half_structure_dirs(structure):
    """Offsets of the neighbors that precede each pixel in raster order.

    The structure must be centro-symmetric and have 3 elements along each axis.
    """
    structure = numpy.asarray(structure)
    elems = numpy.where(structure != 0)
    vecs = [elems[dm] - 1 for dm in range(structure.ndim)]
    offset = vecs[0]
    for dm in range(1, structure.ndim):
        offset = offset * 3 + vecs[dm]
    indxs = numpy.where(offset < 0)[0]
    dirs = [[vecs[dm][dr] for dm in range(structure.ndim)] for dr in indxs]
    return numpy.array(dirs, dtype=numpy.int32).reshape(-1, structure.ndim)


def _connect(x, structure, y, greyscale_mode=False):
    """Initialize ``y`` and merge the neighboring pixels into trees.

    On return ``y[i]`` is the index of the parent of pixel ``i`` (-1 for
    background pixels); the trees are not flattened yet.
    """
    dirs = _half_structure_dirs(structure)
    y_shape = cupy.array(y.shape, dtype=numpy.int32)
    dirs_key = tuple(map(tuple, dirs.tolist()))
    _kernel_init_link(greyscale_mode, x.ndim, dirs_key)(
        x, y_shape, y, size=y.size
    )
    _kernel_connect(greyscale_mode, x.ndim, dirs_key)(
        *((x,) if greyscale_mode else ()), y_shape, y, size=y.size
    )


def _label(x, structure, y, greyscale_mode=False):
    count = cupy.zeros(2, dtype=y.dtype)
    _connect(x, structure, y, greyscale_mode)
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


def _neighbor_code(ndim, dirs):
    """C++ pieces shared by the init-link and connect kernels.

    Returns the declaration of the table of neighbor offsets (a constant of the
    kernel), the code that decomposes the index ``i`` into the coordinates c<d>
    and the strides s<d>, the condition that a neighbor ``dr`` is outside of
    the image and the expression of its index offset.
    """
    table = ", ".join(str(d) for dirvec in dirs for d in dirvec)
    declaration = f"static const int dirs[{len(dirs) * ndim}] = {{{table}}};"
    coords = "\n        ".join(
        ["Y rest = i;", "Y stride = 1;"]
        + [
            f"Y c{dm} = rest % shape[{dm}]; rest /= shape[{dm}];"
            f" Y s{dm} = stride; stride *= shape[{dm}];"
            for dm in range(ndim - 1, -1, -1)
        ]
    )
    outside = " || ".join(
        f"c{dm} + dirs[dr * {ndim} + {dm}] < 0 || "
        f"c{dm} + dirs[dr * {ndim} + {dm}] >= shape[{dm}]"
        for dm in range(ndim)
    )
    offset = " + ".join(
        f"dirs[dr * {ndim} + {dm}] * s{dm}" for dm in range(ndim)
    )
    return declaration, coords, outside, offset


@functools.cache
def _kernel_init_link(greyscale_mode, ndim, dirs):
    """Initialize the forest, linking each pixel to a preceding neighbor.

    Each foreground pixel points to its first neighbor (in raster order) that
    is foreground (and, in greyscale mode, has the same value), or to itself
    when it has none. Only the entry of the pixel itself is written, so no
    atomics are needed, and the links point to smaller indices as the unions
    of the connect kernel do. Background pixels get -1.
    """
    ndirs = len(dirs)
    cond = "x[k] != x[i]" if greyscale_mode else "x[k] == 0"
    declaration, coords, outside, offset = _neighbor_code(ndim, dirs)
    code = f"""
        {declaration}
        if (x[i] == 0) {{ y[i] = -1; continue; }}
        {coords}
        Y link = i;
        for (int dr = 0; dr < {ndirs}; dr++) {{
            if ({outside}) continue;
            Y k = i + {offset};
            if ({cond}) continue;
            link = k;
            break;
        }}
        y[i] = link;
        """
    return cupy.ElementwiseKernel(
        "raw X x, raw int32 shape",
        "raw Y y",
        code,
        "cucim_skimage_measure_label_init_link",
    )


@functools.cache
def _kernel_connect(greyscale_mode, ndim, dirs):
    """
    Notes
    -----
    dirs is a tuple of the relative offsets (one tuple of ndim entries each)
    of the neighbors that precede a pixel in raster order. For example, for
    structure = np.ones((3, 3)):
        dirs = ((-1, -1), (-1, 0), (-1, 1), (0, -1))
    (Implementation assumes a centro-symmetric structure). The offsets are
    compiled into the kernel as constants.

    The coordinates of the pixel are computed once. In the dirs loop, j is the
    current pixel and k the current neighbor.
    """
    ndirs = len(dirs)
    in_params = "raw int32 shape"
    if greyscale_mode:
        # greyscale mode -> different values receive different labels
        x_condition = "if (x[k] != x[j]) continue;"
        in_params = "raw X x, " + in_params
    else:
        # binary mode -> all non-background voxels treated the same
        x_condition = ""

    declaration, coords, outside, offset = _neighbor_code(ndim, dirs)

    code = f"""
        using atomic_t = typename cupy::type_traits::conditional<
            cupy::type_traits::is_same<Y, long long>::value,
            unsigned long long,
            Y
        >::type;
        {declaration}
        if (y[i] < 0) continue;
        {coords}
        for (int dr = 0; dr < {ndirs}; dr++) {{
            if ({outside}) continue;
            Y j = i;
            Y k = i + {offset};
            if (y[k] < 0) continue;
            {x_condition}
            while (1) {{
                // find the roots, halving the paths on the way
                while (1) {{
                    Y p = y[j];
                    if (p == j) break;
                    Y gp = y[p];
                    if (gp != p) y[j] = gp;
                    j = gp;
                }}
                while (1) {{
                    Y p = y[k];
                    if (p == k) break;
                    Y gp = y[p];
                    if (gp != p) y[k] = gp;
                    k = gp;
                }}
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
