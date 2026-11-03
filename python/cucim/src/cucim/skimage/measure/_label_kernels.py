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


def _use_blocks(x, structure, greyscale_mode):
    """Whether the 2 x 2 (x 2) block kernels can be used.

    The blocks need full connectivity, which makes every pair of pixels of a
    block neighbors, and pixels that all have the same value (a binary image).
    """
    return (
        (not greyscale_mode or x.dtype == numpy.bool_)
        and x.ndim in (2, 3)
        and numpy.asarray(structure).shape == (3,) * x.ndim
        and bool(numpy.asarray(structure).all())
    )


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


def _connect_blocks(x, y):
    """Build the forest of the blocks, one node per 2 x 2 (x 2) block of pixels.

    With full connectivity all the foreground pixels of a block are neighbors,
    so each block is one tree: its foreground pixels point to the first of them.
    Two blocks are joined if a foreground pixel of each lies on the sides that
    face each other; only these block pairs need a union.

    Returns the shape of the grid of blocks, the number of blocks and the
    occupancy byte of each block.
    """
    ndim = x.ndim
    shape = cupy.array(x.shape, dtype=numpy.int32)
    bshape_tuple = tuple(-(-s // 2) for s in x.shape)
    bshape = cupy.array(bshape_tuple, dtype=numpy.int32)
    nblocks = int(numpy.prod(bshape_tuple))
    occupancy = cupy.empty(nblocks, dtype=numpy.uint8)
    _kernel_block_init(ndim)(x, shape, bshape, y, occupancy, size=nblocks)
    _kernel_block_connect(ndim)(shape, bshape, occupancy, y, size=nblocks)
    return shape, bshape, nblocks, occupancy


def _label_blocks(x, y):
    """`_label` for the block kernels: all the steps are done per block."""
    shape, bshape, nblocks, occupancy = _connect_blocks(x, y)
    ndim = x.ndim
    count = cupy.zeros(2, dtype=y.dtype)
    block_root = cupy.empty(nblocks, dtype=y.dtype)
    _kernel_block_flatten(ndim)(
        shape, bshape, occupancy, y, block_root, count, size=nblocks
    )
    maxlabel = int(count[0])  # synchronize
    labels = cupy.empty(maxlabel, dtype=y.dtype)
    _kernel_block_collect(ndim)(
        shape, bshape, occupancy, block_root, count, labels, size=nblocks
    )
    _kernel_block_finalize(ndim)(
        shape,
        bshape,
        occupancy,
        block_root,
        maxlabel,
        cupy.sort(labels),
        y,
        size=nblocks,
    )
    return maxlabel


def _label(x, structure, y, greyscale_mode=False):
    count = cupy.zeros(2, dtype=y.dtype)
    if _use_blocks(x, structure, greyscale_mode):
        return _label_blocks(x, y)
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


# The union of the trees of the nodes j and k of the forest y, which is shared
# by the connect kernels. The roots are found with path halving; only roots are
# relinked (with a compare-and-swap) and always to a smaller index. On exit j
# and k are scratch values.
_ATOMIC_T_CODE = """
using atomic_t = typename cupy::type_traits::conditional<
    cupy::type_traits::is_same<Y, long long>::value,
    unsigned long long,
    Y
>::type;
"""

_UNION_CODE = """
while (1) {
    // find the roots, halving the paths on the way
    while (1) {
        Y p = y[j];
        if (p == j) break;
        Y gp = y[p];
        if (gp != p) y[j] = gp;
        j = gp;
    }
    while (1) {
        Y p = y[k];
        if (p == k) break;
        Y gp = y[p];
        if (gp != p) y[k] = gp;
        k = gp;
    }
    if (j == k) break;
    if (j < k) {
        Y old = atomicCAS((atomic_t*)&y[k], k, j);
        if (old == k) break;
        k = old;
    }
    else {
        Y old = atomicCAS((atomic_t*)&y[j], j, k);
        if (old == j) break;
        j = old;
    }
}
"""


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
        {_ATOMIC_T_CODE}
        {declaration}
        if (y[i] < 0) continue;
        {coords}
        for (int dr = 0; dr < {ndirs}; dr++) {{
            if ({outside}) continue;
            Y j = i;
            Y k = i + {offset};
            if (y[k] < 0) continue;
            {x_condition}
            {_UNION_CODE}
        }}
        """

    return cupy.ElementwiseKernel(
        in_params,
        "raw Y y",
        code,
        "cucim_skimage_measure_label_connect",
    )


def _block_code(ndim):
    """C++ pieces shared by the block kernels (blocks of 2 pixels per axis).

    A pixel of a block is identified by its local coordinates l_d in {0, 1},
    stored in bit sum(l_d * 2**(ndim - 1 - d)) of the occupancy byte of the
    block, so that the lowest set bit is the first pixel in raster order.
    """
    coords = "\n        ".join(
        ["Y brest = i;", "Y stride = 1;"]
        + [
            f"Y b{d} = brest % bshape[{d}]; brest /= bshape[{d}];"
            f" Y s{d} = stride; stride *= shape[{d}];"
            for d in range(ndim - 1, -1, -1)
        ]
    )
    origin = " + ".join(f"2 * b{d} * s{d}" for d in range(ndim))
    return coords, origin


def _local_offset(ndim, bit):
    """Expression of the index offset of the pixel in bit ``bit`` of a block."""
    return " + ".join(
        f"((({bit}) >> {ndim - 1 - d}) & 1) * s{d}" for d in range(ndim)
    )


@functools.cache
def _kernel_block_init(ndim):
    """Make each block a tree and store which of its pixels are foreground.

    One thread per block. The foreground pixels of the block point to the
    first of them (the root); the background pixels get -1. The occupancy
    of the block, one bit per pixel, is written to ``occupancy``.
    """
    coords, origin = _block_code(ndim)
    pixels = []
    for c in range(2**ndim):
        local = [(c >> (ndim - 1 - d)) & 1 for d in range(ndim)]
        inside = " && ".join(
            f"2 * b{d} + {lc} < shape[{d}]" for d, lc in enumerate(local)
        )
        index = " + ".join(
            [f"{lc} * s{d}" for d, lc in enumerate(local)] + ["origin"]
        )
        pixels.append((c, inside, index))
    load = "\n        ".join(
        f"Y i{c} = {index}; bool in{c} = {inside}; "
        f"bool f{c} = in{c} && x[i{c}] != 0;"
        for c, inside, index in pixels
    )
    occ = " | ".join(f"(f{c} << {c})" for c, _, _ in pixels)
    root = "-1"
    for c, _, _ in reversed(pixels):
        root = f"(f{c} ? i{c} : {root})"
    store = "\n        ".join(
        f"if (in{c}) y[i{c}] = f{c} ? root : -1;" for c, _, _ in pixels
    )
    code = f"""
        {coords}
        Y origin = {origin};
        {load}
        Y root = {root};
        {store}
        occupancy[i] = {occ};
        """
    return cupy.ElementwiseKernel(
        "raw X x, raw int32 shape, raw int32 bshape",
        "raw Y y, raw uint8 occupancy",
        code,
        "cucim_skimage_measure_label_block_init",
    )


@functools.cache
def _kernel_block_connect(ndim):
    """Join the blocks that have neighboring foreground pixels.

    One thread per block. For each of the neighbor blocks that precede the
    block in raster order, the blocks are joined if both have a foreground
    pixel on the sides that face each other: the pixels with local coordinate
    1 along the axes where the neighbor is at +1, 0 where it is at -1, and
    any along the axes where it is at 0. (Any two such pixels are neighbors.)
    """
    coords, origin = _block_code(ndim)
    deltas = _half_structure_dirs(numpy.ones((3,) * ndim, dtype=bool))
    ndeltas = len(deltas)

    def facing(delta):
        mask = 0
        for c in range(2**ndim):
            local = [(c >> (ndim - 1 - d)) & 1 for d in range(ndim)]
            if all(
                dd == 0 or lc == (1 if dd > 0 else 0)
                for dd, lc in zip(delta, local)
            ):
                mask |= 1 << c
        return mask

    delta_table = ", ".join(str(int(v)) for dv in deltas for v in dv)
    own_table = ", ".join(str(facing(dv)) for dv in deltas.tolist())
    other_table = ", ".join(
        str(facing([-v for v in dv])) for dv in deltas.tolist()
    )
    outside = " || ".join(
        f"b{d} + deltas[dr * {ndim} + {d}] < 0 || "
        f"b{d} + deltas[dr * {ndim} + {d}] >= bshape[{d}]"
        for d in range(ndim)
    )
    block_stride = ["1"]
    for d in range(ndim - 1, 0, -1):
        block_stride.insert(0, f"{block_stride[0]} * bshape[{d}]")
    neighbor_block = " + ".join(
        f"deltas[dr * {ndim} + {d}] * ({block_stride[d]})" for d in range(ndim)
    )
    neighbor_origin = " + ".join(
        f"2 * deltas[dr * {ndim} + {d}] * s{d}" for d in range(ndim)
    )
    code = f"""
        {_ATOMIC_T_CODE}
        static const int deltas[{ndeltas * ndim}] = {{{delta_table}}};
        static const unsigned char own_faces[{ndeltas}] = {{{own_table}}};
        static const unsigned char other_faces[{ndeltas}] = {{{other_table}}};
        unsigned int mine = occupancy[i];
        if (mine == 0) continue;
        {coords}
        Y origin = {origin};
        for (int dr = 0; dr < {ndeltas}; dr++) {{
            unsigned int a = mine & own_faces[dr];
            if (a == 0) continue;
            if ({outside}) continue;
            unsigned int c = occupancy[i + {neighbor_block}] & other_faces[dr];
            if (c == 0) continue;
            int abit = __ffs(a) - 1;
            int cbit = __ffs(c) - 1;
            Y j = origin + {_local_offset(ndim, "abit")};
            Y k = origin + {neighbor_origin} + {_local_offset(ndim, "cbit")};
            {_UNION_CODE}
        }}
        """
    return cupy.ElementwiseKernel(
        "raw int32 shape, raw int32 bshape, raw uint8 occupancy",
        "raw Y y",
        code,
        "cucim_skimage_measure_label_block_connect",
    )


def _block_pixels(ndim):
    """Statements that give, for each pixel c of a block, its index i<c>, whether
    it is inside the image (in<c>) and whether it is foreground (f<c>, from the
    occupancy byte ``occ``)."""
    out = []
    for c in range(2**ndim):
        local = [(c >> (ndim - 1 - d)) & 1 for d in range(ndim)]
        inside = " && ".join(
            f"2 * b{d} + {lc} < shape[{d}]" for d, lc in enumerate(local)
        )
        index = " + ".join(
            [f"{lc} * s{d}" for d, lc in enumerate(local)] + ["origin"]
        )
        out.append(
            f"Y i{c} = {index}; bool in{c} = {inside}; "
            f"bool f{c} = (occ >> {c}) & 1;"
        )
    return "\n        ".join(out)


_BLOCK_PARAMS = "raw int32 shape, raw int32 bshape, raw uint8 occupancy"


@functools.cache
def _kernel_block_flatten(ndim):
    """Find the root of each block and count the roots (one thread per block).

    ``block_root[i]`` is the root of the tree of the block (-1 for an empty
    block). A block is a root if its first foreground pixel is one.
    """
    coords, origin = _block_code(ndim)
    rep = f"origin + {_local_offset(ndim, '(__ffs(occ) - 1)')}"
    code = f"""
        unsigned int occ = occupancy[i];
        if (occ == 0) {{ block_root[i] = -1; continue; }}
        {coords}
        Y origin = {origin};
        Y rep = {rep};
        Y root = rep;
        while (root != y[root]) {{ root = y[root]; }}
        block_root[i] = root;
        if (root == rep) atomicAdd(&count[0], 1);
        """
    return cupy.ElementwiseKernel(
        _BLOCK_PARAMS,
        "raw Y y, raw Y block_root, raw Y count",
        code,
        "cucim_skimage_measure_label_block_flatten",
    )


@functools.cache
def _kernel_block_collect(ndim):
    """Store the index of each root, in no particular order."""
    coords, origin = _block_code(ndim)
    rep = f"origin + {_local_offset(ndim, '(__ffs(occ) - 1)')}"
    code = f"""
        unsigned int occ = occupancy[i];
        if (occ == 0) continue;
        {coords}
        Y origin = {origin};
        Y rep = {rep};
        if (block_root[i] != rep) continue;
        Y j = atomicAdd(&count[1], 1);
        labels[j] = rep;
        """
    return cupy.ElementwiseKernel(
        _BLOCK_PARAMS,
        "raw Y block_root, raw Y count, raw Y labels",
        code,
        "cucim_skimage_measure_label_block_collect",
    )


@functools.cache
def _kernel_block_finalize(ndim):
    """Write the labels (the rank of the root + 1, 0 for the background)."""
    coords, origin = _block_code(ndim)
    pixels = _block_pixels(ndim)
    store = "\n        ".join(
        f"if (in{c}) y[i{c}] = f{c} ? label : 0;" for c in range(2**ndim)
    )
    code = f"""
        unsigned int occ = occupancy[i];
        {coords}
        Y origin = {origin};
        {pixels}
        Y label = 0;
        if (occ != 0) {{
            Y yi = block_root[i];
            Y j_min = 0;
            Y j_max = maxlabel - 1;
            Y j = (j_min + j_max) / 2;
            while (j_min < j_max) {{
                if (yi == labels[j]) break;
                if (yi < labels[j]) j_max = j - 1;
                else j_min = j + 1;
                j = (j_min + j_max) / 2;
            }}
            label = j + 1;
        }}
        {store}
        """
    return cupy.ElementwiseKernel(
        _BLOCK_PARAMS + ", raw Y block_root, Y maxlabel, raw Y labels",
        "raw Y y",
        code,
        "cucim_skimage_measure_label_block_finalize",
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
