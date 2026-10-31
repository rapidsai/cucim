# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import math

import cupy as cp
import numpy as np

from ._regionprops_gpu_utils import _get_label_run_loop, _warp_includes

__all__ = [
    "regionprops_euler",
    "regionprops_perimeter",
    "regionprops_perimeter_crofton",
    "regionprops_surface_area",
]

misc_deps = dict()
misc_deps["perimeter"] = []
misc_deps["perimeter_crofton"] = []
misc_deps["euler_number"] = []
misc_deps["surface_area"] = []


# The perimeter, Crofton perimeter and Euler number of scikit-image sum a
# coefficient over the pixels (perimeter) or the 2 x 2 (x 2) windows (Crofton,
# Euler) of the region's binary image, looked up from the configuration of the
# region's pixels in a small neighborhood. To measure every region in
# isolation in one pass, the kernels below compute these configurations from
# "has the same label as this pixel" instead of "is foreground", and attribute
# each pixel's or window's coefficient to one pixel of the region, so that
# the sums can be accumulated per region by the label-run loop.


def _offset_var(offsets, radius):
    """Name of the boolean variable for the pixel at `offsets`."""
    return "same_" + "_".join(str(o + radius) for o in offsets)


def _get_neighbor_coords_code(ndim, radius):
    """Coordinates of pixel ii, and whether coordinate + offset is inside.

    Uses the cucim_coord_t type and the shape{d} variables declared by
    `_get_shape_code`.
    """
    code = """
          cucim_coord_t rem = ii;"""
    for d in range(ndim - 1, 0, -1):
        code += f"""
          cucim_coord_t coord{d} = rem % shape{d};
          rem /= shape{d};"""
    code += """
          cucim_coord_t coord0 = rem;"""
    for d in range(ndim):
        for o in range(-radius, radius + 1):
            if o < 0:
                code += f"""
          bool inside{d}_{o + radius} = coord{d} >= {-o};"""
            elif o > 0:
                code += f"""
          bool inside{d}_{o + radius} = coord{d} + {o} < shape{d};"""
    return code


def _get_same_label_code(offsets, radius):
    """Whether each pixel at `offsets` from pixel ii is inside the image and
    has label `lab`."""
    code = ""
    for off in offsets:
        var = _offset_var(off, radius)
        if not any(off):
            code += f"""
          bool {var} = true;"""
            continue
        inside = " && ".join(
            f"inside{d}_{o + radius}" for d, o in enumerate(off) if o != 0
        )
        delta = " + ".join(
            f"({o}) * stride{d}" for d, o in enumerate(off) if o != 0
        )
        code += f"""
          bool {var} = {inside}
              && labels[static_cast<long long>(ii) + {delta}] == lab;"""
    return code


def _get_shape_code(ndim, int32_coords):
    """Declarations of the coordinate type, and of the shape and strides (in
    pixels) of `labels`.

    Coordinates are unraveled from the raveled index in 32 bits when the
    image has fewer than 2**31 pixels, as 64-bit divisions are much slower.
    They are signed: with `unsigned int`, NVRTC 13.3 miscompiled the 3D Euler
    kernel for 4-byte labels and 10 pixels per thread (illegal address).
    """
    coord_t = "int" if int32_coords else "long long"
    code = f"""
      typedef {coord_t} cucim_coord_t;"""
    for d in range(ndim):
        code += f"""
      cucim_coord_t shape{d} = labels.shape()[{d}];"""
    stride = "1"
    for d in range(ndim - 1, -1, -1):
        code += f"""
      long long stride{d} = {stride};"""
        stride = f"stride{d} * shape{d}"
    return code


def _get_perimeter_op(neighborhood):
    """Coefficient of pixel ii for skimage.measure.perimeter of region `lab`.

    As in scikit-image, a border pixel has label `lab` and at least one
    neighbor (4 or 8-connected) without it, pixels outside the image count as
    background, and the coefficient of a border pixel is looked up from the
    weighted number of border pixels among its 3 x 3 neighbors.
    """
    offsets1 = list(itertools.product(range(-1, 2), repeat=2))
    offsets2 = [
        o
        for o in itertools.product(range(-2, 3), repeat=2)
        if max(map(abs, o)) == 2
    ]
    if neighborhood == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbors = [o for o in offsets1 if any(o)]

    def is_border(p):
        interior = " && ".join(
            _offset_var((p[0] + n[0], p[1] + n[1]), 2) for n in neighbors
        )
        return f"({_offset_var(p, 2)} && !({interior}))"

    code = _get_neighbor_coords_code(2, 2)
    code += _get_same_label_code(offsets1, 2)
    # only border pixels have a nonzero coefficient
    center_interior = " && ".join(_offset_var(n, 2) for n in neighbors)
    code += f"""
          if (!({center_interior})) {{"""
    code += _get_same_label_code(offsets2, 2)
    # convolution weights [[10, 2, 10], [2, 1, 2], [10, 2, 10]]
    terms = " + ".join(
        f"{2 if 0 in o else 10} * {is_border(o)}" for o in offsets1 if any(o)
    )
    code += f"""
            int config = 1 + {terms};
            acc += coefs[config];
          }}"""
    return code


def _get_windows_op(ndim, omit_image_edges=False):
    """Coefficients of the 2**ndim windows of 2 x 2 (x 2) pixels that contain
    pixel ii, for skimage.measure.perimeter_crofton or euler_number of region
    `lab`.

    The configuration of a window has one bit per pixel with label `lab`
    (bits as in the convolution weights of scikit-image). A window's
    coefficient is added by the first of its pixels (in raster order) that has
    label `lab`, so that each window is counted once per region. Windows
    partly outside the image count, as scikit-image pads the image with
    background, except, if `omit_image_edges`, the windows whose last pixel is
    outside the image.

    The coefficient of the window full of `lab` pixels must be 0, so that
    interior pixels can be skipped.
    """
    offsets1 = list(itertools.product(range(-1, 2), repeat=ndim))
    code = _get_neighbor_coords_code(ndim, 1)
    code += _get_same_label_code(offsets1, 1)
    # bit position of each axis in the configuration
    bit_position = [0, 1] if ndim == 2 else [2, 0, 1]
    all_neighbors = " && ".join(_offset_var(o, 1) for o in offsets1 if any(o))
    code += f"""
          if (!({all_neighbors})) {{"""
    for window in itertools.product((-1, 0), repeat=ndim):
        # pixels of the window in raster order, relative to pixel ii
        pixels = [
            tuple(w + r for w, r in zip(window, rel))
            for rel in itertools.product((0, 1), repeat=ndim)
        ]
        first = pixels.index((0,) * ndim)
        cond = [f"!{_offset_var(p, 1)}" for p in pixels[:first]]
        if omit_image_edges:
            cond += [f"inside{d}_2" for d in range(ndim) if window[d] == 0]
        cond = " && ".join(cond) if cond else "true"
        bits = []
        for p in pixels:
            # offsets of p from the last pixel of the window are 0 or -1
            bit = sum(
                1 << bit_position[d]
                for d in range(ndim)
                if p[d] - window[d] - 1 == -1
            )
            bits.append(f"({_offset_var(p, 1)} ? {1 << bit} : 0)")
        code += f"""
            if ({cond}) {{
              acc += coefs[{" | ".join(bits)}];
            }}"""
    code += """
          }"""
    return code


def _surface_area_groups(ndim, isotropic):
    """Neighbor offsets (as indices into ``itertools.product(range(-1, 2),
    repeat=ndim)``) that share a weight in the surface area: a direction and
    its opposite, or, with isotropic spacing, all directions of one kind
    (axis, face diagonal, body diagonal)."""
    offsets = list(itertools.product(range(-1, 2), repeat=ndim))
    groups = {}
    for k, o in enumerate(offsets):
        if not any(o):
            continue
        if isotropic:
            key = sum(1 for v in o if v)
        else:
            key = max(o, tuple(-v for v in o))
        groups.setdefault(key, []).append(k)
    return [groups[key] for key in sorted(groups)]


def _get_surface_area_op(ndim, isotropic):
    """Crofton surface-area coefficient of pixel ii for region `lab`.

    Each of the 3**ndim - 1 neighbors without label `lab` (other labels,
    background, or outside the image) marks one intercept of the region's
    boundary with the discrete line through pixel ii in that direction, and
    adds that direction's weight (`_get_surface_area_coefs`). The neighbors
    are counted in integers per group of directions with the same weight
    (`_surface_area_groups`), so that a boundary pixel needs one
    double-precision multiply-add per group. Interior pixels add nothing.
    """
    offsets1 = list(itertools.product(range(-1, 2), repeat=ndim))
    code = _get_neighbor_coords_code(ndim, 1)
    code += _get_same_label_code(offsets1, 1)
    all_neighbors = " && ".join(_offset_var(o, 1) for o in offsets1 if any(o))
    code += f"""
          if (!({all_neighbors})) {{"""
    for g, group in enumerate(_surface_area_groups(ndim, isotropic)):
        count = " + ".join(f"!{_offset_var(offsets1[k], 1)}" for k in group)
        code += f"""
            acc += coefs[{g}] * static_cast<double>({count});"""
    code += """
          }"""
    return code


@cp.memoize(for_each_device=True)
def _get_boundary_kernel(
    kind, ndim, option, pixels_per_thread, int32_coords=True
):
    """Per-region sums of the perimeter, Crofton, Euler or surface-area
    coefficients.

    kind is "perimeter" (option: neighborhood), "crofton" (option:
    omit_image_edges), "euler" (option unused) or "surface" (option:
    isotropic spacing).
    """
    if kind == "perimeter":
        op = _get_perimeter_op(option)
    elif kind == "surface":
        op = _get_surface_area_op(ndim, isotropic=option)
    else:
        op = _get_windows_op(ndim, omit_image_edges=option)
    # integer coefficients of the Euler number are summed exactly
    acc_c_type, dtype = (
        ("int", "int32") if kind == "euler" else ("double", "float64")
    )
    pre = (
        _get_shape_code(ndim, int32_coords)
        + f"""
      {acc_c_type} acc;"""
    )
    start = """
          acc = 0;"""
    flush = """
          atomicAdd(&out[lab - 1], acc);"""
    source = _get_label_run_loop(
        pixels_per_thread, pre, start, op, flush, warp_reduce=[("acc", "sum")]
    )
    return cp.ElementwiseKernel(
        f"raw X labels, raw uint64 labels_size, raw {dtype} coefs",
        f"raw {dtype} out",
        source,
        preamble=_warp_includes,
        name=(
            f"cucim_{kind}_{ndim}d_{option}_batch{pixels_per_thread}"
            + ("" if int32_coords else "_coords64")
        ),
    )


def _boundary_sums(kind, labels, coefs, option, max_label, pixels_per_thread):
    if not labels.flags.c_contiguous:
        labels = cp.ascontiguousarray(labels)
    int32_coords = labels.size < 2**31
    kernel = _get_boundary_kernel(
        kind, labels.ndim, option, pixels_per_thread, int32_coords
    )
    out = cp.zeros((max_label,), dtype=coefs.dtype)
    kernel(
        labels,
        labels.size,
        coefs,
        out,
        size=math.ceil(labels.size / pixels_per_thread),
    )
    return out


@cp.memoize(for_each_device=True)
def _get_perimeter_coefs():
    # LUT of skimage.measure.perimeter
    coefs = np.zeros(50, dtype=np.float64)
    coefs[[5, 7, 15, 17, 25, 27]] = 1
    coefs[[21, 33]] = math.sqrt(2)
    coefs[[13, 23]] = (1 + math.sqrt(2)) / 2
    return cp.asarray(coefs)


def regionprops_perimeter(
    labels,
    neighborhood=4,
    *,
    max_label=None,
    robust=True,
    labels_close=None,
    props_dict=None,
    pixels_per_thread=10,
):
    """Calculate the perimeter of each labeled region.

    writes "perimeter" to `props_dict`

    Parameters
    ----------
    labels : (M, N) ndarray
        Label image. Each region is measured separately, as in isolation.
    neighborhood : 4 or 8, optional
        Neighborhood connectivity for border pixel determination. It is used to
        compute the contour. A higher neighborhood widens the border on which
        the perimeter is computed.

    Extra Parameters
    ----------------
    max_label : int or None, optional
        The maximum label in labels can be provided to avoid recomputing it if
        it was already known.
    robust : bool, optional
        Unused. Every region is measured in isolation, as by scikit-image,
        also when it is close to or touches other regions.
    labels_close : numpy.ndarray or sequence of int
        Unused.
    props_dict : dict or None, optional
        Dictionary of properties. The output of this
        function will be stored under key "perimeter" within this dictionary.
    pixels_per_thread : int, optional
        Number of pixels processed per thread on the GPU.

    Returns
    -------
    perimeter : cp.ndarray
        The perimeter of each region (float64, indexed by label - 1).

    Notes
    -----
    Both the `perimeter` and the `perimeter_crofton` methods treat the area
    outside of the image as background, so the boundary along the image edge
    is part of the perimeter. In any case, an object touching the image edge
    likely extends outside of the field of view, so an accurate perimeter
    cannot be measured for such objects.

    If the labeled regions have holes, the hole edges will be included in this
    measurement. If this is not desired, use regionprops_label_filled to fill
    the holes and then pass the filled labels image to this function.

    References
    ----------
    .. [1] K. Benkrid, D. Crookes. Design and FPGA Implementation of
           a Perimeter Estimator. The Queen's University of Belfast.
           http://www.cs.qub.ac.uk/~d.crookes/webpubs/papers/perimeter.doc
    """
    if labels.ndim != 2:
        raise ValueError("labels must be 2D")
    if neighborhood not in [4, 8]:
        raise ValueError("neighborhood must be 4 or 8")
    if max_label is None:
        max_label = int(labels.max())

    perimeters = _boundary_sums(
        "perimeter",
        labels,
        _get_perimeter_coefs(),
        neighborhood,
        max_label,
        pixels_per_thread,
    )
    if props_dict is not None:
        props_dict["perimeter"] = perimeters
    return perimeters


@cp.memoize(for_each_device=True)
def _get_perimeter_crofton_coefs(directions):
    # LUT of skimage.measure.perimeter_crofton
    if directions == 2:
        coefs = [
            0,
            np.pi / 2,
            0,
            0,
            0,
            np.pi / 2,
            0,
            0,
            np.pi / 2,
            np.pi,
            0,
            0,
            np.pi / 2,
            np.pi,
            0,
            0,
        ]
    else:
        sq2 = math.sqrt(2)
        coefs = [
            0,
            np.pi / 4 * (1 + 1 / sq2),
            np.pi / (4 * sq2),
            np.pi / (2 * sq2),
            0,
            np.pi / 4 * (1 + 1 / sq2),
            0,
            np.pi / (4 * sq2),
            np.pi / 4,
            np.pi / 2,
            np.pi / (4 * sq2),
            np.pi / (4 * sq2),
            np.pi / 4,
            np.pi / 2,
            0,
            0,
        ]
    return cp.asarray(coefs, dtype=cp.float64)


def regionprops_perimeter_crofton(
    labels,
    directions=4,
    *,
    max_label=None,
    robust=True,
    omit_image_edges=False,
    labels_close=None,
    props_dict=None,
    pixels_per_thread=10,
):
    """Calculate the Crofton perimeter of each labeled region.

    writes "perimeter_crofton" to `props_dict`

    Parameters
    ----------
    labels : (M, N) ndarray
        Label image. Each region is measured separately, as in isolation.
    directions : 2 or 4, optional
        Number of directions used to approximate the Crofton perimeter. By
        default, 4 is used: it should be more accurate than 2.
        Computation time is the same in both cases.

    Extra Parameters
    ----------------
    max_label : int or None, optional
        The maximum label in labels can be provided to avoid recomputing it if
        it was already known.
    robust : bool, optional
        Unused. Every region is measured in isolation, as by scikit-image,
        also when it is close to or touches other regions.
    omit_image_edges : bool, optional
        If True, the windows of 2 x 2 pixels that extend past the last row or
        column of the image are not counted, so that region edges along those
        image edges do not count as perimeter. We cannot accurately estimate
        the perimeter of objects falling partly outside of `image` anyway. The
        default (False) matches scikit-image, which pads the image with
        background.
    labels_close : numpy.ndarray or sequence of int
        Unused.
    props_dict : dict or None, optional
        Dictionary of properties. The output of this
        function will be stored under key "perimeter_crofton" within this
        dictionary.
    pixels_per_thread : int, optional
        Number of pixels processed per thread on the GPU.

    Returns
    -------
    perimeter : cp.ndarray
        The perimeter of each region (float64, indexed by label - 1).

    Notes
    -----
    This measure is based on Crofton formula [1], which is a measure from
    integral geometry. It is defined for general curve length evaluation via
    a double integral along all directions. In a discrete
    space, 2 or 4 directions give a quite good approximation, 4 being more
    accurate than 2 for more complex shapes.

    Similar to :func:`~.measure.perimeter`, this function returns an
    approximation of the perimeter in continuous space.

    Both the `perimeter` and the `perimeter_crofton` methods treat the area
    outside of the image as background, so the boundary along the image edge
    is part of the perimeter. In any case, an object touching the image edge
    likely extends outside of the field of view, so an accurate perimeter
    cannot be measured for such objects.

    If the labeled regions have holes, the hole edges will be included in this
    measurement. If this is not desired, use regionprops_label_filled to fill
    the holes and then pass the filled labels image to this function.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Crofton_formula
    .. [2] S. Rivollier. Analyse d’image geometrique et morphometrique par
           diagrammes de forme et voisinages adaptatifs generaux. PhD thesis,
           2010.
           Ecole Nationale Superieure des Mines de Saint-Etienne.
           https://tel.archives-ouvertes.fr/tel-00560838
    """
    if labels.ndim != 2:
        raise ValueError("labels must be 2D")
    if directions not in [2, 4]:
        raise ValueError("directions must be 2 or 4")
    if max_label is None:
        max_label = int(labels.max())

    perimeters = _boundary_sums(
        "crofton",
        labels,
        _get_perimeter_crofton_coefs(directions),
        bool(omit_image_edges),
        max_label,
        pixels_per_thread,
    )
    if props_dict is not None:
        props_dict["perimeter_crofton"] = perimeters
    return perimeters


@cp.memoize(for_each_device=True)
def _get_euler_coefs(ndim, connectivity):
    from cucim.skimage.measure._regionprops_utils import (
        EULER_COEFS2D_4,
        EULER_COEFS2D_8,
        EULER_COEFS3D_26,
    )

    if ndim == 2:
        coefs = EULER_COEFS2D_4 if connectivity == 1 else EULER_COEFS2D_8
    elif connectivity == 2:
        raise NotImplementedError(
            "For 3D images, Euler number is implemented "
            "for connectivities 1 and 3 only"
        )
    elif connectivity == 1:
        coefs = EULER_COEFS3D_26[::-1]
    else:
        coefs = EULER_COEFS3D_26
    return cp.asarray(coefs, dtype=cp.int32)


def regionprops_euler(
    labels,
    connectivity=None,
    *,
    max_label=None,
    robust=True,
    labels_close=None,
    props_dict=None,
    pixels_per_thread=10,
):
    """Calculate the Euler characteristic of each labeled region.

    For 2D objects, the Euler number is the number of objects minus the number
    of holes. For 3D objects, the Euler number is obtained as the number of
    objects plus the number of holes, minus the number of tunnels, or loops.

    writes "euler_number" to `props_dict`

    Parameters
    ----------
    labels: (M, N[, P]) cupy.ndarray
        Label image. Each region is measured separately, as in isolation.
    connectivity : int, optional
        Maximum number of orthogonal hops to consider a pixel/voxel
        as a neighbor.
        Accepted values are ranging from  1 to input.ndim. If ``None``, a full
        connectivity of ``input.ndim`` is used.
        4 or 8 neighborhoods are defined for 2D images (connectivity 1 and 2,
        respectively).
        6 or 26 neighborhoods are defined for 3D images, (connectivity 1 and 3,
        respectively). Connectivity 2 is not defined.

    Extra Parameters
    ----------------
    max_label : int or None, optional
        The maximum label in labels can be provided to avoid recomputing it if
        it was already known.
    robust : bool, optional
        Unused. Every region is measured in isolation, as by scikit-image,
        also when it is close to or touches other regions.
    labels_close : numpy.ndarray or sequence of int
        Unused.
    props_dict : dict or None, optional
        Dictionary of properties. The output of this
        function will be stored under key "euler_number" within this dictionary.
    pixels_per_thread : int, optional
        Number of pixels processed per thread on the GPU.

    Returns
    -------
    euler_number : cp.ndarray of int
        Euler characteristic of each region.

    Notes
    -----
    The Euler characteristic is an integer number that describes the
    topology of the set of all objects in the input image. If object is
    4-connected, then background is 8-connected, and conversely.

    The computation of the Euler characteristic is based on an integral
    geometry formula in discretized space. In practice, a neighborhood
    configuration is constructed, and a LUT is applied for each
    configuration. The coefficients used are the ones of Ohser et al.

    It can be useful to compute the Euler characteristic for several
    connectivities. A large relative difference between results
    for different connectivities suggests that the image resolution
    (with respect to the size of objects and holes) is too low.

    References
    ----------
    .. [1] S. Rivollier. Analyse d’image geometrique et morphometrique par
           diagrammes de forme et voisinages adaptatifs generaux. PhD thesis,
           2010. Ecole Nationale Superieure des Mines de Saint-Etienne.
           https://tel.archives-ouvertes.fr/tel-00560838
    .. [2] Ohser J., Nagel W., Schladitz K. (2002) The Euler Number of
           Discretized Sets - On the Choice of Adjacency in Homogeneous
           Lattices. In: Mecke K., Stoyan D. (eds) Morphology of Condensed
           Matter. Lecture Notes in Physics, vol 600. Springer, Berlin,
           Heidelberg.
    """
    if labels.ndim not in [2, 3]:
        raise ValueError("only 2D and 3D images are supported")
    if max_label is None:
        max_label = int(labels.max())
    # check connectivity
    if connectivity is None:
        connectivity = labels.ndim

    euler_number = _boundary_sums(
        "euler",
        labels,
        _get_euler_coefs(labels.ndim, connectivity),
        False,
        max_label,
        pixels_per_thread,
    )
    if labels.ndim == 3:
        # as int(0.125 * coefs @ h) in scikit-image (truncation toward zero)
        euler_number = (euler_number * 0.125).astype(cp.int64)
    else:
        euler_number = euler_number.astype(cp.int64)
    if props_dict is not None:
        props_dict["euler_number"] = euler_number
    return euler_number


# Fraction of the unit sphere in the Voronoi cells of a direction and its
# opposite, among the 26 directions to the neighbors of a voxel: the weights
# of the axis, face-diagonal and body-diagonal directions in the discrete
# Crofton formula for the surface area (Lehmann and Legland, 2012; the same
# values as in ITK's ShapeLabelMapFilter). They are computed by
# scipy.spatial.SphericalVoronoi in the tests.
_CROFTON_3D_WEIGHTS = (
    0.09155578240952184,  # axis (6 of the 26 directions)
    0.07396125575215028,  # face diagonal (12)
    0.07039127956463334,  # body diagonal (8)
)


def _is_isotropic(spacing):
    return spacing is None or len(set(float(v) for v in spacing)) == 1


def _get_surface_area_coefs(spacing):
    """Weight of each group of neighbor directions in the surface area
    (groups of `_surface_area_groups`).

    Crofton's formula estimates the area of a surface from the number of its
    intercepts with lines: S = 4 * sum_k c_k * A_k, where the projected area
    in direction k is A_k = n_k / (2 * lambda_k) for n_k intercepts with lines
    of density lambda_k = |d_k| / V (V the voxel volume, d_k the
    spacing-scaled step between neighbors). Each intercept is found once,
    from the voxel of the region next to it, so it adds
    4 * c_k * V / (2 * |d_k|). As in ITK, the weights c_k do not depend on the
    spacing.
    """
    isotropic = _is_isotropic(spacing)
    spacing = (1.0, 1.0, 1.0) if spacing is None else spacing
    spacing = np.asarray(spacing, dtype=np.float64)
    volume = float(np.prod(spacing))
    offsets = list(itertools.product(range(-1, 2), repeat=3))
    coefs = []
    for group in _surface_area_groups(3, isotropic):
        o = np.asarray(offsets[group[0]])
        length = math.sqrt(float(np.sum((o * spacing) ** 2)))
        weight = _CROFTON_3D_WEIGHTS[np.count_nonzero(o) - 1]
        coefs.append(2.0 * weight * volume / length)
    return cp.asarray(coefs, dtype=cp.float64)


def regionprops_surface_area(
    labels,
    spacing=None,
    *,
    max_label=None,
    props_dict=None,
    pixels_per_thread=10,
):
    """Estimate the surface area of each labeled region of a 3D image.

    writes "surface_area" to `props_dict`

    The estimate is the discrete Crofton formula with the 13 directions to
    the neighbors of a voxel (axes, face diagonals and body diagonals), as
    computed by ITK's ``ShapeLabelMapFilter`` for the ``Perimeter`` of 3D
    label objects [1]_. Each region is measured as in isolation: neighbors
    with another label count as background, and the image is padded with
    background. With anisotropic spacing the direction weights are not
    adapted to the spacing, also as in ITK.

    Parameters
    ----------
    labels : (M, N, P) cupy.ndarray
        Label image.
    spacing : tuple of float, optional
        The voxel spacing along each axis.

    Extra Parameters
    ----------------
    max_label : int or None, optional
        The maximum label in labels can be provided to avoid recomputing it if
        it was already known.
    props_dict : dict or None, optional
        Dictionary of properties. The output of this function will be stored
        under key "surface_area" within this dictionary.
    pixels_per_thread : int, optional
        Number of pixels processed per thread on the GPU.

    Returns
    -------
    surface_area : cp.ndarray
        The estimated surface area of each region, of shape (max_label,).

    References
    ----------
    .. [1] G. Lehmann and D. Legland, "Efficient N-dimensional surface
           estimation using Crofton formula and run-length encoding", The
           Insight Journal, 2012. :DOI:`10.54294/wdu86d`
    """
    if labels.ndim != 3:
        raise ValueError("surface_area is defined for 3D images only")
    if max_label is None:
        max_label = int(labels.max())
    surface_area = _boundary_sums(
        "surface",
        labels,
        _get_surface_area_coefs(spacing),
        _is_isotropic(spacing),
        max_label,
        pixels_per_thread,
    )
    if props_dict is not None:
        props_dict["surface_area"] = surface_area
    return surface_area
