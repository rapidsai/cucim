# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Watershed segmentation by parallel path-cost relaxation (global kernels).

This module implements the global-memory kernels: the nD key relaxation for
standard watershed, label resolution, and compact watershed. The relaxation
is in the spirit of the cellular-automaton (Ford-Bellman) GPU watersheds of
Kauffmann and Piché (2008, 2010). See the Notes and References of
`watershed` for how each part relates to the literature.

Standard watershed (``compactness == 0``)
-----------------------------------------
Each pixel carries a 64-bit path-cost key compared lexicographically::

    key = (ordered_bits(L) << 32) | D

``L`` is the minimax path cost from the nearest marker (the flooding level at
which the pixel is reached) and ``D`` counts the steps taken since the path
last rose to level ``L``. Markers have ``D == 0`` and every other reached
pixel has ``D >= 1``. This is the lexicographic watershed cost of Lotufo and
Falcão (2000), written as ``C = [M, d]`` by Felkel et al. (2001), who note
that a sequential FIFO queue keeps the second component implicitly; it is
the analogue of scikit-image's FIFO ``age`` tie-break. A parallel relaxation
has no queue, so ``D`` is stored explicitly, and packing both components into
one integer makes every comparison and update a single 64-bit operation.

The relaxation computes, for every unlabeled pixel ``q``::

    key(q) = min over neighbors p of max((cost(q) << 32) | 1, key(p) + 1)

This extension is monotone and strictly increases the key along every edge,
so the system has a unique fixed point. It is reached by any update order
(double-buffered, in-place, or block-asynchronous), which is what makes the
result deterministic even though the kernels update keys in place without
synchronization. Both kernels skip pixels (or tiles) unless a neighbor
improved in the previous or current launch, since otherwise nothing they
read can have changed; this is the same idea as the active-vertex masks of
GPU shortest-path relaxation (Harish and Narayanan, 2007) and the pixel
activation map of cudaIFT (Gonçalves et al., 2019).

Labels are resolved afterwards, as root labels of the optimum-path forest of
the image foresting transform (Falcão et al., 2004): each pixel's parent is
its first neighbor (in a fixed order) holding the minimal key, and the parent
pointers are followed to a marker by pointer jumping
(https://en.wikipedia.org/wiki/Pointer_jumping), as in the path-reducing GPU
watershed of Yeghiazaryan and Voiculescu (WACV 2018): each thread walks its
own chain for a bounded number of steps per launch, replacing every pointer
it passes with its grandparent (path halving). This reproduces
scikit-image's choice of the first-popped neighbor up to ties in the full
priority-queue order.

Compact watershed (``compactness > 0``)
---------------------------------------
The priority of reaching a pixel ``q`` from a neighbor ``p`` is
``max(image[q] + compactness * |q - source(p)|, P(p))``, where ``source(p)``
is the marker ``p`` was reached from (following scikit-image's
implementation of Neubert and Protzel, 2014). It depends on that marker, so
it is not a path cost and the fixed point is not unique. The compact path
therefore uses a double-buffered (Jacobi) relaxation of labels, keys and
source coordinates, which is deterministic for a given input. Keys have the
same ``(P, D)`` layout as above, and candidates with equal priority are
ranked by the offering neighbor's key, approximating scikit-image's FIFO
tie-break. A per-pixel "changed in the previous sweep" flag lets quiet
regions skip both the neighbor scan and the copy between buffers.

See `_watershed_ca_block_async.py` for the shared-memory tiled variant of the
standard-watershed key relaxation. `watershed` uses it for images with at most
three dimensions of size greater than one (after dropping dimensions of size
one and treating a 1D image as a single row), so the global key kernel here is
only needed for higher-dimensional images or when explicitly requested.
"""

import itertools

import cupy as cp

# Preamble for CUDA kernels that use fixed-width integer types (libcudacxx)
_KERNEL_PREAMBLE = "#include <cuda/std/cstdint>\n"

# Mapping from CuPy/NumPy integer dtype to fixed-width C type for kernel codegen
_DTYPE_TO_CTYPE = {
    cp.dtype("int8"): "cuda::std::int8_t",
    cp.dtype("uint8"): "cuda::std::uint8_t",
    cp.dtype("int16"): "cuda::std::int16_t",
    cp.dtype("uint16"): "cuda::std::uint16_t",
    cp.dtype("int32"): "cuda::std::int32_t",
    cp.dtype("uint32"): "cuda::std::uint32_t",
    cp.dtype("int64"): "cuda::std::int64_t",
    cp.dtype("uint64"): "cuda::std::uint64_t",
}

# Device helpers shared by the path-cost key kernels.
_KEY_PREAMBLE = r"""
// Ordered cost of a pixel outside the mask. Such pixels never update and
// are never chosen as a parent.
#define WS_COST_MASKED 0xFFFFFFFFu
// Key of a pixel that has not (yet) been reached from any marker.
#define WS_KEY_UNREACHED 0xFFFFFFFFFFFFFFFFull

// Map a float to an unsigned int with the same ordering, so that keys can be
// compared as integers. -0.0 is mapped like +0.0, and NaN sorts above +inf.
__device__ __forceinline__ unsigned int ws_ordered_cost(float v) {
    if (isnan(v)) return WS_COST_MASKED - 1u;
    v += 0.0f;
    unsigned int u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Markers are the only reached pixels with a zero step count.
__device__ __forceinline__ bool ws_is_marker(unsigned long long key) {
    return (key & 0xFFFFFFFFull) == 0ull;
}

// Key of a pixel with ordered cost ``cost`` reached from a neighbor holding
// ``key``: entering a higher level restarts the step count at 1, otherwise
// the path stays on the neighbor's level and takes one more step.
__device__ __forceinline__ unsigned long long ws_extend_key(
    unsigned long long key, unsigned int cost
) {
    unsigned long long entry = (((unsigned long long)cost) << 32) | 1ull;
    unsigned long long step = key + 1ull;
    return step > entry ? step : entry;
}
"""


def _get_neighbor_offsets_1d():
    """Get 1D neighbor offsets (left and right)."""
    return [(-1,), (1,)]


def _get_neighbor_offsets_2d(connectivity):
    """Get 2D neighbor offsets based on connectivity.

    Parameters
    ----------
    connectivity : int
        1 for 4-connectivity, 2 for 8-connectivity

    Returns
    -------
    neighbors : list of tuple
        List of (dy, dx) offsets for neighbors
    """
    # fmt: off
    if connectivity == 1:
        # 4-connectivity (cross pattern)
        return [
            (-1,  0),  # top
            ( 1,  0),  # bottom
            ( 0, -1),  # left
            ( 0,  1),  # right
        ]
    else:
        # 8-connectivity (square pattern)
        return [
            (-1,  0),  # top
            ( 1,  0),  # bottom
            ( 0, -1),  # left
            ( 0,  1),  # right
            (-1, -1),  # top-left
            (-1,  1),  # top-right
            ( 1, -1),  # bottom-left
            ( 1,  1),  # bottom-right
        ]
    # fmt: on


def _get_neighbor_offsets_3d(connectivity):
    """Get 3D neighbor offsets based on connectivity.

    Parameters
    ----------
    connectivity : int
        1 for 6-connectivity (face neighbors)
        2 for 18-connectivity (face + edge neighbors)
        3 for 26-connectivity (face + edge + corner neighbors)

    Returns
    -------
    neighbors : list of tuple
        List of (dz, dy, dx) offsets for neighbors
    """
    # fmt: off
    # Face neighbors (6-connectivity)
    face_neighbors = [
        (-1,  0,  0),
        ( 1,  0,  0),
        ( 0, -1,  0),
        ( 0,  1,  0),
        ( 0,  0, -1),
        ( 0,  0,  1),
    ]

    if connectivity == 1:
        return face_neighbors

    # Edge neighbors (12 additional)
    edge_neighbors = [
        (-1, -1,  0),
        (-1,  1,  0),
        (-1,  0, -1),
        (-1,  0,  1),
        ( 1, -1,  0),
        ( 1,  1,  0),
        ( 1,  0, -1),
        ( 1,  0,  1),
        ( 0, -1, -1),
        ( 0, -1,  1),
        ( 0,  1, -1),
        ( 0,  1,  1),
    ]

    if connectivity == 2:
        return face_neighbors + edge_neighbors

    # Corner neighbors (8 additional)
    corner_neighbors = [
        (-1, -1, -1),
        (-1, -1,  1),
        (-1,  1, -1),
        (-1,  1,  1),
        ( 1, -1, -1),
        ( 1, -1,  1),
        ( 1,  1, -1),
        ( 1,  1,  1),
    ]
    # fmt: on

    # connectivity == 3: 26-connectivity
    return face_neighbors + edge_neighbors + corner_neighbors


def _get_neighbor_offsets(ndim, connectivity):
    """Get neighbor offsets for the given dimensionality and connectivity.

    An offset is included if the number of non-zero components is at most
    ``connectivity``. This matches the convention used by
    ``scipy.ndimage.generate_binary_structure(ndim, connectivity)``.

    Parameters
    ----------
    ndim : int
        Number of dimensions.
    connectivity : int
        Maximum number of orthogonal steps to reach a neighbor.
        Must be in [1, ndim].

    Returns
    -------
    neighbors : list of tuple
        List of offset tuples, each of length ndim. Sorted by
        connectivity level (face neighbors first, then edge, then
        corner, etc.).
    """
    # Use hand-tuned lists for common cases (avoids numpy overhead)
    if ndim == 1:
        return _get_neighbor_offsets_1d()
    elif ndim == 2:
        return _get_neighbor_offsets_2d(connectivity)
    elif ndim == 3:
        return _get_neighbor_offsets_3d(connectivity)

    # General nD case

    # All offsets in {-1, 0, 1}^ndim, excluding the origin, filtered by
    # connectivity (number of non-zero components), sorted by level.
    offsets = []
    for off in itertools.product((-1, 0, 1), repeat=ndim):
        n_nonzero = sum(c != 0 for c in off)
        if 0 < n_nonzero <= connectivity:
            offsets.append((n_nonzero, off))
    offsets.sort(key=lambda x: x[0])
    return [off for _, off in offsets]


@cp.memoize(for_each_device=True)
def _generate_coord_code(ndim):
    """Generate CUDA code to extract nD coordinates from a flat index.

    Returns (dim_names, dim_params, size_expr, coord_code) where:
    - dim_names: list of dimension size variable names in C-order
    - dim_params: kernel parameter declaration string
    - size_expr: C expression for total size
    - coord_code: C code to compute c_0..c_{ndim-1} from idx
    """
    dim_names = [f"dim_{j}" for j in range(ndim)]
    dim_params = ", ".join(f"int {d}" for d in dim_names)
    size_expr = " * ".join(dim_names)

    coord_lines = []
    for j in range(ndim - 1, -1, -1):
        if j == ndim - 1:
            coord_lines.append(f"int c_{j} = idx % {dim_names[j]};")
            if ndim > 1:
                coord_lines.append(f"int _rem_{j} = idx / {dim_names[j]};")
        elif j == 0:
            coord_lines.append(f"int c_0 = _rem_{j + 1};")
        else:
            coord_lines.append(f"int c_{j} = _rem_{j + 1} % {dim_names[j]};")
            coord_lines.append(f"int _rem_{j} = _rem_{j + 1} / {dim_names[j]};")
    coord_code = "\n    ".join(coord_lines)

    return dim_names, dim_params, size_expr, coord_code


def _generate_neighbor_nidx(ndim, dim_names, offsets):
    """Generate CUDA code for neighbor coordinate + flat index computation.

    For each neighbor offset, generates:
    - nc_0..nc_{ndim-1} neighbor coordinate declarations
    - bounds check expression
    - nidx flat index expression

    Returns (nc_decl_str, bounds_str, nidx_expr) for each offset.
    """
    results = []
    for offset in offsets:
        nc_decls = []
        bounds = []
        for j, off in enumerate(offset):
            nc_decls.append(f"int nc_{j} = c_{j} + ({off});")
            bounds.append(f"nc_{j} >= 0 && nc_{j} < {dim_names[j]}")

        nc_decl_str = "\n        ".join(nc_decls)
        bounds_str = " && ".join(bounds)

        nidx_parts = [f"nc_{j}" for j in range(ndim)]
        nidx_expr = nidx_parts[0]
        for j in range(1, ndim):
            nidx_expr = f"({nidx_expr}) * {dim_names[j]} + {nidx_parts[j]}"
        results.append((nc_decl_str, bounds_str, nidx_expr))
    return results


def _generate_neighbor_loop(ndim, connectivity, body):
    """Generate an unrolled loop over in-bounds neighbors of ``idx``.

    ``body`` is CUDA code executed for each in-bounds neighbor, with the
    neighbor's flat index available as ``nidx``.

    Returns (dim_params, size_expr, coord_code, loop_code).
    """
    dim_names, dim_params, size_expr, coord_code = _generate_coord_code(ndim)
    neighbors = _get_neighbor_offsets(ndim, connectivity)
    loop_code = ""
    for nc_decl_str, bounds_str, nidx_expr in _generate_neighbor_nidx(
        ndim, dim_names, neighbors
    ):
        loop_code += f"""
    {{
        {nc_decl_str}
        if ({bounds_str}) {{
            int nidx = {nidx_expr};
            {body}
        }}
    }}"""
    return dim_params, size_expr, coord_code, loop_code


# Every flat-index kernel computes the global thread index in 64-bit so that
# the last block cannot overflow ``int`` before the bounds check.
_FLAT_INDEX_CODE = """
    long long _gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid >= (long long)({size_expr})) return;
    int idx = (int)_gid;"""


def _launch_until_converged(launch, changed, max_launches, check_interval):
    """Launch a relaxation kernel until it reports no change.

    ``changed`` is a one-element device flag that ``launch`` sets to a
    non-zero value whenever a launch modifies any value. It is reset once per
    batch of ``check_interval`` launches and read back (forcing a
    device->host sync) only after each batch. Launches past convergence
    within a batch are no-ops because the relaxations are idempotent at their
    fixed point.
    """
    launched = 0
    while launched < max_launches:
        changed.fill(0)
        batch = min(check_interval, max_launches - launched)
        for _ in range(batch):
            launch()
        launched += batch
        if int(changed[0]) == 0:
            break
    return launched


#############################################
# Standard watershed: lexicographic path keys #
#############################################


@cp.memoize(for_each_device=True)
def _get_key_init_kernel():
    """Elementwise kernel computing ordered costs and initial keys."""
    return cp.ElementwiseKernel(
        "float32 image, M marker, raw uint8 mask, int32 has_mask",
        "uint32 cost, uint64 key",
        """
        if (has_mask && !mask[i]) {
            cost = WS_COST_MASKED;
            key = WS_KEY_UNREACHED;
        } else {
            cost = ws_ordered_cost(image);
            key = (marker != 0) ? (((unsigned long long)cost) << 32)
                                : WS_KEY_UNREACHED;
        }
        """,
        "cucim_watershed_key_init",
        preamble=_KEY_PREAMBLE,
    )


@cp.memoize(for_each_device=True)
def _get_key_relax_kernel(ndim, connectivity):
    """Get the global-memory key relaxation kernel for nD watershed.

    Keys are updated in place. Concurrent launches of other threads may be
    observed or not; either way the keys decrease monotonically toward the
    unique fixed point (see the module docstring).

    ``stamp`` holds, per pixel, the index (mod 256) of the last launch in
    which the pixel's key improved; it starts at 0 and launches are numbered
    from 1. A pixel is only re-evaluated if a neighbor improved in the
    previous or the current launch, because otherwise none of its candidates
    can have changed since it was last evaluated. Wrapped stamps can only
    cause extra (harmless) evaluations, never missed ones.
    """
    dim_params, size_expr, coord_code, neighbor_loop = _generate_neighbor_loop(
        ndim,
        connectivity,
        "unsigned long long nkey = keys[nidx]; if (nkey < best) best = nkey;",
    )
    _, _, _, stamp_loop = _generate_neighbor_loop(
        ndim,
        connectivity,
        "neighbor_changed |= (unsigned char)(now - stamp[nidx]) <= 1;",
    )
    kernel_code = (
        _KERNEL_PREAMBLE
        + _KEY_PREAMBLE
        + f"""
extern "C" __global__
void watershed_key_relax(
    const unsigned int* __restrict__ cost,
    unsigned long long* keys,
    unsigned char* stamp,
    int* __restrict__ changed,
    int launch,
    {dim_params}
) {{
    {_FLAT_INDEX_CODE.format(size_expr=size_expr)}

    {coord_code}

    // Skip pixels whose neighbors did not improve recently.
    const unsigned char now = (unsigned char)launch;
    bool neighbor_changed = false;
    {stamp_loop}
    if (!neighbor_changed) return;

    unsigned int my_cost = cost[idx];
    unsigned long long my_key = keys[idx];
    if (my_cost == WS_COST_MASKED || ws_is_marker(my_key)) return;

    unsigned long long best = WS_KEY_UNREACHED;
    {neighbor_loop}

    if (best == WS_KEY_UNREACHED) return;
    unsigned long long new_key = ws_extend_key(best, my_cost);
    if (new_key < my_key) {{
        keys[idx] = new_key;
        stamp[idx] = now;
        *changed = 1;
    }}
}}
"""
    )
    return cp.RawKernel(kernel_code, "watershed_key_relax")


@cp.memoize(for_each_device=True)
def _get_parent_kernel(ndim, connectivity):
    """Get the kernel assigning each reached pixel its parent pixel.

    The parent is the first neighbor (in ``_get_neighbor_offsets`` order)
    holding the minimal key. Markers are their own parent; masked and
    unreached pixels get -1.
    """
    dim_params, size_expr, coord_code, neighbor_loop = _generate_neighbor_loop(
        ndim,
        connectivity,
        "unsigned long long nkey = keys[nidx];\n"
        "            if (nkey < best) { best = nkey; best_idx = nidx; }",
    )
    kernel_code = (
        _KERNEL_PREAMBLE
        + _KEY_PREAMBLE
        + f"""
extern "C" __global__
void watershed_parent(
    const unsigned int* __restrict__ cost,
    const unsigned long long* __restrict__ keys,
    int* __restrict__ parent,
    {dim_params}
) {{
    {_FLAT_INDEX_CODE.format(size_expr=size_expr)}

    unsigned long long my_key = keys[idx];
    if (cost[idx] == WS_COST_MASKED || my_key == WS_KEY_UNREACHED) {{
        parent[idx] = -1;
        return;
    }}
    if (ws_is_marker(my_key)) {{
        parent[idx] = idx;
        return;
    }}

    {coord_code}

    unsigned long long best = WS_KEY_UNREACHED;
    int best_idx = -1;
    {neighbor_loop}
    parent[idx] = best_idx;
}}
"""
    )
    return cp.RawKernel(kernel_code, "watershed_parent")


# Maximum number of pointer steps one thread takes per resolve launch.
_RESOLVE_MAX_STEPS = 16


@cp.memoize(for_each_device=True)
def _get_resolve_labels_kernel(label_ctype):
    """Get the kernel that follows each pixel's parent chain to its marker.

    Each thread walks from its pixel toward the root (a marker, which is its
    own parent) for at most ``max_steps`` steps, replacing every pointer it
    passes with its grandparent (path halving, a form of pointer jumping).
    It then points its own pixel at the farthest ancestor it reached. Once
    the root is reached, the root marker's label is written; otherwise
    ``changed`` is set and a further launch continues from the advanced
    pointer. Every pointer value a thread reads, stale or freshly written by
    another thread, is an ancestor on the same chain, so the labels do not
    depend on scheduling. Since each launch advances every pointer by at
    least ``max_steps`` (ever longer) hops, the number of launches grows only
    logarithmically with the chain length, while bounding the walk keeps a
    single very long chain from serializing a launch.
    """
    L = label_ctype
    kernel_code = (
        _KERNEL_PREAMBLE
        + f"""
extern "C" __global__
void watershed_resolve_labels(
    int* parent,
    const {L}* __restrict__ markers,
    {L}* __restrict__ labels,
    int* __restrict__ changed,
    int max_steps,
    int size
) {{
    long long _gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid >= size) return;
    int idx = (int)_gid;
    int up = parent[idx];
    if (up < 0) {{
        // Masked or never reached from a marker.
        labels[idx] = 0;
        return;
    }}
    int node = idx;
    for (int step = 0; step < max_steps; step++) {{
        int up2 = parent[up];
        if (up2 == up) {{
            // ``up`` is the root marker.
            if (up != parent[idx]) parent[idx] = up;
            labels[idx] = markers[up];
            return;
        }}
        parent[node] = up2;
        node = up2;
        up = parent[node];
    }}
    parent[idx] = up;
    *changed = 1;
}}
"""
    )
    return cp.RawKernel(kernel_code, "watershed_resolve_labels")


def _init_keys(image_flat, markers_flat, mask_flat, has_mask):
    """Compute ordered per-pixel costs and initial path-cost keys."""
    return _get_key_init_kernel()(
        image_flat, markers_flat, mask_flat, cp.int32(has_mask)
    )


def _relax_keys_global(
    cost,
    keys,
    ndim,
    connectivity,
    image_shape,
    max_iterations,
    convergence_check_interval=None,
):
    """Relax path-cost keys in place with the global-memory nD kernel."""
    size = cost.size
    threads_per_block = 256
    blocks = (size + threads_per_block - 1) // threads_per_block
    kernel = _get_key_relax_kernel(ndim, connectivity)
    changed = cp.zeros(1, dtype=cp.int32)
    # Launches are numbered from 1, so the initial stamps of 0 make every
    # pixel active in the first launch.
    stamp = cp.zeros(size, dtype=cp.uint8)
    dim_args = tuple(int(s) for s in image_shape)
    launch_index = [0]

    def launch():
        launch_index[0] += 1
        kernel(
            (blocks,),
            (threads_per_block,),
            (cost, keys, stamp, changed, cp.int32(launch_index[0]), *dim_args),
        )

    # Each launch is cheap, so the per-batch host sync dominates; checking
    # every 8th launch recovers most of that cost.
    if convergence_check_interval is None:
        convergence_check_interval = 8
    _launch_until_converged(
        launch, changed, max_iterations, convergence_check_interval
    )


def _labels_from_keys(cost, keys, markers_flat, ndim, connectivity, shape):
    """Resolve labels from converged keys via the parent forest."""
    size = cost.size
    threads_per_block = 256
    blocks = (size + threads_per_block - 1) // threads_per_block
    dim_args = tuple(int(s) for s in shape)

    parent = cp.empty(size, dtype=cp.int32)
    _get_parent_kernel(ndim, connectivity)(
        (blocks,), (threads_per_block,), (cost, keys, parent, *dim_args)
    )

    labels = cp.empty(size, dtype=markers_flat.dtype)
    resolve_kernel = _get_resolve_labels_kernel(
        _DTYPE_TO_CTYPE[markers_flat.dtype]
    )
    changed = cp.zeros(1, dtype=cp.int32)
    resolve_args = (
        parent,
        markers_flat,
        labels,
        changed,
        cp.int32(_RESOLVE_MAX_STEPS),
        cp.int32(size),
    )

    def launch():
        resolve_kernel((blocks,), (threads_per_block,), resolve_args)

    # Typically one or two launches; a chain as long as the whole image
    # still needs only a handful. Check after every launch.
    _launch_until_converged(launch, changed, max(1, size), 1)
    return labels


###################################
# Compact n-dimensional watershed #
###################################


@cp.memoize(for_each_device=True)
def _get_compact_init_kernel(ndim, label_ctype="cuda::std::int32_t"):
    """Get initialization kernel for compact CA-watershed.

    Markers get key ``ordered_cost(image) << 32`` (step count 0) and their
    own coordinates as source; other pixels start unreached.

    Parameters
    ----------
    ndim : int
        Number of dimensions (for source coordinate extraction).
    label_ctype : str
        C type for label arrays.
    """
    L = label_ctype
    dim_names, dim_params, size_expr, coord_code = _generate_coord_code(ndim)
    src_names = [f"source_{j}" for j in range(ndim)]
    src_params = ", ".join(f"int* {s}" for s in src_names)
    src_init_unset = "\n        ".join(f"{s}[idx] = -1;" for s in src_names)
    src_init_marker = "\n        ".join(
        f"{s}[idx] = c_{j};" for j, s in enumerate(src_names)
    )

    kernel_code = (
        _KERNEL_PREAMBLE
        + _KEY_PREAMBLE
        + f"""
extern "C" __global__
void watershed_compact_init(
    const float* image,
    const {L}* markers,
    {L}* labels,
    unsigned char* state,
    unsigned long long* keys,
    {src_params},
    const unsigned char* mask,
    int has_mask,
    {dim_params}
) {{
    {_FLAT_INDEX_CODE.format(size_expr=size_expr)}

    {coord_code}

    if (has_mask && !mask[idx]) {{
        labels[idx] = 0;
        state[idx] = 0;
        keys[idx] = WS_KEY_UNREACHED;
        {src_init_unset}
        return;
    }}

    {L} marker_label = markers[idx];

    if (marker_label != 0) {{
        labels[idx] = marker_label;
        state[idx] = 1;  // LABELED
        keys[idx] = ((unsigned long long)ws_ordered_cost(image[idx])) << 32;
        {src_init_marker}
    }} else {{
        labels[idx] = 0;
        state[idx] = 2;  // UNLABELED
        keys[idx] = WS_KEY_UNREACHED;
        {src_init_unset}
    }}
}}
"""
    )
    return cp.RawKernel(kernel_code, "watershed_compact_init")


@cp.memoize(for_each_device=True)
def _get_compact_step_kernel(
    ndim,
    connectivity=1,
    label_ctype="cuda::std::int32_t",
):
    """Get iteration kernel for nD compact CA-watershed.

    Uses a 1D grid for all dimensionalities, recovering nD coordinates
    from the flat index for neighbor bounds checking.

    Each pixel holds a key ``(P, D)`` like the standard-watershed key, where
    ``P`` is the compact priority. A neighbor ``p`` offers the candidate
    priority ``max(image[q] + compactness * |q - source(p)|, P(p))``.
    Candidates are compared by priority first and then by the neighbor's own
    key, which approximates the order in which scikit-image's priority queue
    would have popped those neighbors (its FIFO ``age`` tie-break). A pixel
    only updates when its key strictly decreases.

    The relaxation is double-buffered (a synchronous-CA / Jacobi update):
    the kernel reads from input arrays (``labels_in``/``keys_in``/...) and
    writes to separate output arrays (``labels_out``/...), and the caller
    swaps the buffers each iteration. Because the compact cost depends on the
    source marker, its fixed point depends on the update order; double
    buffering keeps the result deterministic.

    To avoid streaming every cell through memory on every sweep, a
    double-buffered ``changed`` byte records whether each cell changed in the
    previous sweep. A cell whose neighbors did not change cannot find a new
    candidate and skips the neighbor scan, and a cell that did not change in
    the previous sweep already holds its value in the output buffer and
    skips the copy-through. Both shortcuts leave the result of every sweep
    unchanged.

    Parameters
    ----------
    ndim : int
        Number of dimensions.
    connectivity : int
        Neighborhood connectivity (1 to ndim).
    label_ctype : str
        C type for label arrays.

    Returns
    -------
    kernel : cupy.RawKernel
        Compiled CUDA kernel
    """
    L = label_ctype
    src_names = [f"source_{j}" for j in range(ndim)]
    src_params = ", ".join(
        f"const int* __restrict__ {s}_in, int* __restrict__ {s}_out"
        for s in src_names
    )
    src_r = [f"{s}_in" for s in src_names]
    src_w = [f"{s}_out" for s in src_names]
    best_src_decls = "\n    ".join(
        f"int best_src_{j} = -1;" for j in range(ndim)
    )
    final_src_updates = "\n        ".join(
        f"{w}[idx] = best_src_{j};" for j, w in enumerate(src_w)
    )

    # Every output cell must be written each sweep, so cells that do not
    # update copy their input forward.
    copy_lines = [
        "labels_out[idx] = labels_in[idx];",
        "keys_out[idx] = my_key;",
    ]
    copy_lines.extend(f"{w}[idx] = {r}[idx];" for r, w in zip(src_r, src_w))
    copy_through = "\n        ".join(copy_lines)

    src_reads = "\n            ".join(
        f"int nsrc_{j} = {r}[nidx];" for j, r in enumerate(src_r)
    )
    dist_decls = "\n            ".join(
        f"float d_{j} = (float)(c_{j} - nsrc_{j});" for j in range(ndim)
    )
    dist_terms = " + ".join(f"d_{j} * d_{j}" for j in range(ndim))
    best_src_updates = "\n                ".join(
        f"best_src_{j} = nsrc_{j};" for j in range(ndim)
    )
    dim_params, size_expr, coord_code, neighbor_loop = _generate_neighbor_loop(
        ndim,
        connectivity,
        f"""{L} nlabel = labels_in[nidx];
            if (nlabel != 0) {{
            {src_reads}

            {dist_decls}
            float euclidean_dist = sqrtf({dist_terms});

            // Enforce monotonicity: never cheaper than the neighbor itself.
            unsigned long long nkey = keys_in[nidx];
            unsigned int npriority = (unsigned int)(nkey >> 32);
            unsigned int priority = ws_ordered_cost(
                my_image + compactness * euclidean_dist
            );
            if (priority < npriority) priority = npriority;

            if (priority < best_priority ||
                (priority == best_priority && nkey < best_nkey)) {{
                best_priority = priority;
                best_nkey = nkey;
                best_label = nlabel;
                {best_src_updates}
            }}
            }}""",
    )

    _, _, _, neighbor_changed_loop = _generate_neighbor_loop(
        ndim,
        connectivity,
        "neighbor_changed |= changed_in[nidx] != 0;",
    )

    kernel_code = (
        _KERNEL_PREAMBLE
        + _KEY_PREAMBLE
        + f"""
extern "C" __global__
void watershed_compact_step(
    const float* __restrict__ image,
    const unsigned char* __restrict__ state,
    const {L}* __restrict__ labels_in,
    {L}* __restrict__ labels_out,
    const unsigned long long* __restrict__ keys_in,
    unsigned long long* __restrict__ keys_out,
    {src_params},
    const unsigned char* __restrict__ changed_in,
    unsigned char* __restrict__ changed_out,
    int* __restrict__ changed,
    float compactness,
    {dim_params}
) {{
    {_FLAT_INDEX_CODE.format(size_expr=size_expr)}

    // The output buffers hold this cell's value from before the previous
    // sweep, so they only need refreshing if the cell changed in that sweep.
    bool changed_prev = changed_in[idx] != 0;
    changed_out[idx] = 0;
    if (state[idx] != 2) {{
        // Fixed (marker/masked) cell: copy through unchanged.
        if (changed_prev) {{
            unsigned long long my_key = keys_in[idx];
            {copy_through}
        }}
        return;
    }}

    {coord_code}

    // Candidates only change if a neighbor changed in the previous sweep.
    bool neighbor_changed = false;
    {neighbor_changed_loop}
    if (!neighbor_changed) {{
        if (changed_prev) {{
            unsigned long long my_key = keys_in[idx];
            {copy_through}
        }}
        return;
    }}

    unsigned long long my_key = keys_in[idx];
    float my_image = image[idx];
    unsigned int best_priority = WS_COST_MASKED;
    unsigned long long best_nkey = WS_KEY_UNREACHED;
    {L} best_label = 0;
    {best_src_decls}

    {neighbor_loop}

    if (best_label != 0) {{
        unsigned long long new_key =
            (best_priority == (unsigned int)(best_nkey >> 32))
                ? best_nkey + 1ull
                : ((((unsigned long long)best_priority) << 32) | 1ull);
        if (new_key < my_key) {{
            labels_out[idx] = best_label;
            keys_out[idx] = new_key;
            {final_src_updates}
            changed_out[idx] = 1;
            *changed = 1;
            return;
        }}
    }}
    // No improvement this sweep: copy through unchanged if needed.
    if (changed_prev) {{
        {copy_through}
    }}
}}
"""
    )

    return cp.RawKernel(kernel_code, "watershed_compact_step")


def _watershed_compact(
    image_flat,
    markers_flat,
    mask_flat,
    has_mask,
    connectivity,
    ndim,
    image_shape,
    max_iterations,
    compactness,
    convergence_check_interval=None,
):
    """Run the double-buffered compact watershed relaxation.

    The priority of a pixel reached from a marker is its image value plus
    ``compactness`` times the Euclidean distance to that marker.

    Returns
    -------
    labels, keys : cupy.ndarray
        Flat label array (marker dtype) and final uint64 priority keys.
    """
    size = image_flat.size
    label_dtype = markers_flat.dtype
    label_ctype = _DTYPE_TO_CTYPE[label_dtype]
    threads_per_block = 256
    blocks = (size + threads_per_block - 1) // threads_per_block
    dim_args = tuple(int(s) for s in image_shape)

    labels = cp.empty(size, dtype=label_dtype)
    state = cp.empty(size, dtype=cp.uint8)
    keys = cp.empty(size, dtype=cp.uint64)
    sources = [cp.empty(size, dtype=cp.int32) for _ in range(ndim)]

    _get_compact_init_kernel(ndim, label_ctype)(
        (blocks,),
        (threads_per_block,),
        (
            image_flat,
            markers_flat,
            labels,
            state,
            keys,
            *sources,
            mask_flat,
            cp.int32(has_mask),
            *dim_args,
        ),
    )

    step_kernel = _get_compact_step_kernel(ndim, connectivity, label_ctype)

    # Per-cell state that is read and written each iteration, in the order
    # the kernel expects: labels, keys, sources..., changed flags. Marking
    # every cell as changed makes the first sweep evaluate and copy all
    # cells, so the output buffers need no initialization.
    bufs = {"read": [labels, keys, *sources, cp.ones(size, dtype=cp.uint8)]}
    bufs["write"] = [cp.empty_like(b) for b in bufs["read"]]
    changed = cp.zeros(1, dtype=cp.int32)
    trailing = (changed, cp.float32(compactness), *dim_args)

    def launch():
        # Interleave (in, out) for each buffer to match the kernel's paired
        # parameter order.
        paired = []
        for r, w in zip(bufs["read"], bufs["write"]):
            paired.extend((r, w))
        step_kernel(
            (blocks,),
            (threads_per_block,),
            (image_flat, state, *paired, *trailing),
        )
        bufs["read"], bufs["write"] = bufs["write"], bufs["read"]

    if convergence_check_interval is None:
        convergence_check_interval = 8
    _launch_until_converged(
        launch, changed, max_iterations, convergence_check_interval
    )

    # After the final swap, the read buffers hold the latest values.
    return bufs["read"][0], bufs["read"][1]
