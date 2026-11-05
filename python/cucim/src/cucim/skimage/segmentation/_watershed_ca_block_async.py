# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Block-asynchronous key relaxation for standard 2D and 3D watershed.

The `_watershed_ca.py` module describes the lexicographic path-cost keys and
implements the global-memory nD relaxation. The block-asynchronous variant
here follows the tiling scheme of:

P. Quesada-Barriuso, D.B. Heras, F. Argüello, Efficient 2D and 3D watershed on
graphics processing unit: block-asynchronous approaches based on cellular
automata, Computers & Electrical Engineering, Volume 39, Issue 8, 2013,
pp. 2638-2655, ISSN 0045-7906,
:DOI:`10.1016/j.compeleceng.2013.04.020`

As in the plain block-asynchronous algorithm of its Section 4.2, each block
loads a tile plus a one-pixel halo into shared memory, iterates until the tile
is stable (here, or until an iteration cap is reached), writes the tile back
to global memory, and the kernel is relaunched until no tile changes. The rest
differs from that paper:

- Their watershed is an unseeded hill-climbing cellular automaton that
  floods from all regional minima; this one relaxes path-cost keys from
  markers.
- They update each tile in synchronous steps with the halo held fixed
  during a launch, and note that their underlying automaton is
  non-deterministic. Because data then propagate at different speeds inside
  and across tiles, watershed lines on plateaus are misplaced; the
  artifact-free variant of their Section 4.3 fixes this with a separately
  computed distance to the plateau's lower border. Here
  keys are updated in place, so threads may read keys their neighbors wrote
  in the same step and tiles read halos that other tiles are changing,
  plateaus are ordered by the step count in the keys, and because the
  relaxation has a unique fixed point the converged keys (and hence the
  labels) are identical to those of the global-memory kernel and do not
  depend on GPU scheduling.
- Each tile records the last launch in which it improved a key. A block only
  relaxes its tile if the tile or one of its neighboring tiles (which supply
  its halo) improved in the previous or the current launch; otherwise
  nothing it reads can have changed since it last converged, and it exits
  before loading the tile. Late in the relaxation only the tiles around the
  moving flood front do any work.
"""

import math

import cupy as cp

from ._watershed_ca import (
    _KERNEL_PREAMBLE,
    _KEY_PREAMBLE,
    _get_neighbor_offsets,
    _launch_until_converged,
)

# Tile shapes (in C order) processed by one thread block. Each thread owns
# one pixel of the tile; the shared-memory tile adds a one-pixel halo.
TILE_2D = (16, 32)
TILE_3D = (8, 8, 8)
# Single-row images (1D images, or images whose other dimensions have size
# one) use long tiles so the flood can cross many pixels per launch.
TILE_ROW = (1, 512)
HALO = 1

# Default caps on shared-memory iterations per launch. Blocks exit the inner
# loop early once their tile stops changing.
DEFAULT_INNER_ITERATIONS_2D = 64
DEFAULT_INNER_ITERATIONS_3D = 32
DEFAULT_INNER_ITERATIONS_ROW = TILE_ROW[1]


@cp.memoize(for_each_device=True)
def _get_block_key_relax_kernel(ndim, connectivity, inner_iterations, tile):
    """Get the shared-memory tiled key relaxation kernel for 2D or 3D.

    ``tile_stamp`` holds, per tile, the last launch index in which the tile
    improved any key (0 initially). A block skips its tile unless the stamp
    of the tile or of a neighboring tile is at least ``launch - 1``. A change
    made by a neighboring tile after this block read its halo is therefore
    picked up in the next launch, and a converged launch (no improvement in
    any tile) implies every tile is at the fixed point.

    Parameters
    ----------
    ndim : int
        Number of dimensions (2 or 3).
    connectivity : int
        Neighborhood connectivity (1 to ndim).
    inner_iterations : int
        Maximum number of shared-memory iterations per launch.
    tile : tuple of int
        Tile shape in C order. The last axis maps to ``threadIdx.x``.

    Returns
    -------
    kernel : cupy.RawKernel
        Compiled CUDA kernel
    """
    ext = [t + 2 * HALO for t in tile]
    shared_size = math.prod(ext)
    num_threads = math.prod(tile)
    shared_strides = [math.prod(ext[j + 1 :]) for j in range(ndim)]

    # The last array axis maps to x, the one before it to y, then z.
    thread_axes = ["threadIdx.x", "threadIdx.y", "threadIdx.z"]
    block_axes = ["blockIdx.x", "blockIdx.y", "blockIdx.z"]
    axes = range(ndim)

    thread_coords = "\n    ".join(
        f"const int t_{j} = {thread_axes[ndim - 1 - j]};" for j in axes
    )
    # Global coordinate of shared-memory element 0 along each axis.
    origins = "\n    ".join(
        f"const int o_{j} = {block_axes[ndim - 1 - j]} * {tile[j]} - {HALO};"
        for j in axes
    )
    tid_expr = "t_0"
    for j in range(1, ndim):
        tid_expr = f"({tid_expr}) * {tile[j]} + t_{j}"

    # Decompose a linear shared index ``s`` into per-axis shared coordinates.
    decompose = ["int _rem = s;"]
    for j in range(ndim - 1, 0, -1):
        decompose.append(f"const int sc_{j} = _rem % {ext[j]};")
        decompose.append(f"_rem /= {ext[j]};")
    decompose.append("const int sc_0 = _rem;")
    decompose = "\n        ".join(decompose)

    def flat_global(coord):
        expr = f"{coord}_0"
        for j in range(1, ndim):
            expr = f"({expr}) * dim_{j} + {coord}_{j}"
        return expr

    def in_bounds(coord):
        return " && ".join(
            f"{coord}_{j} >= 0 && {coord}_{j} < dim_{j}" for j in axes
        )

    load_coords = "\n        ".join(
        f"const int g_{j} = o_{j} + sc_{j};" for j in axes
    )
    own_coords = "\n        ".join(
        f"const int p_{j} = o_{j} + {HALO} + t_{j};" for j in axes
    )
    sidx_expr = " + ".join(
        f"(t_{j} + {HALO}) * {shared_strides[j]}" for j in axes
    )
    dim_params = ", ".join(f"int dim_{j}" for j in axes)

    neighbor_min = "\n".join(
        f"""                {{
                    unsigned long long nkey = s_key[sidx + ({offset})];
                    if (nkey < best) best = nkey;
                }}"""
        for offset in (
            sum(o * s for o, s in zip(off, shared_strides))
            for off in _get_neighbor_offsets(ndim, connectivity)
        )
    )

    # Neighbor tile offsets from ``tid`` in base 3 (last axis fastest), and
    # the flat tile index in the CUDA grid.
    tile_lines = ["int _rem = tid;"]
    for j in range(ndim - 1, -1, -1):
        tile_lines.append(
            f"const int b_{j} = {block_axes[ndim - 1 - j]} + _rem % 3 - 1;"
        )
        tile_lines.append("_rem /= 3;")
    grid_axes = ["gridDim.x", "gridDim.y", "gridDim.z"]
    tile_bounds = " && ".join(
        f"b_{j} >= 0 && b_{j} < {grid_axes[ndim - 1 - j]}" for j in axes
    )
    tile_index = "b_0"
    tile_flat = block_axes[ndim - 1]
    for j in range(1, ndim):
        tile_index = f"({tile_index}) * {grid_axes[ndim - 1 - j]} + b_{j}"
        tile_flat = (
            f"({tile_flat}) * {grid_axes[ndim - 1 - j]} + "
            f"{block_axes[ndim - 1 - j]}"
        )
    tile_lines.append(
        f"if ({tile_bounds}) "
        f"neighborhood_changed = tile_stamp[{tile_index}] >= launch - 1;"
    )
    tile_neighbor_code = "\n        ".join(tile_lines)

    name = f"watershed_key_relax_block_{ndim}d"
    kernel_code = (
        _KERNEL_PREAMBLE
        + _KEY_PREAMBLE
        + f"""
extern "C" __global__
void {name}(
    const unsigned int* __restrict__ cost,
    unsigned long long* keys,
    int* __restrict__ changed,
    int* tile_stamp,
    int launch,
    {dim_params}
) {{
    __shared__ unsigned long long s_key[{shared_size}];
    __shared__ unsigned int s_cost[{shared_size}];

    {thread_coords}
    {origins}
    const int tid = {tid_expr};

    // ========================================
    // Phase 0: Skip tiles with a quiet neighborhood
    // ========================================
    // One thread per tile in the 3^{ndim} neighborhood (including this one)
    // checks whether that tile improved a key in the previous or current
    // launch.
    int neighborhood_changed = 0;
    if (tid < {3**ndim}) {{
        {tile_neighbor_code}
    }}
    if (!__syncthreads_or(neighborhood_changed)) return;

    // ========================================
    // Phase 1: Cooperative load of tile + halo
    // ========================================
    // Out-of-image elements act as masked pixels: they never update and are
    // never the minimal neighbor.
    for (int s = tid; s < {shared_size}; s += {num_threads}) {{
        {decompose}
        {load_coords}
        if ({in_bounds("g")}) {{
            int gidx = {flat_global("g")};
            s_key[s] = keys[gidx];
            s_cost[s] = cost[gidx];
        }} else {{
            s_key[s] = WS_KEY_UNREACHED;
            s_cost[s] = WS_COST_MASKED;
        }}
    }}
    __syncthreads();

    // ========================================
    // Phase 2: Relax keys in shared memory
    // ========================================
    // Halo elements stay fixed during the launch; tile elements are updated
    // in place until no thread in the block improves its key.
    const int sidx = {sidx_expr};
    const unsigned int my_cost = s_cost[sidx];
    const unsigned long long initial_key = s_key[sidx];
    const bool active = my_cost != WS_COST_MASKED && !ws_is_marker(initial_key);
    unsigned long long my_key = initial_key;

    for (int iter = 0; iter < {inner_iterations}; iter++) {{
        int improved = 0;
        if (active) {{
            unsigned long long best = WS_KEY_UNREACHED;
{neighbor_min}
            if (best != WS_KEY_UNREACHED) {{
                unsigned long long new_key = ws_extend_key(best, my_cost);
                if (new_key < my_key) {{
                    my_key = new_key;
                    s_key[sidx] = new_key;
                    improved = 1;
                }}
            }}
        }}
        if (!__syncthreads_or(improved)) break;
    }}

    // ========================================
    // Phase 3: Write improved keys back
    // ========================================
    // Only in-image pixels can be active, so the write is in bounds.
    const bool any_improved = my_key < initial_key;
    if (any_improved) {{
        {own_coords}
        keys[{flat_global("p")}] = my_key;
    }}
    if (__syncthreads_or(any_improved) && tid == 0) {{
        tile_stamp[{tile_flat}] = launch;
        *changed = 1;
    }}
}}
"""
    )
    return cp.RawKernel(kernel_code, name)


def _relax_keys_block_async(
    cost,
    keys,
    ndim,
    connectivity,
    image_shape,
    max_iterations,
    inner_iterations=None,
    convergence_check_interval=None,
):
    """Relax path-cost keys in place with the tiled 2D/3D kernel."""
    if ndim == 2 and image_shape[0] == 1:
        tile = TILE_ROW
        if inner_iterations is None:
            inner_iterations = DEFAULT_INNER_ITERATIONS_ROW
    elif ndim == 2:
        tile = TILE_2D
        if inner_iterations is None:
            inner_iterations = DEFAULT_INNER_ITERATIONS_2D
    elif ndim == 3:
        tile = TILE_3D
        if inner_iterations is None:
            inner_iterations = DEFAULT_INNER_ITERATIONS_3D
    else:
        raise NotImplementedError(
            f"Block-async only supports 2D and 3D, got {ndim}D"
        )

    kernel = _get_block_key_relax_kernel(
        ndim, connectivity, inner_iterations, tile
    )
    # CUDA block/grid dimensions are ordered (x, y, z), i.e. reversed with
    # respect to the C-order image axes.
    block_size = tuple(reversed(tile))
    grid_size = tuple(
        (int(s) + t - 1) // t
        for s, t in zip(reversed(image_shape), reversed(tile))
    )
    changed = cp.zeros(1, dtype=cp.int32)
    # Launches are numbered from 1, so the initial stamps of 0 make every
    # tile active in the first launch.
    tile_stamp = cp.zeros(math.prod(grid_size), dtype=cp.int32)
    dim_args = tuple(int(s) for s in image_shape)
    launch_index = [0]

    def launch():
        launch_index[0] += 1
        kernel(
            grid_size,
            block_size,
            (
                cost,
                keys,
                changed,
                tile_stamp,
                cp.int32(launch_index[0]),
                *dim_args,
            ),
        )

    # Each convergence check is a device->host sync (~15-20 us per launch).
    # Launches past convergence are cheap because every tile is then quiet
    # and exits before loading, so checking every 4th launch is 7-19% faster
    # overall than checking every launch. A CUDA graph conditional while
    # node could move this loop to the device and remove the remaining
    # syncs for a further modest gain, but CuPy does not expose conditional
    # nodes, so that would require a cuda-python dependency.
    if convergence_check_interval is None:
        convergence_check_interval = 4
    _launch_until_converged(
        launch, changed, max_iterations, convergence_check_interval
    )
