# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests of the 64-bit index kernels of ``marching_cubes``.

Volumes with 2**31 / 3 or more nodes (about 894**3), and meshes of that size,
use kernels with 64-bit indices. The tests run these kernels on small volumes
(by lowering the limit) and compare with the 32-bit kernels, which must give
exactly the same mesh.
"""

import cupy as cp
import numpy as np
import pytest

from cucim.skimage.measure import (
    _marching_cubes_common as common,
    marching_cubes,
)


def _volume(shape, seed):
    rng = np.random.default_rng(seed)
    noise = rng.random(shape).astype(np.float32)
    # a smooth field with a few holes, so that the surface has many cases
    grid = np.meshgrid(*[np.linspace(-1, 1, s) for s in shape], indexing="ij")
    radius = np.sqrt(sum(g**2 for g in grid)).astype(np.float32)
    return cp.asarray(radius + 0.15 * noise)


@pytest.mark.parametrize("method", ["lewiner", "lorensen", "vega"])
@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("allow_degenerate", [True, False])
def test_wide_kernels_match_narrow(
    method, use_mask, allow_degenerate, monkeypatch
):
    volume = _volume((23, 29, 31), seed=0)
    mask = None
    if use_mask:
        mask = cp.asarray(np.random.default_rng(1).random(volume.shape) > 0.2)
    kwargs = dict(
        level=0.7,
        method=method,
        mask=mask,
        allow_degenerate=allow_degenerate,
    )
    narrow = marching_cubes(volume, **kwargs)

    monkeypatch.setattr(common, "_WIDE_LIMIT", 0)
    assert common._wide(1)
    wide = marching_cubes(volume, **kwargs)
    for a, b in zip(narrow, wide):
        assert a.dtype == b.dtype
        cp.testing.assert_array_equal(a, b)


def test_wide_scan_matches_cumsum(monkeypatch):
    monkeypatch.setattr(common, "_WIDE_LIMIT", 0)
    counts = cp.asarray(
        np.random.default_rng(2).integers(0, 4, 3_000_001), dtype=cp.uint8
    )
    result = common._inclusive_scan_uint8(counts, min_size=0)
    cp.testing.assert_array_equal(result, cp.cumsum(counts, dtype=cp.int32))


def test_too_many_vertices_raises():
    with pytest.raises(ValueError, match="too large"):
        common._check_int32_range(2**31, "the number of vertices")
    common._check_int32_range(2**31 - 1, "the number of vertices")


@pytest.mark.parametrize("method", ["lewiner", "lorensen", "vega"])
def test_sphere_in_a_volume_above_the_32_bit_limit(method):
    # 3 * 900**3 edge indices exceed 2**31: the volume needs the 64-bit kernels
    n = 900
    free, _ = cp.cuda.runtime.memGetInfo()
    if free < 16 * 2**30:
        pytest.skip("needs about 16 GB of free device memory")
    center = (n - 1) / 2
    radius = 0.3 * n
    axis = cp.arange(n, dtype=cp.float32) - center
    volume = cp.empty((n, n, n), dtype=cp.float32)
    for i in range(n):  # one plane at a time, to limit the temporary arrays
        volume[i] = cp.sqrt(
            axis[i] ** 2 + axis[:, None] ** 2 + axis[None, :] ** 2
        )
    vertices, faces, normals, values = marching_cubes(
        volume, level=radius, method=method
    )
    del volume
    distance = cp.sqrt(((vertices - center) ** 2).sum(axis=1))
    assert float(cp.abs(distance - radius).max()) < 1e-2
    assert int(faces.min()) == 0
    assert int(faces.max()) < vertices.shape[0]
    # the area of the mesh is the one of the sphere, within the interpolation
    # error of the linear triangles
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    area = float(cp.linalg.norm(cp.cross(b - a, c - a), axis=1).sum() / 2)
    assert abs(area / (4 * np.pi * radius**2) - 1) < 1e-3
