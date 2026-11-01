# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT

"""Tests of ``method="vega"`` of ``marching_cubes``.

The tests do not compare with another implementation of the method. They check
properties that the corrected interior test of Vega et al. gives: the
topology of the surface of each cell and of closed fields equals the one of
the trilinear interpolant (checked on a finer grid), the topology does not
depend on the orientation of the field, and the case selection equals the one
of the reference implementation (reference data in _vega_reference_cells.py).
The mirror fields and the checks of the symmetries follow the tests of the
isoext project (MIT License, Copyright (c) 2023 Guangyan Cai).
"""

import itertools

import cupy as cp
import numpy as np
import pytest
from _vega_reference_cells import VEGA_REFERENCE_CELLS
from cupyx.scipy import ndimage as ndi
from scipy.ndimage import map_coordinates

from cucim.skimage.measure import marching_cubes
from cucim.skimage.measure._marching_cubes_vega import (
    _VEGA_CORNERS,
    _vega_count_cells_gpu,
)


def _cell_volume(vega_values):
    """2 x 2 x 2 volume from the corner values in the Vega corner order."""
    volume = np.zeros((2, 2, 2), dtype=np.float32)
    for corner, value in zip(_VEGA_CORNERS, vega_values):
        volume[corner] = value
    return volume


def _patch_topology(faces):
    """(Euler characteristic, number of components) of a triangle mesh."""
    faces = np.asarray(faces)
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    edges.sort(axis=1)
    n_edges = len(np.unique(edges, axis=0))
    vertices = np.unique(faces)
    euler = len(vertices) - n_edges + len(faces)

    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b, c in faces.tolist():
        parent[find(b)] = find(a)
        parent[find(c)] = find(a)
    components = len({find(a) for a in vertices.tolist()})
    return euler, components


def _edge_multiplicities(faces):
    faces = np.asarray(faces)
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    edges.sort(axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return counts


def _signed_volume(vertices, faces):
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)


def _mesh(volume, method, level=0.0, **kwargs):
    vertices, faces, _, _ = marching_cubes(
        cp.asarray(volume), level, method=method, **kwargs
    )
    return vertices.get(), faces.get()


def _trilinear_upsampled(volume, n_fine):
    """The values of the trilinear interpolant of a volume on a finer grid."""
    axes = [np.linspace(0, s - 1, n_fine * (s - 1) + 1) for s in volume.shape]
    coords = np.meshgrid(*axes, indexing="ij")
    return map_coordinates(volume, coords, order=1).astype(np.float32)


def _closed_field(seed, size, sigma):
    """A smooth random field that is positive on the border."""
    cp.random.seed(seed)
    values = ndi.gaussian_filter(
        cp.random.standard_normal((size,) * 3).astype(cp.float32),
        sigma,
        mode="nearest",
    )
    values[0] = values[-1] = 1.0
    values[:, 0] = values[:, -1] = 1.0
    values[:, :, 0] = values[:, :, -1] = 1.0
    return values.get()


def test_vega_is_accepted_and_the_default_is_unchanged():
    volume = cp.asarray(_closed_field(0, 16, 1.5))
    default = marching_cubes(volume, 0.0)
    lewiner = marching_cubes(volume, 0.0, method="lewiner")
    vega = marching_cubes(volume, 0.0, method="vega")
    for a, b in zip(default, lewiner):
        cp.testing.assert_array_equal(a, b)
    assert vega[0].dtype == cp.float32 and vega[1].dtype == cp.int32
    assert vega[0].shape == vega[2].shape
    assert vega[0].shape[0] == vega[3].shape[0]


def test_vega_matches_lewiner_where_no_cell_is_ambiguous():
    """The vertices are the same, the triangles split the quads differently."""
    n = 32
    grid = np.ogrid[tuple(slice(-1.0, 1.0, complex(n)) for _ in range(3))]
    image = sum((c / s) ** 2 for c, s in zip(grid, (0.75, 0.9, 1.1))) - 0.55
    volume = image.astype(np.float32)
    vl, fl = _mesh(volume, "lewiner")
    vv, fv = _mesh(volume, "vega")
    np.testing.assert_array_equal(np.unique(vl, axis=0), np.unique(vv, axis=0))
    assert vl.shape == vv.shape
    assert fl.shape == fv.shape
    assert _patch_topology(fl) == _patch_topology(fv)
    volume_l, volume_v = _signed_volume(vl, fl), _signed_volume(vv, fv)
    assert np.sign(volume_l) == np.sign(volume_v)
    np.testing.assert_allclose(volume_v, volume_l, rtol=1e-3)


@pytest.mark.parametrize("gradient_direction", ["descent", "ascent"])
def test_vega_winding_follows_gradient_direction_as_lewiner(
    gradient_direction,
):
    volume = _closed_field(3, 20, 1.5)
    volume_signs = []
    for method in ("lewiner", "vega"):
        vertices, faces = _mesh(
            volume,
            method,
            gradient_direction=gradient_direction,
            allow_degenerate=False,
        )
        volume_signs.append(np.sign(_signed_volume(vertices, faces)))
    assert volume_signs[0] == volume_signs[1]
    descent = _mesh(volume, "vega", gradient_direction="descent")
    ascent = _mesh(volume, "vega", gradient_direction="ascent")
    np.testing.assert_array_equal(descent[1], ascent[1][:, ::-1])


def test_vega_matches_the_case_selection_of_the_reference():
    n_ambiguous = 0
    for offset, reversed_, values in VEGA_REFERENCE_CELLS:
        volume = cp.asarray(_cell_volume(values))
        _, _, row_offsets, _ = _vega_count_cells_gpu(volume, 0.0)
        row = int(row_offsets[0])
        assert (row & 0x7FFF, row >> 15) == (offset, reversed_), values
        n_ambiguous += offset >= 159
    assert n_ambiguous > 100


@pytest.mark.parametrize(
    "values, n_triangles",
    [
        # case 10.1.1, the examples of the paper that other implementations
        # misidentify as 10.1.2 (figure 7)
        ([-7, -6, 2, 3, 2, 3, -7, -6], 4),
        ([-12, -10, 2, 3, 4, 6, -6, -5], 4),
        # case 10.2 with a center vertex (figure 9)
        ([-4, -9, 5, 8, 8, 5, -9, -4], 8),
    ],
)
def test_vega_paper_examples(values, n_triangles):
    _, faces = _mesh(_cell_volume(values), "vega")
    assert faces.shape[0] == n_triangles


def test_vega_paper_example_with_two_valid_patterns():
    # figure 8: the products of the opposite vertices of a face are equal,
    # case 3.1 or 3.2 (both patterns are valid)
    _, faces = _mesh(_cell_volume([-6, -3, -3, 6, 4, -2, -2, -4]), "vega")
    assert faces.shape[0] in (2, 4)


def test_vega_cell_topology_equals_the_trilinear_interpolant():
    rng = np.random.default_rng(11)
    n_checked = 0
    while n_checked < 300:
        values = rng.uniform(-1, 1, (2, 2, 2)).astype(np.float32)
        if (values > 0).all() or (values < 0).all():
            continue
        fine = _trilinear_upsampled(values, 16)
        expected = _patch_topology(_mesh(fine, "lewiner", 1e-4)[1])
        assert _patch_topology(_mesh(values, "vega", 1e-4)[1]) == expected
        n_checked += 1


@pytest.mark.parametrize("seed", [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
def test_vega_closed_field_is_watertight_and_has_the_trilinear_topology(seed):
    volume = _closed_field(seed, 5, 0.8)
    vertices, faces = _mesh(volume, "vega", 1e-3)
    assert (_edge_multiplicities(faces) == 2).all()
    fine = _trilinear_upsampled(volume, 8)
    expected = _patch_topology(_mesh(fine, "lewiner", 1e-3)[1])
    assert _patch_topology(faces) == expected


def _cube_symmetries():
    for perm in itertools.permutations(range(3)):
        for r in range(8):
            yield perm, [d for d in range(3) if r >> d & 1]


@pytest.mark.parametrize("seed", range(3))
def test_vega_topology_does_not_depend_on_the_orientation_of_the_field(seed):
    volume = _closed_field(seed, 9, 1.0)
    reference = _patch_topology(_mesh(volume, "vega", 1e-3)[1])
    for perm, flips in _cube_symmetries():
        transformed = np.transpose(volume, perm)
        if flips:
            transformed = np.flip(transformed, flips)
        faces = _mesh(np.ascontiguousarray(transformed), "vega", 1e-3)[1]
        assert _patch_topology(faces) == reference, (perm, flips)


def test_vega_mirrored_cells_have_the_same_topology():
    # a cell for which the topology of the Lewiner method depends on the
    # orientation (the second field of the tests of isoext)
    values = np.asarray(
        [0.5625, -0.4375, 0.0625, 0.5625, 0.1875, 0.125, -0.4375, 0.125],
        dtype=np.float32,
    ).reshape(2, 2, 2)
    mirrored = np.ascontiguousarray(values.transpose(2, 1, 0))
    lewiner = [
        _patch_topology(_mesh(v, "lewiner")[1]) for v in (values, mirrored)
    ]
    assert lewiner[0] != lewiner[1]
    vega = [_patch_topology(_mesh(v, "vega")[1]) for v in (values, mirrored)]
    assert vega[0] == vega[1]
    # the expected topology is the one of the trilinear interpolant
    expected = _patch_topology(
        _mesh(_trilinear_upsampled(values, 16), "lewiner")[1]
    )
    assert vega[0] == expected


def test_vega_mask_and_step_size():
    volume = cp.asarray(_closed_field(5, 24, 1.5))
    full = marching_cubes(volume, 0.0, method="vega")
    all_true = marching_cubes(
        volume, 0.0, method="vega", mask=cp.ones(volume.shape, dtype=bool)
    )
    for a, b in zip(full, all_true):
        cp.testing.assert_array_equal(a, b)
    mask = cp.zeros(volume.shape, dtype=bool)
    mask[:12] = True
    half = marching_cubes(volume, 0.0, method="vega", mask=mask)
    assert 0 < half[1].shape[0] < full[1].shape[0]
    stepped = marching_cubes(volume, 0.0, method="vega", step_size=2)
    assert stepped[1].shape[0] < full[1].shape[0]


def test_vega_allow_degenerate_false_removes_the_degenerate_faces():
    # values equal to the level make triangles of zero area
    volume = np.zeros((4, 4, 4), dtype=np.float32)
    volume[1:3, 1:3, 1:3] = 1.0
    vertices, faces = _mesh(volume, "vega", 0.5, allow_degenerate=False)
    triangles = vertices[faces]
    assert np.all(
        np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            ),
            axis=1,
        )
        > 0
    )
