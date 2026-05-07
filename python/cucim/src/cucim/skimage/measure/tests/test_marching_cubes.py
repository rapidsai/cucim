# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause


import cupy as cp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from skimage.draw import ellipsoid, ellipsoid_stats
from skimage.measure import (
    marching_cubes as skimage_marching_cubes,
    mesh_surface_area,
)

from cucim.skimage.measure import (
    _marching_cubes_lewiner_luts as mcluts,
    marching_cubes,
)
from cucim.skimage.measure._marching_cubes import (
    _LEWINER_LUT_NAMES,
    _decode_cases_classic,
    _decode_named_lut,
    _get_lewiner_luts_cpu,
    _get_lewiner_luts_device,
    _lewiner_case_needs_center_vertex,
    _lewiner_center_vertex,
    _lewiner_count_cells_gpu,
    _lewiner_generate_center_vertices_gpu,
    _lewiner_generate_edge_ids_gpu,
    _lewiner_generate_faces_from_edge_ids_gpu,
    _lewiner_tri_edges_and_trace_for_case_values,
    _lewiner_tri_edges_for_case_values,
)


def _single_voxel_volume():
    volume = cp.zeros((3, 3, 3), dtype=cp.float32)
    volume[1, 1, 1] = 1.0
    return volume


def _case_volume(values):
    volume = np.empty((2, 2, 2), dtype=np.float32)
    volume[0, 0, 0] = values[0]
    volume[1, 0, 0] = values[1]
    volume[1, 1, 0] = values[2]
    volume[0, 1, 0] = values[3]
    volume[0, 0, 1] = values[4]
    volume[1, 0, 1] = values[5]
    volume[1, 1, 1] = values[6]
    volume[0, 1, 1] = values[7]
    return cp.asarray(volume)


def _single_cell_edge_vertex_ids():
    edge_vertex_ids = np.full((2, 2, 2, 3), -1, dtype=np.int32)
    edge_to_axis_slot = {
        0: (0, 0, 0, 0),
        1: (1, 0, 0, 1),
        2: (0, 1, 0, 0),
        3: (0, 0, 0, 1),
        4: (0, 0, 1, 0),
        5: (1, 0, 1, 1),
        6: (0, 1, 1, 0),
        7: (0, 0, 1, 1),
        8: (0, 0, 0, 2),
        9: (1, 0, 0, 2),
        10: (1, 1, 0, 2),
        11: (0, 1, 0, 2),
    }
    for edge, slot in edge_to_axis_slot.items():
        edge_vertex_ids[slot] = 100 + edge
    return cp.asarray(edge_vertex_ids.reshape(-1))


_LEWINER_LOCAL_VALUE_ORDER = (0, 4, 7, 3, 1, 5, 6, 2)
_LEWINER_EDGE_TO_LOCAL = np.asarray(
    [8, 7, 11, 3, 9, 5, 10, 1, 0, 4, 6, 2, 12], dtype=np.int8
)


def _lewiner_expected_local_edges_for_volume(values):
    values = tuple(values[i] for i in _LEWINER_LOCAL_VALUE_ORDER)
    edges = np.asarray(
        _lewiner_tri_edges_for_case_values(values), dtype=np.int8
    )
    return tuple(int(v) for v in _LEWINER_EDGE_TO_LOCAL[edges])


def _case_volume_for_lewiner_values(values):
    local_values = (
        values[0],
        values[4],
        values[7],
        values[3],
        values[1],
        values[5],
        values[6],
        values[2],
    )
    return _case_volume(local_values), local_values


def _skimage_volume_for_lewiner_values(values):
    volume = np.empty((2, 2, 2), dtype=np.float32)
    volume[0, 0, 0] = values[0]
    volume[0, 0, 1] = values[1]
    volume[0, 1, 1] = values[2]
    volume[0, 1, 0] = values[3]
    volume[1, 0, 0] = values[4]
    volume[1, 0, 1] = values[5]
    volume[1, 1, 1] = values[6]
    volume[1, 1, 0] = values[7]
    return volume


def _lewiner_generated_volumes():
    coords = np.ogrid[tuple(slice(-1.0, 1.0, complex(s)) for s in (16, 18, 20))]
    x, y, z = coords
    trig_x = np.linspace(-np.pi, np.pi, 18, dtype=np.float32)[:, None, None]
    trig_y = np.linspace(-np.pi, np.pi, 20, dtype=np.float32)[None, :, None]
    trig_z = np.linspace(-np.pi, np.pi, 22, dtype=np.float32)[None, None, :]
    return [
        ((x / 0.7) ** 2 + (y / 0.9) ** 2 + (z / 1.1) ** 2 - 0.8).astype(
            np.float32
        ),
        (x + 0.35 * y - 0.2 * z).astype(np.float32),
        (np.sin(trig_x) + 0.75 * np.cos(trig_y) - 0.4 * np.sin(trig_z)).astype(
            np.float32
        ),
    ]


def test_default_lewiner_single_voxel_smoke():
    verts, faces, normals, values = marching_cubes(_single_voxel_volume(), 0.5)

    assert verts.ndim == 2
    assert faces.ndim == 2
    assert normals.shape == verts.shape
    assert values.shape == (verts.shape[0],)
    assert verts.dtype == cp.float32
    assert faces.dtype == cp.int32
    assert bool(cp.all(faces >= 0))
    assert bool(cp.all(faces < verts.shape[0]))


def test_invalid_method():
    with pytest.raises(ValueError, match="method should be either"):
        marching_cubes(_single_voxel_volume(), 0.5, method="invalid")


def test_lewiner_lut_scaffolding():
    cpu_luts = _get_lewiner_luts_cpu()
    device_luts = _get_lewiner_luts_device()

    assert set(cpu_luts) == set(_LEWINER_LUT_NAMES)
    assert set(device_luts) == {name.lower() for name in _LEWINER_LUT_NAMES}
    assert len(cpu_luts) == 47

    for name, arr in cpu_luts.items():
        expected_shape = getattr(mcluts, name)[0]
        assert arr.shape == expected_shape
        assert arr.dtype == np.int8
        assert device_luts[name.lower()].shape == expected_shape
        assert device_luts[name.lower()].dtype == cp.int8

    np.testing.assert_array_equal(
        _decode_cases_classic(), _decode_named_lut("CASESCLASSIC")
    )


def test_lewiner_reference_selector_binary_cases():
    for case in range(256):
        values = tuple(1.0 if case & (1 << i) else -1.0 for i in range(8))
        edges = _lewiner_tri_edges_for_case_values(values)
        actual = _triangles_as_sets(np.asarray(edges).reshape(-1, 3))
        expected = _skimage_lewiner_case_triangles(values)
        assert actual == expected, f"case={case}"

        tri_counts, center_flags = _lewiner_count_cells_gpu(
            _case_volume(values), 0.0
        )
        gpu_edges = _lewiner_expected_local_edges_for_volume(values)
        assert int(tri_counts[0]) == len(gpu_edges) // 3, f"case={case}"
        assert int(center_flags[0]) == (12 in gpu_edges), f"case={case}"
        edge_ids = _lewiner_generate_edge_ids_gpu(
            _case_volume(values), 0.0, tri_counts
        )
        np.testing.assert_array_equal(
            cp.asnumpy(edge_ids), np.asarray(gpu_edges)
        )


@pytest.mark.parametrize(
    "expected_table, values",
    [
        (
            "TILING4_2",
            (
                -0.010297325,
                -0.33440364,
                2.0113717,
                -0.51191173,
                21.898249,
                -2.9239168,
                -0.11596733,
                -17.542377,
            ),
        ),
        (
            "TILING7_2",
            (
                2.6393442,
                -0.022573676,
                -0.023431384,
                -0.22414387,
                -4.8741188,
                17.554071,
                -3.9130943,
                0.022010362,
            ),
        ),
        (
            "TILING7_3",
            (
                0.034643568,
                -39.678399,
                0.14743367,
                -0.032561303,
                -0.029669263,
                10.909386,
                0.022582398,
                11.287739,
            ),
        ),
        (
            "TILING7_4_1",
            (
                -0.34109717,
                27.756059,
                -5.2469554,
                -8.006981,
                87.440195,
                -0.01467409,
                0.016275834,
                -0.014158758,
            ),
        ),
        (
            "TILING10_1_1",
            (
                -3.1807288,
                0.061347338,
                -16.319789,
                11.369266,
                -0.13853397,
                0.025723929,
                -0.26795469,
                0.062523282,
            ),
        ),
        (
            "TILING10_1_2",
            (
                -7.0776404,
                0.031258997,
                57.804971,
                -12.298217,
                0.18120949,
                -0.014297345,
                -0.7980851,
                0.040058661,
            ),
        ),
        (
            "TILING10_2",
            (
                22.193552,
                -8.7039667,
                -0.95966374,
                0.015089464,
                -0.765601,
                96.78445,
                0.83834013,
                -0.036854809,
            ),
        ),
        (
            "TILING10_2_",
            (
                -31.097675,
                0.14161598,
                0.018064559,
                -0.23175924,
                12.53511,
                -1.3789733,
                -0.42235556,
                39.678691,
            ),
        ),
        (
            "TILING12_1_1",
            (
                -20.994637,
                0.22184394,
                -5.2393152,
                0.03684127,
                0.044973006,
                -7.0101958,
                -1.0957436,
                2.26831,
            ),
        ),
        (
            "TILING12_2",
            (
                81.384124,
                -51.849902,
                4.555275,
                -1.898468,
                -1.2514439,
                0.021560843,
                0.023074299,
                -5.7886924,
            ),
        ),
        (
            "TILING12_2_",
            (
                -0.030682461,
                0.31162952,
                0.86980283,
                -49.148175,
                31.592069,
                -10.208769,
                0.024636743,
                -0.6325333,
            ),
        ),
        (
            "TILING13_2",
            (
                0.052043343,
                -0.022978098,
                0.027186016,
                -13.625358,
                -58.353135,
                0.14418092,
                -0.027351424,
                6.7745484,
            ),
        ),
        (
            "TILING13_3",
            (
                -0.02928883,
                0.30373054,
                -24.882304,
                91.510656,
                0.024265332,
                -47.611617,
                0.040431047,
                -0.93368651,
            ),
        ),
        (
            "TILING13_4",
            (
                -0.074507588,
                0.05111846,
                -85.477048,
                5.9626186,
                28.537463,
                -0.13654382,
                2.4709994,
                -16.967766,
            ),
        ),
        (
            "TILING13_5_1",
            (
                1.0986236,
                -6.1231812,
                3.2394844,
                -27.677765,
                -0.026985229,
                1.7648155,
                -17.112664,
                12.499956,
            ),
        ),
    ],
)
def test_lewiner_reference_selector_ambiguous_branches(expected_table, values):
    edges, trace = _lewiner_tri_edges_and_trace_for_case_values(values)
    assert expected_table in {item[0] for item in trace}

    actual = _triangles_as_sets(np.asarray(edges).reshape(-1, 3))
    expected = _skimage_lewiner_case_triangles(values)
    assert actual == expected

    tri_counts, center_flags = _lewiner_count_cells_gpu(
        _case_volume(values), 0.0
    )
    gpu_edges = _lewiner_expected_local_edges_for_volume(values)
    assert int(tri_counts[0]) == len(gpu_edges) // 3
    assert int(center_flags[0]) == (12 in gpu_edges)
    edge_ids = _lewiner_generate_edge_ids_gpu(
        _case_volume(values), 0.0, tri_counts
    )
    np.testing.assert_array_equal(cp.asnumpy(edge_ids), np.asarray(gpu_edges))


def test_lewiner_center_vertex_detection():
    center_cases = []
    for case in range(256):
        values = tuple(1.0 if case & (1 << i) else -1.0 for i in range(8))
        needs_center = _lewiner_case_needs_center_vertex(values)
        has_center = 12 in _lewiner_tri_edges_for_case_values(values)
        assert needs_center is has_center
        if needs_center:
            center_cases.append(case)

    assert center_cases


def test_lewiner_center_vertex_formula():
    values = (4.0, -2.0, 1.0, -3.0, 5.0, -7.0, 2.0, -11.0)
    position, normal = _lewiner_center_vertex(values)

    weights = np.asarray([1 / abs(v) for v in values], dtype=np.float64)
    expected_position = np.asarray(
        [
            weights[[1, 2, 5, 6]].sum(),
            weights[[2, 3, 6, 7]].sum(),
            weights[[4, 5, 6, 7]].sum(),
        ]
    )
    expected_position /= weights.sum()

    assert_allclose(position, expected_position)
    assert_allclose(np.linalg.norm(normal), 1.0)


def test_lewiner_gpu_center_vertex_generation():
    values = (
        -0.074507588,
        0.05111846,
        -85.477048,
        5.9626186,
        28.537463,
        -0.13654382,
        2.4709994,
        -16.967766,
    )
    spacing = (2.0, 3.0, 4.0)
    volume, local_values = _case_volume_for_lewiner_values(values)
    tri_counts, center_flags = _lewiner_count_cells_gpu(volume, 0.0)
    assert int(tri_counts[0]) == 12
    assert int(center_flags[0]) == 1

    vertices, normals, out_values, center_vertex_ids = (
        _lewiner_generate_center_vertices_gpu(
            volume, 0.0, spacing, center_flags
        )
    )
    expected_position, expected_normal = _lewiner_center_vertex(local_values)

    cp.testing.assert_array_equal(center_vertex_ids, cp.asarray([0]))
    cp.testing.assert_allclose(
        vertices[0], cp.asarray(expected_position * spacing), rtol=1e-6
    )
    cp.testing.assert_allclose(
        normals[0], cp.asarray(expected_normal), rtol=1e-6, atol=1e-7
    )
    cp.testing.assert_allclose(
        out_values[0], np.max(local_values) - np.min(local_values), rtol=1e-6
    )


def test_lewiner_gpu_faces_from_edge_ids():
    values = (
        -0.074507588,
        0.05111846,
        -85.477048,
        5.9626186,
        28.537463,
        -0.13654382,
        2.4709994,
        -16.967766,
    )
    volume, _ = _case_volume_for_lewiner_values(values)
    tri_counts, center_flags = _lewiner_count_cells_gpu(volume, 0.0)
    assert int(tri_counts[0]) == 12
    assert int(center_flags[0]) == 1

    edge_ids = _lewiner_generate_edge_ids_gpu(volume, 0.0, tri_counts)
    edge_vertex_ids = _single_cell_edge_vertex_ids()
    center_vertex_ids = cp.asarray([112], dtype=cp.int32)
    expected = cp.where(
        edge_ids.reshape(-1, 3) == 12,
        cp.asarray(112, dtype=cp.int32),
        edge_ids.reshape(-1, 3).astype(cp.int32) + 100,
    )

    faces_ascent = _lewiner_generate_faces_from_edge_ids_gpu(
        edge_ids,
        tri_counts,
        edge_vertex_ids,
        center_vertex_ids,
        volume.shape,
        "ascent",
    )
    cp.testing.assert_array_equal(faces_ascent, expected)

    faces_descent = _lewiner_generate_faces_from_edge_ids_gpu(
        edge_ids,
        tri_counts,
        edge_vertex_ids,
        center_vertex_ids,
        volume.shape,
        "descent",
    )
    cp.testing.assert_array_equal(faces_descent, expected[:, ::-1])


def test_lorensen_single_voxel_smoke():
    verts, faces, normals, values = marching_cubes(
        _single_voxel_volume(), 0.5, method="lorensen"
    )

    assert verts.shape == (6, 3)
    assert faces.shape == (8, 3)
    assert normals.shape == verts.shape
    assert values.shape == (6,)
    assert verts.dtype == cp.float32
    assert faces.dtype == cp.int32
    assert bool(cp.all(faces >= 0))
    assert bool(cp.all(faces < verts.shape[0]))
    cp.testing.assert_allclose(values, cp.ones(6, dtype=cp.float32))


def test_lorensen_matches_skimage_single_voxel_mesh():
    volume = cp.asnumpy(_single_voxel_volume())
    verts, faces = marching_cubes(cp.asarray(volume), 0.5, method="lorensen")[
        :2
    ]
    expected_verts, expected_faces = skimage_marching_cubes(
        volume, 0.5, method="lorensen"
    )[:2]

    assert _same_mesh(
        cp.asnumpy(verts), cp.asnumpy(faces), expected_verts, expected_faces
    )


def test_lewiner_matches_skimage_single_voxel_mesh():
    volume = cp.asnumpy(_single_voxel_volume())
    verts, faces = marching_cubes(cp.asarray(volume), 0.5)[:2]
    expected_verts, expected_faces = skimage_marching_cubes(
        volume, 0.5, method="lewiner"
    )[:2]

    assert _same_mesh(
        cp.asnumpy(verts), cp.asnumpy(faces), expected_verts, expected_faces
    )


def test_lewiner_matches_skimage_exact_level_single_voxel_mesh():
    volume = cp.asnumpy(_single_voxel_volume())
    verts, faces = marching_cubes(cp.asarray(volume), 0.0)[:2]
    expected_verts, expected_faces = skimage_marching_cubes(
        volume, 0.0, method="lewiner"
    )[:2]

    assert bool(cp.all(faces >= 0))
    assert _same_mesh(
        cp.asnumpy(verts),
        cp.asnumpy(faces),
        expected_verts,
        expected_faces,
        tol=1e-6,
    )


def test_lewiner_matches_skimage_center_vertex_ambiguous_cell_mesh():
    values = (
        -0.074507588,
        0.05111846,
        -85.477048,
        5.9626186,
        28.537463,
        -0.13654382,
        2.4709994,
        -16.967766,
    )
    volume, _ = _case_volume_for_lewiner_values(values)
    expected_volume = _skimage_volume_for_lewiner_values(values)

    verts, faces, normals, out_values = marching_cubes(volume, 0.0)
    expected_verts, expected_faces = skimage_marching_cubes(
        expected_volume, 0.0, method="lewiner"
    )[:2]

    assert verts.shape == (13, 3)
    assert faces.shape == (12, 3)
    assert normals.shape == verts.shape
    assert out_values.shape == (verts.shape[0],)
    assert bool(cp.all(faces >= 0))
    assert _same_mesh(
        cp.asnumpy(verts),
        cp.asnumpy(faces),
        expected_verts,
        expected_faces,
        tol=1e-5,
    )


@pytest.mark.parametrize("volume", _lewiner_generated_volumes())
def test_lewiner_matches_skimage_generated_volume_stats(volume):
    verts, faces = marching_cubes(cp.asarray(volume), 0.0)[:2]
    expected_verts, expected_faces = skimage_marching_cubes(
        volume, 0.0, method="lewiner"
    )[:2]

    assert verts.shape == expected_verts.shape
    assert faces.shape == expected_faces.shape
    assert bool(cp.all(faces >= 0))
    assert bool(cp.all(faces < verts.shape[0]))
    cp_area = mesh_surface_area(cp.asnumpy(verts), cp.asnumpy(faces))
    expected_area = mesh_surface_area(expected_verts, expected_faces)
    assert_allclose(cp_area, expected_area, rtol=1e-6)


def test_lorensen_spacing():
    spacing = (2.0, 3.0, 4.0)
    verts, _, _, _ = marching_cubes(
        _single_voxel_volume(), 0.5, spacing=spacing, method="lorensen"
    )

    cp.testing.assert_allclose(verts.min(axis=0), cp.asarray([1.0, 1.5, 2.0]))
    cp.testing.assert_allclose(verts.max(axis=0), cp.asarray([3.0, 4.5, 6.0]))


def test_gradient_direction_flips_winding():
    volume = _single_voxel_volume()
    _, faces_descent, _, _ = marching_cubes(
        volume, 0.5, method="lorensen", gradient_direction="descent"
    )
    _, faces_ascent, _, _ = marching_cubes(
        volume, 0.5, method="lorensen", gradient_direction="ascent"
    )

    cp.testing.assert_array_equal(faces_descent, faces_ascent[:, ::-1])


def test_lewiner_gradient_direction_flips_winding():
    volume = _single_voxel_volume()
    _, faces_descent, _, _ = marching_cubes(
        volume, 0.5, gradient_direction="descent"
    )
    _, faces_ascent, _, _ = marching_cubes(
        volume, 0.5, gradient_direction="ascent"
    )

    cp.testing.assert_array_equal(faces_descent, faces_ascent[:, ::-1])


def test_no_surface_found():
    with pytest.raises(RuntimeError, match="No surface found"):
        marching_cubes(
            cp.zeros((3, 3, 3), dtype=cp.float32), 0.0, method="lorensen"
        )


def test_unsupported_options():
    volume = _single_voxel_volume()
    for method in ("lorensen", "lewiner"):
        with pytest.raises(NotImplementedError, match="mask"):
            marching_cubes(
                volume,
                0.5,
                method=method,
                mask=cp.ones(volume.shape, dtype=bool),
            )


def test_step_size_matches_skimage_lewiner():
    volume = ellipsoid(6, 10, 16, levelset=True).astype(np.float32)
    verts, faces = marching_cubes(cp.asarray(volume), 0.0, step_size=2)[:2]
    expected_verts, expected_faces = skimage_marching_cubes(
        volume, 0.0, step_size=2
    )[:2]

    assert _same_mesh(
        cp.asnumpy(verts),
        cp.asnumpy(faces),
        expected_verts,
        expected_faces,
        tol=1e-5,
    )


def test_step_size_lorensen_surface_area():
    volume = ellipsoid(6, 10, 16, levelset=True).astype(np.float32)
    verts, faces = marching_cubes(
        cp.asarray(volume), 0.0, method="lorensen", step_size=2
    )[:2]
    expected_verts, expected_faces = skimage_marching_cubes(
        volume, 0.0, method="lorensen", step_size=2
    )[:2]

    assert verts.shape == expected_verts.shape
    assert faces.shape == expected_faces.shape
    assert_allclose(
        mesh_surface_area(cp.asnumpy(verts), cp.asnumpy(faces)),
        mesh_surface_area(expected_verts, expected_faces),
    )


def test_allow_degenerate_false_removes_zero_area_faces():
    volume = np.array(
        [
            [[1.0, 0.0, 0.0], [-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]],
            [[1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            [[0.0, 1.0, -1.0], [1.0, 1.0, -1.0], [0.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )

    verts, faces = marching_cubes(cp.asarray(volume), 0.0)[:2]
    verts_clean, faces_clean = marching_cubes(
        cp.asarray(volume), 0.0, allow_degenerate=False
    )[:2]
    expected_verts, expected_faces = skimage_marching_cubes(
        volume, 0.0, allow_degenerate=False
    )[:2]

    assert faces_clean.shape[0] < faces.shape[0]
    assert verts_clean.shape[0] < verts.shape[0]
    assert not _has_degenerate_faces(
        cp.asnumpy(verts_clean), cp.asnumpy(faces_clean)
    )
    assert_allclose(
        mesh_surface_area(cp.asnumpy(verts_clean), cp.asnumpy(faces_clean)),
        mesh_surface_area(expected_verts, expected_faces),
    )


def test_allow_degenerate_false_both_algs_same_result_ellipse():
    sphere_small = ellipsoid(1, 1, 1, levelset=True)

    vertices1, faces1 = marching_cubes(
        cp.asarray(sphere_small), 0, allow_degenerate=False
    )[:2]
    vertices2, faces2 = marching_cubes(
        cp.asarray(sphere_small),
        0,
        allow_degenerate=False,
        method="lorensen",
    )[:2]

    assert _same_mesh(
        cp.asnumpy(vertices1),
        cp.asnumpy(faces1),
        cp.asnumpy(vertices2),
        cp.asnumpy(faces2),
    )


def test_marching_cubes_isotropic():
    ellipsoid_isotropic = ellipsoid(6, 10, 16, levelset=True)
    _, surf = ellipsoid_stats(6, 10, 16)

    # Classic
    verts, faces = marching_cubes(
        cp.asarray(ellipsoid_isotropic), 0.0, method="lorensen"
    )[:2]
    surf_calc = mesh_surface_area(cp.asnumpy(verts), cp.asnumpy(faces))
    # Test within 1% tolerance for isotropic. Will always underestimate.
    assert surf > surf_calc and surf_calc > surf * 0.99

    # Lewiner
    verts, faces = marching_cubes(cp.asarray(ellipsoid_isotropic), 0.0)[:2]
    surf_calc = mesh_surface_area(cp.asnumpy(verts), cp.asnumpy(faces))
    # Test within 1% tolerance for isotropic. Will always underestimate.
    assert surf > surf_calc and surf_calc > surf * 0.99


def test_marching_cubes_anisotropic():
    # test spacing as numpy array (and not just tuple)
    spacing = np.array([1.0, 10 / 6.0, 16 / 6.0])
    ellipsoid_anisotropic = ellipsoid(6, 10, 16, spacing=spacing, levelset=True)
    _, surf = ellipsoid_stats(6, 10, 16)

    # Classic
    verts, faces = marching_cubes(
        cp.asarray(ellipsoid_anisotropic),
        0.0,
        spacing=spacing,
        method="lorensen",
    )[:2]
    surf_calc = mesh_surface_area(cp.asnumpy(verts), cp.asnumpy(faces))
    # Test within 1.5% tolerance for anisotropic. Will always underestimate.
    assert surf > surf_calc and surf_calc > surf * 0.985

    # Lewiner
    verts, faces = marching_cubes(
        cp.asarray(ellipsoid_anisotropic), 0.0, spacing=spacing
    )[:2]
    surf_calc = mesh_surface_area(cp.asnumpy(verts), cp.asnumpy(faces))
    # Test within 1.5% tolerance for anisotropic. Will always underestimate.
    assert surf > surf_calc and surf_calc > surf * 0.985

    verts, faces = marching_cubes(
        cp.asarray(ellipsoid_anisotropic),
        0.0,
        spacing=spacing,
        allow_degenerate=False,
    )[:2]
    surf_calc = mesh_surface_area(cp.asnumpy(verts), cp.asnumpy(faces))
    # Test within 1.5% tolerance for anisotropic. Will always underestimate.
    assert surf > surf_calc and surf_calc > surf * 0.985

    # Test marching cube with mask
    with pytest.raises(ValueError):
        marching_cubes(
            cp.asarray(ellipsoid_anisotropic),
            0.0,
            spacing=spacing,
            mask=np.array([]),
            method="lorensen",
        )[:2]
    with pytest.raises(ValueError):
        marching_cubes(
            cp.asarray(ellipsoid_anisotropic),
            0.0,
            spacing=spacing,
            mask=np.array([]),
        )[:2]


def test_invalid_input():
    # Classic
    with pytest.raises(ValueError):
        marching_cubes(cp.zeros((2, 2, 1)), 0, method="lorensen")
    with pytest.raises(ValueError):
        marching_cubes(cp.zeros((2, 2, 1)), 1, method="lorensen")
    with pytest.raises(ValueError):
        marching_cubes(cp.ones((3, 3, 3)), 1, spacing=(1, 2), method="lorensen")
    with pytest.raises(ValueError):
        marching_cubes(cp.zeros((20, 20)), 0, method="lorensen")
    with pytest.raises(ValueError):
        marching_cubes(cp.zeros((3, 3, 3)), 0, method="lorensen", step_size=0)

    # Lewiner
    with pytest.raises(ValueError):
        marching_cubes(cp.zeros((2, 2, 1)), 0)
    with pytest.raises(ValueError):
        marching_cubes(cp.zeros((2, 2, 1)), 1)
    with pytest.raises(ValueError):
        marching_cubes(cp.ones((3, 3, 3)), 1, spacing=(1, 2))
    with pytest.raises(ValueError):
        marching_cubes(cp.zeros((20, 20)), 0)
    with pytest.raises(ValueError):
        marching_cubes(cp.zeros((3, 3, 3)), 0, step_size=0)

    # invalid method name
    ellipsoid_isotropic = ellipsoid(6, 10, 16, levelset=True)
    with pytest.raises(ValueError):
        marching_cubes(cp.asarray(ellipsoid_isotropic), 0.0, method="abcd")


def _same_mesh(vertices1, faces1, vertices2, faces2, tol=1e-10):
    """Compare meshes invariant to face and in-triangle vertex order."""
    triangles1 = vertices1[np.asarray(faces1)]
    triangles2 = vertices2[np.asarray(faces2)]
    triang1 = [
        np.concatenate(sorted(t, key=lambda x: tuple(x))) for t in triangles1
    ]
    triang2 = [
        np.concatenate(sorted(t, key=lambda x: tuple(x))) for t in triangles2
    ]
    triang1 = np.array(sorted([tuple(x) for x in triang1]))
    triang2 = np.array(sorted([tuple(x) for x in triang2]))
    return triang1.shape == triang2.shape and np.allclose(
        triang1, triang2, 0, tol
    )


def _has_degenerate_faces(vertices, faces):
    triangles = vertices[np.asarray(faces)]
    return bool(
        np.any(
            np.all(triangles[:, 0] == triangles[:, 1], axis=1)
            | np.all(triangles[:, 0] == triangles[:, 2], axis=1)
            | np.all(triangles[:, 1] == triangles[:, 2], axis=1)
        )
    )


_SKIMAGE_EDGE_MIDPOINTS = {
    (0.0, 0.0, 0.5): 0,
    (0.0, 0.5, 1.0): 1,
    (0.0, 1.0, 0.5): 2,
    (0.0, 0.5, 0.0): 3,
    (1.0, 0.0, 0.5): 4,
    (1.0, 0.5, 1.0): 5,
    (1.0, 1.0, 0.5): 6,
    (1.0, 0.5, 0.0): 7,
    (0.5, 0.0, 0.0): 8,
    (0.5, 0.0, 1.0): 9,
    (0.5, 1.0, 1.0): 10,
    (0.5, 1.0, 0.0): 11,
    (0.5, 0.5, 0.5): 12,
}


def _skimage_lewiner_case_triangles(values):
    if all(v < 0 for v in values) or all(v > 0 for v in values):
        return []

    volume = np.empty((2, 2, 2), dtype=np.float32)
    volume[0, 0, 0] = values[0]
    volume[0, 0, 1] = values[1]
    volume[0, 1, 1] = values[2]
    volume[0, 1, 0] = values[3]
    volume[1, 0, 0] = values[4]
    volume[1, 0, 1] = values[5]
    volume[1, 1, 1] = values[6]
    volume[1, 1, 0] = values[7]

    vertices, faces = skimage_marching_cubes(volume, 0.0, method="lewiner")[:2]
    edge_ids = [_vertex_to_skimage_edge_id(vertex) for vertex in vertices]
    return _triangles_as_sets(np.asarray(edge_ids, dtype=np.int8)[faces])


def _vertex_to_skimage_edge_id(vertex):
    vertex = np.asarray(vertex, dtype=np.float64)
    is_boundary = np.isclose(vertex, 0.0) | np.isclose(vertex, 1.0)
    if not np.any(is_boundary):
        return 12

    for midpoint, edge_id in _SKIMAGE_EDGE_MIDPOINTS.items():
        midpoint = np.asarray(midpoint)
        variable = np.isclose(midpoint, 0.5)
        fixed = ~variable
        if np.allclose(vertex[fixed], midpoint[fixed]):
            if np.all((vertex[variable] >= 0.0) & (vertex[variable] <= 1.0)):
                return edge_id
    raise AssertionError(f"could not map vertex {vertex} to a cube edge")


def _triangles_as_sets(triangles):
    return sorted(tuple(sorted(int(v) for v in tri)) for tri in triangles)


# TODO: convert remaining tests once Lewiner, mask, step_size and
# allow_degenerate=False support are available.
if False:
    from skimage.draw import ellipsoid, ellipsoid_stats
    from skimage.measure import marching_cubes, mesh_surface_area

    def test_marching_cubes_isotropic():
        ellipsoid_isotropic = ellipsoid(6, 10, 16, levelset=True)
        _, surf = ellipsoid_stats(6, 10, 16)

        # Classic
        verts, faces = marching_cubes(
            ellipsoid_isotropic, 0.0, method="lorensen"
        )[:2]
        surf_calc = mesh_surface_area(verts, faces)
        # Test within 1% tolerance for isotropic. Will always underestimate.
        assert surf > surf_calc and surf_calc > surf * 0.99

        # Lewiner
        verts, faces = marching_cubes(ellipsoid_isotropic, 0.0)[:2]
        surf_calc = mesh_surface_area(verts, faces)
        # Test within 1% tolerance for isotropic. Will always underestimate.
        assert surf > surf_calc and surf_calc > surf * 0.99

    def test_marching_cubes_anisotropic():
        # test spacing as numpy array (and not just tuple)
        spacing = np.array([1.0, 10 / 6.0, 16 / 6.0])
        ellipsoid_anisotropic = ellipsoid(
            6, 10, 16, spacing=spacing, levelset=True
        )
        _, surf = ellipsoid_stats(6, 10, 16)

        # Classic
        verts, faces = marching_cubes(
            ellipsoid_anisotropic, 0.0, spacing=spacing, method="lorensen"
        )[:2]
        surf_calc = mesh_surface_area(verts, faces)
        # Test within 1.5% tolerance for anisotropic. Will always underestimate.
        assert surf > surf_calc and surf_calc > surf * 0.985

        # Lewiner
        verts, faces = marching_cubes(
            ellipsoid_anisotropic, 0.0, spacing=spacing
        )[:2]
        surf_calc = mesh_surface_area(verts, faces)
        # Test within 1.5% tolerance for anisotropic. Will always underestimate.
        assert surf > surf_calc and surf_calc > surf * 0.985

        # Test marching cube with mask
        with pytest.raises(ValueError):
            verts, faces = marching_cubes(
                ellipsoid_anisotropic, 0.0, spacing=spacing, mask=np.array([])
            )[:2]

        # Test spacing together with allow_degenerate=False
        marching_cubes(
            ellipsoid_anisotropic, 0, spacing=spacing, allow_degenerate=False
        )

    def test_invalid_input():
        # Classic
        with pytest.raises(ValueError):
            marching_cubes(np.zeros((2, 2, 1)), 0, method="lorensen")
        with pytest.raises(ValueError):
            marching_cubes(np.zeros((2, 2, 1)), 1, method="lorensen")
        with pytest.raises(ValueError):
            marching_cubes(
                np.ones((3, 3, 3)), 1, spacing=(1, 2), method="lorensen"
            )
        with pytest.raises(ValueError):
            marching_cubes(np.zeros((20, 20)), 0, method="lorensen")

        # Lewiner
        with pytest.raises(ValueError):
            marching_cubes(np.zeros((2, 2, 1)), 0)
        with pytest.raises(ValueError):
            marching_cubes(np.zeros((2, 2, 1)), 1)
        with pytest.raises(ValueError):
            marching_cubes(np.ones((3, 3, 3)), 1, spacing=(1, 2))
        with pytest.raises(ValueError):
            marching_cubes(np.zeros((20, 20)), 0)

        # invalid method name
        ellipsoid_isotropic = ellipsoid(6, 10, 16, levelset=True)
        with pytest.raises(ValueError):
            marching_cubes(ellipsoid_isotropic, 0.0, method="abcd")

    def test_both_algs_same_result_ellipse():
        # Performing this test on data that does not have ambiguities

        sphere_small = ellipsoid(1, 1, 1, levelset=True)

        vertices1, faces1 = marching_cubes(
            sphere_small, 0, allow_degenerate=False
        )[:2]
        vertices2, faces2 = marching_cubes(
            sphere_small, 0, allow_degenerate=False, method="lorensen"
        )[:2]

        # Order is different, best we can do is test equal shape and same
        # vertices present
        assert _same_mesh(vertices1, faces1, vertices2, faces2)

    def _same_mesh(vertices1, faces1, vertices2, faces2, tol=1e-10):
        """Compare two meshes, using a certain tolerance and invariant to
        the order of the faces.
        """
        # Unwind vertices
        triangles1 = vertices1[np.array(faces1)]
        triangles2 = vertices2[np.array(faces2)]
        # Sort vertices within each triangle
        triang1 = [
            np.concatenate(sorted(t, key=lambda x: tuple(x)))
            for t in triangles1
        ]
        triang2 = [
            np.concatenate(sorted(t, key=lambda x: tuple(x)))
            for t in triangles2
        ]
        # Sort the resulting 9-element "tuples"
        triang1 = np.array(sorted([tuple(x) for x in triang1]))
        triang2 = np.array(sorted([tuple(x) for x in triang2]))
        return triang1.shape == triang2.shape and np.allclose(
            triang1, triang2, 0, tol
        )

    def test_both_algs_same_result_donut():
        # Performing this test on data that does not have ambiguities
        n = 48
        a, b = 2.5 / n, -1.25

        vol = np.empty((n, n, n), "float32")
        for iz in range(vol.shape[0]):
            for iy in range(vol.shape[1]):
                for ix in range(vol.shape[2]):
                    # Double-torii formula by Thomas Lewiner
                    z, y, x = (
                        float(iz) * a + b,
                        float(iy) * a + b,
                        float(ix) * a + b,
                    )
                    vol[iz, iy, ix] = (
                        (
                            (8 * x) ** 2
                            + (8 * y - 2) ** 2
                            + (8 * z) ** 2
                            + 16
                            - 1.85 * 1.85
                        )
                        * (
                            (8 * x) ** 2
                            + (8 * y - 2) ** 2
                            + (8 * z) ** 2
                            + 16
                            - 1.85 * 1.85
                        )
                        - 64 * ((8 * x) ** 2 + (8 * y - 2) ** 2)
                    ) * (
                        (
                            (8 * x) ** 2
                            + ((8 * y - 2) + 4) * ((8 * y - 2) + 4)
                            + (8 * z) ** 2
                            + 16
                            - 1.85 * 1.85
                        )
                        * (
                            (8 * x) ** 2
                            + ((8 * y - 2) + 4) * ((8 * y - 2) + 4)
                            + (8 * z) ** 2
                            + 16
                            - 1.85 * 1.85
                        )
                        - 64
                        * (((8 * y - 2) + 4) * ((8 * y - 2) + 4) + (8 * z) ** 2)
                    ) + 1025

        vertices1, faces1 = marching_cubes(vol, 0, method="lorensen")[:2]
        vertices2, faces2 = marching_cubes(vol, 0)[:2]

        # Old and new alg are different
        assert not _same_mesh(vertices1, faces1, vertices2, faces2)

    def test_masked_marching_cubes():
        ellipsoid_scalar = ellipsoid(6, 10, 16, levelset=True)
        mask = np.ones_like(ellipsoid_scalar, dtype=bool)
        mask[:10, :, :] = False
        mask[:, :, 20:] = False
        ver, faces, _, _ = marching_cubes(ellipsoid_scalar, 0, mask=mask)
        area = mesh_surface_area(ver, faces)

        assert_allclose(area, 299.56878662109375, rtol=0.01)

    def test_masked_marching_cubes_empty():
        ellipsoid_scalar = ellipsoid(6, 10, 16, levelset=True)
        mask = np.array([])
        with pytest.raises(ValueError):
            _ = marching_cubes(ellipsoid_scalar, 0, mask=mask)

    def test_masked_marching_cubes_all_true():
        ellipsoid_scalar = ellipsoid(6, 10, 16, levelset=True)
        mask = np.ones_like(ellipsoid_scalar, dtype=bool)
        ver_m, faces_m, _, _ = marching_cubes(ellipsoid_scalar, 0, mask=mask)
        ver, faces, _, _ = marching_cubes(ellipsoid_scalar, 0, mask=mask)
        assert_allclose(ver_m, ver, rtol=0.00001)
        assert_allclose(faces_m, faces, rtol=0.00001)
