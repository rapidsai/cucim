# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import cupy as cp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from skimage.draw import ellipsoid, ellipsoid_stats
from skimage.measure import (
    marching_cubes as skimage_marching_cubes,
    mesh_surface_area as skimage_mesh_surface_area,
)

from cucim.skimage.measure import marching_cubes, mesh_surface_area


def _single_voxel_volume():
    volume = cp.zeros((3, 3, 3), dtype=cp.float32)
    volume[1, 1, 1] = 1.0
    return volume


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
    with pytest.raises(ValueError, match="method should be one of"):
        marching_cubes(_single_voxel_volume(), 0.5, method="invalid")


def test_mesh_surface_area():
    verts = cp.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=cp.float32,
    )
    faces = cp.asarray([[0, 1, 2], [0, 1, 3]], dtype=cp.int32)

    area = mesh_surface_area(verts, faces)

    assert isinstance(area, cp.ndarray)
    cp.testing.assert_allclose(area, cp.asarray(1.5, dtype=cp.float32))

    empty_faces = cp.empty((0, 3), dtype=cp.int32)
    cp.testing.assert_array_equal(
        mesh_surface_area(verts, empty_faces), cp.asarray(0.0, dtype=cp.float32)
    )

    int_verts = verts.astype(cp.int16)
    int_area = mesh_surface_area(int_verts, faces)
    assert int_area.dtype == cp.float32
    cp.testing.assert_allclose(int_area, cp.asarray(1.5, dtype=cp.float32))

    with pytest.raises(TypeError, match="cupy.ndarray"):
        mesh_surface_area(cp.asnumpy(verts), cp.asnumpy(faces))


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
    cp_area = cp.asnumpy(mesh_surface_area(verts, faces))
    expected_area = skimage_mesh_surface_area(expected_verts, expected_faces)
    assert_allclose(cp_area, expected_area, rtol=1e-6)


@pytest.mark.parametrize("smooth", [False, True])
@pytest.mark.parametrize("level", [0.0, 0.1])
def test_lewiner_matches_skimage_noisy_volume_mesh(smooth, level):
    # Noise gives many ambiguous cells, including ones for which the
    # interior test of scikit-image returns false when its extra condition
    # (test 5 or 10) fails.
    rng = np.random.default_rng(5)
    volume = rng.standard_normal((40, 40, 40)).astype(np.float32)
    if smooth:
        from scipy import ndimage as ndi

        volume = ndi.gaussian_filter(volume, 1).astype(np.float32)
    verts, faces = marching_cubes(cp.asarray(volume), level)[:2]
    expected_verts, expected_faces = skimage_marching_cubes(
        volume, level, method="lewiner"
    )[:2]

    assert verts.shape == expected_verts.shape
    assert faces.shape == expected_faces.shape
    cp_area = cp.asnumpy(mesh_surface_area(verts, faces))
    expected_area = skimage_mesh_surface_area(expected_verts, expected_faces)
    assert_allclose(cp_area, expected_area, rtol=1e-5)
    if smooth:
        # (the sorted triangle lists of the unsmoothed noise volume are not
        # stable under float32 rounding)
        assert _same_mesh(
            cp.asnumpy(verts),
            cp.asnumpy(faces),
            expected_verts,
            expected_faces,
            tol=1e-5,
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
    volume = _skimage_volume_for_lewiner_values(values)

    verts, faces, normals, out_values = marching_cubes(cp.asarray(volume), 0.0)
    expected_verts, expected_faces = skimage_marching_cubes(
        volume, 0.0, method="lewiner"
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
        with pytest.raises(ValueError, match="same shape"):
            marching_cubes(
                volume,
                0.5,
                method=method,
                mask=cp.ones((2, 2, 2), dtype=bool),
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
        cp.asnumpy(mesh_surface_area(verts, faces)),
        skimage_mesh_surface_area(expected_verts, expected_faces),
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
        cp.asnumpy(mesh_surface_area(verts_clean, faces_clean)),
        skimage_mesh_surface_area(expected_verts, expected_faces),
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


def test_both_algs_different_result_donut():
    # Upstream scikit-image uses this double-torii formula from Thomas Lewiner
    # as a non-ambiguous volume where Lorensen and Lewiner still differ.
    n = 48
    a, b = 2.5 / n, -1.25
    z, y, x = np.ogrid[
        b : b + n * a : complex(n),
        b : b + n * a : complex(n),
        b : b + n * a : complex(n),
    ]

    first = (8 * x) ** 2 + (8 * y - 2) ** 2 + (8 * z) ** 2 + 16 - 1.85 * 1.85
    second = (
        (8 * x) ** 2
        + ((8 * y - 2) + 4) * ((8 * y - 2) + 4)
        + (8 * z) ** 2
        + 16
        - 1.85 * 1.85
    )
    volume = (
        (first * first - 64 * ((8 * x) ** 2 + (8 * y - 2) ** 2))
        * (second * second - 64 * (((8 * y - 2) + 4) ** 2 + (8 * z) ** 2))
        + 1025
    ).astype(np.float32)

    vertices1, faces1 = marching_cubes(
        cp.asarray(volume), 0, method="lorensen"
    )[:2]
    vertices2, faces2 = marching_cubes(cp.asarray(volume), 0)[:2]

    assert not _same_mesh(
        cp.asnumpy(vertices1),
        cp.asnumpy(faces1),
        cp.asnumpy(vertices2),
        cp.asnumpy(faces2),
    )


def _has_unreferenced_vertices(vertices, faces):
    used = np.zeros(vertices.shape[0], dtype=bool)
    used[np.asarray(faces).ravel()] = True
    return bool(np.any(~used))


def test_masked_marching_cubes():
    volume = ellipsoid(6, 10, 16, levelset=True).astype(np.float32)
    mask = np.ones_like(volume, dtype=bool)
    mask[:10, :, :] = False
    mask[:, :, 20:] = False

    verts, faces = marching_cubes(
        cp.asarray(volume), 0.0, mask=cp.asarray(mask)
    )[:2]
    expected_verts, expected_faces = skimage_marching_cubes(
        volume, 0.0, mask=mask
    )[:2]

    assert faces.shape == expected_faces.shape
    assert not _has_unreferenced_vertices(cp.asnumpy(verts), cp.asnumpy(faces))
    assert_allclose(
        cp.asnumpy(mesh_surface_area(verts, faces)),
        skimage_mesh_surface_area(expected_verts, expected_faces),
        rtol=1e-6,
    )


def test_masked_marching_cubes_all_true():
    volume = ellipsoid(6, 10, 16, levelset=True).astype(np.float32)
    mask = cp.ones(volume.shape, dtype=bool)

    verts_m, faces_m, normals_m, values_m = marching_cubes(
        cp.asarray(volume), 0.0, mask=mask
    )
    verts, faces, normals, values = marching_cubes(cp.asarray(volume), 0.0)

    cp.testing.assert_allclose(verts_m, verts)
    cp.testing.assert_array_equal(faces_m, faces)
    cp.testing.assert_allclose(normals_m, normals)
    cp.testing.assert_allclose(values_m, values)


def test_masked_marching_cubes_empty():
    volume = ellipsoid(6, 10, 16, levelset=True).astype(np.float32)
    mask = cp.zeros(volume.shape, dtype=bool)
    with pytest.raises(RuntimeError, match="No surface found"):
        marching_cubes(cp.asarray(volume), 0.0, mask=mask)


def test_masked_marching_cubes_with_step_size():
    volume = ellipsoid(6, 10, 16, levelset=True).astype(np.float32)
    mask = np.ones_like(volume, dtype=bool)
    mask[:10, :, :] = False
    mask[:, :, 20:] = False

    verts, faces = marching_cubes(
        cp.asarray(volume), 0.0, mask=cp.asarray(mask), step_size=2
    )[:2]
    expected_verts, expected_faces = skimage_marching_cubes(
        volume, 0.0, mask=mask, step_size=2
    )[:2]

    assert verts.shape == expected_verts.shape
    assert faces.shape == expected_faces.shape
    assert_allclose(
        cp.asnumpy(mesh_surface_area(verts, faces)),
        skimage_mesh_surface_area(expected_verts, expected_faces),
        rtol=1e-6,
    )


def test_marching_cubes_isotropic():
    ellipsoid_isotropic = ellipsoid(6, 10, 16, levelset=True)
    _, surf = ellipsoid_stats(6, 10, 16)

    # Classic
    verts, faces = marching_cubes(
        cp.asarray(ellipsoid_isotropic), 0.0, method="lorensen"
    )[:2]
    surf_calc = float(mesh_surface_area(verts, faces))
    # Test within 1% tolerance for isotropic. Will always underestimate.
    assert surf > surf_calc and surf_calc > surf * 0.99

    # Lewiner
    verts, faces = marching_cubes(cp.asarray(ellipsoid_isotropic), 0.0)[:2]
    surf_calc = float(mesh_surface_area(verts, faces))
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
    surf_calc = float(mesh_surface_area(verts, faces))
    # Test within 1.5% tolerance for anisotropic. Will always underestimate.
    assert surf > surf_calc and surf_calc > surf * 0.985

    # Lewiner
    verts, faces = marching_cubes(
        cp.asarray(ellipsoid_anisotropic), 0.0, spacing=spacing
    )[:2]
    surf_calc = float(mesh_surface_area(verts, faces))
    # Test within 1.5% tolerance for anisotropic. Will always underestimate.
    assert surf > surf_calc and surf_calc > surf * 0.985

    verts, faces = marching_cubes(
        cp.asarray(ellipsoid_anisotropic),
        0.0,
        spacing=spacing,
        allow_degenerate=False,
    )[:2]
    surf_calc = float(mesh_surface_area(verts, faces))
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


@pytest.mark.parametrize("method", ["lewiner", "lorensen"])
@pytest.mark.parametrize("gradient_direction", ["descent", "ascent"])
def test_winding_matches_skimage(method, gradient_direction):
    """The orientation (sign of the enclosed volume) follows scikit-image."""
    grid = np.ogrid[-1:1:48j, -1:1:48j, -1:1:48j]
    volume = (
        sum((c / s) ** 2 for c, s in zip(grid, (0.6, 0.75, 0.9))) - 1
    ).astype(np.float32)

    def signed_volume(vertices, faces):
        a, b, c = (np.asarray(vertices)[faces[:, i]] for i in range(3))
        return np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6

    kwargs = dict(
        method=method,
        gradient_direction=gradient_direction,
        allow_degenerate=False,
    )
    verts, faces, _, _ = marching_cubes(cp.asarray(volume), 0.0, **kwargs)
    ref_verts, ref_faces, _, _ = skimage_marching_cubes(volume, 0.0, **kwargs)
    ours = signed_volume(verts.get(), faces.get())
    ref = signed_volume(ref_verts, ref_faces)
    assert np.sign(ours) == np.sign(ref)
    np.testing.assert_allclose(ours, ref, rtol=1e-3)


def _sphere_volume(shape=(18, 19, 20)):
    x, y, z = np.mgrid[
        -1 : 1 : shape[0] * 1j, -1 : 1 : shape[1] * 1j, -1 : 1 : shape[2] * 1j
    ]
    return (x**2 + y**2 + z**2 + 0.05 * np.sin(7 * x * y)).astype(np.float32)


@pytest.mark.parametrize("method", ["lewiner", "lorensen", "vega"])
def test_nan_input_is_rejected(method):
    # a NaN in the far corner used to give face indices outside the vertices
    volume = cp.ones((2, 2, 2), dtype=cp.float32)
    volume[0, 0, 0] = 0
    volume[1, 1, 1] = cp.nan
    with pytest.raises(ValueError, match="NaN"):
        marching_cubes(volume, 0.5, method=method)


def test_level_range_is_checked_as_float32():
    # the kernels use the level as a float32, so the range check does too:
    # 1 / 3 is below its float32 value, which is the minimum of the volume
    # after the conversion, but it is not out of range
    volume = np.ones((4, 4, 4))
    volume[0, 0, 0] = 1 / 3
    try:
        marching_cubes(cp.asarray(volume), 1 / 3)
    except RuntimeError:
        pass  # no surface at the minimum, which is also what scikit-image does


@pytest.mark.parametrize("method", ["lewiner", "lorensen"])
@pytest.mark.parametrize("level", [1.0, 2.0])
@pytest.mark.parametrize("allow_degenerate", [True, False])
def test_level_equal_to_sample_values(method, level, allow_degenerate):
    # samples equal to the level count as below it, in both methods
    rng = np.random.default_rng(3)
    volume = rng.integers(0, 4, (14, 15, 16)).astype(np.float32)
    kwargs = dict(method=method, allow_degenerate=allow_degenerate)
    expected = skimage_marching_cubes(volume, level, **kwargs)
    result = marching_cubes(cp.asarray(volume), level, **kwargs)
    if allow_degenerate:
        assert result[0].shape == expected[0].shape
        assert result[1].shape == expected[1].shape
        area = skimage_mesh_surface_area(expected[0], expected[1])
        cp.testing.assert_allclose(
            mesh_surface_area(result[0], result[1]), area, rtol=1e-5
        )
    else:
        # scikit-image positions the vertices on samples with an error of
        # 1e-16, so its test for coincident vertices finds fewer of them
        assert result[1].shape[0] <= expected[1].shape[0]


@pytest.mark.parametrize("method", ["lewiner", "lorensen"])
def test_normals_have_the_sign_of_scikit_image(method):
    from scipy.spatial import cKDTree

    volume = _sphere_volume()
    expected = skimage_marching_cubes(volume, 0.6, method=method)
    result = marching_cubes(cp.asarray(volume), 0.6, method=method)
    verts, normals = cp.asnumpy(result[0]), cp.asnumpy(result[2])
    _, index = cKDTree(verts).query(expected[0])
    dot = np.sum(expected[2] * normals[index], axis=1)
    # the definitions differ (the gradient is not accumulated over the cells),
    # but the directions agree to a few degrees and have the same sign
    assert dot.min() > 0.99


def test_remove_degenerate_faces_is_repeatable():
    # the vertex merge is a concurrent union-find; the result must not depend
    # on the schedule of the threads
    rng = np.random.default_rng(5)
    volume = cp.asarray(rng.integers(0, 3, (24, 24, 24)).astype(np.float32))
    results = [
        marching_cubes(volume, 1.0, allow_degenerate=False) for _ in range(8)
    ]
    for result in results[1:]:
        assert result[0].shape == results[0][0].shape
        assert result[1].shape == results[0][1].shape


def test_wide_kernels_are_chosen_by_the_volume_size(monkeypatch):
    # a thin volume has few cells but many nodes, and the kernels index both
    from cucim.skimage.measure import _marching_cubes_common as common

    volume = cp.asarray(
        _sphere_volume((2, 2, 600))[:, :, ::-1] * 0
        + cp.asnumpy(cp.linspace(0, 1, 600, dtype=cp.float32))
    )
    expected = marching_cubes(volume, 0.5)
    seen = []
    original = common._wide

    def spy(*sizes):
        seen.append(sizes)
        return original(*sizes)

    monkeypatch.setattr(common, "_wide", spy)
    monkeypatch.setattr(common, "_WIDE_LIMIT", 3 * volume.size)
    result = marching_cubes(volume, 0.5)
    assert all(volume.size in sizes for sizes in seen if len(sizes) > 1)
    for got, want in zip(result, expected):
        cp.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("gradient_direction", ["descent", "ascent"])
@pytest.mark.parametrize("seed", [0, 1])
def test_lorensen_triangles_match_skimage(gradient_direction, seed):
    # the table of scikit-image is applied with its x axis as the last axis
    # of the array; a table applied in the other frame gives the same
    # vertices but other diagonals in many cells
    rng = np.random.default_rng(seed)
    volume = rng.random((14, 15, 16)).astype(np.float32)
    kwargs = dict(method="lorensen", gradient_direction=gradient_direction)
    expected = skimage_marching_cubes(volume, 0.5, **kwargs)
    verts, faces = (
        cp.asnumpy(a)
        for a in marching_cubes(cp.asarray(volume), 0.5, **kwargs)[:2]
    )

    def triangles(v, f):
        # each triangle starting from its first vertex in lexicographic order
        # (keeping the winding), and the triangles sorted
        tri = v[f]
        first = np.argmin(
            tri[:, :, 0] * 1e6 + tri[:, :, 1] * 1e3 + tri[:, :, 2], axis=1
        )
        idx = (first[:, None] + np.arange(3)) % 3
        tri = np.take_along_axis(tri, idx[:, :, None], axis=1)
        flat = np.round(tri.reshape(len(f), 9), 3)
        return flat[np.lexsort(flat.T[::-1])]

    ref = triangles(expected[0], expected[1])
    got = triangles(verts, faces)
    assert ref.shape == got.shape
    assert np.mean(np.all(np.abs(ref - got) < 2e-3, axis=1)) > 0.999
