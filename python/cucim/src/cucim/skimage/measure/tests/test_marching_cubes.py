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

from cucim.skimage.measure import marching_cubes


def _single_voxel_volume():
    volume = cp.zeros((3, 3, 3), dtype=cp.float32)
    volume[1, 1, 1] = 1.0
    return volume


def test_default_lewiner_not_implemented():
    with pytest.raises(NotImplementedError, match="method='lewiner'"):
        marching_cubes(_single_voxel_volume())


def test_invalid_method():
    with pytest.raises(ValueError, match="method should be either"):
        marching_cubes(_single_voxel_volume(), 0.5, method="invalid")


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


def test_no_surface_found():
    with pytest.raises(RuntimeError, match="No surface found"):
        marching_cubes(
            cp.zeros((3, 3, 3), dtype=cp.float32), 0.0, method="lorensen"
        )


def test_unsupported_options():
    volume = _single_voxel_volume()
    with pytest.raises(NotImplementedError, match="step_size"):
        marching_cubes(volume, 0.5, method="lorensen", step_size=2)
    with pytest.raises(NotImplementedError, match="allow_degenerate"):
        marching_cubes(volume, 0.5, method="lorensen", allow_degenerate=False)
    with pytest.raises(NotImplementedError, match="mask"):
        marching_cubes(
            volume,
            0.5,
            method="lorensen",
            mask=cp.ones(volume.shape, dtype=bool),
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

    # Test marching cube with mask
    with pytest.raises(ValueError):
        marching_cubes(
            cp.asarray(ellipsoid_anisotropic),
            0.0,
            spacing=spacing,
            mask=np.array([]),
            method="lorensen",
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
