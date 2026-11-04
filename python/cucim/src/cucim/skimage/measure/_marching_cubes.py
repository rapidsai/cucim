# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2012-2015, P. M. Neila
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import cupy as cp

from ._marching_cubes_lewiner import _marching_cubes_lewiner
from ._marching_cubes_lorensen import _marching_cubes_lorensen
from ._marching_cubes_vega import _marching_cubes_vega


def marching_cubes(
    volume,
    level=None,
    *,
    spacing=(1.0, 1.0, 1.0),
    gradient_direction="descent",
    step_size=1,
    allow_degenerate=True,
    method="lewiner",
    mask=None,
):
    """Marching cubes algorithm to find surfaces in 3D volumetric data.

    The ``method='lewiner'`` default, ``method='lorensen'`` and
    ``method='vega'`` are supported for CuPy inputs. ``'vega'`` is not in
    scikit-image: it is the marching cubes 33 implementation of Vega et al.
    [3]_ with the corrected interior test.

    Differences from scikit-image:

    - The vertices and faces of ``method='lewiner'`` and
      ``method='lorensen'`` are the same as scikit-image's, but in a
      different order.
    - ``verts`` is always float32 (scikit-image returns float64 if ``spacing``
      is not ``(1, 1, 1)``) and ``faces`` is always int32.
    - ``normals`` and ``values`` are computed differently, see Returns.
    - A volume that contains NaN raises a ``ValueError`` (scikit-image
      returns vertices with NaN coordinates).
    - The level is rounded to float32 when the samples are classified, so a
      level that differs from a sample by less than float32 resolution is
      treated as equal to it. A sample equal to the level is below it, as in
      scikit-image.
    - When ``allow_degenerate=False``, zero-area faces are removed and the mesh
      geometry is preserved, but the exact surviving duplicate vertex IDs can
      differ from scikit-image's ordered CPU post-processing. For volumes
      with samples that equal the level (for example integer volumes with an
      integer level) the numbers of vertices and faces can also differ,
      because scikit-image places vertices on a sample with an error of the
      order of 1e-16 and so misses some of the coincident vertices.

    Parameters
    ----------
    volume : (M, N, P) ndarray
        Input data volume to find isosurfaces. Will internally be
        converted to float32 if necessary.
    level : float, optional
        Contour value to search for isosurfaces in `volume`. If not
        given or None, the average of the min and max of vol is used.
    spacing : length-3 tuple of floats, optional
        Voxel spacing in spatial dimensions corresponding to numpy array
        indexing dimensions (M, N, P) as in `volume`.
    gradient_direction : {'descent', 'ascent'}, optional
        Controls if the mesh was generated from an isosurface with gradient
        descent toward objects of interest (the default), or the opposite,
        considering the *left-hand* rule.
        The two options are:
        * descent : Object was greater than exterior
        * ascent : Exterior was greater than object
    step_size : int, optional
        Step size in voxels. Default 1. Larger steps yield faster but
        coarser results. The result will always be topologically correct
        though.
    allow_degenerate : bool, optional
        Whether to allow degenerate (i.e. zero-area) triangles in the
        end-result. Default True. If False, degenerate triangles are
        removed, at the cost of making the algorithm slower.
    method : {'lewiner', 'lorensen', 'vega'}, optional
        Whether the method of Lewiner et al. (the default, as in
        scikit-image), Lorensen et al. or Vega et al. will be used. The
        Lewiner and Vega methods resolve the ambiguous cases of the cells so
        that the surface is topologically consistent; the Vega method uses
        the corrected interior test of [3]_, and its results do not depend on
        the orientation of the cells (the Lewiner method gives different
        meshes for a volume and its mirror image in rare cases). ``'vega'``
        is not available in scikit-image.
    mask : (M, N, P) array, optional
        Boolean array. The marching cube algorithm will be computed only on
        True elements. This will save computational time when interfaces
        are located within certain region of the volume M, N, P-e.g. the top
        half of the cube-and also allow to compute finite surfaces-i.e. open
        surfaces that do not end at the border of the cube.

    Returns
    -------
    verts : (V, 3) array
        Spatial coordinates for V unique mesh vertices. Coordinate order
        matches input `volume` (M, N, P). If ``allow_degenerate`` is set to
        True, then the presence of degenerate triangles in the mesh can make
        this array have duplicate vertices.
    faces : (F, 3) array
        Define triangular faces via referencing vertex indices from ``verts``.
        This algorithm specifically outputs triangles, so each face has
        exactly three indices.
    normals : (V, 3) array
        The (unit) normal direction at each vertex, as calculated from the
        data, with the sign of scikit-image's normals (towards lower values).
        For a vertex on a grid edge it is the interpolation of the
        central-difference gradients at the two end samples; scikit-image
        sums the gradients of the cells that contain the vertex, so the
        directions agree to a few degrees for smooth data but are not equal
        (for white noise the mean cosine of the angle is about 0.7). The center
        vertices of the Lewiner method use scikit-image's cell gradients, and
        those of the Vega method the sum of the central-difference gradients
        at the eight corners of the cell.
    values : (V,) array
        A measure of the data in the local region near each vertex, which can
        be used by visualization tools to apply a colormap to the mesh. For a
        vertex on a grid edge it is the larger of the two sample values of
        the edge (scikit-image uses the range of the samples of the cells
        that contain the vertex), so the values are not the same as
        scikit-image's.

    See Also
    --------
    cucim.skimage.measure.mesh_surface_area
    cucim.skimage.measure.find_contours

    Notes
    -----
    The algorithm [1]_ is an improved version of Chernyaev's Marching
    Cubes 33 algorithm. It is an efficient algorithm that relies on
    heavy use of lookup tables to handle the many different cases,
    keeping the algorithm relatively easy.

    Choosing a method: ``'vega'`` resolves each cell so that its surface has
    the topology of the trilinear interpolant of the cell values, which is
    not always the case for ``'lewiner'``, and it takes about the same time.
    This is a property of each cell and does not make the whole mesh
    manifold: for some noisy volumes the ``'vega'`` mesh has edges that are
    shared by four triangles (the surfaces of neighboring cells overlap along
    an edge), which the ``'lewiner'`` mesh does not have in the same
    volumes. Use ``'lewiner'`` if a manifold, consistently oriented mesh is
    required. ``'lewiner'`` is the default because it reproduces the meshes
    of scikit-image; ``'vega'`` gives different meshes in the cells where the
    Lewiner method is not consistent and splits some quadrilaterals
    differently. ``'lorensen'`` reproduces scikit-image's
    ``method='lorensen'``, can leave holes in the surface where cells are
    ambiguous, and is mainly useful for compatibility and on very noisy
    data, where it is faster than ``'lewiner'``. For binary masks there is no
    underlying field, so the choice of method only selects how the ambiguous
    cells are resolved.

    To quantify the area of an isosurface generated by this algorithm, pass
    verts and faces to `cucim.skimage.measure.mesh_surface_area`.

    Regarding visualization of algorithm output, to contour a volume
    named `myvolume` about the level 0.0, using the ``mayavi`` package::

      >>>
      >> from mayavi import mlab
      >> verts, faces, _, _ = marching_cubes(myvolume, 0.0)
      >> verts = cp.asnumpy(verts)
      >> faces = cp.asnumpy(faces)
      >> mlab.triangular_mesh([vert[0] for vert in verts],
                              [vert[1] for vert in verts],
                              [vert[2] for vert in verts],
                              faces)
      >> mlab.show()

    Similarly using the ``visvis`` package::

      >>>
      >> import visvis as vv
      >> verts, faces, normals, values = marching_cubes(myvolume, 0.0)
      >> verts = cp.asnumpy(verts)
      >> faces = cp.asnumpy(faces)
      >> normals = cp.asnumpy(normals)
      >> values = cp.asnumpy(values)
      >> vv.mesh(np.fliplr(verts), faces, normals, values)
      >> vv.use().Run()

    To reduce the number of triangles in the mesh for better performance,
    see this `example
    <https://docs.enthought.com/mayavi/mayavi/auto/example_julia_set_decimation.html#example-julia-set-decimation>`_
    using the ``mayavi`` package.

    References
    ----------
    .. [1] Thomas Lewiner, Helio Lopes, Antonio Wilson Vieira and Geovan
           Tavares. Efficient implementation of Marching Cubes' cases with
           topological guarantees. Journal of Graphics Tools 8(2)
           pp. 1-15 (december 2003).
           :DOI:`10.1080/10867651.2003.10487582`
    .. [2] Lorensen, William and Harvey E. Cline. Marching Cubes: A High
           Resolution 3D Surface Construction Algorithm. Computer Graphics
           (SIGGRAPH 87 Proceedings) 21(4) July 1987, p. 163-170).
           :DOI:`10.1145/37401.37422`
    .. [3] David Vega, Javier Abache and David Coll. A Fast and Memory-Saving
           Marching Cubes 33 Implementation with the Correct Interior Test.
           Journal of Computer Graphics Techniques 8(3), 2019, p. 1-18.

    """  # noqa: E501
    if method not in ("lewiner", "lorensen", "vega"):
        raise ValueError(
            "method should be one of 'lewiner', 'lorensen' or 'vega'"
        )

    if method == "lewiner":
        return _marching_cubes_lewiner(
            volume,
            level,
            spacing,
            gradient_direction,
            step_size,
            allow_degenerate,
            mask,
        )

    if method == "vega":
        return _marching_cubes_vega(
            volume,
            level,
            spacing,
            gradient_direction,
            step_size,
            allow_degenerate,
            mask,
        )

    return _marching_cubes_lorensen(
        volume,
        level,
        spacing,
        gradient_direction,
        step_size,
        allow_degenerate,
        mask,
    )


_mesh_surface_area_kernel = cp.ReductionKernel(
    "raw T verts, I f0, I f1, I f2",
    "T area",
    r"""[&]() {
        // 64-bit offsets: 3 * f overflows 32 bits for 715,827,883 vertices
        long long o0 = 3LL * (long long)f0;
        long long o1 = 3LL * (long long)f1;
        long long o2 = 3LL * (long long)f2;
        T ax = verts[o0 + 0] - verts[o1 + 0];
        T ay = verts[o0 + 1] - verts[o1 + 1];
        T az = verts[o0 + 2] - verts[o1 + 2];
        T bx = verts[o0 + 0] - verts[o2 + 0];
        T by = verts[o0 + 1] - verts[o2 + 1];
        T bz = verts[o0 + 2] - verts[o2 + 2];
        T cx = ay * bz - az * by;
        T cy = az * bx - ax * bz;
        T cz = ax * by - ay * bx;
        return sqrt(cx * cx + cy * cy + cz * cz) * T(0.5);
    }()""",
    "a + b",
    "area = a",
    "0",
    "cucim_skimage_measure_mesh_surface_area",
)


def mesh_surface_area(verts, faces):
    """Compute surface area, given vertices and triangular faces.

    Parameters
    ----------
    verts : (V, 3) array of floats
        Array containing coordinates for V unique mesh vertices.
    faces : (F, 3) array of ints
        List of length-3 lists of integers, referencing vertex coordinates as
        provided in ``verts``.

    Returns
    -------
    area : cupy scalar
        Surface area of mesh. Units now [coordinate units] ** 2.

    Notes
    -----
    The arguments expected by this function are the first two outputs from
    ``cucim.skimage.measure.marching_cubes``. For unit-correct output, ensure
    correct ``spacing`` was passed to ``marching_cubes``.

    The result is a device scalar. The areas of the triangles are summed in
    the floating point type of ``verts`` (at least float32).

    This algorithm works properly only if the ``faces`` provided are all
    triangles.

    See Also
    --------
    cucim.skimage.measure.marching_cubes

    """
    if not isinstance(verts, cp.ndarray) or not isinstance(faces, cp.ndarray):
        raise TypeError("verts and faces must have type cupy.ndarray")

    float_dtype = cp.promote_types(verts.dtype, cp.float32)
    if verts.dtype != float_dtype:
        verts = verts.astype(float_dtype)
    if faces.shape[0] == 0:
        return cp.zeros((), dtype=float_dtype)

    return _mesh_surface_area_kernel(
        verts.ravel(), faces[:, 0], faces[:, 1], faces[:, 2]
    )
