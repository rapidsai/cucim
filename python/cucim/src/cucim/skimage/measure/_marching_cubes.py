# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2012-2015, P. M. Neila
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause


from ._marching_cubes_lewiner import _marching_cubes_lewiner
from ._marching_cubes_lorensen import _marching_cubes_lorensen


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

    The ``method='lewiner'`` default and ``method='lorensen'`` are supported
    for CuPy inputs.

    When ``allow_degenerate=False``, zero-area faces are removed and the mesh
    geometry is preserved, but the exact surviving duplicate vertex IDs can
    differ from scikit-image's ordered CPU post-processing.
    """
    if method not in ("lewiner", "lorensen"):
        raise ValueError("method should be either 'lewiner' or 'lorensen'")

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

    return _marching_cubes_lorensen(
        volume,
        level,
        spacing,
        gradient_direction,
        step_size,
        allow_degenerate,
        mask,
    )
