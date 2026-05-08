# SPDX-FileCopyrightText: 2009-2022 the scikit-image team
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import cupy as cp
import numpy as np
import pytest
from skimage.measure import (
    marching_cubes as skimage_marching_cubes,
)

from cucim.skimage.measure import _marching_cubes_lewiner_luts as _mcluts
from cucim.skimage.measure._marching_cubes_lewiner import (
    _lewiner_count_cells_gpu,
    _lewiner_generate_center_vertices_gpu,
    _lewiner_generate_faces_direct_gpu,
)
from cucim.skimage.measure._marching_cubes_lorensen import _decode_lut

_LEWINER_LUT_NAMES = _mcluts._LEWINER_LUT_NAMES

_FLT_EPSILON = float(np.spacing(1.0))


def _decode_named_lut(name):
    shape, text = getattr(_mcluts, name)
    return _decode_lut(shape, text)


def _get_lewiner_luts_cpu():
    return {name: _decode_named_lut(name) for name in _LEWINER_LUT_NAMES}


def _append_triangles_from_lut(out, luts, name, config, *args, trace=None):
    if len(args) == 1:
        subconfig = None
        n_triangles = args[0]
    elif len(args) == 2:
        subconfig, n_triangles = args
    else:
        raise TypeError("expected n_triangles or subconfig, n_triangles")

    lut = luts[name]
    n_values = n_triangles * 3
    if subconfig is None:
        values = lut[config, :n_values]
    else:
        values = lut[config, subconfig, :n_values]
    out.extend(int(v) for v in values)
    if trace is not None:
        trace.append(
            (
                name,
                int(config),
                None if subconfig is None else int(subconfig),
                int(n_triangles),
            )
        )


def _lewiner_test_face(values, face):
    abs_face = abs(int(face))
    if abs_face == 1:
        a, b, c, d = values[0], values[4], values[5], values[1]
    elif abs_face == 2:
        a, b, c, d = values[1], values[5], values[6], values[2]
    elif abs_face == 3:
        a, b, c, d = values[2], values[6], values[7], values[3]
    elif abs_face == 4:
        a, b, c, d = values[3], values[7], values[4], values[0]
    elif abs_face == 5:
        a, b, c, d = values[0], values[3], values[2], values[1]
    elif abs_face == 6:
        a, b, c, d = values[4], values[7], values[6], values[5]
    else:
        return False

    ac_bd = a * c - b * d
    if -_FLT_EPSILON < ac_bd < _FLT_EPSILON:
        return face >= 0
    return face * a * ac_bd >= 0


def _lewiner_test_internal(values, luts, case, config, subconfig, s):
    v = values
    at = bt = ct = dt = 0.0

    if case in (4, 10):
        a = (v[4] - v[0]) * (v[6] - v[2]) - (v[7] - v[3]) * (v[5] - v[1])
        b = (
            v[2] * (v[4] - v[0])
            + v[0] * (v[6] - v[2])
            - v[1] * (v[7] - v[3])
            - v[3] * (v[5] - v[1])
        )
        t = -b / (2 * a + _FLT_EPSILON)
        if t < 0 or t > 1:
            return s > 0

        at = v[0] + (v[4] - v[0]) * t
        bt = v[3] + (v[7] - v[3]) * t
        ct = v[2] + (v[6] - v[2]) * t
        dt = v[1] + (v[5] - v[1]) * t
    elif case in (6, 7, 12, 13):
        if case == 6:
            edge = int(luts["TEST6"][config, 2])
        elif case == 7:
            edge = int(luts["TEST7"][config, 4])
        elif case == 12:
            edge = int(luts["TEST12"][config, 3])
        else:
            edge = int(luts["TILING13_5_1"][config, subconfig, 0])

        if edge == 0:
            t = v[0] / (v[0] - v[1] + _FLT_EPSILON)
            bt = v[3] + (v[2] - v[3]) * t
            ct = v[7] + (v[6] - v[7]) * t
            dt = v[4] + (v[5] - v[4]) * t
        elif edge == 1:
            t = v[1] / (v[1] - v[2] + _FLT_EPSILON)
            bt = v[0] + (v[3] - v[0]) * t
            ct = v[4] + (v[7] - v[4]) * t
            dt = v[5] + (v[6] - v[5]) * t
        elif edge == 2:
            t = v[2] / (v[2] - v[3] + _FLT_EPSILON)
            bt = v[1] + (v[0] - v[1]) * t
            ct = v[5] + (v[4] - v[5]) * t
            dt = v[6] + (v[7] - v[6]) * t
        elif edge == 3:
            t = v[3] / (v[3] - v[0] + _FLT_EPSILON)
            bt = v[2] + (v[1] - v[2]) * t
            ct = v[6] + (v[5] - v[6]) * t
            dt = v[7] + (v[4] - v[7]) * t
        elif edge == 4:
            t = v[4] / (v[4] - v[5] + _FLT_EPSILON)
            bt = v[7] + (v[6] - v[7]) * t
            ct = v[3] + (v[2] - v[3]) * t
            dt = v[0] + (v[1] - v[0]) * t
        elif edge == 5:
            t = v[5] / (v[5] - v[6] + _FLT_EPSILON)
            bt = v[4] + (v[7] - v[4]) * t
            ct = v[0] + (v[3] - v[0]) * t
            dt = v[1] + (v[2] - v[1]) * t
        elif edge == 6:
            t = v[6] / (v[6] - v[7] + _FLT_EPSILON)
            bt = v[5] + (v[4] - v[5]) * t
            ct = v[1] + (v[0] - v[1]) * t
            dt = v[2] + (v[3] - v[2]) * t
        elif edge == 7:
            t = v[7] / (v[7] - v[4] + _FLT_EPSILON)
            bt = v[6] + (v[5] - v[6]) * t
            ct = v[2] + (v[1] - v[2]) * t
            dt = v[3] + (v[0] - v[3]) * t
        elif edge == 8:
            t = v[0] / (v[0] - v[4] + _FLT_EPSILON)
            bt = v[3] + (v[7] - v[3]) * t
            ct = v[2] + (v[6] - v[2]) * t
            dt = v[1] + (v[5] - v[1]) * t
        elif edge == 9:
            t = v[1] / (v[1] - v[5] + _FLT_EPSILON)
            bt = v[0] + (v[4] - v[0]) * t
            ct = v[3] + (v[7] - v[3]) * t
            dt = v[2] + (v[6] - v[2]) * t
        elif edge == 10:
            t = v[2] / (v[2] - v[6] + _FLT_EPSILON)
            bt = v[1] + (v[5] - v[1]) * t
            ct = v[0] + (v[4] - v[0]) * t
            dt = v[3] + (v[7] - v[3]) * t
        elif edge == 11:
            t = v[3] / (v[3] - v[7] + _FLT_EPSILON)
            bt = v[2] + (v[6] - v[2]) * t
            ct = v[1] + (v[5] - v[1]) * t
            dt = v[0] + (v[4] - v[0]) * t
        else:
            return s < 0
    else:
        return s < 0

    test = 0
    if at >= 0:
        test += 1
    if bt >= 0:
        test += 2
    if ct >= 0:
        test += 4
    if dt >= 0:
        test += 8

    if test in (0, 1, 2, 3, 4, 6, 8, 9, 12):
        return s > 0
    if test == 5:
        if at * ct - bt * dt < _FLT_EPSILON:
            return s > 0
    elif test == 10:
        if at * ct - bt * dt >= _FLT_EPSILON:
            return s > 0
    return s < 0


def _lewiner_tri_edges_for_case_values(values, luts=None):
    edges, _ = _lewiner_tri_edges_and_trace_for_case_values(values, luts=luts)
    return edges


def _lewiner_tri_edges_and_trace_for_case_values(values, luts=None):
    values = tuple(float(v) for v in values)
    if len(values) != 8:
        raise ValueError("values must contain 8 cube corner values.")
    if luts is None:
        luts = _get_lewiner_luts_cpu()

    index = 0
    for i, value in enumerate(values):
        if value > 0.0:
            index += 1 << i

    case = int(luts["CASES"][index, 0])
    if case == 0:
        return (), ()
    config = int(luts["CASES"][index, 1])
    out = []
    trace = []
    _lewiner_big_switch(values, luts, case, config, out, trace=trace)
    return tuple(out), tuple(trace)


def _lewiner_center_vertex(values):
    values = tuple(float(v) for v in values)
    if len(values) != 8:
        raise ValueError("values must contain 8 cube corner values.")

    weights = [1.0 / (_FLT_EPSILON + abs(v)) for v in values]
    total = sum(weights)
    position = np.asarray(
        [
            (weights[1] + weights[2] + weights[5] + weights[6]) / total,
            (weights[2] + weights[3] + weights[6] + weights[7]) / total,
            (weights[4] + weights[5] + weights[6] + weights[7]) / total,
        ],
        dtype=np.float64,
    )

    v = values
    gradients = np.asarray(
        [
            (v[0] - v[1], v[0] - v[3], v[0] - v[4]),
            (v[0] - v[1], v[1] - v[2], v[1] - v[5]),
            (v[3] - v[2], v[1] - v[2], v[2] - v[6]),
            (v[3] - v[2], v[0] - v[3], v[3] - v[7]),
            (v[4] - v[5], v[4] - v[7], v[0] - v[4]),
            (v[4] - v[5], v[5] - v[6], v[1] - v[5]),
            (v[7] - v[6], v[5] - v[6], v[2] - v[6]),
            (v[7] - v[6], v[4] - v[7], v[3] - v[7]),
        ],
        dtype=np.float64,
    )
    gradient = np.sum(
        np.asarray(weights, dtype=np.float64)[:, np.newaxis] * gradients,
        axis=0,
    )
    norm = np.linalg.norm(gradient)
    if norm > 0:
        gradient = gradient / norm
    return position, gradient


def _lewiner_big_switch(values, luts, case, config, out, trace=None):
    subconfig = 0

    def add(out, luts, name, config, *args):
        _append_triangles_from_lut(out, luts, name, config, *args, trace=trace)

    test_face = _lewiner_test_face
    test_internal = _lewiner_test_internal

    if case == 1:
        add(out, luts, "TILING1", config, 1)
    elif case == 2:
        add(out, luts, "TILING2", config, 2)
    elif case == 3:
        if test_face(values, luts["TEST3"][config]):
            add(out, luts, "TILING3_2", config, 4)
        else:
            add(out, luts, "TILING3_1", config, 2)
    elif case == 4:
        if test_internal(
            values, luts, case, config, subconfig, luts["TEST4"][config]
        ):
            add(out, luts, "TILING4_1", config, 2)
        else:
            add(out, luts, "TILING4_2", config, 6)
    elif case == 5:
        add(out, luts, "TILING5", config, 3)
    elif case == 6:
        if test_face(values, luts["TEST6"][config, 0]):
            add(out, luts, "TILING6_2", config, 5)
        elif test_internal(
            values, luts, case, config, subconfig, luts["TEST6"][config, 1]
        ):
            add(out, luts, "TILING6_1_1", config, 3)
        else:
            add(out, luts, "TILING6_1_2", config, 9)
    elif case == 7:
        if test_face(values, luts["TEST7"][config, 0]):
            subconfig += 1
        if test_face(values, luts["TEST7"][config, 1]):
            subconfig += 2
        if test_face(values, luts["TEST7"][config, 2]):
            subconfig += 4
        if subconfig == 0:
            add(out, luts, "TILING7_1", config, 3)
        elif subconfig in (1, 2, 4):
            add(
                out, luts, "TILING7_2", config, {1: 0, 2: 1, 4: 2}[subconfig], 5
            )
        elif subconfig in (3, 5, 6):
            add(
                out, luts, "TILING7_3", config, {3: 0, 5: 1, 6: 2}[subconfig], 9
            )
        elif test_internal(
            values, luts, case, config, subconfig, luts["TEST7"][config, 3]
        ):
            add(out, luts, "TILING7_4_2", config, 9)
        else:
            add(out, luts, "TILING7_4_1", config, 5)
    elif case == 8:
        add(out, luts, "TILING8", config, 2)
    elif case == 9:
        add(out, luts, "TILING9", config, 4)
    elif case == 10:
        if test_face(values, luts["TEST10"][config, 0]):
            if test_face(values, luts["TEST10"][config, 1]):
                add(out, luts, "TILING10_1_1_", config, 4)
            else:
                add(out, luts, "TILING10_2", config, 8)
        elif test_face(values, luts["TEST10"][config, 1]):
            add(out, luts, "TILING10_2_", config, 8)
        elif test_internal(
            values, luts, case, config, subconfig, luts["TEST10"][config, 2]
        ):
            add(out, luts, "TILING10_1_1", config, 4)
        else:
            add(out, luts, "TILING10_1_2", config, 8)
    elif case == 11:
        add(out, luts, "TILING11", config, 4)
    elif case == 12:
        if test_face(values, luts["TEST12"][config, 0]):
            if test_face(values, luts["TEST12"][config, 1]):
                add(out, luts, "TILING12_1_1_", config, 4)
            else:
                add(out, luts, "TILING12_2", config, 8)
        elif test_face(values, luts["TEST12"][config, 1]):
            add(out, luts, "TILING12_2_", config, 8)
        elif test_internal(
            values, luts, case, config, subconfig, luts["TEST12"][config, 2]
        ):
            add(out, luts, "TILING12_1_1", config, 4)
        else:
            add(out, luts, "TILING12_1_2", config, 8)
    elif case == 13:
        _lewiner_case_13(values, luts, config, out, trace=trace)
    elif case == 14:
        add(out, luts, "TILING14", config, 4)


def _lewiner_case_13(values, luts, config, out, trace=None):
    subconfig = 0
    for i in range(6):
        if _lewiner_test_face(values, luts["TEST13"][config, i]):
            subconfig += 1 << i
    subconfig = int(luts["SUBCONFIG13"][subconfig])

    def add(out, luts, name, config, *args):
        _append_triangles_from_lut(out, luts, name, config, *args, trace=trace)

    if subconfig == 0:
        add(out, luts, "TILING13_1", config, 4)
    elif 1 <= subconfig <= 6:
        add(out, luts, "TILING13_2", config, subconfig - 1, 6)
    elif 7 <= subconfig <= 18:
        add(out, luts, "TILING13_3", config, subconfig - 7, 10)
    elif 19 <= subconfig <= 22:
        add(out, luts, "TILING13_4", config, subconfig - 19, 12)
    elif 23 <= subconfig <= 26:
        internal_subconfig = subconfig - 23
        if _lewiner_test_internal(
            values,
            luts,
            13,
            config,
            internal_subconfig,
            luts["TEST13"][config, 6],
        ):
            add(out, luts, "TILING13_5_1", config, internal_subconfig, 6)
        else:
            add(out, luts, "TILING13_5_2", config, internal_subconfig, 10)
    elif 27 <= subconfig <= 38:
        add(out, luts, "TILING13_3_", config, subconfig - 27, 10)
    elif 39 <= subconfig <= 44:
        add(out, luts, "TILING13_2_", config, subconfig - 39, 6)
    elif subconfig == 45:
        add(out, luts, "TILING13_1_", config, 4)


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


def _expected_single_cell_faces_from_edges(local_edges):
    edge_ids = cp.asarray(local_edges, dtype=cp.int8).reshape(-1, 3)
    return cp.where(
        edge_ids == 12,
        cp.asarray(112, dtype=cp.int32),
        edge_ids.astype(cp.int32) + 100,
    )


def test_lewiner_reference_selector_binary_cases():
    for case in range(256):
        values = tuple(1.0 if case & (1 << i) else -1.0 for i in range(8))
        edges = _lewiner_tri_edges_for_case_values(values)
        actual = _triangles_as_sets(np.asarray(edges).reshape(-1, 3))
        expected = _skimage_lewiner_case_triangles(values)
        assert actual == expected, f"case={case}"

        (
            tri_counts,
            center_flags,
            case_codes,
            n_center_vertices,
        ) = _lewiner_count_cells_gpu(_case_volume(values), 0.0)
        gpu_edges = _lewiner_expected_local_edges_for_volume(values)
        assert int(tri_counts[0]) == len(gpu_edges) // 3, f"case={case}"
        assert int(center_flags[0]) == (12 in gpu_edges), f"case={case}"
        assert n_center_vertices == int(center_flags[0]), f"case={case}"
        expected_case_code = sum(
            ((case >> src) & 1) << dst
            for dst, src in enumerate(_LEWINER_LOCAL_VALUE_ORDER)
        )
        assert int(case_codes[0]) == expected_case_code, f"case={case}"
        faces = _lewiner_generate_faces_direct_gpu(
            _case_volume(values),
            0.0,
            case_codes,
            tri_counts,
            _single_cell_edge_vertex_ids(),
            cp.asarray([112], dtype=cp.int32),
            "ascent",
        )
        cp.testing.assert_array_equal(
            faces, _expected_single_cell_faces_from_edges(gpu_edges)
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

    (
        tri_counts,
        center_flags,
        _case_codes,
        n_center_vertices,
    ) = _lewiner_count_cells_gpu(_case_volume(values), 0.0)
    gpu_edges = _lewiner_expected_local_edges_for_volume(values)
    assert int(tri_counts[0]) == len(gpu_edges) // 3
    assert int(center_flags[0]) == (12 in gpu_edges)
    assert n_center_vertices == int(center_flags[0])
    faces = _lewiner_generate_faces_direct_gpu(
        _case_volume(values),
        0.0,
        _case_codes,
        tri_counts,
        _single_cell_edge_vertex_ids(),
        cp.asarray([112], dtype=cp.int32),
        "ascent",
    )
    cp.testing.assert_array_equal(
        faces, _expected_single_cell_faces_from_edges(gpu_edges)
    )


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
    (
        tri_counts,
        center_flags,
        _case_codes,
        n_center_vertices,
    ) = _lewiner_count_cells_gpu(volume, 0.0)
    assert int(tri_counts[0]) == 12
    assert int(center_flags[0]) == 1
    assert n_center_vertices == 1

    vertices, normals, out_values, center_vertex_ids = (
        _lewiner_generate_center_vertices_gpu(
            volume, 0.0, spacing, center_flags, n_centers=n_center_vertices
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


def test_lewiner_gpu_faces_direct_vertex_ids():
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
    volume, local_values = _case_volume_for_lewiner_values(values)
    (
        tri_counts,
        center_flags,
        _case_codes,
        n_center_vertices,
    ) = _lewiner_count_cells_gpu(volume, 0.0)
    assert int(tri_counts[0]) == 12
    assert int(center_flags[0]) == 1
    assert n_center_vertices == 1

    edge_vertex_ids = _single_cell_edge_vertex_ids()
    center_vertex_ids = cp.asarray([112], dtype=cp.int32)
    expected = _expected_single_cell_faces_from_edges(
        _lewiner_expected_local_edges_for_volume(local_values)
    )

    faces_ascent = _lewiner_generate_faces_direct_gpu(
        volume,
        0.0,
        _case_codes,
        tri_counts,
        edge_vertex_ids,
        center_vertex_ids,
        "ascent",
    )
    cp.testing.assert_array_equal(faces_ascent, expected)

    faces_descent = _lewiner_generate_faces_direct_gpu(
        volume,
        0.0,
        _case_codes,
        tri_counts,
        edge_vertex_ids,
        center_vertex_ids,
        "descent",
    )
    cp.testing.assert_array_equal(faces_descent, expected[:, ::-1])


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


def _triangles_as_sets(triangles):
    return sorted(tuple(sorted(int(v) for v in tri)) for tri in triangles)


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
