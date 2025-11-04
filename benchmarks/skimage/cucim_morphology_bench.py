# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import copy
import functools
import math
import operator
import os
import pickle

import cupy as cp
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import skimage
import skimage.data
import skimage.morphology
from _image_bench import ImageBench

import cucim.skimage
import cucim.skimage.morphology


class BinaryMorphologyBench(ImageBench):
    def __init__(
        self,
        function_name,
        shape,
        footprint=None,
        dtypes=[bool],
        fixed_kwargs={},
        index_str="",
        var_kwargs={},
        module_cpu=skimage.morphology,
        module_gpu=cucim.skimage.morphology,
        run_cpu=True,
    ):
        array_kwargs = dict(footprint=footprint)
        if "footprint" in fixed_kwargs:
            raise ValueError("fixed_kwargs cannot contain 'footprint'")
        fixed_kwargs = copy.deepcopy(fixed_kwargs)
        fixed_kwargs.update(array_kwargs)

        super().__init__(
            function_name=function_name,
            shape=shape,
            dtypes=dtypes,
            fixed_kwargs=fixed_kwargs,
            var_kwargs=var_kwargs,
            index_str=index_str,
            module_cpu=module_cpu,
            module_gpu=module_gpu,
            run_cpu=run_cpu,
        )

    def set_args(self, dtype):
        imaged = (cp.random.standard_normal(self.shape) > 0).astype(dtype)
        image = cp.asnumpy(imaged)
        self.args_cpu = (image,)
        self.args_gpu = (imaged,)


class IsotropicMorphologyBench(ImageBench):
    def set_args(self, dtype):
        imaged = (cp.random.standard_normal(self.shape) > 0).astype(dtype)
        image = cp.asnumpy(imaged)
        self.args_cpu = (image,)
        self.args_gpu = (imaged,)


class SkeletonizeBench(ImageBench):
    def set_args(self, dtype):
        h = ~skimage.data.horse()
        nrow = math.ceil(self.shape[0] / h.shape[0])
        ncol = math.ceil(self.shape[1] / h.shape[1])
        image = np.tile(h, (nrow, ncol))[: self.shape[0], : self.shape[1]]
        imaged = cp.asarray(image)
        self.args_cpu = (image,)
        self.args_gpu = (imaged,)


class ReconstructionBench(ImageBench):
    def set_args(self, dtype):
        coords = cp.meshgrid(
            *[cp.linspace(0, 6 * cp.pi, s) for s in self.shape], sparse=True
        )
        bumps = functools.reduce(operator.add, [cp.sin(c) for c in coords])
        h = 0.6
        seed = bumps - h
        self.args_cpu = (cp.asnumpy(seed), cp.asnumpy(bumps))
        self.args_gpu = (seed, bumps)


class RemoveSmallObjectsBench(ImageBench):
    def _init_test_data(self, dtype):
        ndim = len(self.shape)
        if ndim < 2 or ndim > 3:
            raise ValueError("only 2d and 3d test cases are available")
        a = cp.array([[0, 0, 0, 1, 0], [1, 1, 1, 0, 0], [1, 1, 1, 0, 1]], dtype)
        if ndim == 3:
            a = a[..., cp.newaxis]
            a = cp.tile(a, (1, 1, 2))
            a = cp.pad(a, ((0, 0), (0, 0), (1, 1)), mode="constant")
        ntile = [math.ceil(self.shape[i] / a.shape[i]) for i in range(ndim)]
        a = cp.tile(a, tuple(ntile))
        return a[tuple([slice(s) for s in self.shape])]

    def set_args(self, dtype):
        a = self._init_test_data(dtype)
        self.args_cpu = (cp.asnumpy(a), 6)
        self.args_gpu = (a, 6)


class RemoveSmallHolesBench(RemoveSmallObjectsBench):
    def set_args(self, dtype):
        a = ~self._init_test_data(dtype)
        self.args_cpu = (cp.asnumpy(a), 5)
        self.args_gpu = (a, 5)


def main(args):
    pfile = "cucim_morphology_results.pickle"
    if os.path.exists(pfile):
        with open(pfile, "rb") as f:
            all_results = pickle.load(f)
    else:
        all_results = pd.DataFrame()

    dtypes = [np.dtype(args.dtype)]
    # image sizes/shapes
    shape = tuple(list(map(int, (args.img_size.split(",")))))
    run_cpu = not args.no_cpu

    for function_name, fixed_kwargs, var_kwargs, allow_color, allow_nd in [
        # binary.py
        ("binary_erosion", dict(), dict(), False, True),
        ("binary_dilation", dict(), dict(), False, True),
        ("binary_opening", dict(), dict(), False, True),
        ("binary_closing", dict(), dict(), False, True),
        ("isotropic_erosion", dict(), dict(radius=[5, 10, 20]), False, True),
        ("isotropic_dilation", dict(), dict(radius=[5, 10, 20]), False, True),
        ("isotropic_opening", dict(), dict(radius=[5, 10, 20]), False, True),
        ("isotropic_closing", dict(), dict(radius=[5, 10, 20]), False, True),
        # misc.py
        ("remove_small_objects", dict(), dict(), False, True),
        ("remove_small_holes", dict(), dict(), False, True),
        # gray.py
        ("erosion", dict(), dict(), False, True),
        ("dilation", dict(), dict(), False, True),
        ("opening", dict(), dict(), False, True),
        ("closing", dict(), dict(), False, True),
        ("white_tophat", dict(), dict(), False, True),
        ("black_tophat", dict(), dict(), False, True),
        # _skeletonize.py
        (
            "medial_axis",
            dict(rng=123),
            dict(return_distance=[False, True]),
            False,
            False,
        ),
        ("thin", dict(), dict(), False, True),
        # grayreconstruct.py
        ("reconstruction", dict(), dict(), False, True),
        # footprints.py
        # OMIT the functions from this file (each creates a structuring element)
    ]:
        if function_name != args.func_name:
            continue

        if not allow_color:
            if len(shape) > 2 and shape[-1] == 3:
                continue

        ndim = len(shape)
        if function_name in ["thin", "medial_axis"]:
            if ndim != 2:
                raise ValueError("only 2d benchmark data has been implemented")

            if not allow_nd and ndim > 2:
                continue

            B = SkeletonizeBench(
                function_name=function_name,
                shape=shape,
                dtypes=[bool],
                fixed_kwargs={},
                var_kwargs=var_kwargs,
                module_cpu=skimage.morphology,
                module_gpu=cucim.skimage.morphology,
                run_cpu=run_cpu,
            )

        if function_name.startswith("binary"):
            if not allow_nd and ndim > 2:
                continue

            for connectivity in range(1, ndim + 1):
                index_str = f"conn={connectivity}"
                footprint = ndi.generate_binary_structure(ndim, connectivity)

                B = BinaryMorphologyBench(
                    function_name=function_name,
                    shape=shape,
                    dtypes=[bool],
                    footprint=footprint,
                    fixed_kwargs={},
                    var_kwargs=var_kwargs,
                    index_str=index_str,
                    module_cpu=skimage.morphology,
                    module_gpu=cucim.skimage.morphology,
                    run_cpu=run_cpu,
                )

        elif function_name.startswith("isotropic"):
            if not allow_nd and ndim > 2:
                continue

            B = IsotropicMorphologyBench(
                function_name=function_name,
                shape=shape,
                dtypes=[bool],
                fixed_kwargs=fixed_kwargs,
                var_kwargs=var_kwargs,
                module_cpu=skimage.morphology,
                module_gpu=cucim.skimage.morphology,
                run_cpu=run_cpu,
            )

        elif function_name in ["remove_small_holes", "remove_small_objects"]:
            if not allow_nd and ndim > 2:
                continue

            if function_name == "remove_small_objects":
                TestClass = RemoveSmallObjectsBench
            elif function_name == "remove_small_holes":
                TestClass = RemoveSmallHolesBench

            else:
                raise ValueError(f"unknown function: {function_name}")
            B = TestClass(
                function_name=function_name,
                shape=shape,
                dtypes=[bool],
                fixed_kwargs=fixed_kwargs,
                var_kwargs=var_kwargs,
                module_cpu=skimage.morphology,
                module_gpu=cucim.skimage.morphology,
                run_cpu=run_cpu,
            )
        else:
            if not allow_nd:
                if not allow_color:
                    if ndim > 2:
                        continue
                else:
                    if ndim > 3 or (ndim == 3 and shape[-1] not in [3, 4]):
                        continue

            if shape[-1] == 3 and not allow_color:
                continue

            if function_name == "reconstruction":
                TestClass = ReconstructionBench
            else:
                TestClass = ImageBench

            B = TestClass(
                function_name=function_name,
                shape=shape,
                dtypes=dtypes,
                fixed_kwargs=fixed_kwargs,
                var_kwargs=var_kwargs,
                module_cpu=skimage.morphology,
                module_gpu=cucim.skimage.morphology,
                run_cpu=run_cpu,
            )
        results = B.run_benchmark(duration=args.duration)
        all_results = pd.concat([all_results, results["full"]])

    fbase = os.path.splitext(pfile)[0]
    all_results.to_csv(fbase + ".csv")
    all_results.to_pickle(pfile)
    try:
        import tabular  # noqa: F401

        with open(fbase + ".md", "w") as f:
            f.write(all_results.to_markdown())
    except ImportError:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmarking cuCIM morphology functions"
    )
    # fmt: off
    func_name_choices = [
        'binary_erosion', 'binary_dilation', 'binary_opening',
        'binary_closing', 'isotropic_erosion', 'isotropic_dilation',
        'isotropic_opening', 'isotropic_closing','remove_small_objects',
        'remove_small_holes', 'erosion', 'dilation', 'opening', 'closing',
        'white_tophat', 'black_tophat', 'thin', 'medial_axis', 'reconstruction'
    ]
    # fmt: on
    dtype_choices = [
        "float16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    ]
    parser.add_argument(
        "-i",
        "--img_size",
        type=str,
        help=(
            "Size of input image (omit color channel, it will be appended "
            "as needed)"
        ),
        required=True,
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=str,
        help="Dtype of input image",
        choices=dtype_choices,
        required=True,
    )
    parser.add_argument(
        "-f",
        "--func_name",
        type=str,
        help="function to benchmark",
        choices=func_name_choices,
        required=True,
    )
    parser.add_argument(
        "-t",
        "--duration",
        type=int,
        help="time to run benchmark",
        required=True,
    )
    parser.add_argument(
        "--no_cpu",
        action="store_true",
        help="disable cpu measurements",
        default=False,
    )

    args = parser.parse_args()
    main(args)
