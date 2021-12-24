import copy
import functools
import math
import operator
import os
import pickle

import cucim.skimage
import cucim.skimage.morphology
import cupy as cp
import numpy as np
import pandas as pd
import skimage
import skimage.data
import skimage.morphology
import scipy.ndimage as ndi

from _image_bench import ImageBench


class BinaryMorphologyBench(ImageBench):
    def __init__(
        self,
        function_name,
        shape,
        selem=None,
        dtypes=[bool],
        fixed_kwargs={},
        index_str="",
        var_kwargs={},
        module_cpu=skimage.morphology,
        module_gpu=cucim.skimage.morphology,
    ):

        array_kwargs = dict(selem=selem)
        if "selem" in fixed_kwargs:
            raise ValueError("fixed_kwargs cannot contain 'selem'")
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
        )

    def set_args(self, dtype):
        imaged = cp.random.standard_normal(self.shape).astype(dtype) > 0
        image = cp.asnumpy(imaged)
        self.args_cpu = (image,)
        self.args_gpu = (imaged,)


class SkeletonizeBench(ImageBench):
    def set_args(self, dtype):
        h = ~skimage.data.horse()
        nrow = math.ceil(self.shape[0] / h.shape[0])
        ncol = math.ceil(self.shape[1] / h.shape[1])
        image = np.tile(h, (nrow, ncol))[:self.shape[0], :self.shape[1]]
        imaged = cp.asarray(image)
        self.args_cpu = (image,)
        self.args_gpu = (imaged,)


class ReconstructionBench(ImageBench):
    def set_args(self, dtype):
        coords = cp.meshgrid(*[cp.linspace(0, 6 * cp.pi, s) for s in self.shape], sparse=True)
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


pfile = "cucim_morphology_results.pickle"
if os.path.exists(pfile):
    with open(pfile, "rb") as f:
        all_results = pickle.load(f)
else:
    all_results = pd.DataFrame()
dtypes_gray = [np.float32]


for function_name, fixed_kwargs, var_kwargs, allow_nd in [
    ("thin", dict(), dict(), True),
]:

    for shape in [(512, 512), (3840, 2160)]:

        ndim = len(shape)
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
        )
        results = B.run_benchmark(duration=1)
        all_results = all_results.append(results["full"])


for function_name, fixed_kwargs, var_kwargs, allow_nd in [
    ("binary_erosion", dict(), dict(), True),
    ("binary_dilation", dict(), dict(), True),
    ("binary_opening", dict(), dict(), True),
    ("binary_closing", dict(), dict(), True),
]:

    for shape in [(512, 512), (3840, 2160), (192, 192, 192)]:

        ndim = len(shape)
        if not allow_nd and ndim > 2:
            continue

        for connectivity in range(1, ndim + 1):
            index_str = f"conn={connectivity}"
            selem = ndi.generate_binary_structure(ndim, connectivity)

            B = BinaryMorphologyBench(
                function_name=function_name,
                shape=shape,
                dtypes=[bool],
                selem=selem,
                fixed_kwargs={},
                var_kwargs=var_kwargs,
                index_str=index_str,
                module_cpu=skimage.morphology,
                module_gpu=cucim.skimage.morphology,
            )
            results = B.run_benchmark(duration=1)
            all_results = all_results.append(results["full"])


for function_name, fixed_kwargs, var_kwargs, allow_nd in [
    # misc.py
    ("remove_small_objects", dict(), dict(), True),
    ("remove_small_holes", dict(), dict(), True),
]:

    for shape in [(512, 512), (3840, 2160), (192, 192, 192)]:

        ndim = len(shape)
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
        )
        results = B.run_benchmark(duration=1)
        all_results = all_results.append(results["full"])


for function_name, fixed_kwargs, var_kwargs, allow_color, allow_nd in [
    # grey.py
    ("erosion", dict(), dict(), False, True),
    ("dilation", dict(), dict(), False, True),
    ("opening", dict(), dict(), False, True),
    ("closing", dict(), dict(), False, True),
    ("white_tophat", dict(), dict(), False, True),
    ("black_tophat", dict(), dict(), False, True),
    # greyreconstruct.py
    ("reconstruction", dict(), dict(), False, True),
    # selem.py
    # OMIT the functions from this file (each creates a structuring element)
]:

    for shape in [(512, 512), (3840, 2160), (3840, 2160, 3), (192, 192, 192)]:

        ndim = len(shape)
        if not allow_nd:
            if not allow_color:
                if ndim > 2:
                    continue
            else:
                if ndim > 3 or (ndim == 3 and shape[-1] not in [3, 4]):
                    continue
        if shape[-1] == 3 and not allow_color:
            continue

        if function_name == "gabor" and np.prod(shape) > 1000000:
            # avoid cases that are too slow on the CPU
            var_kwargs["frequency"] = [f for f in var_kwargs["frequency"] if f >= 0.1]

        if function_name == "median":
            selems = []
            ndim = len(shape)
            selem_sizes = [3, 5, 7, 9] if ndim == 2 else [3, 5, 7]
            for selem_size in [3, 5, 7, 9]:
                selems.append(np.ones((selem_sizes,) * ndim, dtype=bool))
            var_kwargs["selem"] = selems

        if function_name in ["gaussian", "unsharp_mask"]:
            fixed_kwargs["channel_axis"] = -1 if shape[-1] == 3 else None

        if function_name == "reconstruction":
            TestClass = ReconstructionBench
        else:
            TestClass = ImageBench

        B = TestClass(
            function_name=function_name,
            shape=shape,
            dtypes=dtypes_gray,
            fixed_kwargs=fixed_kwargs,
            var_kwargs=var_kwargs,
            module_cpu=skimage.morphology,
            module_gpu=cucim.skimage.morphology,
        )
        results = B.run_benchmark(duration=1)
        all_results = all_results.append(results["full"])


fbase = os.path.splitext(pfile)[0]
all_results.to_csv(fbase + ".csv")
all_results.to_pickle(pfile)
with open(fbase + ".md", "wt") as f:
    f.write(all_results.to_markdown())
