import math
import os
import pickle

import cupy
import cupy as cp
import cupyx.scipy.ndimage
import numpy as np
import pandas as pd
import scipy
import scipy.ndimage as ndi

from _image_bench import ImageBench


class LabelBench(ImageBench):
    def __init__(
        self,
        function_name,
        shape,
        structure=None,
        contiguous_labels=True,
        dtypes=np.float32,
        fixed_kwargs={},
        var_kwargs={},
        index_str=None,
        module_cpu=scipy.ndimage,
        module_gpu=cupyx.scipy.ndimage,
    ):

        self.contiguous_labels = contiguous_labels
        array_kwargs = dict(structure=structure)
        if "structure" in fixed_kwargs:
            raise ValueError("fixed_kwargs cannot contain 'structure'")
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
        a = np.array(
            [
                [0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 4, 0],
                [2, 2, 0, 0, 3, 0, 4, 4],
                [0, 0, 0, 0, 0, 5, 0, 0],
            ]
        )
        tiling = tuple(s // a_s for s, a_s in zip(shape, a.shape))
        if self.contiguous_labels:
            image = np.kron(a, np.ones(tiling, dtype=a.dtype))
        else:
            image = np.tile(a, tiling)
        imaged = cp.asarray(image)
        self.args_cpu = (image,)
        self.args_gpu = (imaged,)


class MeasurementsBench(ImageBench):
    def __init__(
        self,
        function_name,
        shape,
        use_index=False,
        nlabels=16,
        dtypes=[np.float32],
        fixed_kwargs={},
        var_kwargs={},
        index_str=None,
        module_cpu=scipy.ndimage,
        module_gpu=cupyx.scipy.ndimage,
    ):

        self.nlabels = nlabels
        self.use_index = use_index
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
        size = math.prod(self.shape)
        imaged = cupy.arange(size, dtype=dtype).reshape(self.shape)
        labelsd = cupy.random.choice(self.nlabels, size)
        labelsd = labelsd.reshape(self.shape) + 1

        image = cp.asnumpy(imaged)
        labels = cp.asnumpy(labelsd)
        if self.use_index:
            indexd = cupy.arange(1, self.nlabels + 1, dtype=cupy.intp)
            index = cp.asnumpy(indexd)
        else:
            indexd = None
            index = None

        self.args_cpu = (image,)
        self.args_gpu = (imaged,)
        # store labels and index as fixed_kwargs since histogram does not use
        # them in the same position
        self.fixed_kwargs_gpu.update(dict(labels=labelsd, index=indexd))
        self.fixed_kwargs_cpu.update(dict(labels=labels, index=index))


pfile = "measurements_results.pickle"
if os.path.exists(pfile):
    with open(pfile, "rb") as f:
        all_results = pickle.load(f)
else:
    all_results = pd.DataFrame()

dtypes = [np.float32]
for shape in [(512, 512), (3840, 2160), (192, 192, 192)]:
    ndim = len(shape)

    for fname, var_kwargs in [
        ("label", {}),  # dict(greyscale_mode=[False, True]) not available in cupyx
    ]:
        for contiguous_labels in [True, False]:
            if contiguous_labels:
                index_str = "contiguous"
            else:
                index_str = None
            B = LabelBench(
                function_name=fname,
                shape=shape,
                dtypes=dtypes,
                structure=ndi.generate_binary_structure(ndim, ndim),
                contiguous_labels=contiguous_labels,
                index_str=index_str,
                fixed_kwargs=dict(output=None),
                var_kwargs=var_kwargs,
            )
            results = B.run_benchmark(duration=1)
            all_results = all_results.append(results["full"])

    for fname in [
        "sum",
        "mean",
        "variance",
        "standard_deviation",
        "minimum",
        "minimum_position",
        "maximum",
        "maximum_position",
        "median",
        "extrema",
        "center_of_mass",
    ]:
        for use_index in [True, False]:
            if use_index:
                nlabels_cases = [4, 16, 64, 256]
            else:
                nlabels_cases = [16]
            for nlabels in nlabels_cases:
                if use_index:
                    index_str = f"{nlabels} labels, no index"
                else:
                    index_str = f"{nlabels} labels, with index"
                B = MeasurementsBench(
                    function_name=fname,
                    shape=shape,
                    dtypes=dtypes,
                    use_index=use_index,
                    nlabels=nlabels,
                    index_str=index_str,
                    var_kwargs=var_kwargs,
                )
                results = B.run_benchmark(duration=1)
                all_results = all_results.append(results["full"])

fbase = os.path.splitext(pfile)[0]
all_results.to_csv(fbase + ".csv")
all_results.to_pickle(pfile)
with open(fbase + ".md", "wt") as f:
    f.write(all_results.to_markdown())
