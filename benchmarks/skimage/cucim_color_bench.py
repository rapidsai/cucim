import os
import pickle

import cucim.skimage
import cucim.skimage.color
import cupy
import cupy as cp
import cupyx.scipy.ndimage
import numpy as np
import pandas as pd
import scipy
import skimage
import skimage.color

from _image_bench import ImageBench


class ColorBench(ImageBench):
    def set_args(self, dtype):
        if self.shape[-1] != 3:
            raise ValueError("shape must be 3 on the last axis")
        imaged = cupy.testing.shaped_random(self.shape, xp=cp, dtype=dtype, scale=1.0)
        image = cp.asnumpy(imaged)
        self.args_cpu = (image,)
        self.args_gpu = (imaged,)


class RGBABench(ImageBench):
    def set_args(self, dtype):
        if self.shape[-1] != 4:
            raise ValueError("shape must be 4 on the last axis")
        imaged = cupy.testing.shaped_random(self.shape, xp=cp, dtype=dtype, scale=1.0)
        image = cp.asnumpy(imaged)
        self.args_cpu = (image,)
        self.args_gpu = (imaged,)


class LabelBench(ImageBench):
    def __init__(
        self,
        function_name,
        shape,
        contiguous_labels=True,
        dtypes=np.float32,
        fixed_kwargs={},
        var_kwargs={},
        index_str=None,
        module_cpu=scipy.ndimage,
        module_gpu=cupyx.scipy.ndimage,
    ):
        self.contiguous_labels = contiguous_labels
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
            ],
            dtype=int,
        )
        tiling = tuple(s // a_s for s, a_s in zip(shape, a.shape))
        if self.contiguous_labels:
            label = np.kron(a, np.ones(tiling, dtype=a.dtype))
        else:
            label = np.tile(a, tiling)
        labeld = cp.asarray(label)
        if self.shape[-1] != 3:
            raise ValueError("shape must be 3 on the last axis")
        imaged = cupy.testing.shaped_random(labeld.shape, xp=cp, dtype=dtype, scale=1.0)
        image = cp.asnumpy(imaged)
        self.args_cpu = (
            label,
            image,
        )
        self.args_gpu = (
            labeld,
            imaged,
        )


pfile = "cucim_color_results.pickle"
if os.path.exists(pfile):
    with open(pfile, "rb") as f:
        all_results = pickle.load(f)
else:
    all_results = pd.DataFrame()
dtypes = [np.float32]
all_colorspaces = True

for shape in [(256, 256, 3), (3840, 2160, 3), (192, 192, 192, 3)]:
    ndim = len(shape)

    if all_colorspaces:
        color_spaces = ["RGB", "HSV", "RGB CIE", "XYZ", "YUV", "YIQ", "YPbPr", "YCbCr", "YDbDr"]
    else:
        color_spaces = ["RGB", "HSV", "YUV", "XYZ"]
    for fromspace in color_spaces:
        for tospace in color_spaces:
            if fromspace == tospace:
                continue

            B = ColorBench(
                function_name="convert_colorspace",
                shape=shape,
                dtypes=dtypes,
                fixed_kwargs=dict(fromspace=fromspace, tospace=tospace),
                var_kwargs={},
                index_str=f"{fromspace.lower()}2{tospace.lower()}",
                module_cpu=skimage.color,
                module_gpu=cucim.skimage.color,
            )
            results = B.run_benchmark(duration=1)
            all_results = pd.concat([all_results, results["full"]])

    # rgb2hed and hed2rgb test combine_stains and separate_stains and all other
    # stains should have equivalent performance.
    #
    # Probably not necessary to benchmark combine_stains and separate_stains
    # e.g.
    #    ihc_hdx = separate_stains(ihc, hdx_from_rgb)
    #    ihc = combine_stains(ihc_hdx, rgb_from_hdx)
    #

    for fname in ["rgb2hed", "hed2rgb", "lab2lch", "lch2lab", "xyz2lab",
                  "lab2xyz"]:
        B = ColorBench(
            function_name=fname,
            shape=shape,
            dtypes=dtypes,
            fixed_kwargs={},
            var_kwargs={},
            module_cpu=skimage.color,
            module_gpu=cucim.skimage.color,
        )
        results = B.run_benchmark(duration=1)
        all_results = pd.concat([all_results, results["full"]])

    B = RGBABench(
        function_name="rgba2rgb",
        shape=shape[:-1] + (4,),
        dtypes=dtypes,
        fixed_kwargs={},
        var_kwargs={},
        module_cpu=skimage.color,
        module_gpu=cucim.skimage.color,
    )
    results = B.run_benchmark(duration=1)
    all_results = pd.concat([all_results, results["full"]])

    for contiguous_labels in [True, False]:
        if contiguous_labels:
            index_str = "contiguous"
        else:
            index_str = None
        B = LabelBench(
            function_name="label2rgb",
            shape=shape,
            dtypes=dtypes,
            contiguous_labels=contiguous_labels,
            index_str=index_str,
            fixed_kwargs=dict(bg_label=0),
            var_kwargs=dict(kind=["avg", "overlay"]),
            module_cpu=skimage.color,
            module_gpu=cucim.skimage.color,
        )
        results = B.run_benchmark(duration=1)
        all_results = pd.concat([all_results, results["full"]])

fbase = os.path.splitext(pfile)[0]
all_results.to_csv(fbase + ".csv")
all_results.to_pickle(pfile)
with open(fbase + ".md", "wt") as f:
    f.write(all_results.to_markdown())
