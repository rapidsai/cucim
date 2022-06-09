import os
import pickle

import cucim.skimage
import cucim.skimage.feature
import cupy as cp
import numpy as np
import pandas as pd
import skimage
import skimage.feature

from _image_bench import ImageBench


class MatchTemplateBench(ImageBench):
    def set_args(self, dtype):
        rstate = cp.random.RandomState(5)
        imaged = rstate.standard_normal(self.shape) > 2
        imaged = imaged.astype(dtype)
        templated = cp.zeros((3,) * imaged.ndim, dtype=dtype)
        templated[(1,) * imaged.ndim] = 1
        image = cp.asnumpy(imaged)
        template = cp.asnumpy(templated)
        assert imaged.dtype == dtype
        assert imaged.shape == self.shape
        self.args_cpu = (image, template)
        self.args_gpu = (imaged, templated)


pfile = "cucim_feature_results.pickle"
if os.path.exists(pfile):
    with open(pfile, "rb") as f:
        all_results = pickle.load(f)
else:
    all_results = pd.DataFrame()
dtypes = [np.float32]

for function_name, fixed_kwargs, var_kwargs, allow_color, allow_nd in [
    ("multiscale_basic_features", dict(edges=True), dict(texture=[True, False]), True, True),
    ("canny", dict(sigma=1.8), dict(), False, False),
    # reduced default rings, histograms, orientations to fit daisy at (3840, 2160) into GPU memory
    (
        "daisy",
        dict(step=4, radius=15, rings=2, histograms=5, orientations=4),
        dict(normalization=["l1", "l2", "daisy"]),
        False,
        False,
    ),
    ("structure_tensor", dict(sigma=1, mode="reflect", order="rc"), dict(), False, True),
    ("hessian_matrix", dict(sigma=1, mode="reflect", order="rc"), dict(), False, True),
    ("hessian_matrix_det", dict(sigma=1, approximate=False), dict(), False, True),
    ("shape_index", dict(sigma=1, mode="reflect"), dict(), False, False),
    ("corner_kitchen_rosenfeld", dict(mode="reflect"), dict(), False, False),
    ("corner_harris", dict(k=0.05, eps=1e-6, sigma=1), dict(method=["k", "eps"]), False, False),
    ("corner_shi_tomasi", dict(sigma=1), dict(), False, False),
    ("corner_foerstner", dict(sigma=1), dict(), False, False),
    ("corner_peaks", dict(), dict(min_distance=(2, 3, 5)), False, True),
]:

    for shape in [(128, 128, 128), (512, 512), (3840, 2160), (3840, 2160, 3), (192, 192, 192)]:

        if function_name in ["corner_peaks", "peak_local_max"] and np.prod(shape) > 1000000:
            # skip any large sizes that take too long
            continue
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

        if function_name == "multiscale_basic_features":
            fixed_kwargs["channel_axis"] = -1 if shape[-1] == 3 else None
            if ndim == 3 and shape[-1] != 3:
                # Omit texture=True case to avoid excessive GPU memory usage
                var_kwargs["texture"] = [False]

        B = ImageBench(
            function_name=function_name,
            shape=shape,
            dtypes=dtypes,
            fixed_kwargs=fixed_kwargs,
            var_kwargs=var_kwargs,
            module_cpu=skimage.feature,
            module_gpu=cucim.skimage.feature,
        )
        results = B.run_benchmark(duration=1)
        all_results = pd.concat([all_results, results["full"]])


for function_name, fixed_kwargs, var_kwargs, allow_color, allow_nd in [
    ("match_template", dict(), dict(pad_input=[False], mode=["reflect"]), False, True),
]:
    for shape in [(512, 512), (3840, 2160), (3840, 2160, 3), (192, 192, 192)]:

        ndim = len(shape)
        if not allow_nd:
            if allow_color:
                if ndim > 2:
                    continue
            else:
                if ndim > 3 or (ndim == 3 and shape[-1] not in [3, 4]):
                    continue
        if shape[-1] == 3 and not allow_color:
            continue

        B = MatchTemplateBench(
            function_name=function_name,
            shape=shape,
            dtypes=dtypes,
            fixed_kwargs=fixed_kwargs,
            var_kwargs=var_kwargs,
            module_cpu=skimage.feature,
            module_gpu=cucim.skimage.feature,
        )
        results = B.run_benchmark(duration=1)
        all_results = pd.concat([all_results, results["full"]])

fbase = os.path.splitext(pfile)[0]
all_results.to_csv(fbase + ".csv")
all_results.to_pickle(pfile)
with open(fbase + ".md", "wt") as f:
    f.write(all_results.to_markdown())
