import os
import pickle

import cucim.skimage
import cucim.skimage.exposure
import cupy
import cupy as cp
import numpy as np
import pandas as pd
import skimage
import skimage.exposure

from _image_bench import ImageBench


class ExposureBench(ImageBench):
    def set_args(self, dtype):
        if np.dtype(dtype).kind in "iu":
            scale = 256
        else:
            scale = 1.0
        imaged = cupy.testing.shaped_random(self.shape, xp=cp, dtype=dtype, scale=scale)
        image = cp.asnumpy(imaged)
        self.args_cpu = (image,)
        self.args_gpu = (imaged,)


class MatchHistogramBench(ImageBench):
    def set_args(self, dtype):
        if np.dtype(dtype).kind in "iu":
            scale = 256
        else:
            scale = 1.0
        imaged = cupy.testing.shaped_random(self.shape, xp=cp, dtype=dtype, scale=scale)
        imaged2 = cupy.testing.shaped_random(self.shape, xp=cp, dtype=dtype, scale=scale)
        image = cp.asnumpy(imaged)
        image2 = cp.asnumpy(imaged2)
        self.args_cpu = (image, image2)
        self.args_gpu = (imaged, imaged2)


pfile = "cucim_exposure_results.pickle"
if os.path.exists(pfile):
    with open(pfile, "rb") as f:
        all_results = pickle.load(f)
else:
    all_results = pd.DataFrame()
dtypes = [np.uint8, np.float32]

exposure_config = {
    "equalize_adapthist": dict(
        fixed_kwargs=dict(clip_limit=0.01, nbins=256),
        variable_kwargs=dict(),
        color_required=False,
        grayscale_only=False,
        dtypes=None,
        shapes=None,
    ),
    "histogram": dict(
        fixed_kwargs=dict(source_range="image"),
        variable_kwargs=dict(nbins=[16, 256], normalize=[True, False]),
        color_required=False,
        grayscale_only=True,
        dtypes=None,
        shapes=None,
    ),
}

for function_name, fixed_kwargs, var_kwargs, allow_color in [
    ("equalize_adapthist", dict(clip_limit=0.01, nbins=256), dict(), True),
    (
        "histogram",
        dict(source_range="image"),
        dict(nbins=[16, 256], normalize=[True, False]),
        False,
    ),
    ("cumulative_distribution", dict(), dict(nbins=[16, 256]), False),
    ("equalize_hist", dict(mask=None), dict(nbins=[16, 256]), False),
    ("rescale_intensity", dict(in_range="image", out_range="dtype"), dict(), False),
    ("adjust_gamma", dict(), dict(), False),
    ("adjust_log", dict(), dict(), False),
    ("adjust_sigmoid", dict(), dict(inv=[False, True]), False),
    ("is_low_contrast", dict(), dict(), False),
]:

    for shape in [(512, 512), (3840, 2160), (3840, 2160, 3), (192, 192, 192)]:
        ndim = len(shape)
        if shape[-1] == 3 and not allow_color:
            continue

        if function_name == "equalize_adapthist":
            #  TODO: fix equalize_adapthist for size (3840, 2160) and kernel_size = [16, 16]
            size_factors = [4, 8, 16]
            kernel_sizes = []
            for size_factor in size_factors:
                kernel_sizes.append([max(s // size_factor, 1) for s in shape if s != 3])
            var_kwargs.update(dict(kernel_size=kernel_sizes))

        B = ExposureBench(
            function_name=function_name,
            shape=shape,
            dtypes=dtypes,
            fixed_kwargs=fixed_kwargs,
            var_kwargs=var_kwargs,
            module_cpu=skimage.exposure,
            module_gpu=cucim.skimage.exposure,
        )
        results = B.run_benchmark(duration=1)
        all_results = all_results.append(results["full"])


for shape in [(512, 512), (3840, 2160), (3840, 2160, 3), (192, 192, 192)]:
    ndim = len(shape)

    channel_axis = -1 if shape[-1] in [3, 4] else None

    B = MatchHistogramBench(
        function_name="match_histograms",
        shape=shape,
        dtypes=dtypes,
        fixed_kwargs=dict(channel_axis=channel_axis),
        var_kwargs=dict(),
        module_cpu=skimage.exposure,
        module_gpu=cucim.skimage.exposure,
    )
    results = B.run_benchmark(duration=1)
    all_results = all_results.append(results["full"])

fbase = os.path.splitext(pfile)[0]
all_results.to_csv(fbase + ".csv")
all_results.to_pickle(pfile)
with open(fbase + ".md", "wt") as f:
    f.write(all_results.to_markdown())
