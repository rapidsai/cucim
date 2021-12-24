import math
import os
import pickle

import cucim.skimage
import cucim.skimage.restoration
import cupy as cp
import cupyx.scipy.ndimage as ndi
import numpy as np
import pandas as pd
import skimage
import skimage.restoration
from cucim.skimage.restoration import denoise_tv_chambolle as tv_gpu
from skimage.restoration import denoise_tv_chambolle as tv_cpu

from _image_bench import ImageBench


class DenoiseBench(ImageBench):
    def set_args(self, dtype):
        if np.dtype(dtype).kind in "iu":
            im1 = skimage.data.camera()
        else:
            im1 = skimage.data.camera() / 255.0
            im1 = im1.astype(dtype)
        if len(self.shape) == 3:
            im1 = im1[..., np.newaxis]

        # add noise
        if np.dtype(dtype).kind in "iu":
            sigma = 0.05 * 255
            im1 = im1 + sigma * np.random.randn(*im1.shape)
            im1 = np.clip(im1, 0, 255).astype(dtype)
        else:
            sigma = 0.05
            im1 = im1 + sigma * np.random.randn(*im1.shape)

        n_tile = [math.ceil(s / im_s) for s, im_s in zip(self.shape, im1.shape)]
        slices = tuple([slice(s) for s in self.shape])
        image = np.tile(im1, n_tile)[slices]
        imaged = cp.asarray(image)

        self.args_cpu = (image,)
        self.args_gpu = (imaged,)


class CalibratedDenoiseBench(ImageBench):
    def set_args(self, dtype):
        if np.dtype(dtype).kind in "iu":
            im1 = skimage.data.camera()
        else:
            im1 = skimage.data.camera() / 255.0
            im1 = im1.astype(dtype)
        if len(self.shape) == 3:
            im1 = im1[..., np.newaxis]

        # add noise
        if np.dtype(dtype).kind in "iu":
            sigma = 0.05 * 255
            im1 = im1 + sigma * np.random.randn(*im1.shape)
            im1 = np.clip(im1, 0, 255).astype(dtype)
        else:
            sigma = 0.05
            im1 = im1 + sigma * np.random.randn(*im1.shape)

        n_tile = [math.ceil(s / im_s) for s, im_s in zip(self.shape, im1.shape)]
        slices = tuple([slice(s) for s in self.shape])
        image = np.tile(im1, n_tile)[slices]
        imaged = cp.asarray(image)

        denoise_parameters = {"weight": np.linspace(0.01, 0.4, 10)}
        self.args_cpu = (image, tv_cpu, denoise_parameters)
        self.args_gpu = (imaged, tv_gpu, denoise_parameters)


class DeconvolutionBench(ImageBench):
    def set_args(self, dtype):
        if np.dtype(dtype).kind in "iu":
            im1 = skimage.data.camera()
        else:
            im1 = skimage.data.camera() / 255.0
            im1 = im1.astype(dtype)
        if len(self.shape) == 3:
            im1 = im1[..., np.newaxis]
        im1 = cp.array(im1)
        n_tile = [math.ceil(s / im_s) for s, im_s in zip(self.shape, im1.shape)]
        slices = tuple([slice(s) for s in self.shape])
        imaged = cp.tile(im1, n_tile)[slices]

        psfd = cp.ones((5,) * imaged.ndim) / 25
        imaged = ndi.convolve(imaged, psfd)

        image = cp.asnumpy(imaged)
        psf = cp.asnumpy(psfd)

        self.args_cpu = (image, psf)
        self.args_gpu = (imaged, psfd)


pfile = "cucim_restoration_results.pickle"
if os.path.exists(pfile):
    with open(pfile, "rb") as f:
        all_results = pickle.load(f)
else:
    all_results = pd.DataFrame()
dtypes = [np.float32]


for function_name, fixed_kwargs, var_kwargs, allow_color, allow_nd in [
    # _denoise.py
    ("denoise_tv_chambolle", dict(), dict(weight=[0.02]), True, True),
    # j_invariant.py
    ("calibrate_denoiser", dict(), dict(), False, True),
]:

    for shape in [(512, 512), (1980, 1080), (1980, 1080, 3), (128, 128, 128)]:

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

        if function_name == "denoise_tv_chambolle":
            fixed_kwargs["channel_axis"] = -1 if shape[-1] == 3 else None

        if function_name == "calibrate_denoiser":
            denoise_class = CalibratedDenoiseBench
        else:
            denoise_class = DenoiseBench

        B = denoise_class(
            function_name=function_name,
            shape=shape,
            dtypes=dtypes,
            fixed_kwargs=fixed_kwargs,
            var_kwargs=var_kwargs,
            module_cpu=skimage.restoration,
            module_gpu=cucim.skimage.restoration,
        )
        results = B.run_benchmark(duration=1)
        all_results = all_results.append(results["full"])


# function_name, fixed_kwargs, var_kwargs, allow_color, allow_nd = ('unsupervised_wiener', dict(), dict(), False, True)
for function_name, fixed_kwargs, var_kwargs, allow_color, allow_nd in [
    # deconvolution.py
    ("wiener", dict(balance=100.0), dict(), False, False),
    ("unsupervised_wiener", dict(), dict(), False, False),
    ("richardson_lucy", dict(), dict(iterations=[5]), False, True),
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

        B = DeconvolutionBench(
            function_name=function_name,
            shape=shape,
            dtypes=dtypes,
            fixed_kwargs=fixed_kwargs,
            var_kwargs=var_kwargs,
            module_cpu=skimage.restoration,
            module_gpu=cucim.skimage.restoration,
        )
        results = B.run_benchmark(duration=1)
        all_results = all_results.append(results["full"])


fbase = os.path.splitext(pfile)[0]
all_results.to_csv(fbase + ".csv")
all_results.to_pickle(pfile)
with open(fbase + ".md", "wt") as f:
    f.write(all_results.to_markdown())
