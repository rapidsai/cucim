# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import math
import os
import pickle

import cupy as cp
import cupyx.scipy.ndimage as ndi
import numpy as np
import pandas as pd
import skimage
import skimage.restoration
from _image_bench import ImageBench
from skimage.restoration import denoise_tv_chambolle as tv_cpu

import cucim.skimage
import cucim.skimage.restoration
from cucim.skimage.restoration import denoise_tv_chambolle as tv_gpu


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


def main(args):
    pfile = "cucim_restoration_results.pickle"
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
        # _denoise.py
        ("denoise_tv_chambolle", dict(), dict(weight=[0.02]), True, True),
        # j_invariant.py
        ("calibrate_denoiser", dict(), dict(), False, True),
        # deconvolution.py
        ("wiener", dict(balance=100.0), dict(), False, False),
        ("unsupervised_wiener", dict(), dict(), False, False),
        ("richardson_lucy", dict(), dict(num_iter=[5]), False, True),
    ]:
        if function_name != args.func_name:
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

        if function_name in ["denoise_tv_chambolle", "calibrate_denoiser"]:
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
                run_cpu=run_cpu,
            )
            results = B.run_benchmark(duration=args.duration)
            all_results = pd.concat([all_results, results["full"]])

        elif function_name in [
            "wiener",
            "unsupervised_wiener",
            "richardson_lucy",
        ]:
            B = DeconvolutionBench(
                function_name=function_name,
                shape=shape,
                dtypes=dtypes,
                fixed_kwargs=fixed_kwargs,
                var_kwargs=var_kwargs,
                module_cpu=skimage.restoration,
                module_gpu=cucim.skimage.restoration,
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
        description="Benchmarking cuCIM restoration functions"
    )
    func_name_choices = [
        "denoise_tv_chambolle",
        "calibrate_denoiser",
        "wiener",
        "unsupervised_wiener",
        "richardson_lucy",
    ]
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
