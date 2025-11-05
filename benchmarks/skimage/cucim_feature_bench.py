# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import math
import os
import pickle

import cupy as cp
import numpy as np
import pandas as pd
import skimage
import skimage.feature
from _image_bench import ImageBench
from skimage import data, draw

import cucim.skimage
import cucim.skimage.feature
from cucim.skimage import exposure


class BlobDetectionBench(ImageBench):
    def set_args(self, dtype):
        ndim = len(self.shape)
        if ndim == 2:
            # create 2D image by tiling the coins image
            img = cp.array(data.coins())
            img = exposure.equalize_hist(img)  # improves detection
            n_tile = (math.ceil(s / s0) for s, s0 in zip(self.shape, img.shape))
            img = cp.tile(img, n_tile)
            img = img[tuple(slice(s) for s in img.shape)]
            img = img.astype(dtype, copy=False)
        elif ndim == 3:
            # create 3D volume with randomly positioned ellipses
            e = cp.array(draw.ellipsoid(*(max(s // 20, 1) for s in self.shape)))
            e = e.astype(dtype, copy=False)
            img = cp.zeros(self.shape, dtype=dtype)
            rng = np.random.default_rng(5)
            num_ellipse = 64
            offsets = rng.integers(0, np.prod(img.shape), num_ellipse)
            locs = np.unravel_index(offsets, img.shape)
            for loc in zip(*locs):
                loc = tuple(min(p, s - es) for p, s, es in zip(loc, img.shape, e.shape))
                sl = tuple(slice(p, p + es) for p, es in zip(loc, e.shape))
                img[sl] = e
        else:
            raise NotImplementedError("unsupported ndim")
        self.args_gpu = (img,)
        self.args_cpu = (cp.asnumpy(img),)


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


def main(args):
    pfile = "cucim_feature_results.pickle"
    if os.path.exists(pfile):
        with open(pfile, "rb") as f:
            all_results = pickle.load(f)
    else:
        all_results = pd.DataFrame()

    dtypes = [np.dtype(args.dtype)]

    for function_name, fixed_kwargs, var_kwargs, allow_color, allow_nd in [
        (
            "multiscale_basic_features",
            dict(edges=True),
            dict(texture=[True, False]),
            True,
            True,
        ),
        ("canny", dict(sigma=1.8), dict(), False, False),
        # reduced default rings, histograms, orientations to fit daisy at
        # (3840, 2160) into GPU memory
        (
            "daisy",
            dict(step=4, radius=15, rings=2, histograms=5, orientations=4),
            dict(normalization=["l1", "l2", "daisy"]),
            False,
            False,
        ),
        (
            "structure_tensor",
            dict(sigma=1, mode="reflect", order="rc"),
            dict(),
            False,
            True,
        ),
        (
            "hessian_matrix",
            dict(sigma=1, mode="reflect", order="rc"),
            dict(),
            False,
            True,
        ),
        (
            "hessian_matrix_det",
            dict(sigma=1, approximate=False),
            dict(),
            False,
            True,
        ),
        ("shape_index", dict(sigma=1, mode="reflect"), dict(), False, False),
        (
            "corner_kitchen_rosenfeld",
            dict(mode="reflect"),
            dict(),
            False,
            False,
        ),
        (
            "corner_harris",
            dict(k=0.05, eps=1e-6, sigma=1),
            dict(method=["k", "eps"]),
            False,
            False,
        ),
        ("corner_shi_tomasi", dict(sigma=1), dict(), False, False),
        ("corner_foerstner", dict(sigma=1), dict(), False, False),
        ("corner_peaks", dict(), dict(min_distance=(2, 3, 5)), False, True),
        (
            "match_template",
            dict(),
            dict(pad_input=[False], mode=["reflect"]),
            False,
            True,
        ),
        # blob detectors, fixed kwargs are taken from the docstring examples
        (
            "blob_dog",
            dict(threshold=0.05, min_sigma=10, max_sigma=40),
            dict(),
            False,
            True,
        ),
        ("blob_log", dict(threshold=0.3), dict(), False, True),
        ("blob_doh", dict(), dict(), False, False),
    ]:
        if function_name == args.func_name:
            shape = tuple(list(map(int, (args.img_size.split(",")))))
        else:
            continue

        ndim = len(shape)
        run_cpu = not args.no_cpu
        if not allow_nd:
            if not allow_color:
                if ndim > 2:
                    continue
            else:
                if ndim > 3 or (ndim == 3 and shape[-1] not in [3, 4]):
                    continue

        if shape[-1] == 3 and not allow_color:
            continue

        if function_name.startswith("blob"):
            if ndim == 3:
                # set more reasonable threshold for the synthetic 3d data
                if function_name == "blob_log":
                    fixed_kwargs = {"threshold": 0.6}
                else:
                    fixed_kwargs = {}

            B = BlobDetectionBench(
                function_name=function_name,
                shape=shape,
                dtypes=dtypes,
                fixed_kwargs=fixed_kwargs,
                var_kwargs=var_kwargs,
                module_cpu=skimage.feature,
                module_gpu=cucim.skimage.feature,
                run_cpu=run_cpu,
            )
        elif function_name != "match_template":
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
                run_cpu=run_cpu,
            )
        else:
            B = MatchTemplateBench(
                function_name=function_name,
                shape=shape,
                dtypes=dtypes,
                fixed_kwargs=fixed_kwargs,
                var_kwargs=var_kwargs,
                module_cpu=skimage.feature,
                module_gpu=cucim.skimage.feature,
                run_cpu=run_cpu,
            )

        results = B.run_benchmark(duration=args.duration)
        all_results = pd.concat([all_results, results["full"]])

    fbase = os.path.splitext(pfile)[0]
    all_results.to_csv(fbase + ".csv")
    all_results.to_pickle(pfile)
    with open(fbase + ".md", "w") as f:
        f.write(all_results.to_markdown())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking cuCIM Feature")
    # fmt: off
    func_name_choices = [
        "multiscale_basic_features", "canny", "daisy", "structure_tensor",
        "hessian_matrix", "hessian_matrix_det", "shape_index",
        "corner_kitchen_rosenfeld", "corner_harris", "corner_shi_tomasi",
        "corner_foerstner", "corner_peaks", "match_template", "blob_dog",
        "blob_log", "blob_doh",
    ]
    dtype_choices = [
        "float16", "float32", "float64", "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64"
    ]
    parser.add_argument(
        "-i", "--img_size", type=str, help="Size of input image",
        required=True
    )
    parser.add_argument(
        "-d", "--dtype", type=str, help="Dtype of input image",
        choices=dtype_choices, required=True
    )
    parser.add_argument(
        "-f", "--func_name", type=str, help="function to benchmark",
        choices=func_name_choices, required=True
    )
    parser.add_argument(
        "-t", "--duration", type=int, help="time to run benchmark",
        required=True
    )
    parser.add_argument(
        "--no_cpu", action="store_true", help="disable cpu measurements",
        default=False
    )
    # fmt: on
    args = parser.parse_args()
    main(args)
