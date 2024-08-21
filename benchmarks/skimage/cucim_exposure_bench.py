import argparse
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
        imaged = cupy.testing.shaped_random(
            self.shape, xp=cp, dtype=dtype, scale=scale
        )
        image = cp.asnumpy(imaged)
        self.args_cpu = (image,)
        self.args_gpu = (imaged,)


class MatchHistogramBench(ImageBench):
    def set_args(self, dtype):
        if np.dtype(dtype).kind in "iu":
            scale = 256
        else:
            scale = 1.0
        imaged = cupy.testing.shaped_random(
            self.shape, xp=cp, dtype=dtype, scale=scale
        )
        imaged2 = cupy.testing.shaped_random(
            self.shape, xp=cp, dtype=dtype, scale=scale
        )
        image = cp.asnumpy(imaged)
        image2 = cp.asnumpy(imaged2)
        self.args_cpu = (image, image2)
        self.args_gpu = (imaged, imaged2)


def main(args):
    pfile = "cucim_exposure_results.pickle"
    if os.path.exists(pfile):
        with open(pfile, "rb") as f:
            all_results = pickle.load(f)
    else:
        all_results = pd.DataFrame()

    dtypes = [np.dtype(args.dtype)]
    # image sizes/shapes
    shape = tuple(list(map(int, (args.img_size.split(",")))))
    run_cpu = not args.no_cpu

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
        (
            "rescale_intensity",
            dict(in_range="image", out_range="dtype"),
            dict(),
            False,
        ),
        ("adjust_gamma", dict(), dict(), False),
        ("adjust_log", dict(), dict(), False),
        ("adjust_sigmoid", dict(), dict(inv=[False, True]), False),
        ("is_low_contrast", dict(), dict(), False),
    ]:
        if function_name != args.func_name:
            continue

        if function_name == "match_histograms":
            channel_axis = -1 if shape[-1] in [3, 4] else None

            B = MatchHistogramBench(
                function_name="match_histograms",
                shape=shape,
                dtypes=dtypes,
                fixed_kwargs=dict(channel_axis=channel_axis),
                var_kwargs=dict(),
                module_cpu=skimage.exposure,
                module_gpu=cucim.skimage.exposure,
                run_cpu=run_cpu,
            )
            results = B.run_benchmark(duration=args.duration)
            all_results = pd.concat([all_results, results["full"]])

        else:
            if shape[-1] == 3 and not allow_color:
                continue

            if function_name == "equalize_adapthist":
                #  TODO: fix equalize_adapthist for size (3840, 2160)
                #        and kernel_size = [16, 16]
                size_factors = [4, 8, 16]
                kernel_sizes = []
                for size_factor in size_factors:
                    kernel_sizes.append(
                        [max(s // size_factor, 1) for s in shape if s != 3]
                    )
                var_kwargs.update(dict(kernel_size=kernel_sizes))

            B = ExposureBench(
                function_name=function_name,
                shape=shape,
                dtypes=dtypes,
                fixed_kwargs=fixed_kwargs,
                var_kwargs=var_kwargs,
                module_cpu=skimage.exposure,
                module_gpu=cucim.skimage.exposure,
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
    parser = argparse.ArgumentParser(
        description="Benchmarking cuCIM exposure functions"
    )
    func_name_choices = [
        "equalize_adapthist",
        "cumulative_distribution",
        "equalize_hist",
        "rescale_intensity",
        "adjust_gamma",
        "adjust_log",
        "adjust_sigmoid",
        "is_low_contrast",
        "match_histograms",
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
        "-i", "--img_size", type=str, help="Size of input image", required=True
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
