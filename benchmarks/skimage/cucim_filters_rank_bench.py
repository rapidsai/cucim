# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os

import numpy as np
import pandas as pd
import skimage.filters.rank
from _image_bench import ImageBench
from skimage.morphology import disk

import cucim.skimage.filters.rank

RANK_FILTERS = [
    # generic.py
    ("autolevel", dict(), dict()),
    ("enhance_contrast", dict(), dict()),
    ("entropy", dict(), dict()),
    ("equalize", dict(), dict()),
    ("geometric_mean", dict(), dict()),
    ("gradient", dict(), dict()),
    ("majority", dict(), dict()),
    ("maximum", dict(), dict()),
    ("mean", dict(), dict()),
    ("median", dict(), dict()),
    ("minimum", dict(), dict()),
    ("modal", dict(), dict()),
    ("noise_filter", dict(), dict()),
    ("pop", dict(), dict()),
    ("subtract_mean", dict(), dict()),
    ("sum", dict(), dict()),
    ("threshold", dict(), dict()),
    # percentile.py
    ("autolevel_percentile", dict(p0=0.1, p1=0.9), dict()),
    ("enhance_contrast_percentile", dict(p0=0.1, p1=0.9), dict()),
    ("gradient_percentile", dict(p0=0.1, p1=0.9), dict()),
    ("mean_percentile", dict(p0=0.1, p1=0.9), dict()),
    ("percentile", dict(p0=0.5), dict()),
    ("pop_percentile", dict(p0=0.1, p1=0.9), dict()),
    ("subtract_mean_percentile", dict(p0=0.1, p1=0.9), dict()),
    ("sum_percentile", dict(p0=0.1, p1=0.9), dict()),
    ("threshold_percentile", dict(p0=0.5), dict()),
    # bilateral.py
    ("mean_bilateral", dict(s0=10, s1=10), dict()),
    ("pop_bilateral", dict(s0=10, s1=10), dict()),
    ("sum_bilateral", dict(s0=10, s1=10), dict()),
]


def _parse_shape(img_size):
    return tuple(list(map(int, img_size.split(","))))


def _parse_radii(radii):
    return [int(r) for r in radii.split(",")]


def main(args):
    cfile = "cucim_filters_rank_results.csv"
    if getattr(args, "no_resume", False) or not os.path.exists(cfile):
        all_results = pd.DataFrame()
    else:
        all_results = pd.read_csv(cfile, index_col=0)
    dtypes = [np.dtype(args.dtype)]

    shape = _parse_shape(args.img_size)
    if len(shape) != 2:
        raise ValueError("rank filter benchmarks use 2D disk footprints")

    footprints = [disk(radius).astype(bool) for radius in _parse_radii(args.radii)]

    for function_name, fixed_kwargs, var_kwargs in RANK_FILTERS:
        if function_name != args.func_name:
            continue

        var_kwargs = dict(var_kwargs)
        var_kwargs["footprint"] = footprints

        B = ImageBench(
            function_name=function_name,
            shape=shape,
            dtypes=dtypes,
            fixed_kwargs=fixed_kwargs,
            var_kwargs=var_kwargs,
            module_cpu=skimage.filters.rank,
            module_gpu=cucim.skimage.filters.rank,
            run_cpu=not args.no_cpu,
        )
        results = B.run_benchmark(duration=args.duration)
        all_results = pd.concat([all_results, results["full"]])

    fbase = os.path.splitext(cfile)[0]
    all_results.to_csv(cfile, index=True)
    with open(fbase + ".md", "w") as f:
        f.write(all_results.to_markdown())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking cuCIM rank filters")
    func_name_choices = [filter_spec[0] for filter_spec in RANK_FILTERS]
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
        "--radii",
        type=str,
        help="Comma-separated disk footprint radii to benchmark",
        default="1,3,5,7",
    )
    parser.add_argument(
        "--no_cpu",
        action="store_true",
        help="disable cpu measurements",
        default=False,
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="do not load existing results CSV; save only this run's results (overwrite)",
        default=False,
    )

    args = parser.parse_args()
    main(args)
