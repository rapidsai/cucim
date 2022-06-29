import argparse
import os
import pickle

import cucim.skimage
import cucim.skimage.metrics
import cupy as cp
import numpy as np
import pandas as pd
import skimage
import skimage.metrics

from _image_bench import ImageBench


class MetricsBench(ImageBench):
    def set_args(self, dtype):
        imaged = cp.testing.shaped_arange(self.shape, dtype=dtype)
        imaged2 = cp.testing.shaped_arange(self.shape, dtype=dtype)
        imaged2 = imaged2 + 0.05 * cp.random.standard_normal(self.shape)
        imaged = imaged / imaged.max()
        imaged2 = imaged2 / imaged2.max()
        imaged2 = imaged2.clip(0, 1.0)
        self.args_cpu = (cp.asnumpy(imaged), cp.asnumpy(imaged2))
        self.args_gpu = (imaged, imaged2)


def main(args):

    pfile = "cucim_metrics_results.pickle"
    if os.path.exists(pfile):
        with open(pfile, "rb") as f:
            all_results = pickle.load(f)
    else:
        all_results = pd.DataFrame()

    dtypes = [np.dtype(args.dtype)]
    # image sizes/shapes
    shape = tuple(list(map(int,(args.img_size.split(',')))))
    run_cpu = not args.no_cpu

    for function_name, fixed_kwargs, var_kwargs, allow_color, allow_nd in [
        # _structural_similarity.py
        (
            "structural_similarity",
            dict(data_range=1.0),
            dict(gradient=[False, True], gaussian_weights=[False, True]),
            True,
            True,
        ),
        # simple_metrics.py
        ("mean_squared_error", dict(), dict(), True, True),
        (
            "normalized_root_mse",
            dict(),
            dict(normalization=["euclidean", "min-max", "mean"]),
            True,
            True,
        ),
        ("peak_signal_noise_ratio", dict(data_range=1.0), dict(), True, True),
        ("normalized_mutual_information", dict(bins=100), dict(), True, True),

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

        if function_name in ["structural_similarity"]:
            fixed_kwargs["channel_axis"] = -1 if shape[-1] == 3 else None

        B = MetricsBench(
            function_name=function_name,
            shape=shape,
            dtypes=dtypes,
            fixed_kwargs=fixed_kwargs,
            var_kwargs=var_kwargs,
            module_cpu=skimage.metrics,
            module_gpu=cucim.skimage.metrics,
            run_cpu=run_cpu,
        )
        results = B.run_benchmark(duration=1)
        all_results = pd.concat([all_results, results["full"]])


    fbase = os.path.splitext(pfile)[0]
    all_results.to_csv(fbase + ".csv")
    all_results.to_pickle(pfile)
    try:
        import tabulate

        with open(fbase + ".md", "wt") as f:
            f.write(all_results.to_markdown())
    except ImportError:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmarking cuCIM metrics functions')
    func_name_choices = ['structural_similarity', 'mean_squared_error', 'normalized_root_mse', 'peak_signal_noise_ratio', 'normalized_mutual_information']
    dtype_choices = ['float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64']
    parser.add_argument('-i','--img_size', type=str, help='Size of input image', required=True)
    parser.add_argument('-d','--dtype', type=str, help='Dtype of input image', choices=dtype_choices, required=True)
    parser.add_argument('-f','--func_name', type=str, help='function to benchmark', choices=func_name_choices, required=True)
    parser.add_argument('-t','--duration', type=int, help='time to run benchmark', required=True)
    parser.add_argument('--no_cpu', action='store_true', help='disable cpu measurements', default=False)

    args = parser.parse_args()
    main(args)
