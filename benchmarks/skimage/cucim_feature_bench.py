import os
import pickle
import argparse

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

def main(args):

    pfile = "cucim_feature_results.pickle"
    if os.path.exists(pfile):
        with open(pfile, "rb") as f:
            all_results = pickle.load(f)
    else:
        all_results = pd.DataFrame()

    dtype_dict = {'fp64': np.float64, 'fp32': np.float32, 'fp16': np.float16}
    dtypes = [dtype_dict[args.dtype]]

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
        ("match_template", dict(), dict(pad_input=[False], mode=["reflect"]), False, True)
    ]:

        if function_name == args.func_name:
            shape = tuple(list(map(int,(args.img_size.split(',')))))
        else:
            continue

        #if function_name in ["corner_peaks", "peak_local_max"] and np.prod(shape) > 1000000:
            # skip any large sizes that take too long
        ndim = len(shape)
        run_cpu = not args.no_cpu

        if function_name != "match_template":
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
                run_cpu=run_cpu,
            )
        else:
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
                run_cpu=run_cpu,
            )

        results = B.run_benchmark(duration=args.duration)
        all_results = all_results.append(results["full"])

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
    parser = argparse.ArgumentParser(description='Benchmarking cuCIM Feature')
    func_name_choices = ["multiscale_basic_features","canny","daisy","structure_tensor","hessian_matrix","hessian_matrix_det","shape_index","corner_kitchen_rosenfeld","corner_harris","corner_shi_tomasi","corner_foerstner","corner_peaks","match_template"]
    parser.add_argument('-i','--img_size', type=str, help='Size of input image', required=True)
    parser.add_argument('-d','--dtype', type=str, help='Dtype of input image', choices = ['fp64','fp32','fp16'], required=True)
    parser.add_argument('-f','--func_name', type=str, help='function to benchmark', choices = func_name_choices, required=True)
    parser.add_argument('-t','--duration', type=int, help='time to run benchmark', required=True)
    parser.add_argument('--no_cpu', action='store_true', help='disable cpu measurements', default=False)

    args = parser.parse_args()
    main(args)
