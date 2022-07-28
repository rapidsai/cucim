import argparse
import os
import pickle

import cucim.skimage
import cucim.skimage.transform
import numpy as np
import pandas as pd
import skimage
import skimage.transform

from _image_bench import ImageBench


def main(args):

    pfile = "cucim_transform_results.pickle"
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
        # _warps.py
        (
            "resize",
            dict(preserve_range=True),
            dict(order=[0, 1, 3], mode=["reflect"], anti_aliasing=[True]),
            True,
            True,
        ),  # scale handled in loop below
        (
            "resize_local_mean",
            dict(preserve_range=True),
            {},
            True,
            True,
        ),  # scale handled in loop below
        (
            "rescale",
            dict(preserve_range=True),
            dict(order=[0, 1, 3], mode=["reflect"], anti_aliasing=[True]),
            True,
            True,
        ),  # output_shape handled in loop below
        (
            "rotate",
            dict(angle=15, preserve_range=True),
            dict(order=[0, 1, 3], mode=["reflect"], resize=[False, True]),
            False,
            False,
        ),
        ("downscale_local_mean", dict(), dict(), True, True),  # factors handled in loop below
        (
            "swirl",
            dict(strength=1, preserve_range=True),
            dict(order=[0, 1, 3], mode=["reflect"]),
            False,
            False,
        ),
        # TODO : warp? already indirectly benchmarked via swirl, etc
        ("warp_polar", dict(), dict(scaling=["linear", "log"]), True, False),
        # integral.py
        ("integral_image", dict(), dict(), False, True),
        # TODO: integrate
        # pyramids.py
        (
            "pyramid_gaussian",
            dict(max_layer=6, downscale=2, preserve_range=True),
            dict(order=[0, 1, 3]),
            True,
            True,
        ),
        (
            "pyramid_laplacian",
            dict(max_layer=6, downscale=2, preserve_range=True),
            dict(order=[0, 1, 3]),
            True,
            True,
        ),
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

        ndim_spatial = ndim - 1 if shape[-1] == 3 else ndim

        if function_name in ["rescale", "warp_polar", "pyramid_gaussian", "pyramid_laplacian"]:
            fixed_kwargs["channel_axis"] = -1 if ndim_spatial < ndim else None

        function_is_generator = function_name in ["pyramid_gaussian", "pyramid_laplacian"]

        if function_name in ["rescale", "resize", "resize_local_mean"]:
            scales = [0.75, 1.25]
            if function_name == "rescale":
                var_kwargs["scale"] = [(s,) * ndim_spatial for s in scales]
            elif function_name.startswith("resize"):
                out_shapes = [[int(s_ * s) for s_ in shape] for s in scales]
                if ndim_spatial < ndim:
                    # don't resize along channels dimension
                    out_shapes = [
                        tuple([int(s_ * s) for s_ in shape[:-1]]) + (shape[-1],) for s in scales
                    ]
                else:
                    out_shapes = [tuple([int(s_ * s) for s_ in shape]) for s in scales]
                var_kwargs["output_shape"] = out_shapes

        elif function_name == "downscale_local_mean":
            if ndim_spatial < ndim:
                # no downscaling along channels axis
                factors = [(2,) * (ndim - 1) + (1,)]
            else:
                factors = [(2,) * (ndim - 1) + (4,)]
            var_kwargs["factors"] = factors

        B = ImageBench(
            function_name=function_name,
            shape=shape,
            dtypes=dtypes,
            fixed_kwargs=fixed_kwargs,
            var_kwargs=var_kwargs,
            module_cpu=skimage.transform,
            module_gpu=cucim.skimage.transform,
            function_is_generator=function_is_generator,
        )
        results = B.run_benchmark(duration=args.duration)
        all_results = pd.concat([all_results, results["full"]])

    fbase = os.path.splitext(pfile)[0]
    all_results.to_csv(fbase + ".csv")
    all_results.to_pickle(pfile)
    try:
        import tabular

        with open(fbase + ".md", "wt") as f:
            f.write(all_results.to_markdown())
    except ImportError:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmarking cuCIM transform functions')
    func_name_choices = ['resize', 'resize_local_mean', 'rescale', 'rotate', 'downscale_local_mean', 'warp_polar', 'integral_image', 'pyramid_gaussian', 'pyramid_laplacian']
    dtype_choices = ['float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64']
    parser.add_argument('-i','--img_size', type=str, help='Size of input image (omit color channel, it will be appended as needed)', required=True)
    parser.add_argument('-d','--dtype', type=str, help='Dtype of input image', choices = dtype_choices, required=True)
    parser.add_argument('-f','--func_name', type=str, help='function to benchmark', choices = func_name_choices, required=True)
    parser.add_argument('-t','--duration', type=int, help='time to run benchmark', required=True)
    parser.add_argument('--no_cpu', action='store_true', help='disable cpu measurements', default=False)

    args = parser.parse_args()
    main(args)
