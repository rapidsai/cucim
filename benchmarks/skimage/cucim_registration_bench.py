import argparse
import math
import os
import pickle

import cupy as cp
import numpy as np
import pandas as pd
import skimage
import skimage.registration
from _image_bench import ImageBench

import cucim.skimage
import cucim.skimage.registration


class RegistrationBench(ImageBench):
    def set_args(self, dtype):
        if np.dtype(dtype).kind in "iu":
            im1 = skimage.data.camera()
        else:
            im1 = skimage.data.camera() / 255.0
            im1 = im1.astype(dtype)
        if len(self.shape) == 3:
            im1 = im1[..., np.newaxis]
        n_tile = [math.ceil(s / im_s) for s, im_s in zip(self.shape, im1.shape)]
        slices = tuple([slice(s) for s in self.shape])
        image = np.tile(im1, n_tile)[slices]
        image2 = np.roll(image, (10, 20))
        imaged = cp.asarray(image)
        imaged2 = cp.asarray(image2)

        self.args_cpu = (image, image2)
        self.args_gpu = (imaged, imaged2)


def main(args):
    pfile = "cucim_registration_results.pickle"
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
        # _phase_cross_correlation.py
        ("phase_cross_correlation", dict(), dict(), False, True),
        # optical flow functions
        (
            "optical_flow_tvl1",
            dict(),
            dict(num_iter=[10], num_warp=[5]),
            False,
            True,
        ),
        (
            "optical_flow_ilk",
            dict(),
            dict(
                radius=[3, 7],
                num_warp=[10],
                gaussian=[False, True],
                prefilter=[False, True],
            ),
            False,
            True,
        ),
    ]:
        if function_name != args.func_name:
            continue

        if function_name == "phase_cross_correlation":
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

            for masked in [True, False]:
                index_str = f"masked={masked}"
                if masked:
                    moving_mask = cp.ones(shape, dtype=bool)
                    moving_mask[20:-20, :] = 0
                    moving_mask[:, 20:-20] = 0
                    reference_mask = cp.ones(shape, dtype=bool)
                    reference_mask[80:-80, :] = 0
                    reference_mask[:, 80:-80] = 0
                    fixed_kwargs["moving_mask"] = moving_mask
                    fixed_kwargs["reference_mask"] = reference_mask
                else:
                    fixed_kwargs["moving_mask"] = None
                    fixed_kwargs["reference_mask"] = None

                B = RegistrationBench(
                    function_name=function_name,
                    shape=shape,
                    dtypes=dtypes,
                    fixed_kwargs=fixed_kwargs,
                    var_kwargs=var_kwargs,
                    index_str=index_str,
                    module_cpu=skimage.registration,
                    module_gpu=cucim.skimage.registration,
                    run_cpu=run_cpu,
                )
            results = B.run_benchmark(duration=args.duration)
            all_results = pd.concat([all_results, results["full"]])

        else:
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

            B = RegistrationBench(
                function_name=function_name,
                shape=shape,
                dtypes=dtypes,
                fixed_kwargs=fixed_kwargs,
                var_kwargs=var_kwargs,
                module_cpu=skimage.registration,
                module_gpu=cucim.skimage.registration,
                run_cpu=run_cpu,
            )
            results = B.run_benchmark(duration=args.duration)
            all_results = pd.concat([all_results, results["full"]])

    fbase = os.path.splitext(pfile)[0]
    all_results.to_csv(fbase + ".csv")
    all_results.to_pickle(pfile)
    try:
        import tabular  # noqa: F401

        with open(fbase + ".md", "wt") as f:
            f.write(all_results.to_markdown())
    except ImportError:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmarking cuCIM registration functions"
    )
    func_name_choices = [
        "phase_cross_correlation",
        "optical_flow_tvl1",
        "optical_flow_ilk",
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
