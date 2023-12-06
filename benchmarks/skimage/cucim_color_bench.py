import argparse
import os
import pickle

import cupy
import cupy as cp
import cupyx.scipy.ndimage
import numpy as np
import pandas as pd
import scipy
import skimage
import skimage.color
from _image_bench import ImageBench

import cucim.skimage
import cucim.skimage.color

func_name_choices = [
    "convert_colorspace",
    "rgb2hed",
    "hed2rgb",
    "lab2lch",
    "lch2lab",
    "xyz2lab",
    "lab2xyz",
    "rgba2rgb",
    "label2rgb",
]


class ColorBench(ImageBench):
    def set_args(self, dtype):
        if self.shape[-1] != 3:
            raise ValueError("shape must be 3 on the last axis")
        imaged = cupy.testing.shaped_random(
            self.shape, xp=cp, dtype=dtype, scale=1.0
        )
        image = cp.asnumpy(imaged)
        self.args_cpu = (image,)
        self.args_gpu = (imaged,)


class RGBABench(ImageBench):
    def set_args(self, dtype):
        if self.shape[-1] != 4:
            raise ValueError("shape must be 4 on the last axis")
        imaged = cupy.testing.shaped_random(
            self.shape, xp=cp, dtype=dtype, scale=1.0
        )
        image = cp.asnumpy(imaged)
        self.args_cpu = (image,)
        self.args_gpu = (imaged,)


class LabelBench(ImageBench):
    def __init__(
        self,
        function_name,
        shape,
        contiguous_labels=True,
        dtypes=np.float32,
        fixed_kwargs={},
        var_kwargs={},
        index_str=None,
        module_cpu=scipy.ndimage,
        module_gpu=cupyx.scipy.ndimage,
        run_cpu=True,
    ):
        self.contiguous_labels = contiguous_labels
        super().__init__(
            function_name=function_name,
            shape=shape,
            dtypes=dtypes,
            fixed_kwargs=fixed_kwargs,
            var_kwargs=var_kwargs,
            index_str=index_str,
            module_cpu=module_cpu,
            module_gpu=module_gpu,
            run_cpu=run_cpu,
        )

    def set_args(self, dtype):
        a = np.array(
            [
                [0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 4, 0],
                [2, 2, 0, 0, 3, 0, 4, 4],
                [0, 0, 0, 0, 0, 5, 0, 0],
            ],
            dtype=int,
        )
        tiling = tuple(s // a_s for s, a_s in zip(self.shape, a.shape))
        if self.contiguous_labels:
            label = np.kron(a, np.ones(tiling, dtype=a.dtype))
        else:
            label = np.tile(a, tiling)
        labelled = cp.asarray(label)
        imaged = cupy.testing.shaped_random(
            labelled.shape, xp=cp, dtype=dtype, scale=1.0
        )
        image = cp.asnumpy(imaged)
        self.args_cpu = (
            label,
            image,
        )
        self.args_gpu = (
            labelled,
            imaged,
        )


def main(args):
    pfile = "cucim_color_results.pickle"
    if os.path.exists(pfile):
        with open(pfile, "rb") as f:
            all_results = pickle.load(f)
    else:
        all_results = pd.DataFrame()

    dtypes = [np.dtype(args.dtype)]
    # image sizes/shapes
    shape = tuple(list(map(int, (args.img_size.split(",")))))
    run_cpu = not args.no_cpu

    all_colorspaces = False

    for function_name in func_name_choices:
        if function_name != args.func_name:
            continue

        if function_name == "convert_colorspace":
            if all_colorspaces:
                color_spaces = [
                    "RGB",
                    "HSV",
                    "RGB CIE",
                    "XYZ",
                    "YUV",
                    "YIQ",
                    "YPbPr",
                    "YCbCr",
                    "YDbDr",
                ]
            else:
                color_spaces = ["RGB", "HSV", "YUV", "XYZ"]
            for fromspace in color_spaces:
                for tospace in color_spaces:
                    if fromspace == tospace:
                        continue

                    B = ColorBench(
                        function_name="convert_colorspace",
                        shape=shape + (3,),
                        dtypes=dtypes,
                        fixed_kwargs=dict(fromspace=fromspace, tospace=tospace),
                        var_kwargs={},
                        index_str=f"{fromspace.lower()}2{tospace.lower()}",
                        module_cpu=skimage.color,
                        module_gpu=cucim.skimage.color,
                        run_cpu=run_cpu,
                    )
                    results = B.run_benchmark(duration=args.duration)
                    all_results = pd.concat([all_results, results["full"]])

        elif function_name == "rgba2rgb":
            B = RGBABench(
                function_name="rgba2rgb",
                shape=shape[:-1] + (4,),
                dtypes=dtypes,
                fixed_kwargs={},
                var_kwargs={},
                module_cpu=skimage.color,
                module_gpu=cucim.skimage.color,
                run_cpu=run_cpu,
            )
            results = B.run_benchmark(duration=args.duration)
            all_results = pd.concat([all_results, results["full"]])

        elif function_name == "label2rgb":
            for contiguous_labels in [True, False]:
                if contiguous_labels:
                    index_str = "contiguous"
                else:
                    index_str = None
                B = LabelBench(
                    function_name="label2rgb",
                    shape=shape,
                    dtypes=dtypes,
                    contiguous_labels=contiguous_labels,
                    index_str=index_str,
                    fixed_kwargs=dict(bg_label=0),
                    var_kwargs=dict(kind=["avg", "overlay"]),
                    module_cpu=skimage.color,
                    module_gpu=cucim.skimage.color,
                    run_cpu=run_cpu,
                )
                results = B.run_benchmark(duration=args.duration)
                all_results = pd.concat([all_results, results["full"]])

        elif function_name in [
            "rgb2hed",
            "hed2rgb",
            "lab2lch",
            "lch2lab",
            "xyz2lab",
            "lab2xyz",
        ]:
            B = ColorBench(
                function_name=function_name,
                shape=shape + (3,),
                dtypes=dtypes,
                fixed_kwargs={},
                var_kwargs={},
                module_cpu=skimage.color,
                module_gpu=cucim.skimage.color,
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
        description="Benchmarking cuCIM color conversion functions"
    )
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
