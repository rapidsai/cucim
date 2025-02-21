import argparse
import math
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
from cucim.skimage import data, measure

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
    "deltaE_cie76",
    "deltaE_ciede94",
    "deltaE_ciede2000",
    "deltaE_cmc",
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


class DeltaEBench(ImageBench):
    def set_args(self, dtype):
        from skimage import color, data

        # create synthetic lab image pair
        rgb1 = data.astronaut()
        lab1 = color.rgb2lab(rgb1)
        lab2 = color.rgb2lab(np.roll(rgb1, (1, 1), axis=(0, 1)))

        # change to desired dtype
        lab1 = lab1.astype(dtype, copy=False)
        lab2 = lab2.astype(dtype, copy=False)

        # tile then crop as needed to get the expected size
        n_tile0 = math.ceil(self.shape[0] / lab1.shape[0])
        n_tile1 = math.ceil(self.shape[1] / lab1.shape[1])
        lab1 = np.tile(lab1, (n_tile0, n_tile1, 1))
        lab1 = lab1[: self.shape[0], : self.shape[1], :]
        lab2 = np.tile(lab2, (n_tile0, n_tile1, 1))
        lab2 = lab2[: self.shape[0], : self.shape[1], :]

        print(f"{lab1.shape=}")
        self.args_cpu = (lab1, lab2)
        self.args_gpu = (cp.asarray(lab1), cp.asarray(lab2))


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
        image_none=False,
        dtypes=np.float32,
        fixed_kwargs={},
        var_kwargs={},
        index_str=None,
        module_cpu=scipy.ndimage,
        module_gpu=cupyx.scipy.ndimage,
        run_cpu=True,
    ):
        self.image_none = image_none
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

    def _generate_labels(self, dtype):
        ndim = len(self.shape)
        blobs_kwargs = dict(
            blob_size_fraction=0.05, volume_fraction=0.35, rng=5
        )
        # binary blobs only creates square outputs
        labels = measure.label(
            data.binary_blobs(max(self.shape), n_dim=ndim, **blobs_kwargs)
        )
        print(f"# labels generated = {labels.max()}")

        # crop to rectangular
        labels = labels[tuple(slice(s) for s in self.shape)]
        return labels.astype(dtype, copy=False)

    def set_args(self, dtype):
        labels_d = self._generate_labels(dtype=np.int32)
        labels = cp.asnumpy(labels_d)
        if self.image_none:
            self.args_cpu = (labels,)
            self.args_gpu = (labels_d,)
        else:
            imaged = cupy.testing.shaped_random(
                labels.shape + (3,), xp=cp, dtype=dtype, scale=1.0
            )
            image = cp.asnumpy(imaged)
            self.args_cpu = (
                labels,
                image,
            )
            self.args_gpu = (
                labels_d,
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

        elif function_name.startswith("deltaE"):
            # only run these functions for floating point data types
            float_dtypes = [t for t in dtypes if np.dtype(t).kind == "f"]

            B = DeltaEBench(
                function_name=function_name,
                shape=shape + (3,),
                dtypes=float_dtypes,
                fixed_kwargs={},
                var_kwargs={},
                # index_str=f"{fromspace.lower()}2{tospace.lower()}",
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
            for image_none in [True, False]:
                if image_none:
                    # "avg" case cannot be used without an image
                    var_kwargs = dict(kind=["overlay"])
                    index_str = "without_image"
                else:
                    var_kwargs = dict(kind=["avg", "overlay"])
                    index_str = "with_image"

                B = LabelBench(
                    function_name="label2rgb",
                    shape=shape,
                    dtypes=dtypes,
                    image_none=image_none,
                    index_str=index_str,
                    fixed_kwargs=dict(bg_label=0),
                    var_kwargs=var_kwargs,
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

        with open(fbase + ".md", "w") as f:
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
