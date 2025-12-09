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
import skimage.segmentation
from _image_bench import ImageBench

import cucim.skimage
from cucim.skimage import data, measure


class LabelBench(ImageBench):
    def __init__(
        self,
        function_name,
        shape,
        dtypes=np.float32,
        fixed_kwargs={},
        var_kwargs={},
        index_str=None,
        module_cpu=skimage.measure,
        module_gpu=cucim.skimage.measure,
        run_cpu=True,
    ):
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
        blobs_kwargs = dict(blob_size_fraction=0.05, volume_fraction=0.35, rng=5)
        # binary blobs only creates square outputs
        labels = measure.label(
            data.binary_blobs(max(self.shape), n_dim=ndim, **blobs_kwargs)
        )
        print(f"# labels generated = {labels.max()}")

        # crop to rectangular
        labels = labels[tuple(slice(s) for s in self.shape)]
        return labels.astype(dtype, copy=False)

    def set_args(self, dtype):
        labels_d = self._generate_labels(dtype)
        labels = cp.asnumpy(labels_d)
        self.args_cpu = (labels,)
        self.args_gpu = (labels_d,)


class LabelAndImageBench(LabelBench):
    def set_args(self, dtype):
        labels_d = self._generate_labels(dtype)
        labels = cp.asnumpy(labels_d)
        image_d = cp.random.standard_normal(labels.shape).astype(np.float32)
        image = cp.asnumpy(image_d)
        self.args_cpu = (image, labels)
        self.args_gpu = (image_d, labels_d)


class MorphGeodesicBench(ImageBench):
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

        # need this preprocessing for morphological_geodesic_active_contour
        imaged = cp.array(
            skimage.segmentation.inverse_gaussian_gradient(cp.asnumpy(imaged))
        )

        image = cp.asnumpy(imaged)
        assert imaged.dtype == dtype
        assert imaged.shape == self.shape
        self.args_cpu = (image,)
        self.args_gpu = (imaged,)


class RandomWalkerBench(ImageBench):
    def set_args(self, dtype):
        # Note: dtype only used for merkers array, data is hard-coded as float32

        if np.dtype(dtype).kind not in "iu":
            raise ValueError("random_walker markers require integer dtype")

        n_dim = len(self.shape)
        data = cucim.skimage.img_as_float(
            cucim.skimage.data.binary_blobs(length=max(self.shape), n_dim=n_dim, rng=1)
        )
        data = data[tuple(slice(s) for s in self.shape)]
        sigma = 0.35
        rng = np.random.default_rng(5)
        data += cp.array(rng.normal(loc=0, scale=sigma, size=data.shape))
        data = cucim.skimage.exposure.rescale_intensity(
            data, in_range=(-sigma, 1 + sigma), out_range=(-1, 1)
        )
        data = data.astype(cp.float32)
        data_cpu = cp.asnumpy(data)

        # The range of the binary image spans over (-1, 1).
        # We choose the hottest and the coldest pixels as markers.
        markers = cp.zeros(data.shape, dtype=dtype)
        markers[data < -0.95] = 1
        markers[data > 0.95] = 2
        markers_cpu = cp.asnumpy(markers)
        self.args_cpu = (data_cpu, markers_cpu)
        self.args_gpu = (data, markers)


def main(args):
    pfile = "cucim_segmentation_results.pickle"
    if os.path.exists(pfile):
        with open(pfile, "rb") as f:
            all_results = pickle.load(f)
    else:
        all_results = pd.DataFrame()

    dtypes = [np.dtype(args.dtype)]
    dtypes_label = [np.dtype(args.dtype_label)]
    # image sizes/shapes
    shape = tuple(list(map(int, (args.img_size.split(",")))))
    run_cpu = not args.no_cpu

    for function_name, fixed_kwargs, var_kwargs, allow_color, allow_nd in [
        # _clear_border.py
        (
            "clear_border",
            dict(),
            dict(),
            False,
            True,
        ),
        # _expand_labels.py
        (
            "expand_labels",
            dict(distance=1),
            dict(),
            False,
            True,
        ),
        # _join.py
        (
            "relabel_sequential",
            dict(offset=5),
            dict(),
            False,
            True,
        ),
        # boundaries.py
        (
            "find_boundaries",
            dict(),
            dict(connectivity=[1], mode=["thick", "inner", "outer", "subpixel"]),
            False,
            True,
        ),
        (
            "mark_boundaries",
            dict(),
            dict(),
            False,
            True,
        ),
        (
            "random_walker",
            dict(beta=4, tol=1.0e-4, prob_tol=1.0e-2),
            dict(mode=["cg", "cg_j"]),
            False,
            True,
        ),
        # morphsnakes.py
        ("inverse_gaussian_gradient", dict(), dict(), False, True),
        (
            "morphological_geodesic_active_contour",
            dict(),
            dict(num_iter=[16], init_level_set=["checkerboard", "disk"]),
            False,
            False,
        ),
        (
            "morphological_chan_vese",
            dict(),
            dict(num_iter=[16], init_level_set=["checkerboard", "disk"]),
            False,
            False,
        ),
        (
            "chan_vese",
            dict(),
            # Reduced number of iterations so scikit-image comparison will not
            # take minutes to complete. Empirically, approximately the same
            # acceleration was measured for 10 or 100 iterations.
            dict(max_num_iter=[10], init_level_set=["checkerboard", "disk"]),
            False,
            False,
        ),
        # omit: disk_level_set (simple array generation function)
        # omit: checkerboard_level_set (simple array generation function)
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

        if function_name in [
            "clear_border",
            "expand_labels",
            "relabel_sequential",
            "find_boundaries",
            "mark_boundaries",
            "random_walker",
        ]:
            if function_name == "random_walker":
                fixed_kwargs["channel_axis"] = -1 if shape[-1] == 3 else None

            if function_name == "mark_boundaries":
                bench_func = LabelAndImageBench
            elif function_name == "random_walker":
                bench_func = RandomWalkerBench
            else:
                bench_func = LabelBench

            B = bench_func(
                function_name=function_name,
                shape=shape,
                dtypes=dtypes_label,
                fixed_kwargs=fixed_kwargs,
                var_kwargs=var_kwargs,
                module_cpu=skimage.segmentation,
                module_gpu=cucim.skimage.segmentation,
                run_cpu=run_cpu,
            )
            results = B.run_benchmark(duration=args.duration)
            all_results = pd.concat([all_results, results["full"]])

        elif function_name in [
            "inverse_gaussian_gradient",
            "morphological_geodesic_active_contour",
            "morphological_chan_vese",
            "chan_vese",
        ]:
            if function_name == "morphological_geodesic_active_contour":
                bench_class = MorphGeodesicBench
            else:
                bench_class = ImageBench

            B = bench_class(
                function_name=function_name,
                shape=shape,
                dtypes=dtypes,
                fixed_kwargs=fixed_kwargs,
                var_kwargs=var_kwargs,
                module_cpu=skimage.segmentation,
                module_gpu=cucim.skimage.segmentation,
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
        description="Benchmarking cuCIM segmentation functions"
    )
    func_name_choices = [
        "clear_border",
        "expand_labels",
        "relabel_sequential",
        "find_boundaries",
        "mark_boundaries",
        "random_walker",
        "inverse_gaussian_gradient",
        "morphological_geodesic_active_contour",
        "morphological_chan_vese",
        "chan_vese",
    ]
    label_dtype_choices = [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
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
            "Size of input image (omit color channel, it will be appended as needed)"
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
        "--dtype_label",
        type=str,
        help="Dtype of input image",
        choices=label_dtype_choices,
        required=False,
        default="uint8",
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
