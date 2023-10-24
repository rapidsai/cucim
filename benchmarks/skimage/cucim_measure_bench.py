import argparse
import math
import os
import pickle

import cupy as cp
import numpy as np
import pandas as pd
import skimage
import skimage.measure
from _image_bench import ImageBench
from cucim_metrics_bench import MetricsBench

import cucim.skimage
import cucim.skimage.measure


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
        module_cpu=skimage.measure,
        module_gpu=cucim.skimage.measure,
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
            ]
        )
        tiling = tuple(s // a_s for s, a_s in zip(self.shape, a.shape))
        if self.contiguous_labels:
            image = np.kron(a, np.ones(tiling, dtype=a.dtype))
        else:
            image = np.tile(a, tiling)
        imaged = cp.asarray(image)
        self.args_cpu = (image,)
        self.args_gpu = (imaged,)


class RegionpropsBench(ImageBench):
    def __init__(
        self,
        function_name,
        shape,
        contiguous_labels=True,
        dtypes=np.float32,
        fixed_kwargs={},
        var_kwargs={},
        index_str=None,
        module_cpu=skimage.measure,
        module_gpu=cucim.skimage.measure,
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
            ]
        )
        tiling = tuple(s // a_s for s, a_s in zip(self.shape, a.shape))
        if self.contiguous_labels:
            image = np.kron(a, np.ones(tiling, dtype=a.dtype))
        else:
            image = np.tile(a, tiling)
        imaged = cp.asarray(image)
        label_dev = cucim.skimage.measure.label(imaged).astype(int)
        label = cp.asnumpy(label_dev)

        self.args_cpu = (label, image)
        self.args_gpu = (label_dev, imaged)


class FiltersBench(ImageBench):
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
        imaged = cp.asarray(image)
        assert imaged.dtype == dtype
        assert imaged.shape == self.shape
        self.args_cpu = (image,)
        self.args_gpu = (imaged,)


class BinaryImagePairBench(ImageBench):
    def set_args(self, dtype):
        rng = cp.random.default_rng(seed=123)
        img0d = rng.integers(0, 2, self.shape, dtype=cp.uint8).view(bool)
        img1d = rng.integers(0, 2, self.shape, dtype=cp.uint8).view(bool)
        img0 = cp.asnumpy(img0d)
        img1 = cp.asnumpy(img1d)

        self.args_cpu = (img0, img1)
        self.args_gpu = (img0d, img1d)


class MandersColocBench(ImageBench):
    def set_args(self, dtype):
        # image
        imaged = cp.testing.shaped_arange(self.shape, dtype=dtype)
        imaged = imaged / imaged.max()

        # binary mask
        rng = cp.random.default_rng(seed=123)
        maskd = rng.integers(0, 2, self.shape, dtype=cp.uint8).view(bool)

        self.args_cpu = (cp.asnumpy(imaged), cp.asnumpy(maskd))
        self.args_gpu = (imaged, maskd)


def main(args):

    pfile = "cucim_measure_results.pickle"
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
        # _gaussian.py
        (
            "label",
            dict(return_num=False, background=0),
            dict(connectivity=[1, 2]),
            False,
            True,
        ),
        # regionprops.py
        ("regionprops", dict(), dict(), False, True),
        # _moments.py
        ("moments", dict(), dict(order=[1, 2, 3, 4]), False, False),
        ("moments_central", dict(), dict(order=[1, 2, 3]), False, True),
        # omitted from benchmarks (only tiny arrays): moments_normalized, moments_hu
        ("centroid", dict(), dict(), False, True),
        ("inertia_tensor", dict(), dict(), False, True),
        ("inertia_tensor_eigvals", dict(), dict(), False, True),
        # _polygon.py
        # TODO: approximate_polygon, subdivide_polygon
        # block.py
        (
            "block_reduce",
            dict(),
            dict(
                func=[
                    cp.sum,
                ]
            ),
            True,
            True,
        ),  # variable block_size configured below
        # entropy.py
        ("shannon_entropy", dict(base=2), dict(), True, True),
        # profile.py
        (
            "profile_line",
            dict(src=(5, 7)),
            dict(reduce_func=[cp.mean], linewidth=[1, 2, 4], order=[1, 3]),
            True,
            False,
        ),  # variable block_size configured below

        # binary image overlap measures
        ("intersection_coeff", dict(mask=None), dict(), False, True),
        ("manders_coloc_coeff", dict(mask=None), dict(), False, True),
        ("manders_overlap_coeff", dict(mask=None), dict(), False, True),
        ("pearson_corr_coeff", dict(mask=None), dict(), False, True),
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

        if function_name in ['label', 'regionprops']:

            Tester = LabelBench if function_name == "label" else RegionpropsBench

            for contiguous_labels in [True, False]:
                if contiguous_labels:
                    index_str = f"contiguous"
                else:
                    index_str = None
                B = Tester(
                    function_name=function_name,
                    shape=shape,
                    dtypes=dtypes,
                    contiguous_labels=contiguous_labels,
                    index_str=index_str,
                    fixed_kwargs=fixed_kwargs,
                    var_kwargs=var_kwargs,
                    module_cpu=skimage.measure,
                    module_gpu=cucim.skimage.measure,
                    run_cpu=run_cpu,
                )
        elif function_name in ['intersection_coeff', 'manders_coloc_coeff',
                               'manders_overlap_coeff', 'pearson_corr_coeff']:

            if function_name in ["pearson_corr_coeff", "manders_overlap_coeff"]:
                # arguments are two images of matching dtype
                Tester = MetricsBench
            elif function_name == "manders_coloc_coeff":
                # arguments are one image + binary mask
                Tester = MandersColocBench
            else:
                # arguments are two binary images
                Tester = BinaryImagePairBench

            B = Tester(
                function_name=function_name,
                shape=shape,
                dtypes=dtypes,
                fixed_kwargs=fixed_kwargs,
                var_kwargs=var_kwargs,
                module_cpu=skimage.measure,
                module_gpu=cucim.skimage.measure,
                run_cpu=run_cpu,
            )
        else:


            if function_name == "gabor" and np.prod(shape) > 1000000:
                # avoid cases that are too slow on the CPU
                var_kwargs["frequency"] = [f for f in var_kwargs["frequency"] if f >= 0.1]

            if function_name == "block_reduce":
                ndim = len(shape)
                if shape[-1] == 3:
                    block_sizes = [(b,) * (ndim - 1) + (3,) for b in (16, 32, 64)]
                else:
                    block_sizes = [(b,) * ndim for b in (16, 32, 64)]
                var_kwargs["block_size"] = block_sizes

            if function_name == "profile_line":
                fixed_kwargs["dst"] = (shape[0] - 32, shape[1] + 9)

            if function_name == "median":
                footprints = []
                ndim = len(shape)
                footprint_sizes = [3, 5, 7, 9] if ndim == 2 else [3, 5, 7]
                for footprint_size in [3, 5, 7, 9]:
                    footprints.append(
                        np.ones((footprint_sizes,) * ndim, dtype=bool)
                    )
                var_kwargs["footprint"] = footprints

            if function_name in ["gaussian", "unsharp_mask"]:
                fixed_kwargs["channel_axis"] = -1 if shape[-1] == 3 else None

            B = FiltersBench(
                function_name=function_name,
                shape=shape,
                dtypes=dtypes,
                fixed_kwargs=fixed_kwargs,
                var_kwargs=var_kwargs,
                module_cpu=skimage.measure,
                module_gpu=cucim.skimage.measure,
                run_cpu=run_cpu,
            )
        results = B.run_benchmark(duration=args.duration)
        all_results = pd.concat([all_results, results["full"]])

    fbase = os.path.splitext(pfile)[0]
    all_results.to_csv(fbase + ".csv")
    all_results.to_pickle(pfile)
    with open(fbase + ".md", "wt") as f:
        f.write(all_results.to_markdown())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmarking cuCIM measure functions')
    func_name_choices = ['label', 'regionprops', 'moments', 'moments_central', 'centroid', 'inertia_tensor', 'inertia_tensor_eigvals', 'block_reduce', 'shannon_entropy', 'profile_line', 'intersection_coeff', 'manders_coloc_coeff', 'manders_overlap_coeff', 'pearson_corr_coeff']
    dtype_choices = ['bool', 'float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64']
    parser.add_argument('-i','--img_size', type=str, help='Size of input image', required=True)
    parser.add_argument('-d','--dtype', type=str, help='Dtype of input image', choices=dtype_choices, required=True)
    parser.add_argument('-f','--func_name', type=str, help='function to benchmark', choices=func_name_choices, required=True)
    parser.add_argument('-t','--duration', type=int, help='time to run benchmark', required=True)
    parser.add_argument('--no_cpu', action='store_true', help='disable cpu measurements', default=False)

    args = parser.parse_args()
    main(args)
