import math
import os
import pickle

import cucim.skimage
import cucim.skimage.data
import cucim.skimage.exposure
import cucim.skimage.segmentation
import cupy as cp
import numpy as np
import pandas as pd
import skimage
import skimage.segmentation

from _image_bench import ImageBench


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
        )

    def set_args(self, dtype):
        a = np.array(
            [
                [0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 4, 0],
                [2, 2, 0, 0, 3, 0, 4, 4],
                [0, 0, 0, 0, 0, 5, 0, 0],
            ],
            dtype=dtype,
        )
        tiling = tuple(s // a_s for s, a_s in zip(shape, a.shape))
        if self.contiguous_labels:
            labels = np.kron(a, np.ones(tiling, dtype=a.dtype))
        else:
            labels = np.tile(a, tiling)
        labels_d = cp.asarray(labels)
        self.args_cpu = (labels,)
        self.args_gpu = (labels_d,)


class LabelAndImageBench(LabelBench):

    def set_args(self, dtype):
        a = np.array(
            [
                [0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 4, 0],
                [2, 2, 0, 0, 3, 0, 4, 4],
                [0, 0, 0, 0, 0, 5, 0, 0],
            ],
            dtype=dtype,
        )
        tiling = tuple(s // a_s for s, a_s in zip(shape, a.shape))
        if self.contiguous_labels:
            labels = np.kron(a, np.ones(tiling, dtype=a.dtype))
        else:
            labels = np.tile(a, tiling)
        labels_d = cp.asarray(labels)
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

        if np.dtype(dtype).kind not in 'iu':
            raise ValueError("random_walker markers require integer dtype")

        n_dim = len(self.shape)
        data = cucim.skimage.img_as_float(
            cucim.skimage.data.binary_blobs(
                length=max(self.shape), n_dim=n_dim, seed=1
            )
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


pfile = "cucim_segmentation_results.pickle"
if os.path.exists(pfile):
    with open(pfile, "rb") as f:
        all_results = pickle.load(f)
else:
    all_results = pd.DataFrame()

dtypes = [np.int32]
for function_name, fixed_kwargs, var_kwargs, allow_color, allow_nd in [
    # _clear_border.py
    (
        "clear_border",
        dict(),
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
        dict(beta=4, tol=1.e-4, prob_tol=1.e-2),
        dict(mode=['cg', 'cg_j']),
        False,
        True,
    ),
]:

    for shape in [
        (64, 64), (512, 512), (1980, 1080), (1980, 1080, 3), (128, 128, 128)]:

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

        if function_name == 'random_walker':
            fixed_kwargs['channel_axis'] = -1 if shape[-1] == 3 else None

        if function_name == 'mark_boundaries':
            bench_func = LabelAndImageBench
        elif function_name == 'random_walker':
            bench_func = RandomWalkerBench
        else:
            bench_func = LabelBench

        B = bench_func(
            function_name=function_name,
            shape=shape,
            dtypes=dtypes,
            fixed_kwargs=fixed_kwargs,
            var_kwargs=var_kwargs,
            module_cpu=skimage.segmentation,
            module_gpu=cucim.skimage.segmentation,
        )
        results = B.run_benchmark(duration=1)
        all_results = pd.concat([all_results, results["full"]])


dtypes = [np.float32]
# function_name, fixed_kwargs, var_kwargs, allow_color, allow_nd = ('unsupervised_wiener', dict(), dict(), False, True)
for function_name, fixed_kwargs, var_kwargs, allow_color, allow_nd in [
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
    # omit: disk_level_set (simple array generation function)
    # omit: checkerboard_level_set (simple array generation function)

]:

    for shape in [(512, 512), ]: # (3840, 2160), (3840, 2160, 3), (192, 192, 192)]:

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
