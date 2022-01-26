import math
import os
import pickle

import cucim.skimage
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
            image = np.kron(a, np.ones(tiling, dtype=a.dtype))
        else:
            image = np.tile(a, tiling)
        imaged = cp.asarray(image)
        self.args_cpu = (image,)
        self.args_gpu = (imaged,)


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
        imaged = skimage.segmentation.inverse_gaussian_gradient(imaged)

        image = cp.asnumpy(imaged)
        assert imaged.dtype == dtype
        assert imaged.shape == self.shape
        self.args_cpu = (image,)
        self.args_gpu = (imaged,)


pfile = "cucim_segmentation_results.pickle"
if os.path.exists(pfile):
    with open(pfile, "rb") as f:
        all_results = pickle.load(f)
else:
    all_results = pd.DataFrame()
dtypes = [np.int32]


for function_name, fixed_kwargs, var_kwargs, allow_color, allow_nd in [
    # _denoise.py
    (
        "find_boundaries",
        dict(),
        dict(connectivity=[1], mode=["thick", "inner", "outer", "subpixel"]),
        False,
        True,
    ),
]:

    for shape in [
        (64, 64),
    ]:  # (512, 512), (1980, 1080), (1980, 1080, 3), (128, 128, 128)]:

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

        B = LabelBench(
            function_name=function_name,
            shape=shape,
            dtypes=dtypes,
            fixed_kwargs=fixed_kwargs,
            var_kwargs=var_kwargs,
            module_cpu=skimage.segmentation,
            module_gpu=cucim.skimage.segmentation,
        )
        results = B.run_benchmark(duration=1)
        all_results = all_results.append(results["full"])


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
]:

    for shape in [(512, 512), (3840, 2160), (3840, 2160, 3), (192, 192, 192)]:

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

        B = ImageBench(
            function_name=function_name,
            shape=shape,
            dtypes=dtypes,
            fixed_kwargs=fixed_kwargs,
            var_kwargs=var_kwargs,
            module_cpu=skimage.segmentation,
            module_gpu=cucim.skimage.segmentation,
        )
        results = B.run_benchmark(duration=1)
        all_results = all_results.append(results["full"])


fbase = os.path.splitext(pfile)[0]
all_results.to_csv(fbase + ".csv")
all_results.to_pickle(pfile)
with open(fbase + ".md", "wt") as f:
    f.write(all_results.to_markdown())
