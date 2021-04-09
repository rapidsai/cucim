import math
import os
import pickle

import cupy
import cupy as cp
import numpy as np
import pandas as pd

from _image_bench import ImageBench


class InterpolationBench(ImageBench):
    def set_args(self, dtype):
        imaged = cupy.testing.shaped_random(self.shape, xp=cp, dtype=dtype)
        image = cp.asnumpy(imaged)
        self.args_cpu = (image,)
        self.args_gpu = (imaged,)


class MapCoordinatesBench(ImageBench):
    def set_args(self, dtype):

        imaged = cupy.testing.shaped_random(self.shape, xp=cp, dtype=dtype)
        image = cp.asnumpy(imaged)

        rstate = cp.random.RandomState(5)
        ndim = len(self.shape)
        coordsd = cp.indices(self.shape) + 0.1 * rstate.standard_normal((ndim,) + self.shape)
        coords = cupy.asnumpy(coordsd)

        self.args_cpu = (image, coords)
        self.args_gpu = (imaged, coordsd)


pfile = "interp_results.pickle"
if os.path.exists(pfile):
    with open(pfile, "rb") as f:
        all_results = pickle.load(f)
else:
    all_results = pd.DataFrame()

orders = [0, 1, 3, 5]  # 2, 3, 4, 5]
modes = ["constant", "reflect"]
prefilter = True
dtypes = [np.float32]
# (4608, 3456) = 16MP as in IPOL paper
for shape in [(512, 512), (3840, 2160), (4608, 3456), (192, 192, 192)]:
    ndim = len(shape)

    B = MapCoordinatesBench(
        function_name="map_coordinates",
        shape=shape,
        dtypes=dtypes,
        fixed_kwargs=dict(output=None, prefilter=prefilter),
        var_kwargs=dict(mode=modes, order=orders),
    )
    results = B.run_benchmark(duration=1)
    all_results = all_results.append(results["full"])

    for fname, fixed_kwargs, var_kwargs in [
        (
            "affine_transform1",  # see special case below
            dict(output=None, output_shape=None, prefilter=prefilter),
            dict(mode=modes, order=orders),
        ),
        (
            "affine_transform2",  # see special case below
            dict(output=None, output_shape=None, prefilter=prefilter),
            dict(mode=modes, order=orders),
        ),
        ("zoom", dict(output=None, zoom=1.1, prefilter=prefilter), dict(mode=modes, order=orders)),
        (
            "shift",
            dict(output=None, shift=1.5, prefilter=prefilter),
            dict(mode=modes, order=orders),
        ),
        (
            "rotate",
            dict(output=None, reshape=True, axes=(0, 1), angle=30, prefilter=prefilter),
            dict(mode=modes, order=orders),
        ),
        (
            "spline_filter",
            dict(output=np.float32),
            dict(
                mode=[
                    "mirror",
                ],
                order=[2, 3, 4, 5],
            ),
        ),
        (
            "spline_filter1d",
            dict(output=np.float32),
            dict(
                mode=[
                    "mirror",
                ],
                order=[2, 3, 4, 5],
                axis=[0, -1],
            ),
        ),
    ]:

        if fname == "affine_transform1":
            # affine_transform case 1: the general affine matrix code path
            fname = fname[:-1]
            ndim = len(shape)
            angle = np.deg2rad(30)
            cos = math.cos(angle)
            sin = math.cos(angle)
            matrix = np.identity(ndim)
            axes = (0, 1)
            matrix[axes[0], axes[0]] = cos
            matrix[axes[0], axes[1]] = sin
            matrix[axes[1], axes[0]] = -sin
            matrix[axes[1], axes[1]] = cos
            offset = np.full((ndim,), 1.5, dtype=float)
            fixed_kwargs["matrix"] = matrix
            fixed_kwargs["offset"] = offset
        elif fname == "affine_transform2":
            # affine_transform case 2: exercises the zoom + shift code path
            fname = fname[:-1]
            if len(shape) == 2:
                matrix = np.asarray([0.5, 2.0])
                offset = np.asarray([20.0, -25.0])
            elif len(shape) == 3:
                matrix = np.asarray([0.5, 2.0, 0.6])
                offset = np.asarray([0.0, -5.0, 15])
            fixed_kwargs["matrix"] = matrix
            fixed_kwargs["offset"] = offset

        B = InterpolationBench(
            function_name=fname,
            shape=shape,
            dtypes=dtypes,
            fixed_kwargs=fixed_kwargs,
            var_kwargs=var_kwargs,
        )
        results = B.run_benchmark(duration=1)
        all_results = all_results.append(results["full"])

fbase = os.path.splitext(pfile)[0]
all_results.to_csv(fbase + ".csv")
all_results.to_pickle(pfile)
with open(fbase + ".md", "wt") as f:
    f.write(all_results.to_markdown())
