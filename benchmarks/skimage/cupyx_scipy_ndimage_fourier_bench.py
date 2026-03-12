# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import cupy
import cupy as cp
import numpy as np
import pandas as pd
from _image_bench import ImageBench

cfile = "fourier_results.csv"
if os.path.exists(cfile):
    all_results = pd.read_csv(cfile, index_col=0)
else:
    all_results = pd.DataFrame()

dtypes = [np.float32]

for shape in [(512, 512), (3840, 2160), (192, 192, 192)]:
    # shape = (200, 200, 200)
    ndim = len(shape)

    class FourierBench(ImageBench):
        def set_args(self, dtype):
            cplx_dt = np.promote_types(dtype, np.complex64)
            imaged = cupy.testing.shaped_random(self.shape, xp=cp, dtype=cplx_dt)
            image = cp.asnumpy(imaged)
            self.args_cpu = (image,)
            self.args_gpu = (imaged,)

    for fname, fixed_kwargs, var_kwargs in [
        ("fourier_gaussian", dict(sigma=5), {}),
        ("fourier_uniform", dict(size=16), {}),
        ("fourier_shift", dict(shift=5), {}),
        ("fourier_ellipsoid", dict(size=15.0), {}),
    ]:
        B = FourierBench(
            function_name=fname,
            shape=shape,
            dtypes=dtypes,
            fixed_kwargs=fixed_kwargs,
            var_kwargs=var_kwargs,
        )
        results = B.run_benchmark(duration=1)
        all_results = pd.concat([all_results, results["full"]])


fbase = os.path.splitext(cfile)[0]
all_results.to_csv(cfile, index=True)
with open(fbase + ".md", "w") as f:
    f.write(all_results.to_markdown())
