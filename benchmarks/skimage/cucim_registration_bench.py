import math
import os
import pickle

import cucim.skimage
import cucim.skimage.registration
import cupy as cp
import numpy as np
import pandas as pd
import skimage
import skimage.registration

from _image_bench import ImageBench


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


pfile = "cucim_registration_results.pickle"
if os.path.exists(pfile):
    with open(pfile, "rb") as f:
        all_results = pickle.load(f)
else:
    all_results = pd.DataFrame()
dtypes = [np.float32]


for function_name, fixed_kwargs, var_kwargs, allow_color, allow_nd in [
    # _phase_cross_correlation.py
    ("phase_cross_correlation", dict(), dict(), False, True),
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
            )
            results = B.run_benchmark(duration=1)
            all_results = all_results.append(results["full"])


for function_name, fixed_kwargs, var_kwargs, allow_color, allow_nd in [
    # _phase_cross_correlation.py
    ("optical_flow_tvl1", dict(), dict(num_iter=[10], num_warp=[5]), False, True),
    (
        "optical_flow_ilk",
        dict(),
        dict(radius=[3, 7], num_warp=[10], gaussian=[False, True], prefilter=[False, True]),
        False,
        True,
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

        B = RegistrationBench(
            function_name=function_name,
            shape=shape,
            dtypes=dtypes,
            fixed_kwargs=fixed_kwargs,
            var_kwargs=var_kwargs,
            module_cpu=skimage.registration,
            module_gpu=cucim.skimage.registration,
        )
        results = B.run_benchmark(duration=1)
        all_results = all_results.append(results["full"])


fbase = os.path.splitext(pfile)[0]
all_results.to_csv(fbase + ".csv")
all_results.to_pickle(pfile)
with open(fbase + ".md", "wt") as f:
    f.write(all_results.to_markdown())
