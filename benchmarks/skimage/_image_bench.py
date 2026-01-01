# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import math
import re
import subprocess
import time
import types
from collections import abc

import cupy as cp
import cupyx.scipy.ndimage
import numpy as np
import pandas as pd
import scipy.ndimage
import skimage.data

from cucim.time import repeat


def product_dict(**kwargs):
    # https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


class ImageBench:
    def __init__(
        self,
        function_name,
        shape,
        dtypes=[np.float32],
        fixed_kwargs={},
        var_kwargs={},
        index_str=None,  # extra string to append to dataframe index
        module_cpu=scipy.ndimage,
        module_gpu=cupyx.scipy.ndimage,
        function_is_generator=False,
        run_cpu=True,
    ):
        self.shape = shape
        self.function_name = function_name
        self.fixed_kwargs_cpu = self._update_kwargs_arrays(fixed_kwargs, "cpu")
        self.fixed_kwargs_gpu = self._update_kwargs_arrays(fixed_kwargs, "gpu")
        self.var_kwargs = var_kwargs
        self.index_str = index_str
        # self.set_args_kwargs = set_args_kwargs
        if not isinstance(dtypes, abc.Sequence):
            dtypes = [dtypes]
        self.dtypes = [np.dtype(d) for d in dtypes]
        if not function_is_generator:
            self.func_cpu = getattr(module_cpu, function_name)
            self.func_gpu = getattr(module_gpu, function_name)
        else:
            # benchmark by generating all values
            def gen_cpu(*args, **kwargs):
                generator = getattr(module_cpu, function_name)(*args, **kwargs)
                return list(generator)

            def gen_gpu(*args, **kwargs):
                generator = getattr(module_gpu, function_name)(*args, **kwargs)
                return list(generator)

            self.func_cpu = gen_cpu
            self.func_gpu = gen_gpu

        self.module_name_cpu = module_cpu.__name__
        self.module_name_gpu = module_gpu.__name__

        self.run_cpu = run_cpu

    def set_args(self, dtype):
        if np.dtype(dtype).kind in "iu":
            im1 = skimage.data.camera()
            im1 = im1.astype(dtype)
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

    def _update_array(self, array, target="cpu"):
        if target == "gpu" and isinstance(array, np.ndarray):
            array = cp.asarray(array)
        elif target == "cpu" and isinstance(array, cp.ndarray):
            array = cp.asnumpy(array)
        return array

    def _update_kwargs_arrays(self, kwargs, target="cpu"):
        new_dict = {}
        for k, v in kwargs.items():
            new_dict[k] = self._update_array(v, target=target)
        return new_dict

    def _index(self, name, var_kwargs, dtype=None, shape=None):
        index = name
        if var_kwargs:
            index += " ("
        params = []
        for k, v in var_kwargs.items():
            if isinstance(v, types.FunctionType):
                params.append(f"{k}={v.__name__}")
            elif isinstance(v, (np.ndarray, cp.ndarray)):
                params.append(f"{k}=array,shape={v.shape},dtype={v.dtype.name}")
            else:
                params.append(f"{k}={v}")
        if dtype is not None:
            params.append(f", {np.dtype(dtype).name}")
        if shape is not None:
            params.append(f"s={shape}")
        index += ", ".join(params)
        index.replace(",,", ",")
        if var_kwargs:
            index += ") "
        if self.index_str is not None:
            index += ", " + self.index_str
        return index

    def _prep_kwargs_string(self, kwargs):
        params = []
        for k, v in kwargs.items():
            if isinstance(v, types.FunctionType):
                params.append(f"{k}={v.__name__}")
            elif isinstance(v, (np.ndarray, cp.ndarray)):
                params.append(f"{k}=array,shape={v.shape},dtype={v.dtype.name}")
            else:
                params.append(f"{k}={v}")
        return ", ".join(params)

    def get_reps(self, func, args, kwargs, target_duration=5, cpu=True):
        if not cpu:
            # dry run
            func(*args, **kwargs)
        # time 1 repetition
        d = cp.cuda.Device()
        tstart = time.time()
        func(*args, **kwargs)
        d.synchronize()
        dur = time.time() - tstart
        n_repeat = max(1, math.ceil(target_duration / dur))
        if cpu:
            n_warmup = 0
        else:
            n_warmup = max(1, math.ceil(n_repeat / 5))
        reps = dict(n_warmup=n_warmup, n_repeat=n_repeat)
        return reps

    def run_benchmark(self, duration=3, verbose=True):
        df = pd.DataFrame()
        self.df = df
        kw_lists = self.var_kwargs
        pdict = list(product_dict(**kw_lists))
        for dtype in self.dtypes:
            self.set_args(dtype)
            for i, var_kwargs1 in enumerate(pdict):
                # arr_index = indices[i]
                index = self._index(self.function_name, var_kwargs1)

                # transfer any arrays in kwargs to the appropriate device
                var_kwargs_cpu = self._update_kwargs_arrays(var_kwargs1, "cpu")
                var_kwargs_gpu = self._update_kwargs_arrays(var_kwargs1, "gpu")

                # Note: brute_force=True on 'gpu' because False is not
                # implemented
                if "brute_force" in var_kwargs_gpu:
                    var_kwargs_gpu["brute_force"] = True

                kw_gpu = {**self.fixed_kwargs_gpu, **var_kwargs_gpu}
                rep_kwargs_gpu = self.get_reps(
                    self.func_gpu, self.args_gpu, kw_gpu, duration, cpu=False
                )
                print("Number of Repetitions (GPU): ", rep_kwargs_gpu)

                if self.run_cpu is True:
                    kw_cpu = {**self.fixed_kwargs_cpu, **var_kwargs_cpu}
                    rep_kwargs_cpu = self.get_reps(
                        self.func_cpu, self.args_cpu, kw_cpu, duration, cpu=True
                    )
                perf_gpu = repeat(
                    self.func_gpu, self.args_gpu, kw_gpu, **rep_kwargs_gpu
                )

                df.at[index, "GPU: kwargs"] = self._prep_kwargs_string(kw_gpu)
                df.at[index, "shape"] = f"{self.shape}"
                # df.at[index,  "description"] = index
                df.at[index, "function_name"] = self.function_name
                df.at[index, "dtype"] = np.dtype(dtype).name
                df.at[index, "ndim"] = len(self.shape)

                if self.run_cpu is True:
                    perf = repeat(
                        self.func_cpu, self.args_cpu, kw_cpu, **rep_kwargs_cpu
                    )
                    df.at[index, "GPU accel"] = (
                        perf.cpu_times.mean() / perf_gpu.gpu_times.mean()
                    )
                    df.at[index, "CPU: host (mean)"] = perf.cpu_times.mean()
                    df.at[index, "CPU: host (std)"] = perf.cpu_times.std()

                    df.at[index, "CPU: kwargs"] = self._prep_kwargs_string(kw_cpu)
                df.at[index, "GPU: host (mean)"] = perf_gpu.cpu_times.mean()
                df.at[index, "GPU: host (std)"] = perf_gpu.cpu_times.std()
                df.at[index, "GPU: device (mean)"] = perf_gpu.gpu_times.mean()
                df.at[index, "GPU: device (std)"] = perf_gpu.gpu_times.std()
                with cp.cuda.Device() as device:
                    props = cp.cuda.runtime.getDeviceProperties(device.id)
                    gpu_name = props["name"].decode()

                df.at[index, "GPU: DEV Name"] = [gpu_name for i in range(len(df))]
                cmd = "cat /proc/cpuinfo"
                cpuinfo = subprocess.check_output(cmd, shell=True).strip()
                cpu_name = (
                    re.search("\nmodel name.*\n", cpuinfo.decode()).group(0).strip("\n")
                )
                cpu_name = cpu_name.replace("model name\t: ", "")
                df.at[index, "CPU: DEV Name"] = [cpu_name for i in range(len(df))]

                # accelerations[arr_index] = df.at[index,  "GPU accel"]
                if verbose:
                    print(df.loc[index])

        results = {}
        results["full"] = df
        results["var_kwargs_names"] = list(self.var_kwargs.keys())
        results["var_kwargs_values"] = list(self.var_kwargs.values())
        results["function_name"] = self.function_name
        results["module_name_cpu"] = self.module_name_cpu
        results["module_name_gpu"] = self.module_name_gpu
        return results
