# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import os

import cupy as cp
import kvikio.defaults
import numpy as np
from cupyx.profiler import benchmark
from demo_implementation import cupy_to_zarr, get_n_tiles, read_tiled
from tifffile import TiffFile

from cucim.core.operations.color import image_to_absorbance

data_dir = os.environ.get("WHOLE_SLIDE_DATA_DIR", os.path.dirname("__file__"))
fname = os.path.join(data_dir, "resize.tiff")
if not os.path.exists(fname):
    raise RuntimeError(f"Could not find data file: {fname}")

# make sure we are not in compatibility mode to ensure cuFile is being used
# (when compat_mode() is True, POSIX will be used instead of libcufile.so)
kvikio.defaults.compat_mode_reset(False)
assert not kvikio.defaults.compat_mode()

# set the number of threads to use
kvikio.defaults.num_threads_reset(16)

print(f"\t{kvikio.defaults.compat_mode() = }")
print(f"\t{kvikio.defaults.get_num_threads() = }")
print(f"\tkvikio task size = {kvikio.defaults.task_size() / 1024**2} MB")


n_buffer = 128
max_duration = 4
level = 0
compressor = None

with TiffFile(fname) as tif:
    page = tif.pages[level]
    page_shape = page.shape
    tile_shape = (page.tilelength, page.tilewidth, page.samplesperpixel)
    total_tiles = math.prod(get_n_tiles(page))
print(f"Resolution level {level}\n")
print(f"\tshape: {page_shape}")
print(f"\tstored as {total_tiles} tiles of shape {tile_shape}")


# read the uint8 TIFF
kwargs = dict(levels=[level], backend="kvikio-pread", n_buffer=n_buffer)
image_gpu = read_tiled(fname, **kwargs)[0]

# read the uint8 TIFF applying tile-wise processing to give a float32 array
kwargs = dict(
    levels=[level],
    backend="kvikio-pread",
    n_buffer=n_buffer,
    tile_func=image_to_absorbance,
    out_dtype=cp.float32,
)
preprocessed_gpu = read_tiled(fname, **kwargs)[0]

# benchmark writing these CuPy outputs to Zarr with various chunk sizes
dtypes = ["uint8", "float32"]
chunk_shapes = [(512, 512, 3), (1024, 1024, 3), (2048, 2048, 3)]
backends = ["dask", "kvikio-raw_write", "kvikio-pwrite"]
kvikio.defaults.num_threads_reset(16)
write_time_means = np.zeros(
    ((len(dtypes), len(chunk_shapes), len(backends), 2)), dtype=float
)
write_time_stds = np.zeros_like(write_time_means)
for i, dtype in enumerate(dtypes):
    if dtype == "uint8":
        img = image_gpu
        assert img.dtype == cp.uint8
    elif dtype == "float32":
        img = preprocessed_gpu
        assert img.dtype == cp.float32
    else:
        raise NotImplementedError("only testing for uint8 and float32")
    for j, chunk_shape in enumerate(chunk_shapes):
        for k, backend in enumerate(backends):
            kwargs = dict(
                output_path=f"./image-{dtype}.zarr",
                chunk_shape=chunk_shape,
                zarr_kwargs=dict(overwrite=True, compressor=compressor),
                n_buffer=64,
                backend=backend,
            )
            for m, gds_enabled in enumerate([False, True]):
                kvikio.defaults.compat_mode_reset(not gds_enabled)
                perf_write_float32 = benchmark(
                    cupy_to_zarr,
                    (img,),
                    kwargs=kwargs,
                    n_warmup=1,
                    n_repeat=7,
                    max_duration=15,
                )
                t = perf_write_float32.gpu_times
                write_time_means[i, j, k, m] = t.mean()
                write_time_stds[i, j, k, m] = t.std()
                print(
                    f"Duration ({cp.dtype(dtype).name} write, {chunk_shape=}, {backend=}, {gds_enabled=}): "  # noqa: E501
                    f"{t.mean()} s +/- {t.std()} s"
                )
out_name = "write_times.npz"
# auto-increment filename to avoid overwriting old results
cnt = 1
while os.path.exists(out_name):
    out_name = f"write_times{cnt}.npz"
    cnt += 1

np.savez(out_name, write_time_means=write_time_means, write_time_stds=write_time_stds)


"""
Output on local system:


"""
