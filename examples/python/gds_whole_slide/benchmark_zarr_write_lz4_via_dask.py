"""
TODO: Currently LZ4-compressed images are a tiny bit larger than uncompressed.
Need to look into why this is!

"""

import math
import os

import cupy as cp
import kvikio.defaults
import numpy as np
from cupyx.profiler import benchmark
from demo_implementation import cupy_to_zarr, get_n_tiles, read_tiled
from lz4_nvcomp import LZ4NVCOMP
from tifffile import TiffFile

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
print(f"\tkvikio task size = {kvikio.defaults.task_size()/1024**2} MB")


n_buffer = 128
max_duration = 8
level = 1

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

# benchmark writing these CuPy outputs to Zarr with various chunk sizes
# Note: nvcomp only supports integer and unsigned dtypes.
# https://github.com/rapidsai/kvikio/blob/b0c6cedf43d1bc240c3ef1b38ebb9d89574a08ee/python/kvikio/nvcomp.py#L12-L21  # noqa: E501

dtypes = ["uint16"]
chunk_shapes = [
    (512, 512, 3),
    (1024, 1024, 3),
    (2048, 2048, 3),
    (4096, 4096, 3),
]
backend = "dask"
compressors = [None, LZ4NVCOMP()]
kvikio.defaults.num_threads_reset(16)
write_time_means = np.zeros(
    ((len(dtypes), len(chunk_shapes), len(compressors), 2)), dtype=float
)
write_time_stds = np.zeros_like(write_time_means)
for i, dtype in enumerate(dtypes):
    dtype = np.dtype(dtype)
    if dtype == np.uint8:
        img = image_gpu
        assert img.dtype == cp.uint8
    elif dtype == np.uint16:
        img = image_gpu.astype(dtype)
    else:
        raise NotImplementedError(
            "LZ4 compression can only be tested for uint8 and uint16"
        )
    for j, chunk_shape in enumerate(chunk_shapes):
        for k, compressor in enumerate(compressors):
            kwargs = dict(
                output_path=f"./image-{dtype}-chunk{chunk_shape[0]}.zarr"
                if compressor is None
                else f"./image-{dtype}-chunk{chunk_shape[0]}-lz4.zarr",
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
                    max_duration=max_duration,
                )
                t = perf_write_float32.gpu_times
                write_time_means[i, j, k, m] = t.mean()
                write_time_stds[i, j, k, m] = t.std()
                print(
                    f"Duration ({cp.dtype(dtype).name} write, {chunk_shape=}, {compressor=}, {gds_enabled=}): "  # noqa: E501
                    f"{t.mean()} s +/- {t.std()} s"
                )

out_name = "write_times_lz4.npz"
# auto-increment filename to avoid overwriting old results
cnt = 1
while os.path.exists(out_name):
    out_name = f"write_times_lz4{cnt}.npz"
    cnt += 1

np.savez(
    out_name, write_time_means=write_time_means, write_time_stds=write_time_stds
)


"""

421M    image-uint8-chunk2048.zarr/
295M    image-uint8-chunk2048-lz4.zarr/

Output on local system:

    kvikio.defaults.compat_mode() = False
    kvikio.defaults.get_num_threads() = 16
    kvikio task size = 4.0 MB
Resolution level 1

Duration (uint8 write, chunk_shape=(512, 512, 3), compressor=None, gds_enabled=False): 0.9078736816406249 s +/- 0.014021576325619447 s
Duration (uint8 write, chunk_shape=(512, 512, 3), compressor=None, gds_enabled=True): 0.7154334309895835 s +/- 0.012911493047009337 s
Duration (uint8 write, chunk_shape=(512, 512, 3), compressor=LZ4NVCOMP, gds_enabled=False): 4.5858623046875 s +/- 0.0 s
Duration (uint8 write, chunk_shape=(512, 512, 3), compressor=LZ4NVCOMP, gds_enabled=True): 4.7737607421875 s +/- 0.0 s
Duration (uint8 write, chunk_shape=(1024, 1024, 3), compressor=None, gds_enabled=False): 0.33250243268694196 s +/- 0.033499521893803265 s
Duration (uint8 write, chunk_shape=(1024, 1024, 3), compressor=None, gds_enabled=True): 0.1679712175641741 s +/- 0.00943557539273183 s
Duration (uint8 write, chunk_shape=(1024, 1024, 3), compressor=LZ4NVCOMP, gds_enabled=False): 1.308169403076172 s +/- 0.009489436283604222 s
Duration (uint8 write, chunk_shape=(1024, 1024, 3), compressor=LZ4NVCOMP, gds_enabled=True): 1.3726735026041668 s +/- 0.0047038287720149695 s
Duration (uint8 write, chunk_shape=(2048, 2048, 3), compressor=None, gds_enabled=False): 0.25327674865722655 s +/- 0.06120098715153658 s
Duration (uint8 write, chunk_shape=(2048, 2048, 3), compressor=None, gds_enabled=True): 0.17847256687709265 s +/- 0.008324480608522849 s
Duration (uint8 write, chunk_shape=(2048, 2048, 3), compressor=LZ4NVCOMP, gds_enabled=False): 0.5187601710728237 s +/- 0.007874048300727562 s
Duration (uint8 write, chunk_shape=(2048, 2048, 3), compressor=LZ4NVCOMP, gds_enabled=True): 0.45903962925502234 s +/- 0.019780733967690416 s
Duration (uint8 write, chunk_shape=(4096, 4096, 3), compressor=None, gds_enabled=False): 0.2989522748674665 s +/- 0.01275530846360682 s
Duration (uint8 write, chunk_shape=(4096, 4096, 3), compressor=None, gds_enabled=True): 0.3445836966378349 s +/- 0.010775083201675684 s
Duration (uint8 write, chunk_shape=(4096, 4096, 3), compressor=LZ4NVCOMP, gds_enabled=False): 0.3000637948172433 s +/- 0.04060511008994888 s
Duration (uint8 write, chunk_shape=(4096, 4096, 3), compressor=LZ4NVCOMP, gds_enabled=True): 0.2680468902587891 s +/- 0.023300006446218775 s

    shape: (13210, 9960, 3)
    stored as 520 tiles of shape (512, 512, 3)
Duration (uint16 write, chunk_shape=(512, 512, 3), compressor=None, gds_enabled=False): 1.0883130187988281 s +/- 0.029807589266119452 s
Duration (uint16 write, chunk_shape=(512, 512, 3), compressor=None, gds_enabled=True): 0.7361140238444012 s +/- 0.02239347882169216 s
Duration (uint16 write, chunk_shape=(512, 512, 3), compressor=LZ4NVCOMP, gds_enabled=False): 4.56396484375 s +/- 0.0 s
Duration (uint16 write, chunk_shape=(512, 512, 3), compressor=LZ4NVCOMP, gds_enabled=True): 4.76813818359375 s +/- 0.0 s
Duration (uint16 write, chunk_shape=(1024, 1024, 3), compressor=None, gds_enabled=False): 0.45565206037248884 s +/- 0.14820532739770373 s
Duration (uint16 write, chunk_shape=(1024, 1024, 3), compressor=None, gds_enabled=True): 0.31185867309570314 s +/- 0.013444509106645106 s
Duration (uint16 write, chunk_shape=(1024, 1024, 3), compressor=LZ4NVCOMP, gds_enabled=False): 1.5407437337239582 s +/- 0.12621846871213385 s
Duration (uint16 write, chunk_shape=(1024, 1024, 3), compressor=LZ4NVCOMP, gds_enabled=True): 1.298984100341797 s +/- 0.025558385966707807 s
Duration (uint16 write, chunk_shape=(2048, 2048, 3), compressor=None, gds_enabled=False): 0.5751916896275112 s +/- 0.1870959869706265 s
Duration (uint16 write, chunk_shape=(2048, 2048, 3), compressor=None, gds_enabled=True): 0.36480389404296876 s +/- 0.008747247240530139 s
Duration (uint16 write, chunk_shape=(2048, 2048, 3), compressor=LZ4NVCOMP, gds_enabled=False): 0.7505391642252603 s +/- 0.19527720554541175 s
Duration (uint16 write, chunk_shape=(2048, 2048, 3), compressor=LZ4NVCOMP, gds_enabled=True): 0.5359197431291852 s +/- 0.08704735860133013 s
Duration (uint16 write, chunk_shape=(4096, 4096, 3), compressor=None, gds_enabled=False): 1.5224884440104167 s +/- 0.16454276798775797 s
Duration (uint16 write, chunk_shape=(4096, 4096, 3), compressor=None, gds_enabled=True): 1.2984992370605468 s +/- 0.028939276482940306 s
Duration (uint16 write, chunk_shape=(4096, 4096, 3), compressor=LZ4NVCOMP, gds_enabled=False): 0.8166022583007813 s +/- 0.19712617258152443 s
Duration (uint16 write, chunk_shape=(4096, 4096, 3), compressor=LZ4NVCOMP, gds_enabled=True): 0.7342889607747396 s +/- 0.025796103217999546 s

"""  # noqa: E501
