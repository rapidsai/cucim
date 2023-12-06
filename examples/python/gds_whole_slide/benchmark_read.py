import math
import os

import kvikio
import kvikio.defaults
import numpy as np
from cupyx.profiler import benchmark
from demo_implementation import (
    get_n_tiles,
    get_tile_buffers,
    read_openslide,
    read_tifffile,
    read_tiled,
)
from tifffile import TiffFile

data_dir = os.environ.get("WHOLE_SLIDE_DATA_DIR", os.path.dirname("__file__"))
fname = os.path.join(data_dir, "resize.tiff")
if not os.path.exists(fname):
    raise RuntimeError(f"Could not find data file: {fname}")

level = 0
max_duration = 8

with TiffFile(fname) as tif:
    page = tif.pages[level]
    page_shape = page.shape
    tile_shape = (page.tilelength, page.tilewidth, page.samplesperpixel)
    total_tiles = math.prod(get_n_tiles(page))
print(f"Resolution level {level}\n")
print(f"\tshape: {page_shape}")
print(f"\tstored as {total_tiles} tiles of shape {tile_shape}")

# make sure we are not in compatibility mode to ensure cuFile is being used
# (when compat_mode() is True, POSIX will be used instead of libcufile.so)
kvikio.defaults.compat_mode_reset(False)
assert not kvikio.defaults.compat_mode()

# set the number of threads to use
kvikio.defaults.num_threads_reset(16)


print(f"\t{kvikio.defaults.compat_mode() = }")
print(f"\t{kvikio.defaults.get_num_threads() = }")

preregister_buffers = False
if preregister_buffers:
    tile_buffers = get_tile_buffers(fname, level, n_buffer=256)
    for b in tile_buffers:
        kvikio.memory_register(b)
else:
    tile_buffers = None

# print(f"\tkvikio task size = {kvikio.defaults.task_size()/1024**2} MB")

times = []
labels = []
perf_openslide = benchmark(
    read_openslide,
    (fname, level),
    n_warmup=0,
    n_repeat=100,
    max_duration=max_duration,
)
times.append(perf_openslide.gpu_times.mean())
labels.append("openslide")
print(f"duration ({labels[-1]}) = {times[-1]}")

perf_tifffile = benchmark(
    read_tifffile,
    (fname, level),
    n_warmup=0,
    n_repeat=100,
    max_duration=max_duration,
)
times.append(perf_tifffile.gpu_times.mean())
labels.append("tifffile")
print(f"duration ({labels[-1]}) = {times[-1]}")

for gds_enabled in [False, True]:
    kvikio.defaults.compat_mode_reset(not gds_enabled)
    assert kvikio.defaults.compat_mode() == (not gds_enabled)

    p = benchmark(
        read_tiled,
        (fname, [level]),
        kwargs=dict(backend="kvikio-raw_read", tile_buffers=tile_buffers),
        n_warmup=1,
        n_repeat=100,
        max_duration=max_duration,
    )
    if gds_enabled:
        perf_kvikio_raw = p
    else:
        perf_kvikio_raw_nogds = p
    times.append(p.gpu_times.mean())
    labels.append(f"kvikio-read_raw ({gds_enabled=})")
    print(f"duration ({labels[-1]}) = {times[-1]}")

    for mm in [8, 16, 32, 64]:
        kvikio.defaults.task_size_reset(4096 * mm)

        p = benchmark(
            read_tiled,
            (fname, [level]),
            kwargs=dict(backend="kvikio-read", tile_buffers=tile_buffers),
            n_warmup=1,
            n_repeat=100,
            max_duration=max_duration,
        )
        if gds_enabled:
            perf_kvikio_read = p
        else:
            perf_kvikio_read_nogds = p
        times.append(p.gpu_times.mean())
        labels.append(
            f"kvikio-read (task size={kvikio.defaults.task_size() // 1024} kB)"
            f" ({gds_enabled=})"
        )
        print(f"duration ({labels[-1]}) = {times[-1]}")

    # Go back to 4MB task size in pread case
    kvikio.defaults.task_size_reset(512 * 1024)
    if gds_enabled:
        perf_kvikio_pread = []
    else:
        perf_kvikio_pread_nogds = []
    n_buffers = [1, 4, 16, 64, 256]
    for n_buffer in n_buffers:
        p = benchmark(
            read_tiled,
            (fname, [level]),
            kwargs=dict(
                backend="kvikio-pread",
                n_buffer=n_buffer,
                tile_buffers=tile_buffers,
            ),
            n_warmup=1,
            n_repeat=100,
            max_duration=max_duration,
        )
        if gds_enabled:
            perf_kvikio_pread.append(p)
        else:
            perf_kvikio_pread_nogds.append(p)
        times.append(p.gpu_times.mean())
        labels.append(f"kvikio-pread ({n_buffer=}) ({gds_enabled=})")
        print(f"duration ({labels[-1]}) = {times[-1]}")

if preregister_buffers:
    for b in tile_buffers:
        kvikio.memory_deregister(b)

kvikio.defaults.compat_mode_reset(False)

out_name = "read_times.npz"
# auto-increment filename to avoid overwriting old results
cnt = 1
while os.path.exists(out_name):
    out_name = f"read_times{cnt}.npz"
    cnt += 1
np.savez(out_name, times=np.asarray(times), labels=np.asarray(labels))


"""
Resolution level 0 with Cache clearing, but reads are not 4096-byte aligned

    shape: (26420, 19920, 3)
    stored as 2028 tiles of shape (512, 512, 3)
    kvikio.defaults.compat_mode() = False
    kvikio.defaults.get_num_threads() = 18
    kvikio task size = 4.0 MB
duration (openslide) = 28.921716796875
duration (tifffile) = 3.818202718098958
duration (tiled-tifffile) = 3.885939778645833
duration (kvikio-read_raw (gds_enabled=False)) = 3.4184929199218748
duration (kvikio-read (gds_enabled=False)) = 3.813303955078125
duration (kvikio-pread (n_buffer=1) (gds_enabled=False)) = 3.9369333496093746
duration (kvikio-pread (n_buffer=2) (gds_enabled=False)) = 4.028409342447917
duration (kvikio-pread (n_buffer=4) (gds_enabled=False)) = 2.785054626464844
duration (kvikio-pread (n_buffer=8) (gds_enabled=False)) = 1.7379150390625
duration (kvikio-pread (n_buffer=16) (gds_enabled=False)) = 1.2908187103271485
duration (kvikio-pread (n_buffer=32) (gds_enabled=False)) = 1.0635023193359374
duration (kvikio-pread (n_buffer=64) (gds_enabled=False)) = 0.9369119762073862
duration (kvikio-pread (n_buffer=128) (gds_enabled=False)) = 0.8773154449462891
duration (kvikio-read_raw (gds_enabled=True)) = 3.4003018391927085
duration (kvikio-read (gds_enabled=True)) = 3.763134847005208
duration (kvikio-pread (n_buffer=1) (gds_enabled=True)) = 3.7581602376302086
duration (kvikio-pread (n_buffer=2) (gds_enabled=True)) = 4.107709065755208
duration (kvikio-pread (n_buffer=4) (gds_enabled=True)) = 2.609207336425781
duration (kvikio-pread (n_buffer=8) (gds_enabled=True)) = 1.744682902018229
duration (kvikio-pread (n_buffer=16) (gds_enabled=True)) = 1.2838030700683594
duration (kvikio-pread (n_buffer=32) (gds_enabled=True)) = 1.05522587890625
duration (kvikio-pread (n_buffer=64) (gds_enabled=True)) = 0.9214399691495029
duration (kvikio-pread (n_buffer=128) (gds_enabled=True)) = 0.8695069885253907


Resolution level 0 with 4096-byte aligned reads

    shape: (26420, 19920, 3)
    stored as 2028 tiles of shape (512, 512, 3)
    kvikio.defaults.compat_mode() = False
    kvikio.defaults.get_num_threads() = 18
    kvikio task size = 4.0 MB
duration (kvikio-read_raw (gds_enabled=False)) = 3.4100815429687494
duration (kvikio-read (gds_enabled=False)) = 3.8238279622395837
duration (kvikio-pread (n_buffer=1) (gds_enabled=False)) = 3.740669270833333
duration (kvikio-pread (n_buffer=4) (gds_enabled=False)) = 2.672812255859375
duration (kvikio-pread (n_buffer=16) (gds_enabled=False)) = 1.3131573791503905
duration (kvikio-pread (n_buffer=64) (gds_enabled=False)) = 0.9273524225408379
duration (kvikio-pread (n_buffer=256) (gds_enabled=False)) = 0.8461123250325521
duration (kvikio-read_raw (gds_enabled=True)) = 4.179492513020834
duration (kvikio-read (gds_enabled=True)) = 4.889711263020834
duration (kvikio-pread (n_buffer=1) (gds_enabled=True)) = 4.816523600260417
duration (kvikio-pread (n_buffer=4) (gds_enabled=True)) = 2.2351694824218753
duration (kvikio-pread (n_buffer=16) (gds_enabled=True)) = 1.1082978149414064
duration (kvikio-pread (n_buffer=64) (gds_enabled=True)) = 0.670870166015625
duration (kvikio-pread (n_buffer=256) (gds_enabled=True)) = 0.5998859683766086


    pread with default 4MB "task size"
        Resolution level 0
            shape: (26420, 19920, 3)
            stored as 2028 tiles of shape (512, 512, 3)
            kvikio.defaults.compat_mode() = False
            kvikio.defaults.get_num_threads() = 18
            kvikio task size = 4 MB
        duration (kvikio-pread (n_buffer=1) (gds_enabled=True)) = 4.8583107096354174
        duration (kvikio-pread (n_buffer=4) (gds_enabled=True)) = 2.1224323242187504
        duration (kvikio-pread (n_buffer=16) (gds_enabled=True)) = 1.1164629991319446
        duration (kvikio-pread (n_buffer=64) (gds_enabled=True)) = 0.6734547526041668
        duration (kvikio-pread (n_buffer=256) (gds_enabled=True)) = 0.601566697064568
        (cucim) grelee@grelee-dt:~/Dropbox/NVIDIA/demos/gds/gds-cucim-demo$ python benchmark_read.py
        Resolution level 0

    pread with 64kB "task size"
            shape: (26420, 19920, 3)
            stored as 2028 tiles of shape (512, 512, 3)
            kvikio.defaults.compat_mode() = False
            kvikio.defaults.get_num_threads() = 18
            kvikio task size = 0.064 MB
        duration (kvikio-pread (n_buffer=1) (gds_enabled=True)) = 3.0912179565429687
        duration (kvikio-pread (n_buffer=4) (gds_enabled=True)) = 1.3932305145263673
        duration (kvikio-pread (n_buffer=16) (gds_enabled=True)) = 0.9027577819824221
        duration (kvikio-pread (n_buffer=64) (gds_enabled=True)) = 0.7827104492187501
        duration (kvikio-pread (n_buffer=256) (gds_enabled=True)) = 0.756464599609375
"""  # noqa: E501
