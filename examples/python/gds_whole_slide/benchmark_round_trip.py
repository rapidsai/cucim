import os
from time import time

import cupy as cp
import kvikio
import kvikio.defaults
import numpy as np
from cupyx.profiler import benchmark
from demo_implementation import cupy_to_zarr, read_tiled

import cucim.skimage.filters
from cucim.core.operations.color import image_to_absorbance

data_dir = os.environ.get("WHOLE_SLIDE_DATA_DIR", os.path.dirname("__file__"))
fname = os.path.join(data_dir, "resize.tiff")
if not os.path.exists(fname):
    raise RuntimeError(f"Could not find data file: {fname}")

level = 0
max_duration = 8
compressor = None

# set the number of threads to use
kvikio.defaults.num_threads_reset(16)

# Go back to 4MB task size in pread case
kvikio.defaults.task_size_reset(4 * 1024 * 1024)


def round_trip(
    fname,
    level=0,
    n_buffer=64,
    kernel_func=None,
    kernel_func_kwargs={},
    apply_kernel_tilewise=True,
    out_dtype=cp.uint8,
    zarr_chunk_shape=(2048, 2048, 3),
    output_path=None,
    zarr_kwargs=dict(overwrite=True, compressor=None),
    verbose_times=False,
):
    if output_path is None:
        output_path = f"./image-{cp.dtype(out_dtype).name}.zarr"

    if apply_kernel_tilewise:
        tile_func = kernel_func
        tile_func_kwargs = kernel_func_kwargs
    else:
        tile_func = None
        tile_func_kwargs = {}

    if verbose_times:
        tstart = time()
    data_gpu = read_tiled(
        fname,
        levels=[level],
        backend="kvikio-pread",
        n_buffer=n_buffer,
        tile_func=tile_func,
        tile_func_kwargs=tile_func_kwargs,
        out_dtype=out_dtype,
    )[0]
    if verbose_times:
        dur_read = time() - tstart
        if not apply_kernel_tilewise:
            dur_read = time() - tstart
            print(f"{dur_read=}")
        else:
            dur_read_and_comp = time() - tstart
            print(f"{dur_read_and_comp=}")

    if not apply_kernel_tilewise:
        if verbose_times:
            tstart = time()
        data_gpu = kernel_func(data_gpu, **kernel_func_kwargs)
        if verbose_times:
            dur_comp = time() - tstart
            print(f"{dur_comp=}")

    if verbose_times:
        tstart = time()
    cupy_to_zarr(
        data_gpu,
        backend="dask",  # 'kvikio-pwrite',
        output_path=output_path,
        chunk_shape=zarr_chunk_shape,
        zarr_kwargs=zarr_kwargs,
    )
    if verbose_times:
        dur_write = time() - tstart
        print(f"{dur_write=}")

    return output_path


gds_enabled = True
apply_kernel_tilewise = True
times = []
labels = []
n_buffer = 32
for zarr_chunk_shape in [
    (512, 512, 3),
    (1024, 1024, 3),
    (2048, 2048, 3),
    (4096, 4096, 3),
]:
    for computation in ["absorbance", "median", "gaussian", "sobel"]:
        if computation is None:
            kernel_func = None
            kernel_func_kwargs = {}
            out_dtype = cp.uint8
        elif computation == "absorbance":
            kernel_func = image_to_absorbance
            kernel_func_kwargs = {}
            out_dtype = cp.float32
        elif computation == "median":
            kernel_func = cucim.skimage.filters.median
            kernel_func_kwargs = dict(footprint=cp.ones((5, 5, 1), dtype=bool))
            out_dtype = cp.uint8
        elif computation == "gaussian":
            kernel_func = cucim.skimage.filters.gaussian
            kernel_func_kwargs = dict(sigma=2.5, channel_axis=-1)
            out_dtype = cp.uint8
        elif computation == "sobel":
            kernel_func = cucim.skimage.filters.sobel
            kernel_func_kwargs = dict(axis=(0, 1))
            out_dtype = cp.float32

        for apply_kernel_tilewise in [False, True]:
            for gds_enabled in [False, True]:
                kvikio.defaults.compat_mode_reset(not gds_enabled)
                assert kvikio.defaults.compat_mode() == (not gds_enabled)

                kwargs = dict(
                    level=0,
                    n_buffer=n_buffer,
                    kernel_func=kernel_func,
                    kernel_func_kwargs=kernel_func_kwargs,
                    out_dtype=out_dtype,
                    apply_kernel_tilewise=apply_kernel_tilewise,
                    zarr_chunk_shape=zarr_chunk_shape,
                    zarr_kwargs=dict(overwrite=True, compressor=compressor),
                    verbose_times=False,
                )

                perf = benchmark(
                    round_trip,
                    (fname,),
                    kwargs=kwargs,
                    n_warmup=1,
                    n_repeat=100,
                    max_duration=max_duration,
                )
                t = perf.gpu_times

                kernel_description = (
                    "tiled" if apply_kernel_tilewise else "global"
                )
                gds_description = "with GDS" if gds_enabled else "without GDS"
                label = f"{computation=}, {kernel_description}, chunk_shape={zarr_chunk_shape}, {gds_description}"
                print(f"Duration ({label}): {t.mean()} s +/- {t.std()} s")

                times.append(t.mean())
                labels.append(label)

out_name = "round_trip_times.npz"
# auto-increment filename to avoid overwriting old results
cnt = 1
while os.path.exists(out_name):
    out_name = f"round_trip_times{cnt}.npz"
    cnt += 1
np.savez(out_name, times=np.asarray(times), labels=np.asarray(labels))


"""
on dgx-02

(gds_demo) apollo@dgx-02:/mnt/nvme0/cucim/gds-cucim-demo$ python benchmark_round_trip.py
    Duration (computation='absorbance', global, chunk_shape=(512, 512, 3), without GDS): 3.1191054687500004 s +/- 0.225595815853941 s
    Duration (computation='absorbance', global, chunk_shape=(512, 512, 3), with GDS): 2.1042802734375003 s +/- 0.027930034537340144 s
    Duration (computation='absorbance', tiled, chunk_shape=(512, 512, 3), without GDS): 3.141532592773437 s +/- 0.04939690170558264 s
    Duration (computation='absorbance', tiled, chunk_shape=(512, 512, 3), with GDS): 2.231423193359375 s +/- 0.16606144818711904 s
    Duration (computation='median', global, chunk_shape=(512, 512, 3), without GDS): 2.0818441894531245 s +/- 0.021839879963544123 s
    Duration (computation='median', global, chunk_shape=(512, 512, 3), with GDS): 2.035376928710938 s +/- 0.029007739628238723 s
    Duration (computation='median', tiled, chunk_shape=(512, 512, 3), without GDS): 2.567677185058594 s +/- 0.1512259361762735 s
    Duration (computation='median', tiled, chunk_shape=(512, 512, 3), with GDS): 2.258326318359375 s +/- 0.03280750045711541 s
    Duration (computation='gaussian', global, chunk_shape=(512, 512, 3), without GDS): 3.1225366821289064 s +/- 0.03711702097752233 s
    Duration (computation='gaussian', global, chunk_shape=(512, 512, 3), with GDS): 2.2253964843750005 s +/- 0.2325274037827292 s
    Duration (computation='gaussian', tiled, chunk_shape=(512, 512, 3), without GDS): 2.4777058593750003 s +/- 0.009240477773701704 s
    Duration (computation='gaussian', tiled, chunk_shape=(512, 512, 3), with GDS): 2.3227379394531256 s +/- 0.009206045295060663 s
    Duration (computation='sobel', global, chunk_shape=(512, 512, 3), without GDS): 3.2487342529296876 s +/- 0.27136693727180555 s
    Duration (computation='sobel', global, chunk_shape=(512, 512, 3), with GDS): 2.10892294921875 s +/- 0.02097221308256493 s
    Duration (computation='sobel', tiled, chunk_shape=(512, 512, 3), without GDS): 4.1089834798177085 s +/- 0.01915263510603932 s
    Duration (computation='sobel', tiled, chunk_shape=(512, 512, 3), with GDS): 2.938178039550781 s +/- 0.014859342085037834 s
    Duration (computation='absorbance', global, chunk_shape=(1024, 1024, 3), without GDS): 2.7858702392578127 s +/- 0.019651934337578617 s
    Duration (computation='absorbance', global, chunk_shape=(1024, 1024, 3), with GDS): 1.9969252726236981 s +/- 0.027131826130877872 s
    Duration (computation='absorbance', tiled, chunk_shape=(1024, 1024, 3), without GDS): 2.805119934082031 s +/- 0.04126104227305295 s
    Duration (computation='absorbance', tiled, chunk_shape=(1024, 1024, 3), with GDS): 2.000840478515625 s +/- 0.018744582209408268 s
    Duration (computation='median', global, chunk_shape=(1024, 1024, 3), without GDS): 1.1235912679036462 s +/- 0.017105640884020744 s
    Duration (computation='median', global, chunk_shape=(1024, 1024, 3), with GDS): 0.797370131272536 s +/- 0.15744958452772662 s
    Duration (computation='median', tiled, chunk_shape=(1024, 1024, 3), without GDS): 1.437654331752232 s +/- 0.021581864835220975 s
    Duration (computation='median', tiled, chunk_shape=(1024, 1024, 3), with GDS): 0.9701279130415483 s +/- 0.02103963138394191 s
    Duration (computation='gaussian', global, chunk_shape=(1024, 1024, 3), without GDS): 2.8036607666015625 s +/- 0.012992702272646189 s
    Duration (computation='gaussian', global, chunk_shape=(1024, 1024, 3), with GDS): 2.0812043945312504 s +/- 0.11895250430711447 s
    Duration (computation='gaussian', tiled, chunk_shape=(1024, 1024, 3), without GDS): 1.5445246407645088 s +/- 0.015315103050140895 s
    Duration (computation='gaussian', tiled, chunk_shape=(1024, 1024, 3), with GDS): 1.0669870971679687 s +/- 0.02188820345886527 s
    Duration (computation='sobel', global, chunk_shape=(1024, 1024, 3), without GDS): 2.9324288330078128 s +/- 0.22276139967810918 s
    Duration (computation='sobel', global, chunk_shape=(1024, 1024, 3), with GDS): 2.0065390625000004 s +/- 0.013617705944694957 s
    Duration (computation='sobel', tiled, chunk_shape=(1024, 1024, 3), without GDS): 3.6309766438802087 s +/- 0.05874521968221261 s
    Duration (computation='sobel', tiled, chunk_shape=(1024, 1024, 3), with GDS): 2.984893737792969 s +/- 0.25415558271992245 s
    Duration (computation='absorbance', global, chunk_shape=(2048, 2048, 3), without GDS): 2.9885695800781247 s +/- 0.026795940572500634 s
    Duration (computation='absorbance', global, chunk_shape=(2048, 2048, 3), with GDS): 1.965831502278646 s +/- 0.0168025999020846 s
    Duration (computation='absorbance', tiled, chunk_shape=(2048, 2048, 3), without GDS): 3.1813931884765627 s +/- 0.2428613856732194 s
    Duration (computation='absorbance', tiled, chunk_shape=(2048, 2048, 3), with GDS): 1.9277428181966145 s +/- 0.04373333981541544 s
    Duration (computation='median', global, chunk_shape=(2048, 2048, 3), without GDS): 1.0377198669433594 s +/- 0.012093431782665117 s
    Duration (computation='median', global, chunk_shape=(2048, 2048, 3), with GDS): 0.707420703125 s +/- 0.01371743462230204 s
    Duration (computation='median', tiled, chunk_shape=(2048, 2048, 3), without GDS): 1.4118775634765626 s +/- 0.18244415183128557 s
    Duration (computation='median', tiled, chunk_shape=(2048, 2048, 3), with GDS): 0.9218176491477272 s +/- 0.01591385754055359 s
    Duration (computation='gaussian', global, chunk_shape=(2048, 2048, 3), without GDS): 2.9058739013671877 s +/- 0.016099121671779952 s
    Duration (computation='gaussian', global, chunk_shape=(2048, 2048, 3), with GDS): 2.0523116210937498 s +/- 0.18981050500282015 s
    Duration (computation='gaussian', tiled, chunk_shape=(2048, 2048, 3), without GDS): 1.4494330357142857 s +/- 0.014561240480290236 s
    Duration (computation='gaussian', tiled, chunk_shape=(2048, 2048, 3), with GDS): 1.0200827880859376 s +/- 0.019787969209851694 s
    Duration (computation='sobel', global, chunk_shape=(2048, 2048, 3), without GDS): 3.119711364746094 s +/- 0.22031551398713953 s
    Duration (computation='sobel', global, chunk_shape=(2048, 2048, 3), with GDS): 1.958296569824219 s +/- 0.02411710745257342 s
    Duration (computation='sobel', tiled, chunk_shape=(2048, 2048, 3), without GDS): 3.8941464843749998 s +/- 0.049486723671958784 s
    Duration (computation='sobel', tiled, chunk_shape=(2048, 2048, 3), with GDS): 2.9164300537109376 s +/- 0.1746352677421199 s
    Duration (computation='absorbance', global, chunk_shape=(4096, 4096, 3), without GDS): 4.78259814453125 s +/- 0.08003375317781046 s
    Duration (computation='absorbance', global, chunk_shape=(4096, 4096, 3), with GDS): 2.1372686035156248 s +/- 0.043582537001700936 s
    Duration (computation='absorbance', tiled, chunk_shape=(4096, 4096, 3), without GDS): 4.702880371093751 s +/- 0.036472302464017906 s
    Duration (computation='absorbance', tiled, chunk_shape=(4096, 4096, 3), with GDS): 2.1466302734375 s +/- 0.04854747600437549 s
    Duration (computation='median', global, chunk_shape=(4096, 4096, 3), without GDS): 1.2019552951388888 s +/- 0.041718545478099486 s
    Duration (computation='median', global, chunk_shape=(4096, 4096, 3), with GDS): 0.7432061026436942 s +/- 0.024861430042890365 s
    Duration (computation='median', tiled, chunk_shape=(4096, 4096, 3), without GDS): 1.5174084472656253 s +/- 0.19087705973449015 s
    Duration (computation='median', tiled, chunk_shape=(4096, 4096, 3), with GDS): 0.9612706076882103 s +/- 0.047138270409090556 s
    Duration (computation='gaussian', global, chunk_shape=(4096, 4096, 3), without GDS): 4.438681803385417 s +/- 0.07821072441047229 s
    Duration (computation='gaussian', global, chunk_shape=(4096, 4096, 3), with GDS): 2.292842724609375 s +/- 0.17273283392695543 s
    Duration (computation='gaussian', tiled, chunk_shape=(4096, 4096, 3), without GDS): 1.5610232805524553 s +/- 0.04472800111712806 s
    Duration (computation='gaussian', tiled, chunk_shape=(4096, 4096, 3), with GDS): 1.053441522216797 s +/- 0.047010837833830116 s
    Duration (computation='sobel', global, chunk_shape=(4096, 4096, 3), without GDS): 4.608963867187501 s +/- 0.27347076419226174 s
    Duration (computation='sobel', global, chunk_shape=(4096, 4096, 3), with GDS): 2.2000548339843746 s +/- 0.07589894027639234 s
    Duration (computation='sobel', tiled, chunk_shape=(4096, 4096, 3), without GDS): 5.430656982421875 s +/- 0.01277270507812478 s
    Duration (computation='sobel', tiled, chunk_shape=(4096, 4096, 3), with GDS): 3.1940713500976563 s +/- 0.19852391299904837 s

"""
