"""
Benchmark locally modified ndimage functions vs. their CuPy counterparts
"""
import cupy as cp
import cupyx.scipy.ndimage as ndi
import pytest
from cupyx.profiler import benchmark

from cucim.skimage._vendored.ndimage import (
    convolve1d, correlate1d, gaussian_filter, gaussian_filter1d,
    gaussian_gradient_magnitude, gaussian_laplace, laplace, prewitt, sobel,
    uniform_filter, uniform_filter1d,
)

d = cp.cuda.Device()


def _get_image(shape, dtype, seed=123):

    rng = cp.random.default_rng(seed)
    dtype = cp.dtype(dtype)
    if dtype.kind == 'b':
        image = rng.integers(0, 1, shape, dtype=cp.uint8).astype(bool)
    elif dtype.kind in 'iu':
        image = rng.integers(0, 128, shape, dtype=dtype)
    elif dtype.kind in 'c':
        real_dtype = cp.asarray([], dtype=dtype).real.dtype
        image = rng.standard_normal(shape, dtype=real_dtype)
        image = image + 1j * rng.standard_normal(shape, dtype=real_dtype)
    else:
        if dtype == cp.float16:
            image = rng.standard_normal(shape).astype(dtype)
        else:
            image = rng.standard_normal(shape, dtype=dtype)
    return image


def _compare_implementations(
    shape, kernel_size, axis, dtype, mode, cval=0.0, origin=0,
    output_dtype=None, kernel_dtype=None, output_preallocated=False,
    function=convolve1d, max_duration=1
):
    dtype = cp.dtype(dtype)
    if kernel_dtype is None:
        kernel_dtype = dtype
    image = _get_image(shape, dtype)
    kernel = _get_image((kernel_size,), kernel_dtype)
    kwargs = dict(axis=axis, mode=mode, cval=cval, origin=origin)
    if output_dtype is not None:
        output_dtype = cp.dtype(output_dtype)
    function_ref = getattr(ndi, function.__name__)
    if output_preallocated:
        if output_dtype is None:
            output_dtype = image.dtype
        output1 = cp.empty(image.shape, dtype=output_dtype)
        output2 = cp.empty(image.shape, dtype=output_dtype)
        kwargs.update(dict(output=output1))
        perf1 = benchmark(function_ref, (image, kernel), kwargs=kwargs, n_warmup=10, n_repeat=10000, max_duration=max_duration)
        kwargs.update(dict(output=output2, algorithm='shared_memory'))
        perf2 = benchmark(function, (image, kernel), kwargs=kwargs, n_warmup=10, n_repeat=10000, max_duration=max_duration)
        return perf1, perf2
    kwargs.update(dict(output=output_dtype))
    perf1 = benchmark(function_ref, (image, kernel), kwargs=kwargs, n_warmup=10, n_repeat=10000, max_duration=max_duration)
    kwargs.update(dict(output=output_dtype, algorithm='shared_memory'))
    perf2 = benchmark(function, (image, kernel), kwargs=kwargs, n_warmup=10, n_repeat=10000, max_duration=max_duration)
    return perf1, perf2


def _compare_implementations_other(
    shape, dtype, mode, cval=0.0,
    output_dtype=None, kernel_dtype=None, output_preallocated=False,
    function=convolve1d, func_kwargs={}, max_duration=1,
):
    dtype = cp.dtype(dtype)
    image = _get_image(shape, dtype)
    kwargs = dict(mode=mode, cval=cval)
    if func_kwargs:
        kwargs.update(func_kwargs)
    if output_dtype is not None:
        output_dtype = cp.dtype(output_dtype)
    function_ref = getattr(ndi, function.__name__)
    if output_preallocated:
        if output_dtype is None:
            output_dtype = image.dtype
        output1 = cp.empty(image.shape, dtype=output_dtype)
        output2 = cp.empty(image.shape, dtype=output_dtype)
        kwargs.update(dict(output=output1))
        perf1 = benchmark(function_ref, (image,), kwargs=kwargs, n_warmup=10, n_repeat=10000, max_duration=max_duration)
        kwargs.update(dict(output=output1, algorithm='shared_memory'))
        perf2 = benchmark(function, (image,), kwargs=kwargs, n_warmup=10, n_repeat=10000, max_duration=max_duration)
        return perf1, perf2
    kwargs.update(dict(output=output_dtype))
    perf1 = benchmark(function_ref, (image,), kwargs=kwargs, n_warmup=10, n_repeat=10000, max_duration=max_duration)
    kwargs.update(dict(output=output_dtype, algorithm='shared_memory'))
    perf2 = benchmark(function, (image,), kwargs=kwargs, n_warmup=10, n_repeat=10000, max_duration=max_duration)
    return perf1, perf2


print("\n\n")
print("function | shape | dtype | mode | kernel size | preallocated | axis | dur (ms), CuPy | dur (ms), cuCIM | acceleration ")
print("---------|-------|-------|------|-------------|--------------|------|----------------|-----------------|--------------")
for function in [convolve1d]:
    for shape in [(512, 512), (3840, 2160), (64, 64, 64), (256, 256, 256)]:
        for dtype in [cp.float32, cp.uint8]:
            for mode in ['nearest']:
                for kernel_size in [3, 7, 11, 41]:
                    for output_preallocated in [False]:  # , True]:
                        for axis in range(len(shape)):
                            output_dtype = dtype
                            perf1, perf2 = _compare_implementations(shape=shape, kernel_size=kernel_size, mode=mode, axis=axis, dtype=dtype, output_dtype=output_dtype, output_preallocated=output_preallocated, function=function)
                            t_elem = perf1.gpu_times * 1000.
                            t_shared = perf2.gpu_times * 1000.
                            print(f"{function.__name__} | {shape} | {cp.dtype(dtype).name} | {mode} | {kernel_size=} | prealloc={output_preallocated} | {axis=} | {t_elem.mean():0.3f} +/- {t_elem.std():0.3f}  | {t_shared.mean():0.3f} +/- {t_shared.std():0.3f} | {t_elem.mean() / t_shared.mean():0.3f}")


print("function | kwargs | shape | dtype | mode | preallocated | dur (ms), CuPy | dur (ms), cuCIM | acceleration ")
print("---------|--------|-------|-------|------|--------------|----------------|-----------------|--------------")
for function, func_kwargs in [
    # (gaussian_filter1d, dict(sigma=1.0, axis=0)),
    # (gaussian_filter1d, dict(sigma=1.0, axis=-1)),
    # (gaussian_filter1d, dict(sigma=4.0, axis=0)),
    # (gaussian_filter1d, dict(sigma=4.0, axis=-1)),
    (gaussian_filter, dict(sigma=1.0)),
    (gaussian_filter, dict(sigma=4.0)),
    (uniform_filter, dict(size=11)),
    (prewitt, dict(axis=0)),
    (sobel, dict(axis=0)),
    (prewitt, dict(axis=-1)),
    (sobel, dict(axis=-1)),
]:
    for shape in [(512, 512), (3840, 2160), (64, 64, 64), (256, 256, 256)]:
        for (dtype, output_dtype) in [(cp.float32, cp.float32), (cp.uint8, cp.float32)]:
            for mode in ['nearest']:
                for output_preallocated in [False, True]:
                    perf1, perf2 = _compare_implementations_other(shape=shape, mode=mode, dtype=dtype, output_dtype=output_dtype, output_preallocated=output_preallocated, function=function, func_kwargs=func_kwargs)
                    t_elem = perf1.gpu_times * 1000.
                    t_shared = perf2.gpu_times * 1000.
                    print(f"{function.__name__} | {func_kwargs} | {shape} | {cp.dtype(dtype).name} | {mode} | {output_preallocated} | {t_elem.mean():0.3f} +/- {t_elem.std():0.3f}  | {t_shared.mean():0.3f} +/- {t_shared.std():0.3f} | {t_elem.mean() / t_shared.mean():0.3f}")

