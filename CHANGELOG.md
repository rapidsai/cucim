# cuCIM 22.02.00 (2 Feb 2022)

## 🚨 Breaking Changes

- Update cucim.skimage API to match scikit-image 0.19 ([#190](https://github.com/rapidsai/cucim/pull/190)) [@glee77](https://github.com/glee77)

## 🐛 Bug Fixes

- Fix a bug in [v21.12.01](https://github.com/rapidsai/cucim/wiki/release_notes_v21.12.01) ([#191](https://github.com/rapidsai/cucim/pull/191)) [@gigony](https://github.com/gigony)
  - Fix GPU memory leak when using nvJPEG API (when `device='cuda'` parameter is used in `read_region` method).
- Fix segfault for preferred_memory_capacity in Python 3.9+ ([#214](https://github.com/rapidsai/cucim/pull/214)) [@gigony](https://github.com/gigony)

## 📖 Documentation

- PyPI v21.12.00 release ([#182](https://github.com/rapidsai/cucim/pull/182)) [@gigony](https://github.com/gigony)

## 🚀 New Features

1. Update cucim.skimage API to match scikit-image 0.19 ([#190](https://github.com/rapidsai/cucim/pull/190)) [@glee77](https://github.com/glee77)
2. Support multi-threads and batch, and support nvJPEG for JPEG-compressed images ([#191](https://github.com/rapidsai/cucim/pull/191)) [@gigony](https://github.com/gigony)
3. Allow CuPy 10 ([#195](https://github.com/rapidsai/cucim/pull/195)) [@jakikham](https://github.com/jakikham)

### 1. Update cucim.skimage API to match scikit-image 0.19 (🚨 Breaking Changes)

#### channel_axis support

scikit-image 0.19 adds a `channel_axis` argument that should now be used instead of the `multichannel` boolean.

In scikit-image 1.0, the `multichannel` argument will likely be removed so we start supporting `channel_axis` in cuCIM.

This pulls changes from many scikit-image 0.19.0 PRs related to deprecating `multichannel` in favor of `channel_axis`. A few other minor PRs related to deprecations and updates to `color.label2rgb` are incorporated here as well.

The changes are mostly non-breaking, although a couple of deprecated functions have been removed (`rgb2grey`, `grey2rgb`) and a change in the default value of `label2rgb`'s `bg_label` argument. The deprecated `alpha` argument was removed from gray2rgb.

Implements:

- [Add saturation parameter to color.label2rgb #5156](https://github.com/scikit-image/scikit-image/pull/5156)
- [Decorators for helping with the multichannel->channel_axis transition #5228](https://github.com/scikit-image/scikit-image/pull/5228)
- [multichannel to channel_axis (1 of 6): features and draw #5284](https://github.com/scikit-image/scikit-image/pull/5284)
- [multichannel to channel_axis (2 of 6): transform functions #5285](https://github.com/scikit-image/scikit-image/pull/5285)
- [multichannel to channel_axis (3 of 6): filters #5286](https://github.com/scikit-image/scikit-image/pull/5286)
- [multichannel to channel_axis (4 of 6): metrics and measure #5287](https://github.com/scikit-image/scikit-image/pull/5287)
- [multichannel to channel_axis (5 of 6): restoration #5288](https://github.com/scikit-image/scikit-image/pull/5288)
- [multichannel to channel_axis (6 of 6): segmentation #5289](https://github.com/scikit-image/scikit-image/pull/5289)
- [channel_as_last_axis decorator fix #5348](https://github.com/scikit-image/scikit-image/pull/5348)
- [fix wrong error for metric.structural_similarity when image is too small #5395](https://github.com/scikit-image/scikit-image/pull/5395)
- [Add a channel_axis argument to functions in the skimage.color module #5462](https://github.com/scikit-image/scikit-image/pull/5462)
- [Remove deprecated functions and arguments for the 0.19 release #5463](https://github.com/scikit-image/scikit-image/pull/5463)
- [Support nD images and labels in label2rgb #5550](https://github.com/scikit-image/scikit-image/pull/5550)
- [remove need for channel_as_last_axis decorator in skimage.filters #5584](https://github.com/scikit-image/scikit-image/pull/5584)
- [Preserve backwards compatibility for `channel_axis` parameter in transform functions #6095](https://github.com/scikit-image/scikit-image/pull/6095)

#### Update float32 dtype support to match scikit-image 0.19 behavior

Makes float32 and float16 handling consistent with scikit-image 0.19. (must functions support float32, float16 gets promoted to float32)

#### Deprecate APIs

Introduces new deprecations as in scikit-image 0.19.

Specifically:

- `selem` -> `footprint`
- `grey` -> `gray`
- `iterations` -> `num_iter`
- `max_iter` -> `max_num_iter`
- `min_iter` -> `min_num_iter`

### 2. Supporting Multithreading and Batch Processing

cuCIM now supports loading the entire image with multi-threads. It also supports batch loading of images.

If `device` parameter of `read_region()` method is `"cuda"`, it loads a relevant portion of the image file (compressed tile data) into GPU memory using cuFile(GDS, GPUDirect Storage), then decompress those data using nvJPEG's [Batched Image Decoding API](https://docs.nvidia.com/cuda/nvjpeg/index.html#nvjpeg-batched-image-decoding).

Current implementations are not efficient and performance is poor compared to CPU implementations. However, we plan to improve it over the next versions.

#### Example API Usages

The following parameters would be added in the `read_region` method:

- `num_workers`: number of workers(threads) to use for loading the image. (default: `1`)
- `batch_size`: number of images to load at once. (default: `1`)
- `drop_last`: whether to drop the last batch if the batch size is not divisible by the number of images. (default: `False`)
- `preferch_factor`: number of samples loaded in advance by each worker. (default: `2`)
- `shuffle`: whether to shuffle the input locations (default: `False`)
- `seed`: seed value for random value generation (default: 0)

**Loading entire image by using multithreads**

```python
from cucim import CuImage

img = CuImage("input.tif")

region = img.read_region(level=1, num_workers=8)  # read whole image at level 1 using 8 workers
```

**Loading batched image using multithreads**

You can feed locations of the region through the list/tuple of locations or through the NumPy array of locations.
(e.g., `((<x for loc 1>, <y for loc 1>), (<x for loc 2>, <y for loc 2>)])`).
Each element in the location should be int type (int64) and the dimension of the location should be
equal to the dimension of the size.
You can feed any iterator of locations (dimensions of the input don't matter, flattening the item in the iterator once if the item is also an iterator).

For example, you can feed the following iterator:

- `[0, 0, 100, 0]` or `(0, 0, 100, 0)` would be interpreted as a list of `(0, 0)` and `(100, 0)`.
- `((sx, sy) for sy in range(0, height, patch_size) for sx in range(0, width, patch_size))` would iterate over the locations of the patches.
- `[(0, 100), (0, 200)]` would be interpreted as a list of `(0, 0)` and `(100, 0)`.
- Numpy array such as `np.array(((0, 100), (0, 200)))` or `np.array((0, 100, 0, 200))` would be also available and using Numpy array object would be faster than using python list/tuple.

```python
import numpy as np
from cucim import CuImage

cache = CuImage.cache("per_process", memory_capacity=1024)

img = CuImage("image.tif")

locations = [[0,   0], [100,   0], [200,   0], [300,   0],
             [0, 200], [100, 200], [200, 200], [300, 200]]
# locations = np.array(locations)

region = img.read_region(locations, (224, 224), batch_size=4, num_workers=8)

for batch in region:
    img = np.asarray(batch)
    print(img.shape)
    for item in img:
        print(item.shape)

# (4, 224, 224, 3)
# (224, 224, 3)
# (224, 224, 3)
# (224, 224, 3)
# (224, 224, 3)
# (4, 224, 224, 3)
# (224, 224, 3)
# (224, 224, 3)
# (224, 224, 3)
# (224, 224, 3)
```

**Loading image using nvJPEG and cuFile (GDS, GPUDirect Storage)**

If `cuda` argument is specified in `device` parameter of `read_region()` method, it uses nvJPEG with GPUDirect Storage to load images.

Use CuPy instead of Numpy, and Image Cache (`CuImage.cache`) wouldn't be used in the case.

```python
import cupy as cp
from cucim import CuImage

img = CuImage("image.tif")

locations = [[0,   0], [100,   0], [200,   0], [300,   0],
             [0, 200], [100, 200], [200, 200], [300, 200]]
# locations = np.array(locations)

region = img.read_region(locations, (224, 224), batch_size=4, device="cuda")

for batch in region:
    img = cp.asarray(batch)
    print(img.shape)
    for item in img:
        print(item.shape)

# (4, 224, 224, 3)
# (224, 224, 3)
# (224, 224, 3)
# (224, 224, 3)
# (224, 224, 3)
# (4, 224, 224, 3)
# (224, 224, 3)
# (224, 224, 3)
# (224, 224, 3)
# (224, 224, 3)
```

#### Experimental Results

We have compared performance against Tifffile for loading the entire image.

##### System Information

- OS: Ubuntu 18.04
- CPU: [Intel(R) Core(TM) i7-7800X CPU @ 3.50GHz](https://www.cpubenchmark.net/cpu.php?cpu=Intel+Core+i7-7800X+%40+3.50GHz&id=3037), 12 processors.
- Memory: 64GB (G-Skill DDR4 2133 16GB X 4)
- Storage
  - SATA SSD: [Samsung SSD 850 EVO 1TB](https://www.samsung.com/us/computing/memory-storage/solid-state-drives/ssd-850-evo-2-5-sata-iii-1tb-mz-75e1t0b-am/)
  
##### Experiment Setup

Benchmarked loading several images with [Tifffile](https://github.com/cgohlke/tifffile).
+ Use read_region() APIs to read the entire image (.svs/.tiff) at the largest resolution level.
    - Performed on the following images that use a different compression method
        * JPEG2000 YCbCr: [TUPAC-TR-467.svs](https://drive.google.com/drive/u/0/folders/0B--ztKW0d17XYlBqOXppQmw0M2M), 55MB, 19920x26420, tile size 240x240
        * JPEG: image.tif (256x256 multi-resolution/tiled TIF conversion of TUPAC-TR-467.svs), 238MB, 19920x26420, tile size 256x256
        * JPEG2000 RGB: [CMU-1-JP2K-33005.svs](https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/), 126MB, 46000x32893, tile size 240x240
        * JPEG: [0005f7aaab2800f6170c399693a96917.tiff](https://www.kaggle.com/c/prostate-cancer-grade-assessment/data) in [Prostate cANcer graDe Assessment (PANDA) Challenge](https://www.kaggle.com/c/prostate-cancer-grade-assessment/data), 46MB, 27648x29440, tile size 512x512
        * JPEG: [000920ad0b612851f8e01bcc880d9b3d.tiff](https://www.kaggle.com/c/prostate-cancer-grade-assessment/data) in [Prostate cANcer graDe Assessment (PANDA) Challenge](https://www.kaggle.com/c/prostate-cancer-grade-assessment/data), 14MB, 15360x13312, tile size 512x512
        * JPEG: [001d865e65ef5d2579c190a0e0350d8f.tiff](https://www.kaggle.com/c/prostate-cancer-grade-assessment/data) in [Prostate cANcer graDe Assessment (PANDA) Challenge](https://www.kaggle.com/c/prostate-cancer-grade-assessment/data), 71MB, 28672x34560, tile size 512x512

+ Use the same number of workers (threads) for both cuCIM and Tifffile.
    - Tifffile uses half of the available processors by default (6 in the test system)
    - Tested with 6 and 12 threads
+ Use the average time of 5 samples.
+ Test code is available at [here](https://gist.github.com/gigony/260d152a83519614ca8c46df551f0d57)

##### Results

+ JPEG2000 YCbCr: [TUPAC-TR-467.svs](https://drive.google.com/drive/u/0/folders/0B--ztKW0d17XYlBqOXppQmw0M2M), 55MB, 19920x26420, tile size 240x240
  - cuCIM [6 threads]: 2.7688472287729384
  - tifffile [6 threads]: 7.4588409311138095
  - cuCIM [12 threads]: 2.1468488964252175
  - tifffile [12 threads]: 6.142562598735094
+ JPEG: image.tif (256x256 multi-resolution/tiled TIF conversion of TUPAC-TR-467.svs), 238MB, 19920x26420, tile size 256x256
  - cuCIM [6 threads]: 0.6951584462076426
  - tifffile [6 threads]: 1.0252630705013872
  - cuCIM [12 threads]: 0.5354489935562015
  - tifffile [12 threads]: 1.5688881931826473
+ JPEG2000 RGB: [CMU-1-JP2K-33005.svs](https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/), 126MB, 46000x32893, tile size 240x240
  - cuCIM [6 threads]: 9.2361351958476
  - tifffile [6 threads]: 27.936951795965435
  - cuCIM [12 threads]: 7.4136177686043085
  - tifffile [12 threads]: 22.46532293939963
+ JPEG: [0005f7aaab2800f6170c399693a96917.tiff](https://www.kaggle.com/c/prostate-cancer-grade-assessment/data), 46MB, 27648x29440, tile size 512x512
  - cuCIM [6 threads]: 0.7972335423342883
  - tifffile [6 threads]: 0.926042037177831
  - cuCIM [12 threads]: 0.6366931471042335
  - tifffile [12 threads]: 0.9512427857145667
+ JPEG: [000920ad0b612851f8e01bcc880d9b3d.tiff](https://www.kaggle.com/c/prostate-cancer-grade-assessment/data), 14MB, 15360x13312, tile size 512x512
  - cuCIM [6 threads]: 0.2257618647068739
  - tifffile [6 threads]: 0.25579613661393524
  - cuCIM [12 threads]: 0.1840262952260673
  - tifffile [12 threads]: 0.2717844221740961
+ JPEG: [001d865e65ef5d2579c190a0e0350d8f.tiff](https://www.kaggle.com/c/prostate-cancer-grade-assessment/data), 71MB, 28672x34560, tile size 512x512
  - cuCIM [6 threads]: 0.9925791253335774
  - tifffile [6 threads]: 1.131185239739716
  - cuCIM [12 threads]: 0.8037087645381689
  - tifffile [12 threads]: 1.1474561678245663

### 3. Allow CuPy 10

Relaxes version constraints to allow CuPy 10 (in meta.yaml).

`cupy 9.*` => `cupy >=9,<11.0.0a0`

## 🛠️ Improvements

- Add missing imports tests ([#183](https://github.com/rapidsai/cucim/pull/183)) [@Ethyling](https://github.com/Ethyling)
- Allow installation with CuPy 10 ([#197](https://github.com/rapidsai/cucim/pull/197)) [@glee77](https://github.com/glee77)
- Upgrade Numpy to 1.18 for Python 3.9 support ([#196](https://github.com/rapidsai/cucim/pull/196)) [@Ethyling](https://github.com/Ethyling)
- Upgrade Numpy to 1.19 for Python 3.9 support ([#203](https://github.com/rapidsai/cucim/pull/203)) [@Ethyling](https://github.com/Ethyling)

# cuCIM 21.12.00 (9 Dec 2021)

## 🚀 New Features

1. Support Aperio SVS with CPU LZW and jpeg2k decoder ([#141](https://github.com/rapidsai/cucim/pull/141)) [[@gigony](https://github.com/gigony)](https://github.com/gigony](https://github.com/gigony))
2. Add NVTX support for performance analysis ([#144](https://github.com/rapidsai/cucim/pull/144)) [[@gigony](https://github.com/gigony)](https://github.com/gigony](https://github.com/gigony))
3. Normalize operation ([#150](https://github.com/rapidsai/cucim/pull/150)) [[@shekhardw](https://github.com/shekhardw)](https://github.com/shekhardw](https://github.com/shekhardw))

### 1. Support Aperio SVS (.svs)

cuCIM now supports [Aperio SVS format](https://openslide.org/formats/aperio/) with help of [OpenJpeg](https://www.openjpeg.org/) for decoding jpeg2k-compressed data.

Please check [this notebook](https://nbviewer.org/github/rapidsai/cucim/blob/branch-21.12/notebooks/Supporting_Aperio_SVS_Format.ipynb) to see how to use the feature.

#### Unaligned Case (`per_process`, JPEG-compressed SVS file)

![image](https://user-images.githubusercontent.com/1928522/141350490-06fdd8cb-5be2-42e4-9774-c7b76fab6f9a.png)

#### Unaligned Case (`per_process`, JPEG2000 RGB-compressed SVS file)

![image](https://user-images.githubusercontent.com/1928522/141093324-574b532e-ad42-4d61-8473-4c3e07e3feae.png)

#### Unaligned Case (`per_process`, JPEG2000 YCbCr-compressed SVS file)

![image](https://user-images.githubusercontent.com/1928522/141093381-8ab0161d-1b17-4e80-a680-86abfbf2fa65.png)

The detailed data is available [here](https://docs.google.com/spreadsheets/d/15D1EqNI_E9x_S8i3kJLwBxMcEmwk8SafW0WryMrAm6A/edit#gid=369408723).

### 2. Add NVTX support for performance analysis

Important methods in cuCIM are instrumented with [NVTX](https://docs.nvidia.com/gameworks/index.html#gameworkslibrary/nvtx/nvidia_tools_extension_library_nvtx.htm) so can see performance bottlenecks easily with [NSight systems](https://developer.nvidia.com/nsight-systems).

Tracing can be enabled through config file or environment variable or through API and less than 1% performance overheads in normal execution.

#### Enabling Tracing
##### Through `.cucim.json` file

```json
{
        "profiler" : { "trace": true }
}
```

##### Through Environment variable

```bash
CUCIM_TRACE=1 python
```

##### Through API

```python
from cucim import CuImage

CuImage.profiler(trace=True)
#or
CuImage.profiler().trace(True)

CuImage.profiler().config
# {'trace': True}
CuImage.profiler().trace()
# True
CuImage.is_trace_enabled # this is simpler method.
# True
```

#### Profiling with NVIDIA Nsight Systems

```bash
nsys profile -f true -t cuda,nvtx,osrt -s cpu -x true --trace-fork-before-exec true -o my_profile `which python` benchmark.py
# can add `--stats true`
```

Then, execute `nsight-sys` to open the profile results (my_profile.qdrep).

![image](https://user-images.githubusercontent.com/1928522/141221297-2ff5224b-e99b-4fe6-af7d-69452141d71d.png)

With this feature, a bug in cuCIM [v21.10.01](https://github.com/rapidsai/cucim/wiki/release_notes_v21.10.01) (thread contention in Cache) was found and fixed ([#145](https://github.com/rapidsai/cucim/pull/145)).

### 3. Normalize operation

CUDA-based normalization operation is added. Normalization supports the following types.

1. Simple range based normalization
2. Arctangent based normalization

Arctangent-based normalization helps to stretch lower intensity pixels in the image slightly better than range-based normalization. If you look at its [graph](https://mathworld.wolfram.com/InverseTangent.html), there is a huge variation at certain lower intensities, but as intensities become higher, the curve becomes flatter. This helps in isolating regions like lungs (and regions within lungs) more efficiently. There can be separate use cases depending on the modality and the application.

Please check the [test cases](https://github.com/rapidsai/cucim/blob/branch-21.12/python/cucim/src/cucim/core/operations/intensity/tests/test_normalize.py) to see how you can use the operation.


## 🐛 Bug Fixes

- Load libcufile.so with RTLD_NODELETE flag ([#177](https://github.com/rapidsai/cucim/pull/177)) [[@gigony](https://github.com/gigony)](https://github.com/gigony](https://github.com/gigony))
- Remove rmm/nvcc dependencies to fix cudaErrorUnsupportedPtxVersion error ([#175](https://github.com/rapidsai/cucim/pull/175)) [[@gigony](https://github.com/gigony)](https://github.com/gigony](https://github.com/gigony))
- Do not compile code with nvcc if no CUDA kernel exists ([#171](https://github.com/rapidsai/cucim/pull/171)) [[@gigony](https://github.com/gigony)](https://github.com/gigony](https://github.com/gigony))
- Fix a segmentation fault due to unloaded libcufile ([#158](https://github.com/rapidsai/cucim/pull/158)) [[@gigony](https://github.com/gigony)](https://github.com/gigony](https://github.com/gigony))
- Fix thread contention in Cache ([#145](https://github.com/rapidsai/cucim/pull/145)) [[@gigony](https://github.com/gigony)](https://github.com/gigony](https://github.com/gigony))
- Build with NumPy 1.17 ([#148](https://github.com/rapidsai/cucim/pull/148)) [[@jakirkham](https://github.com/jakirkham)](https://github.com/jakirkham](https://github.com/jakirkham))

## 📖 Documentation

- Add Jupyter notebook for SVS Support ([#147](https://github.com/rapidsai/cucim/pull/147)) [[@gigony](https://github.com/gigony)](https://github.com/gigony](https://github.com/gigony))
- Update change log for v21.10.01 ([#142](https://github.com/rapidsai/cucim/pull/142)) [[@gigony](https://github.com/gigony)](https://github.com/gigony](https://github.com/gigony))
- update docs theme to pydata-sphinx-theme ([#138](https://github.com/rapidsai/cucim/pull/138)) [[@quasiben](https://github.com/quasiben)](https://github.com/quasiben](https://github.com/quasiben))
- Update Github links in README.md through script ([#132](https://github.com/rapidsai/cucim/pull/132)) [[@gigony](https://github.com/gigony)](https://github.com/gigony](https://github.com/gigony))
- Fix GDS link in Jupyter notebook ([#131](https://github.com/rapidsai/cucim/pull/131)) [[@gigony](https://github.com/gigony)](https://github.com/gigony](https://github.com/gigony))
- Update notebook for the interoperability with DALI ([#127](https://github.com/rapidsai/cucim/pull/127)) [[@gigony](https://github.com/gigony)](https://github.com/gigony](https://github.com/gigony))



## 🛠️ Improvements

- Update `conda` recipes for Enhanced Compatibility effort by ([#164](https://github.com/rapidsai/cucim/pull/164)) [[@ajschmidt8](https://github.com/ajschmidt8)](https://github.com/ajschmidt8](https://github.com/ajschmidt8))
- Fix Changelog Merge Conflicts for `branch-21.12` ([#156](https://github.com/rapidsai/cucim/pull/156)) [[@ajschmidt8](https://github.com/ajschmidt8)](https://github.com/ajschmidt8](https://github.com/ajschmidt8))
- Add cucim.kit.cumed plugin with skeleton ([#129](https://github.com/rapidsai/cucim/pull/129)) [[@gigony](https://github.com/gigony)](https://github.com/gigony](https://github.com/gigony))
- Update initial cpp unittests ([#128](https://github.com/rapidsai/cucim/pull/128)) [[@gigony](https://github.com/gigony)](https://github.com/gigony](https://github.com/gigony))
- Optimize zoom out implementation with separate padding kernel ([#125](https://github.com/rapidsai/cucim/pull/125)) [[@chirayuG-nvidia](https://github.com/chirayuG-nvidia)](https://github.com/chirayuG-nvidia](https://github.com/chirayuG-nvidia))
- Do not force install linux-64 version of openslide-python ([#124](https://github.com/rapidsai/cucim/pull/124)) [[@Ethyling](https://github.com/Ethyling)](https://github.com/Ethyling](https://github.com/Ethyling))

# cuCIM 21.10.00 (7 Oct 2021)

## 🐛 Bug Fixes

- fix failing regionprops test cases ([#110](https://github.com/rapidsai/cucim/pull/110)) [@grlee77](https://github.com/grlee77)

## 📖 Documentation

- Forward-merge branch-21.08 to branch-21.10 ([#88](https://github.com/rapidsai/cucim/pull/88)) [@jakirkham](https://github.com/jakirkham)
- Update PyPI cuCIM v21.08.01 README.md and CHANGELOG.md ([#87](https://github.com/rapidsai/cucim/pull/87)) [@gigony](https://github.com/gigony)

## 🚀 New Features

- Support raw RGB tiled TIFF ([#108](https://github.com/rapidsai/cucim/pull/108)) [@gigony](https://github.com/gigony)
- Add a mechanism for user to know the availability of cucim.CuImage ([#107](https://github.com/rapidsai/cucim/pull/107)) [@gigony](https://github.com/gigony)
- Enable GDS and Support Runtime Context (__enter__, __exit__) for CuFileDriver and CuImage ([#106](https://github.com/rapidsai/cucim/pull/106)) [@gigony](https://github.com/gigony)
- Add transforms for Digital Pathology ([#100](https://github.com/rapidsai/cucim/pull/100)) [@shekhardw](https://github.com/shekhardw)

## 🛠️ Improvements

- ENH Replace gpuci_conda_retry with gpuci_mamba_retry ([#69](https://github.com/rapidsai/cucim/pull/69)) [@dillon-cullinan](https://github.com/dillon-cullinan)

# cuCIM 21.08.00 (4 Aug 2021)

## 🐛 Bug Fixes

- Remove int-type bug on Windows in skimage.measure.label ([#72](https://github.com/rapidsai/cucim/pull/72)) [@grlee77](https://github.com/grlee77)
- Fix missing array interface for associated_image() ([#65](https://github.com/rapidsai/cucim/pull/65)) [@gigony](https://github.com/gigony)
- Handle zero-padding version string ([#59](https://github.com/rapidsai/cucim/pull/59)) [@gigony](https://github.com/gigony)
- Remove invalid conda environment activation ([#58](https://github.com/rapidsai/cucim/pull/58)) [@ajschmidt8](https://github.com/ajschmidt8)

## 📖 Documentation

- Fix a typo in cache document ([#66](https://github.com/rapidsai/cucim/pull/66)) [@gigony](https://github.com/gigony)

## 🚀 New Features

- Pin `isort` hook to 5.6.4 ([#73](https://github.com/rapidsai/cucim/pull/73)) [@charlesbluca](https://github.com/charlesbluca)
- Add skimage.morphology.thin ([#27](https://github.com/rapidsai/cucim/pull/27)) [@grlee77](https://github.com/grlee77)

## 🛠️ Improvements

- Add SciPy 2021 to README ([#79](https://github.com/rapidsai/cucim/pull/79)) [@jakirkham](https://github.com/jakirkham)
- Use more descriptive ElementwiseKernel names in cucim.skimage ([#75](https://github.com/rapidsai/cucim/pull/75)) [@grlee77](https://github.com/grlee77)
- Add initial Python unit/performance tests for TIFF loader module ([#62](https://github.com/rapidsai/cucim/pull/62)) [@gigony](https://github.com/gigony)
- Fix `21.08` forward-merge conflicts ([#57](https://github.com/rapidsai/cucim/pull/57)) [@ajschmidt8](https://github.com/ajschmidt8)

# cuCIM 21.06.00 (9 Jun 2021)

## 🐛 Bug Fixes

- Update `update-version.sh` ([#42](https://github.com/rapidsai/cucim/pull/42)) [@ajschmidt8](https://github.com/ajschmidt8)

## 🛠️ Improvements

- Update environment variable used to determine `cuda_version` ([#43](https://github.com/rapidsai/cucim/pull/43)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update version script to remove bump2version dependency ([#41](https://github.com/rapidsai/cucim/pull/41)) [@gigony](https://github.com/gigony)
- Update changelog ([#40](https://github.com/rapidsai/cucim/pull/40)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update docs build script ([#39](https://github.com/rapidsai/cucim/pull/39)) [@ajschmidt8](https://github.com/ajschmidt8)

# cuCIM 0.19.0 (15 Apr 2021)

Initial release of cuCIM including cuClaraImage and [cupyimg](https://github.com/mritools/cupyimg).
