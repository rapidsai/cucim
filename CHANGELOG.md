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
