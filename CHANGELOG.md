# cuCIM 22.10.00 (Date TBD)

Please see https://github.com/rapidsai/cucim/releases/tag/v22.10.00a for the latest changes to this development branch.

# cuCIM 22.08.00 (17 Aug 2022)

## 🚨 Breaking Changes

- Stain extraction: use a less strict condition across channels when thresholding ([#316](https://github.com/rapidsai/cucim/pull/316)) [@grlee77](https://github.com/grlee77)

## 🐛 Bug Fixes

- create SimilarityTransform using CuPy 9.x-compatible indexing ([#365](https://github.com/rapidsai/cucim/pull/365)) [@grlee77](https://github.com/grlee77)
- Add `__init__.py` in `cucim.core` ([#359](https://github.com/rapidsai/cucim/pull/359)) [@jakirkham](https://github.com/jakirkham)
- Stain extraction: use a less strict condition across channels when thresholding ([#316](https://github.com/rapidsai/cucim/pull/316)) [@grlee77](https://github.com/grlee77)
- Incorporate bug fixes from skimage 0.19.3 ([#312](https://github.com/rapidsai/cucim/pull/312)) [@grlee77](https://github.com/grlee77)
- fix RawKernel bug for canny filter when quantiles are used ([#310](https://github.com/rapidsai/cucim/pull/310)) [@grlee77](https://github.com/grlee77)

## 📖 Documentation

- Defer loading of `custom.js` ([#383](https://github.com/rapidsai/cucim/pull/383)) [@galipremsagar](https://github.com/galipremsagar)
- add cucim.core.morphology to API docs + other docstring fixes ([#367](https://github.com/rapidsai/cucim/pull/367)) [@grlee77](https://github.com/grlee77)
- Update README.md ([#361](https://github.com/rapidsai/cucim/pull/361)) [@HesAnEasyCoder](https://github.com/HesAnEasyCoder)
- remove unimplemented functions from See Also and fix version numbers in deprecation warnings ([#356](https://github.com/rapidsai/cucim/pull/356)) [@grlee77](https://github.com/grlee77)
- Forward-merge branch-22.06 to branch-22.08 ([#344](https://github.com/rapidsai/cucim/pull/344)) [@grlee77](https://github.com/grlee77)
- Update README.md ([#315](https://github.com/rapidsai/cucim/pull/315)) [@HesAnEasyCoder](https://github.com/HesAnEasyCoder)
- Update index.rst ([#314](https://github.com/rapidsai/cucim/pull/314)) [@HesAnEasyCoder](https://github.com/HesAnEasyCoder)
- Update PyPI package documentation for v22.06.00 ([#311](https://github.com/rapidsai/cucim/pull/311)) [@gigony](https://github.com/gigony)

## 🚀 New Features

- Add segmentation with the Chan-Vese active contours method ([#343](https://github.com/rapidsai/cucim/pull/343)) [@grlee77](https://github.com/grlee77)
- Add cucim.skimage.morphology.medial_axis ([#342](https://github.com/rapidsai/cucim/pull/342)) [@grlee77](https://github.com/grlee77)
- Add cucim.skimage.segmentation.expand_labels ([#341](https://github.com/rapidsai/cucim/pull/341)) [@grlee77](https://github.com/grlee77)
- Add Euclidean distance transform for images/volumes ([#318](https://github.com/rapidsai/cucim/pull/318)) [@grlee77](https://github.com/grlee77)

## 🛠️ Improvements

- Revert &quot;Allow CuPy 11&quot; ([#362](https://github.com/rapidsai/cucim/pull/362)) [@galipremsagar](https://github.com/galipremsagar)
- Fix issues with day &amp; night modes in python docs ([#360](https://github.com/rapidsai/cucim/pull/360)) [@galipremsagar](https://github.com/galipremsagar)
- Allow CuPy 11 ([#357](https://github.com/rapidsai/cucim/pull/357)) [@jakirkham](https://github.com/jakirkham)
- more efficient separable convolution ([#355](https://github.com/rapidsai/cucim/pull/355)) [@grlee77](https://github.com/grlee77)
- Support resolution and spacing metadata ([#349](https://github.com/rapidsai/cucim/pull/349)) [@gigony](https://github.com/gigony)
- Performance optimizations to morphological segmentation functions ([#340](https://github.com/rapidsai/cucim/pull/340)) [@grlee77](https://github.com/grlee77)
- benchmarks: avoid use of deprecated pandas API ([#339](https://github.com/rapidsai/cucim/pull/339)) [@grlee77](https://github.com/grlee77)
- Reduce memory overhead and improve performance of normalize_colors_pca ([#328](https://github.com/rapidsai/cucim/pull/328)) [@grlee77](https://github.com/grlee77)
- Protect against obscure divide by zero error in edge case of `normalize_colors_pca` ([#327](https://github.com/rapidsai/cucim/pull/327)) [@grlee77](https://github.com/grlee77)
- complete parametrization of cucim.skimage benchmarks ([#324](https://github.com/rapidsai/cucim/pull/324)) [@grlee77](https://github.com/grlee77)
- parameterization of `filters` and `features` benchmarks (v2) ([#322](https://github.com/rapidsai/cucim/pull/322)) [@grlee77](https://github.com/grlee77)
- Add a fast histogram-based median filter ([#317](https://github.com/rapidsai/cucim/pull/317)) [@grlee77](https://github.com/grlee77)
- Remove custom compiler environment variables ([#307](https://github.com/rapidsai/cucim/pull/307)) [@ajschmidt8](https://github.com/ajschmidt8)

# cuCIM 22.06.00 (7 Jun 2022)

## 🚨 Breaking Changes

- Promote small integer types to single rather than double precision ([#278](https://github.com/rapidsai/cucim/pull/278)) [@grlee77](https://github.com/grlee77)

## 🐛 Bug Fixes

- Populate correct channel names for RGBA image ([#294](https://github.com/rapidsai/cucim/pull/294)) [@gigony](https://github.com/gigony)
- Merge branch-22.04 into branch-22.06 ([#258](https://github.com/rapidsai/cucim/pull/258)) [@jakirkham](https://github.com/jakirkham)

## 📖 Documentation

- update outdated links to example data ([#289](https://github.com/rapidsai/cucim/pull/289)) [@grlee77](https://github.com/grlee77)
- Add missing API docs ([#275](https://github.com/rapidsai/cucim/pull/275)) [@grlee77](https://github.com/grlee77)

## 🚀 New Features

- add missing `cucim.skimage.segmentation.clear_border` function ([#267](https://github.com/rapidsai/cucim/pull/267)) [@grlee77](https://github.com/grlee77)
- add `cucim.core.operations.color.stain_extraction_pca` and `cucim.core.operations.color.normalize_colors_pca` for digital pathology H&E stain extraction and normalization ([#273](https://github.com/rapidsai/cucim/pull/273)) [@grlee77](https://github.com/grlee77), [@drbeh](https://github.com/drbeh)

## 🛠️ Improvements

- Update to use DLPack v0.6 ([#295](https://github.com/rapidsai/cucim/pull/295)) [@gigony](https://github.com/gigony)
- Remove plugin-related messages temporarily ([#291](https://github.com/rapidsai/cucim/pull/291)) [@gigony](https://github.com/gigony)
- Simplify recipes ([#286](https://github.com/rapidsai/cucim/pull/286)) [@Ethyling](https://github.com/Ethyling)
- Use cupy.fuse to improve efficiency hessian_matrix_eigvals ([#280](https://github.com/rapidsai/cucim/pull/280)) [@grlee77](https://github.com/grlee77)
- Promote small integer types to single rather than double precision ([#278](https://github.com/rapidsai/cucim/pull/278)) [@grlee77](https://github.com/grlee77)
- improve efficiency of histogram-based thresholding functions ([#276](https://github.com/rapidsai/cucim/pull/276)) [@grlee77](https://github.com/grlee77)
- Remove unused dependencies in GPU tests job ([#268](https://github.com/rapidsai/cucim/pull/268)) [@Ethyling](https://github.com/Ethyling)
- Enable footprint decomposition for morphology ([#274](https://github.com/rapidsai/cucim/pull/274)) [@grlee77](https://github.com/grlee77)
- Use conda compilers ([#232](https://github.com/rapidsai/cucim/pull/232)) [@Ethyling](https://github.com/Ethyling)
- Build packages using mambabuild ([#216](https://github.com/rapidsai/cucim/pull/216)) [@Ethyling](https://github.com/Ethyling)

# cuCIM 22.04.00 (6 Apr 2022)

## 🚨 Breaking Changes

- Apply fixes to skimage.transform scheduled for scikit-image 0.19.2 ([#208](https://github.com/rapidsai/cucim/pull/208)) [@grlee77](https://github.com/grlee77)

## 🐛 Bug Fixes

- Fix ImportError from vendored code ([#252](https://github.com/rapidsai/cucim/pull/252)) [@grlee77](https://github.com/grlee77)
- Fix wrong dimension in metadata ([#248](https://github.com/rapidsai/cucim/pull/248)) [@gigony](https://github.com/gigony)
- Handle file descriptor ownership and update documents for GDS ([#234](https://github.com/rapidsai/cucim/pull/234)) [@gigony](https://github.com/gigony)
- Check nullptr of handler in CuFileDriver::close() ([#229](https://github.com/rapidsai/cucim/pull/229)) [@gigony](https://github.com/gigony)
- Fix docs builds ([#218](https://github.com/rapidsai/cucim/pull/218)) [@ajschmidt8](https://github.com/ajschmidt8)
- Apply fixes to skimage.transform scheduled for scikit-image 0.19.2 ([#208](https://github.com/rapidsai/cucim/pull/208)) [@grlee77](https://github.com/grlee77)

## 📖 Documentation

- Update PyPI cuCIM v22.02.01 CHANGELOG.md ([#249](https://github.com/rapidsai/cucim/pull/249)) [@gigony](https://github.com/gigony)
- Update GTC 2021 Spring video links ([#227](https://github.com/rapidsai/cucim/pull/227)) [@gigony](https://github.com/gigony)
- Update documents for v22.02.00 ([#226](https://github.com/rapidsai/cucim/pull/226)) [@gigony](https://github.com/gigony)
- Merge branch-22.02 into branch-22.04 ([#220](https://github.com/rapidsai/cucim/pull/220)) [@jakirkham](https://github.com/jakirkham)

## 🛠️ Improvements

- Expose data type of CuImage object for interoperability with NumPy ([#246](https://github.com/rapidsai/cucim/pull/246)) [@gigony](https://github.com/gigony)
- Temporarily disable new `ops-bot` functionality ([#239](https://github.com/rapidsai/cucim/pull/239)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add `.github/ops-bot.yaml` config file ([#236](https://github.com/rapidsai/cucim/pull/236)) [@ajschmidt8](https://github.com/ajschmidt8)
- randomization per image per batch ([#231](https://github.com/rapidsai/cucim/pull/231)) [@shekhardw](https://github.com/shekhardw)

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

- Update cucim.skimage API to match scikit-image 0.19 ([#190](https://github.com/rapidsai/cucim/pull/190)) [@glee77](https://github.com/glee77)
- Support multi-threads and batch, and support nvJPEG for JPEG-compressed images ([#191](https://github.com/rapidsai/cucim/pull/191)) [@gigony](https://github.com/gigony)
- Allow CuPy 10 ([#195](https://github.com/rapidsai/cucim/pull/195)) [@jakikham](https://github.com/jakikham)

## 🛠️ Improvements

- Add missing imports tests ([#183](https://github.com/rapidsai/cucim/pull/183)) [@Ethyling](https://github.com/Ethyling)
- Allow installation with CuPy 10 ([#197](https://github.com/rapidsai/cucim/pull/197)) [@glee77](https://github.com/glee77)
- Upgrade Numpy to 1.18 for Python 3.9 support ([#196](https://github.com/rapidsai/cucim/pull/196)) [@Ethyling](https://github.com/Ethyling)
- Upgrade Numpy to 1.19 for Python 3.9 support ([#203](https://github.com/rapidsai/cucim/pull/203)) [@Ethyling](https://github.com/Ethyling)

# cuCIM 21.12.00 (9 Dec 2021)

## 🚀 New Features

- Support Aperio SVS with CPU LZW and jpeg2k decoder ([#141](https://github.com/rapidsai/cucim/pull/141)) [[@gigony](https://github.com/gigony)](https://github.com/gigony](https://github.com/gigony))
- Add NVTX support for performance analysis ([#144](https://github.com/rapidsai/cucim/pull/144)) [[@gigony](https://github.com/gigony)](https://github.com/gigony](https://github.com/gigony))
- Normalize operation ([#150](https://github.com/rapidsai/cucim/pull/150)) [[@shekhardw](https://github.com/shekhardw)](https://github.com/shekhardw](https://github.com/shekhardw))

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

- Initial release of cuCIM including cuClaraImage and [cupyimg](https://github.com/mritools/cupyimg).