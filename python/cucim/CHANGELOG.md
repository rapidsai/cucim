
# Changelog (See [Release Notes](https://github.com/rapidsai/cucim/wiki/Release-Notes))

## [22.08.01](https://github.com/rapidsai/cucim/wiki/release_notes_v22.08.01)

- Euclidean distance transform: fix bad choice of block parameters ([#393](https://github.com/rapidsai/cucim/pull/393)) [@grlee77](https://github.com/grlee77))

## [22.08.00](https://github.com/rapidsai/cucim/wiki/release_notes_v22.08.00)

## 🚨 Breaking Changes

- Stain extraction: use a less strict condition across channels when thresholding ([#316](https://github.com/rapidsai/cucim/pull/278)) [@grlee77](https://github.com/grlee77)

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
- Allow CuPy 11 ([#357](https://github.com/rapidsai/cucim/pull/357)) [@jakirkham](https://github.com/grlee77)
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

## [22.02.06](https://github.com/rapidsai/cucim/wiki/release_notes_v22.06.00)

- [Update/Breaking] Promote small integer types to single rather than double precision ([#278](https://github.com/rapidsai/cucim/pull/278)) [@grlee77](https://github.com/grlee77)
- [Bug] Populate correct channel names for RGBA image ([#294](https://github.com/rapidsai/cucim/pull/294)) [@gigony](https://github.com/gigony)
- [Bug] Merge branch-22.04 into branch-22.06 ([#258](https://github.com/rapidsai/cucim/pull/258)) [@jakirkham](https://github.com/jakirkham)
- [New] add missing `cucim.skimage.segmentation.clear_border` function ([#267](https://github.com/rapidsai/cucim/pull/267)) [@grlee77](https://github.com/grlee77)
- [New] add `cucim.core.operations.color.stain_extraction_pca` and `cucim.core.operations.color.normalize_colors_pca` for digital pathology H&E stain extraction and normalization ([#273](https://github.com/rapidsai/cucim/pull/273)) [@grlee77](https://github.com/grlee77), [@drbeh](https://github.com/drbeh)
- [Update] Update to use DLPack v0.6 ([#295](https://github.com/rapidsai/cucim/pull/295)) [@gigony](https://github.com/gigony)
- [Update] Remove plugin-related messages temporarily ([#291](https://github.com/rapidsai/cucim/pull/291)) [@gigony](https://github.com/gigony)
- [Update] Simplify recipes ([#286](https://github.com/rapidsai/cucim/pull/286)) [@Ethyling](https://github.com/Ethyling)
- [Update] Use cupy.fuse to improve efficiency hessian_matrix_eigvals ([#280](https://github.com/rapidsai/cucim/pull/280)) [@grlee77](https://github.com/grlee77)
- [Update] improve efficiency of histogram-based thresholding functions ([#276](https://github.com/rapidsai/cucim/pull/276)) [@grlee77](https://github.com/grlee77)
- [Update] Remove unused dependencies in GPU tests job ([#268](https://github.com/rapidsai/cucim/pull/268)) [@Ethyling](https://github.com/Ethyling)
- [Update] Enable footprint decomposition for morphology ([#274](https://github.com/rapidsai/cucim/pull/274)) [@grlee77](https://github.com/grlee77)
- [Update] Use conda compilers ([#232](https://github.com/rapidsai/cucim/pull/232)) [@Ethyling](https://github.com/Ethyling)
- [Update] Build packages using mambabuild ([#216](https://github.com/rapidsai/cucim/pull/216)) [@Ethyling](https://github.com/Ethyling)
- [Doc] update outdated links to example data ([#289](https://github.com/rapidsai/cucim/pull/289)) [@grlee77](https://github.com/grlee77)
- [Doc] Add missing API docs ([#275](https://github.com/rapidsai/cucim/pull/275)) [@grlee77](https://github.com/grlee77)

## [22.02.04](https://github.com/rapidsai/cucim/wiki/release_notes_v22.04.00)

- [Bug] Fix ImportError from vendored code ([#252](https://github.com/rapidsai/cucim/pull/252)) [@grlee77](https://github.com/grlee77)
- [Bug] Fix wrong dimension in metadata ([#248](https://github.com/rapidsai/cucim/pull/248)) [@gigony](https://github.com/gigony)

## [22.02.01](https://github.com/rapidsai/cucim/wiki/release_notes_v22.02.01)

- [Bug] Check nullptr of handler in CuFileDriver::close() ([#229](https://github.com/rapidsai/cucim/pull/229)) [@gigony](https://github.com/gigony)
- [Bug] Handle file descriptor ownership and update documents for GDS ([#234](https://github.com/rapidsai/cucim/pull/234)) [@gigony](https://github.com/gigony)
- [Bug] Apply fixes to skimage.transform scheduled for scikit-image 0.19.2 ([#208](https://github.com/rapidsai/cucim/pull/208)) [@grlee7](https://github.com/grlee7)
- [New] Randomization of transforms per image per batch ([#231](https://github.com/rapidsai/cucim/pull/231)) [@shekhardw](https://github.com/shekhardw)
- [New] Expose data type of CuImage object for interoperability with NumPy ([#246](https://github.com/rapidsai/cucim/pull/246)) [@gigony](https://github.com/gigony)
- [Update] Remove verbose plugin messages temporarily. Address [#109](https://github.com/rapidsai/cucim/issues/109) ([BUG] - Info messages appearing as warnings in Jupyter notebooks)
- [Doc] Fix docs builds ([#218](https://github.com/rapidsai/cucim/pull/218)) [@ajschmidt8](https://github.com/ajschmidt8)
- [Doc] Update GTC 2021 Spring video links ([#227](https://github.com/rapidsai/cucim/pull/227)) [@gigony](https://github.com/gigony)
- [Doc] Update documents for v22.02.00 ([#226](https://github.com/rapidsai/cucim/pull/226)) [@gigony](https://github.com/gigony)

## [22.02.00](https://github.com/rapidsai/cucim/wiki/release_notes_v22.02.00)

- [New/Breaking] Update cucim.skimage API to match scikit-image 0.19 ([#190](https://github.com/rapidsai/cucim/pull/190)) [@glee77](https://github.com/glee77)
- [Bug] Fix a bug in [v21.12.01](https://github.com/rapidsai/cucim/wiki/release_notes_v21.12.01) ([#191](https://github.com/rapidsai/cucim/pull/191)) [@gigony](https://github.com/gigony)
  - Fix GPU memory leak when using nvJPEG API (when `device='cuda'` parameter is used in `read_region` method).
- [Bug] Fix segfault for preferred_memory_capacity in Python 3.9+ ([#214](https://github.com/rapidsai/cucim/pull/214)) [@gigony](https://github.com/gigony)
- [Doc] PyPI v21.12.00 release ([#182](https://github.com/rapidsai/cucim/pull/182)) [@gigony](https://github.com/gigony)
- [New] Allow CuPy 10 ([#195](https://github.com/rapidsai/cucim/pull/195)) [@jakikham](https://github.com/jakikham)
- [New] Support multi-threads and batch, and support nvJPEG for JPEG-compressed images ([#191](https://github.com/rapidsai/cucim/pull/191)) [@gigony](https://github.com/gigony)
- [New] Update cucim.skimage API to match scikit-image 0.19 ([#190](https://github.com/rapidsai/cucim/pull/190)) [@glee77](https://github.com/glee77)
- [Update] Add missing imports tests ([#183](https://github.com/rapidsai/cucim/pull/183)) [@Ethyling](https://github.com/Ethyling)
- [Update] Allow installation with CuPy 10 ([#197](https://github.com/rapidsai/cucim/pull/197)) [@glee77](https://github.com/glee77)
- [Update] Upgrade Numpy to 1.18 for Python 3.9 support ([#196](https://github.com/rapidsai/cucim/pull/196)) [@Ethyling](https://github.com/Ethyling)
- [Update] Upgrade Numpy to 1.19 for Python 3.9 support ([#203](https://github.com/rapidsai/cucim/pull/203)) [@Ethyling](https://github.com/Ethyling)

## [21.12.01](https://github.com/rapidsai/cucim/wiki/release_notes_v21.12.01)

- [New] Supporting Multithreading and Batch Processing ([#149](https://github.com/rapidsai/cucim/issues/149)) [@gigony](https://github.com/gigony)

## [21.12.00](https://github.com/rapidsai/cucim/wiki/release_notes_v21.12.00)

- [New] Support Aperio SVS with CPU LZW and jpeg2k decoder ([#141](https://github.com/rapidsai/cucim/pull/141)) [@gigony](https://github.com/gigony)
- [New] Add NVTX support for performance analysis ([#144](https://github.com/rapidsai/cucim/pull/144)) [@gigony](https://github.com/gigony)
- [New] Normalize operation ([#150](https://github.com/rapidsai/cucim/pull/150)) [@shekhardw](https://github.com/shekhardw)
- [Bug] Load libcufile.so with RTLD_NODELETE flag ([#177](https://github.com/rapidsai/cucim/pull/177)) [@gigony](https://github.com/gigony)
- [Bug] Remove rmm/nvcc dependencies to fix cudaErrorUnsupportedPtxVersion error ([#175](https://github.com/rapidsai/cucim/pull/175)) [@gigony](https://github.com/gigony)
- [Bug] Do not compile code with nvcc if no CUDA kernel exists ([#171](https://github.com/rapidsai/cucim/pull/171)) [@gigony](https://github.com/gigony)
- [Bug] Fix a segmentation fault due to unloaded libcufile ([#158](https://github.com/rapidsai/cucim/pull/158)) [@gigony](https://github.com/gigony)
- [Bug] Fix thread contention in Cache ([#145](https://github.com/rapidsai/cucim/pull/145)) [@gigony](https://github.com/gigony)
- [Bug] Build with NumPy 1.17 ([#148](https://github.com/rapidsai/cucim/pull/148)) [@jakirkham](https://github.com/jakirkham)
- [Doc] Add Jupyter notebook for SVS Support ([#147](https://github.com/rapidsai/cucim/pull/147)) [@gigony](https://github.com/gigony)
- [Doc] Update change log for v21.10.01 ([#142](https://github.com/rapidsai/cucim/pull/142)) [@gigony](https://github.com/gigony)
- [Doc] update docs theme to pydata-sphinx-theme ([#138](https://github.com/rapidsai/cucim/pull/138)) [@quasiben](https://github.com/quasiben)
- [Doc] Update Github links in README.md through script ([#132](https://github.com/rapidsai/cucim/pull/132)) [@gigony](https://github.com/gigony)
- [Doc] Fix GDS link in Jupyter notebook ([#131](https://github.com/rapidsai/cucim/pull/131)) [@gigony](https://github.com/gigony)
- [Doc] Update notebook for the interoperability with DALI ([#127](https://github.com/rapidsai/cucim/pull/127)) [@gigony](https://github.com/gigony)
- [Update] Update `conda` recipes for Enhanced Compatibility effort by ([#164](https://github.com/rapidsai/cucim/pull/164)) [@ajschmidt8](https://github.com/ajschmidt8)
- [Update] Fix Changelog Merge Conflicts for `branch-21.12` ([#156](https://github.com/rapidsai/cucim/pull/156)) [@ajschmidt8](https://github.com/ajschmidt8)
- [Update] Add cucim.kit.cumed plugin with skeleton ([#129](https://github.com/rapidsai/cucim/pull/129)) [@gigony](https://github.com/gigony)
- [Update] Update initial cpp unittests ([#128](https://github.com/rapidsai/cucim/pull/128)) [@gigony](https://github.com/gigony)
- [Update] Optimize zoom out implementation with separate padding kernel ([#125](https://github.com/rapidsai/cucim/pull/125)) [@chirayuG-nvidia](https://github.com/chirayuG-nvidia)
- [Update] Do not force install linux-64 version of openslide-python ([#124](https://github.com/rapidsai/cucim/pull/124)) [@Ethyling](https://github.com/Ethyling)

## [21.10.01](https://github.com/rapidsai/cucim/wiki/release_notes_v21.10.01)

- [New] Support Aperio SVS with CPU LZW and jpeg2k decoder ([#141](https://github.com/rapidsai/cucim/pull/141))

## [21.10.00](https://github.com/rapidsai/cucim/wiki/release_notes_v21.10.00)

- [New] Add transforms for Digital Pathology ([#100](https://github.com/rapidsai/cucim/pull/100)) [@shekhardw](https://github.com/shekhardw) [@chirayuG-nvidia](https://github.com/chirayuG-nvidia)
- [New] Enable GDS and Support Runtime Context (__enter__, __exit__) for CuFileDriver and CuImage ([#106](https://github.com/rapidsai/cucim/pull/106)) [@gigony](https://github.com/gigony)
- [New] Add a mechanism for user to know the availability of cucim.CuImage ([#107](https://github.com/rapidsai/cucim/pull/107)) [@gigony](https://github.com/gigony)
- [New] Support raw RGB tiled TIFF ([#108](https://github.com/rapidsai/cucim/pull/108)) [@gigony](https://github.com/gigony)
- [Bug] fix failing regionprops test cases ([#110](https://github.com/rapidsai/cucim/pull/110)) [@grlee77](https://github.com/grlee77)
- [Doc] Forward-merge branch-21.08 to branch-21.10 ([#88](https://github.com/rapidsai/cucim/pull/88)) [@jakirkham](https://github.com/jakirkham)
- [Doc] Update PyPI cuCIM v21.08.01 README.md and CHANGELOG.md ([#87](https://github.com/rapidsai/cucim/pull/87)) [@gigony](https://github.com/gigony)
- [Update] ENH Replace gpuci_conda_retry with gpuci_mamba_retry ([#69](https://github.com/rapidsai/cucim/pull/69)) [@dillon-cullinan](https://github.com/dillon-cullinan)

## [21.08.02](https://github.com/rapidsai/cucim/wiki/release_notes_v21.08.02)

- [New] Add transforms for Digital Pathology ([#100](https://github.com/rapidsai/cucim/pull/100)) [@shekhardw](https://github.com/shekhardw) [@chirayuG-nvidia](https://github.com/chirayuG-nvidia)

## [21.08.01](https://github.com/rapidsai/cucim/wiki/release_notes_v21.08.01)

- [New] Add skimage.morphology.thin ([#27](https://github.com/rapidsai/cucim/pull/27))
- [Bug] Fix missing `__array_interface__` for associated_image(): ([#48](https://github.com/rapidsai/cucim/pull/48), [#65](https://github.com/rapidsai/cucim/pull/65))
- [Testing] Added unit and performance tests for TIFF loaders ([#62](https://github.com/rapidsai/cucim/pull/62))
- [Bug] Fix Windows int-type Bug: ([#72](https://github.com/rapidsai/cucim/pull/72))
- [Update] Use more descriptive ElementwiseKernel names in cucim.skimage: ([#75](https://github.com/rapidsai/cucim/pull/75))

## [21.06.00](https://github.com/rapidsai/cucim/wiki/release_notes_v21.06.00)

- Implement cache mechanism
- Add `__cuda_array_interface`.
- Fix a memory leak in Deflate decoder.
