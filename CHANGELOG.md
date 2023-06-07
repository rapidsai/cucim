# cuCIM 23.06.00 (7 Jun 2023)

## 🚨 Breaking Changes

- Support Python 3.9 build/tests ([#547](https://github.com/rapidsai/cucim/pull/547)) [@shwina](https://github.com/shwina)

## 🐛 Bug Fixes

- Fix SHA256 check failure in test suite ([#564](https://github.com/rapidsai/cucim/pull/564)) [@grlee77](https://github.com/grlee77)
- Handle space character in ./run download_testdata ([#556](https://github.com/rapidsai/cucim/pull/556)) [@gigony](https://github.com/gigony)
- Fix `return_error=&#39;always&#39;` behavior in phase_cross_correlation ([#549](https://github.com/rapidsai/cucim/pull/549)) [@grlee77](https://github.com/grlee77)
- Only load versioned `libcufile` ([#548](https://github.com/rapidsai/cucim/pull/548)) [@jakirkham](https://github.com/jakirkham)
- add a 20 minute timeout for pytest runs on CI ([#545](https://github.com/rapidsai/cucim/pull/545)) [@grlee77](https://github.com/grlee77)
- protect against possible out of bounds memory access in 2D distance transform ([#540](https://github.com/rapidsai/cucim/pull/540)) [@grlee77](https://github.com/grlee77)

## 📖 Documentation

- Fix doc building via `run build_package` ([#553](https://github.com/rapidsai/cucim/pull/553)) [@grlee77](https://github.com/grlee77)
- update changelog for release 23.04.00 and 23.04.01 ([#552](https://github.com/rapidsai/cucim/pull/552)) [@grlee77](https://github.com/grlee77)

## 🛠️ Improvements

- Allow numpy 1.24. ([#563](https://github.com/rapidsai/cucim/pull/563)) [@bdice](https://github.com/bdice)
- run docs nightly too ([#560](https://github.com/rapidsai/cucim/pull/560)) [@AyodeAwe](https://github.com/AyodeAwe)
- Update cupy dependency ([#558](https://github.com/rapidsai/cucim/pull/558)) [@vyasr](https://github.com/vyasr)
- Remove libjpeg dependency ([#557](https://github.com/rapidsai/cucim/pull/557)) [@gigony](https://github.com/gigony)
- Enable sccache hits from local builds ([#551](https://github.com/rapidsai/cucim/pull/551)) [@AyodeAwe](https://github.com/AyodeAwe)
- Revert shared workflows branch ([#550](https://github.com/rapidsai/cucim/pull/550)) [@ajschmidt8](https://github.com/ajschmidt8)
- Support Python 3.9 build/tests ([#547](https://github.com/rapidsai/cucim/pull/547)) [@shwina](https://github.com/shwina)
- Remove usage of rapids-get-rapids-version-from-git ([#546](https://github.com/rapidsai/cucim/pull/546)) [@jjacobelli](https://github.com/jjacobelli)
- Use ARC V2 self-hosted runners for GPU jobs ([#538](https://github.com/rapidsai/cucim/pull/538)) [@jjacobelli](https://github.com/jjacobelli)
- Remove underscore in build string. ([#528](https://github.com/rapidsai/cucim/pull/528)) [@bdice](https://github.com/bdice)

# cuCIM 23.04.01 (14 Apr 2023)

## 🛠️ Improvements

- Pin libwebp-base ([#541](https://github.com/rapidsai/cucim/pull/541)) [@ajschmidt8](https://github.com/ajschmidt8)

# cuCIM 23.04.00 (6 Apr 2023)

## 🚨 Breaking Changes

- Fix inefficiency in handling clipping of image range in `resize` and other transforms ([#516](https://github.com/rapidsai/cucim/pull/516)) [@grlee77](https://github.com/grlee77)

## 🐛 Bug Fixes

- Fix bug in median filter with non-uniform footprint ([#521](https://github.com/rapidsai/cucim/pull/521)) [@grlee77](https://github.com/grlee77)
- use cp.around instead of cp.round for CuPy 10.x compatiblity ([#508](https://github.com/rapidsai/cucim/pull/508)) [@grlee77](https://github.com/grlee77)
- Fix error in LZ4-compressed Zarr writing demo ([#506](https://github.com/rapidsai/cucim/pull/506)) [@grlee77](https://github.com/grlee77)
- Normalize whitespace. ([#474](https://github.com/rapidsai/cucim/pull/474)) [@bdice](https://github.com/bdice)

## 🛠️ Improvements

- allow scikit-image 0.20 as well ([#536](https://github.com/rapidsai/cucim/pull/536)) [@grlee77](https://github.com/grlee77)
- Pass `AWS_SESSION_TOKEN` and `SCCACHE_S3_USE_SSL` vars to conda build ([#525](https://github.com/rapidsai/cucim/pull/525)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update aarch64 to GCC 11 ([#524](https://github.com/rapidsai/cucim/pull/524)) [@bdice](https://github.com/bdice)
- Update to GCC 11 ([#522](https://github.com/rapidsai/cucim/pull/522)) [@bdice](https://github.com/bdice)
- Upgrade dockcross and pybind11 ([#519](https://github.com/rapidsai/cucim/pull/519)) [@gigony](https://github.com/gigony)
- Binary morphology: omit weights array when possible ([#517](https://github.com/rapidsai/cucim/pull/517)) [@grlee77](https://github.com/grlee77)
- Fix inefficiency in handling clipping of image range in `resize` and other transforms ([#516](https://github.com/rapidsai/cucim/pull/516)) [@grlee77](https://github.com/grlee77)
- Fix GHA build workflow ([#515](https://github.com/rapidsai/cucim/pull/515)) [@AjayThorve](https://github.com/AjayThorve)
- Reduce error handling verbosity in CI tests scripts ([#511](https://github.com/rapidsai/cucim/pull/511)) [@AjayThorve](https://github.com/AjayThorve)
- Update shared workflow branches ([#510](https://github.com/rapidsai/cucim/pull/510)) [@ajschmidt8](https://github.com/ajschmidt8)
- Remove gpuCI scripts. ([#505](https://github.com/rapidsai/cucim/pull/505)) [@bdice](https://github.com/bdice)
- Move date to build string in `conda` recipe ([#497](https://github.com/rapidsai/cucim/pull/497)) [@ajschmidt8](https://github.com/ajschmidt8)

# cuCIM 23.02.00 (9 Feb 2023)

## 🚨 Breaking Changes

- Add disambiguation option to phase_cross_correlation (skimage 0.20 feature) ([#486](https://github.com/rapidsai/cucim/pull/486)) [@grlee77](https://github.com/grlee77)

## 🐛 Bug Fixes

- apply bug fix to vendored ndimage code ([#494](https://github.com/rapidsai/cucim/pull/494)) [@grlee77](https://github.com/grlee77)
- Closes #490 -- fixes bug in hue jitter ([#491](https://github.com/rapidsai/cucim/pull/491)) [@benlansdell](https://github.com/benlansdell)
- Fix random seed used in test_3d_similarity_estimation ([#472](https://github.com/rapidsai/cucim/pull/472)) [@grlee77](https://github.com/grlee77)

## 📖 Documentation

- Fix documentation author ([#475](https://github.com/rapidsai/cucim/pull/475)) [@bdice](https://github.com/bdice)

## 🚀 New Features

- Add colocalization measures ([#488](https://github.com/rapidsai/cucim/pull/488)) [@grlee77](https://github.com/grlee77)
- Add disambiguation option to phase_cross_correlation (skimage 0.20 feature) ([#486](https://github.com/rapidsai/cucim/pull/486)) [@grlee77](https://github.com/grlee77)

## 🛠️ Improvements

- Update shared workflow branches ([#501](https://github.com/rapidsai/cucim/pull/501)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update `isort` version to 5.12.0 ([#492](https://github.com/rapidsai/cucim/pull/492)) [@ajschmidt8](https://github.com/ajschmidt8)
- Improve rank filtering performance by removing use of footprint kernel when possible ([#485](https://github.com/rapidsai/cucim/pull/485)) [@grlee77](https://github.com/grlee77)
- use vendored version of cupy.pad with added performance optimizations ([#482](https://github.com/rapidsai/cucim/pull/482)) [@grlee77](https://github.com/grlee77)
- add docs builds to Github Actions ([#481](https://github.com/rapidsai/cucim/pull/481)) [@AjayThorve](https://github.com/AjayThorve)
- Update `numpy` version specifier ([#480](https://github.com/rapidsai/cucim/pull/480)) [@ajschmidt8](https://github.com/ajschmidt8)
- Build CUDA `11.8` and Python `3.10` Packages ([#476](https://github.com/rapidsai/cucim/pull/476)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add GitHub Actions Workflows. ([#471](https://github.com/rapidsai/cucim/pull/471)) [@bdice](https://github.com/bdice)
- Fix conflicts in &quot;Forward-merge branch-22.12 to branch-23.02&quot; ([#468](https://github.com/rapidsai/cucim/pull/468)) [@jakirkham](https://github.com/jakirkham)
- Enable copy_prs. ([#465](https://github.com/rapidsai/cucim/pull/465)) [@bdice](https://github.com/bdice)

# cuCIM 22.12.00 (8 Dec 2022)

## 🚨 Breaking Changes

- Implement additional deprecations carried out for scikit-image 0.20 ([#451](https://github.com/rapidsai/cucim/pull/451)) [@grlee77](https://github.com/grlee77)
- improved implementation of ridge filters (bug fixes and reduced memory footprint) ([#423](https://github.com/rapidsai/cucim/pull/423)) [@grlee77](https://github.com/grlee77)

## 🐛 Bug Fixes

- pin to cmake !3.25.0 on CI to avoid bug with CUDA+conda during build ([#444](https://github.com/rapidsai/cucim/pull/444)) [@grlee77](https://github.com/grlee77)
- update incorrect argument and deprecated function for tifffile.TiffWriter ([#433](https://github.com/rapidsai/cucim/pull/433)) [@JoohyungLee0106](https://github.com/JoohyungLee0106)
- Fix rotate behavior for ndim &gt; 2 ([#432](https://github.com/rapidsai/cucim/pull/432)) [@grlee77](https://github.com/grlee77)

## 📖 Documentation

- add whole-slide tiled read/write demos for measuring GPUDirect Storage (GDS) I/O performance ([#452](https://github.com/rapidsai/cucim/pull/452)) [@grlee77](https://github.com/grlee77)
- Add demo for distance_transform_edt ([#394](https://github.com/rapidsai/cucim/pull/394)) [@grlee77](https://github.com/grlee77)

## 🚀 New Features

- Support no-compression method in converter ([#443](https://github.com/rapidsai/cucim/pull/443)) [@gigony](https://github.com/gigony)
- add three segmentation metrics ([#425](https://github.com/rapidsai/cucim/pull/425)) [@grlee77](https://github.com/grlee77)
- add isotropic binary morphology functions ([#421](https://github.com/rapidsai/cucim/pull/421)) [@grlee77](https://github.com/grlee77)
- Add blob feature detectors (blob_dog, blob_log, blob_doh) ([#413](https://github.com/rapidsai/cucim/pull/413)) [@monzelr](https://github.com/monzelr)

## 🛠️ Improvements

- additional minor updates (skimage 0.20) ([#455](https://github.com/rapidsai/cucim/pull/455)) [@grlee77](https://github.com/grlee77)
- Implement additional deprecations carried out for scikit-image 0.20 ([#451](https://github.com/rapidsai/cucim/pull/451)) [@grlee77](https://github.com/grlee77)
- Faster `hessian_matrix_*` and `structure_tensor_eigvals` via analytical eigenvalues for the 3D case ([#434](https://github.com/rapidsai/cucim/pull/434)) [@grlee77](https://github.com/grlee77)
- use fused kernels to reduce overhead in corner detector implementations ([#426](https://github.com/rapidsai/cucim/pull/426)) [@grlee77](https://github.com/grlee77)
- Misc updates for consistency with scikit-image 0.20 ([#424](https://github.com/rapidsai/cucim/pull/424)) [@grlee77](https://github.com/grlee77)
- improved implementation of ridge filters (bug fixes and reduced memory footprint) ([#423](https://github.com/rapidsai/cucim/pull/423)) [@grlee77](https://github.com/grlee77)
- analytical moments computations, support pixel spacings in moments and regionprops ([#422](https://github.com/rapidsai/cucim/pull/422)) [@grlee77](https://github.com/grlee77)
- Forward merge branch-22.10 to branch-22.12 ([#420](https://github.com/rapidsai/cucim/pull/420)) [@grlee77](https://github.com/grlee77)
- Support `sampling` kwarg for `distance_transform_edt` (take pixel/voxel sizes into account) ([#407](https://github.com/rapidsai/cucim/pull/407)) [@grlee77](https://github.com/grlee77)
- Improve performance of Euclidean distance transform ([#406](https://github.com/rapidsai/cucim/pull/406)) [@grlee77](https://github.com/grlee77)

# cuCIM 22.10.00 (12 Oct 2022)

## 🐛 Bug Fixes

- Correctly use dtype when computing shared memory requirements of separable convolution ([#409](https://github.com/rapidsai/cucim/pull/409)) [@grlee77](https://github.com/grlee77)
- Forward-merge branch-22.08 to branch-22.10 ([#403](https://github.com/rapidsai/cucim/pull/403)) [@jakirkham](https://github.com/jakirkham)
- Add missing imports of euler_number and perimeter_crofton ([#386](https://github.com/rapidsai/cucim/pull/386)) [@grlee77](https://github.com/grlee77)

## 📖 Documentation

- update pypi CHANGELOG.md for releases 22.08.00 and 22.08.01 ([#404](https://github.com/rapidsai/cucim/pull/404)) [@grlee77](https://github.com/grlee77)
- Update README.md ([#396](https://github.com/rapidsai/cucim/pull/396)) [@HesAnEasyCoder](https://github.com/HesAnEasyCoder)

## 🚀 New Features

- Allow cupy 11 ([#399](https://github.com/rapidsai/cucim/pull/399)) [@galipremsagar](https://github.com/galipremsagar)
- Add cucim.skimage.feature.match_descriptors ([#338](https://github.com/rapidsai/cucim/pull/338)) [@grlee77](https://github.com/grlee77)

## 🛠️ Improvements

- Merge docs and add links ([#415](https://github.com/rapidsai/cucim/pull/415)) [@jakirkham](https://github.com/jakirkham)
- Add benchmarks for scikit-image functions introduced in 22.08 ([#378](https://github.com/rapidsai/cucim/pull/378)) [@grlee77](https://github.com/grlee77)

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
