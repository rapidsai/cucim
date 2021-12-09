
# Changelog (See [Release Notes](https://github.com/rapidsai/cucim/wiki/Release-Notes))

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
