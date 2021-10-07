
# Changelog (See [Release Notes](https://github.com/rapidsai/cucim/wiki/Release-Notes))

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

