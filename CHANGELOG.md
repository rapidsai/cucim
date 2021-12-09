# cuCIM 22.02.00 (Date TBD)

Please see https://github.com/rapidsai/cucim/releases/tag/v22.02.00a for the latest changes to this development branch.

# cuCIM 21.12.00 (8 Dec 2021)

Please see https://github.com/rapidsai/cucim/releases/tag/v21.12.00 for the details.

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
