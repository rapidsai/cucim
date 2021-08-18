
# Changelog

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

## 0.19.0 (2021-04-19)

- The first release of cuClaraImage + [cupyimg](https://github.com/mritools/cupyimg) as a single project `cuCIM`.
  - `cucim.skimage` package is added from `cupyimg`.
  - CuPy (>=9.0.0b3), scipy, scikit-image is required to use cuCIM's scikit-image-compatible API.

## 0.18.3 (2021-04-16)

- Fix memory leaks that occur when reading completely out-of-boundary regions.

## 0.18.2 (2021-03-29)

- Use the white background only for Philips TIFF file.
  - Generic TIFF file would have the black background by default.
- Fix upside-downed image for TIFF file if the image is not RGB & tiled image with JPEG/Deflate-compressed tiles.
  - Use slow path if the image is not RGB & tiled image with JPEG/Deflate-compressed tiles.
    - Show an error message if the out-of-boundary cases are requested with the slow path.
    - `ValueError: Cannot handle the out-of-boundary cases for a non-RGB image or a non-Jpeg/Deflate-compressed image.`

## 0.18.1 (2021-03-17)

- Disable using cuFile
  - Remove warning messages when libcufile.so is not available.
    - `[warning] CuFileDriver cannot be open. Falling back to use POSIX file IO APIs.`

## 0.18.0 (2021-03-16)

- First release on PyPI with only cuClaraImage features.
- The namespace of the project is changed from `cuimage` to `cucim` and project name is now `cuCIM`
- Support Deflate(zlib) compression in Generic TIFF Format.
  - [libdeflate](https://github.com/ebiggers/libdeflate) library is used to decode the deflate-compressed data.
