# Roadmap

<!-- https://fontawesome.com/icons?d=listing&m=free -->
<!-- https://getbootstrap.com/docs/4.0/utilities/colors/ -->
<!-- https://myst-parser.readthedocs.io/en/latest/using/syntax.html -->
<!-- {fa}`cart-plus,text-info mr-1` -->

```{eval-rst}
The following list is on the road |:smile:|
```

## cuCIM

### {fa}`calendar-alt,text-info mr-1` `v0.1`

- {fa}`check,text-success mr-1` Abstract C++ API -- [v0.1.0](../release_notes/v0.1.0.md)
- {fa}`check,text-success mr-1` Benchmark with openslide (for generic tiff file) : link -- [v0.1.0](../release_notes/v0.1.0.md)
- {fa}`check,text-success mr-1` Usage with C++ API -- [v0.1.0](../release_notes/v0.1.0.md)
- {fa}`check,text-success mr-1` Implement Python API -- [v0.1.0](../release_notes/v0.1.0.md)
- {fa}`check,text-success mr-1` Usage with Python API -- [v0.1.0](../release_notes/v0.1.0.md)
  1. Setup document/build system
  1. Package it
- {fa}`check,text-success mr-1` Sort resolution levels (level 0: the largest resolution) for `CuImage::read_region()` method -- [v0.1.1](../release_notes/v0.1.1.md)
- {fa}`check,text-success mr-1` Fix a crash that occurs when opening a non-existing file -- [v0.1.1](../release_notes/v0.1.1.md)
- {fa}`check,text-success mr-1` Fix an error that occurs when loading a TIFF image that has `TIFFTAG_JPEGTABLES` tag -- [v0.1.1](../release_notes/v0.1.1.md)
  - `Quantization table 0x00 was not defined` message can be shown
- {fa}`check,text-success mr-1` Sort resolution levels (level 0: the largest resolution) for `CuImage::read_region()` method -- [v0.1.1](../release_notes/v0.1.1.md)
- {fa}`check,text-success mr-1` Pass SWIPAT -- [v0.1.1](../release_notes/v0.1.1.md)
- {fa}`check,text-success mr-1` Ignore link check for relative link with header that starts with `/` or `..` -- [v0.1.1](../release_notes/v0.1.1.md)

### {fa}`calendar-alt,text-info mr-1` `v0.2`

- {fa}`check,text-success mr-1` Make it work with various CUDA versions -- [v0.2.0](../release_notes/v0.2.0.md)
- {fa}`check,text-success mr-1` Develop a wrapper for cufile API -- [v0.2.0](../release_notes/v0.2.0.md)
- {fa}`check,text-success mr-1` Support loading [Philips TIFF](https://openslide.org/formats/philips/) files
  - {fa}`check,text-success mr-1` Support Philips TIFF multi-resolution images -- [v0.2.0](../release_notes/v0.2.0.md)
  - {fa}`check,text-success mr-1` Support Philips TIFF associated image from IFD -- [v0.2.0](../release_notes/v0.2.0.md)
- {fa}`check,text-success mr-1` Provide an example/plan for the interoperability with DALI -- [v0.2.0](../release_notes/v0.2.0.md)
- {fa}`check,text-success mr-1` Fix again for the error that occurs when loading a TIFF image that has `TIFFTAG_JPEGTABLES` tag -- [v0.2.0](../release_notes/v0.2.0.md)
- {fa}`check,text-success mr-1` Force-reinstall cucim Python package in the Tox environment whenever `gen_docs` or `gen_docs_dev` command is executed -- [v0.2.0](../release_notes/v0.2.0.md)

### {fa}`calendar-alt,text-info mr-1` `v0.3`

- {fa}`check,text-success mr-1` Add metadata and associated images for Philips TIFF Format
  - {fa}`check,text-success mr-1` Support Philips TIFF associated image from XML metadata -- [v0.3.0](../release_notes/v0.3.0.md)
- {fa}`check,text-success mr-1` Expose metadata of the image as JSON -- [v0.3.0](../release_notes/v0.3.0.md)
- {fa}`check,text-success mr-1` Support reading out of boundary region -- [v0.3.0](../release_notes/v0.3.0.md)
- {fa}`check,text-success mr-1` Showcase the interoperability with DALI -- [v0.3.0](../release_notes/v0.3.0.md)


### {fa}`calendar-alt,text-info mr-1` `v0.18`

- {fa}`check,text-success mr-1` Support Deflate(zlib)-compressed RGB Tiff Image -- [v0.18.0](../release_notes/v0.18.0.md)
- {fa}`check,text-success mr-1` Change the namespaces (`cuimage` to `cucim`) -- [v0.18.0](../release_notes/v0.18.0.md)

### {fa}`calendar-alt,text-info mr-1` `v0.19`

- Refactor the cupyimg package to incorporate it in the adaption layer of cuCIM. Change the namespaces
- Support `__cuda_array_interface__` and DLPack object in Python API
- Support loading data to CUDA memory
- Implement cache mechanism for tile-based image formats

### {fa}`calendar-alt,text-info mr-1` `v0.20`

- Make use of nvJPEG to decode TIFF Files
- Support .svs format with nvJPEG2000
- Design a plug-in mechanism for developing CUDA based 2D/3D imaging filters
- Implement a filter (example: Otsu Thresholding)
- Support loading MHD files

### {fa}`calendar-alt,text-info mr-1` `v0.21`

- Support JPEG, Jpeg 2000, PNG, BMP formats
- Support MIRAX/3DHISTECH (.mrxs) format
- Support LEICA (.scn) format

### {fa}`calendar-alt,text-info mr-1` `v0.22`

- Design a CT bone segmentation filter
- Provide a robust CI/CD system
- Define KPIs and publish report
- Update project to use the latest [Carbonite SDK](https://docs.omniverse.nvidia.com/prod_kit/prod_kit/developer_api.html#carbonite-sdk) for supporting plug-in architecture

## TODOs

### Image Format

#### Generic TIFF(.tif)

- {fa}`check,text-success mr-1` Access image data through container() API (in C++) or as a numpy array (using `__array_interface__` in Python) -- [v0.1.1](../release_notes/v0.1.1.md)
- {fa}`check,text-success mr-1` Fix a crash that occurs when opening a non-existing file -- [v0.1.1](../release_notes/v0.1.1.md)
- {fa}`check,text-success mr-1` Fix an error that occurs when loading a TIFF image that has `TIFFTAG_JPEGTABLES` tag -- [v0.1.1](../release_notes/v0.1.1.md)
  - `Quantization table 0x00 was not defined` message can be shown
- {fa}`check,text-success mr-1` Fix again for the error that occurs when loading a TIFF image that has `TIFFTAG_JPEGTABLES` tag -- [v0.2.0](../release_notes/v0.2.0.md)
  - `ERROR in line 126 while reading JPEG header tables: Not a JPEG file: starts with 0x01 0x00` message can be shown
- {fa}`check,text-success mr-1` Expose metadata of the TIFF file as JSON -- [v0.3.0](../release_notes/v0.3.0.md)
- {fa}`check,text-success mr-1` Support reading out of boundary region -- [v0.3.0](../release_notes/v0.3.0.md)
- {fa}`check,text-success mr-1` Support Deflate(zlib)-compressed RGB Tiff Image -- [v0.18.0](../release_notes/v0.18.0.md)
- Implement cache mechanism for tile-based image formats -- [v0.19.1](../release_notes/v0.19.1.md)
- Use CuFileDriver class for reading files
- Make use of nvJPEG to decode TIFF Files -- [v0.20.0](../release_notes/v0.20.0.md)

- Remove hard-coded metadata (Fill correct values for `cucim::io::format::ImageMetadataDesc`)
  - {fa}`check,text-success mr-1` `resolutions` -- [v0.1.1](../release_notes/v0.1.1.md)
  - `metadata`
- Check if the `tile_rester` memory is freed by jpeg-turbo or not
  - {fa}`check,text-success mr-1` `cpp/plugins/cucim.kit.cuslide/src/cuslide/tiff/ifd.cpp:365` in `IFD::read_region_tiles_libjpeg()` -- [v0.3.0](../release_notes/v0.3.0.md)
    - `cpp/plugins/cucim.kit.cuslide/src/cuslide/cuslide.cpp:123` in `parser_parse` -- [v0.19.1](../release_notes/v0.19.1.md)
- Fill correct metadata information for `CuImage::read_region()`
  - `cpp/src/cucim.cpp:417` -- [v0.19.1](../release_notes/v0.19.1.md)
- Check and use `ifd->samples_per_pixel()` once we can get RGB data instead of RGBA
  - `cpp/plugins/cucim.kit.cuslide/src/cuslide/tiff/ifd.cpp:280` in `IFD::read_region_tiles_libjpeg()` -- [v0.19.1](../release_notes/v0.19.1.md)
- Consider endianness of the .tif file
  - `cpp/plugins/cucim.kit.cuslide/src/cuslide/tiff/ifd.cpp:329` in `IFD::read_region_tiles_libjpeg()`
- Consider tile's depth tag
  - `cpp/plugins/cucim.kit.cuslide/src/cuslide/tiff/ifd.cpp:329` in `IFD::read_region_tiles_libjpeg()`
- Make `file_handle_` object to pointer
  - `cpp/plugins/cucim.kit.cuslide/src/cuslide/tiff/tiff.cpp:50` in `TIFF::TIFF()`
- Remove assumption of sub-resolution dims to 2
  - `cpp/plugins/cucim.kit.cuslide/src/cuslide/tiff/tiff.cpp:140` in `TIFF::read()`

#### [Philips TIFF](https://openslide.org/formats/philips/) (.tif)

- {fa}`check,text-success mr-1` Support Philips TIFF multi-resolution images -- [v0.2.0](../release_notes/v0.2.0.md)
- {fa}`check,text-success mr-1` Support Philips TIFF associated image from IFD -- [v0.2.0](../release_notes/v0.2.0.md)
- {fa}`check,text-success mr-1` Support Philips TIFF associated image from XML metadata -- [v0.3.0](../release_notes/v0.3.0.md)
- {fa}`check,text-success mr-1` Expose XML metadata of the Philips TIFF file as JSON -- [v0.3.0](../release_notes/v0.3.0.md)

#### .mhd

- Support loading MHD files -- [v0.20.0](../release_notes/v0.20.0.md)

#### .svs

- Support .svs format with nvJPEG2000 -- [v0.20.0](../release_notes/v0.20.0.md)

#### .png

- Support .png with [libspng](https://github.com/randy408/libspng/) -- [v0.21.0](../release_notes/v0.21.0.md)
  - **libspng** is faster than **libpng** (but doesn't support encoding)

#### .jpg

- Support .jpg with libjpeg-turbo and nvJPEG -- [v0.21.0](../release_notes/v0.21.0.md)

#### .jp2/.j2k

- Support .jp2/.j2k files with OpenJpeg and nvJPEG2000 -- [v0.21.0](../release_notes/v0.21.0.md)

#### .bmp

- Support .bmp file natively -- [v0.21.0](../release_notes/v0.21.0.md)

#### .mrxs

- Support MIRAX/3DHISTECH (.mrxs) format -- [v0.21.0](../release_notes/v0.21.0.md)

#### .scn

- Support LEICA (.scn) format -- [v0.21.0](../release_notes/v0.21.0.md)

#### .dcm

- Support DICOM format
- Support reading segmentation image instead of main pixel array
  - `examples/cpp/dicom_image/main.cpp:37`

#### .iSyntax

- Support Philips iSyntax format
  - <https://thepathologist.com/fileadmin/issues/App_Notes/0016-024-app-note-Philips__iSyntax_for_Digital_Pathology.pdf>
  - <https://www.openpathology.philips.com/wp-content/uploads/isyntax/4522%20207%2043941_2020_04_24%20Pathology%20iSyntax%20image%20format.pdf>

### Image Filter

#### Basic Filter

- Design a plug-in mechanism for developing CUDA based 2D/3D imaging filters -- [v0.20.0](../release_notes/v0.20.0.md)
- Implement a filter (example: Otsu Thresholding) -- [v0.20.0](../release_notes/v0.20.0.md)

#### Medical-specific Filter

- Design a CT bone segmentation filter -- [v0.22.0](../release_notes/v0.22.0.md)

### Performance Improvements

- {fa}`check,text-success mr-1` Copy data using `std::vector::insert()` instead of `std::vector::push_back()` -- [v0.3.0](../release_notes/v0.3.0.md)
  - `cpp/plugins/cucim.kit.cuslide/src/cuslide/tiff/ifd.cpp:78` in `IFD::IFD()`
  - `cpp/plugins/cucim.kit.cuslide/src/cuslide/tiff/ifd.cpp:98` in `IFD::IFD()`
  - Benchmark result showed that time for assigning 50000 tile offset/size (uint64_t) is reduced from 118 us to 8 us.
- {fa}`check,text-success mr-1` Replace malloc with better allocator for small-sized memory -- [v0.3.0](../release_notes/v0.3.0.md)
  - Use a custom allocator(pmr) for metadata data.
- Try to use `__array_struct__`. Access to array interface could be faster
  - <https://numpy.org/doc/stable/reference/arrays.interface.html#c-struct-access>
  - Check the performance difference between python int vs python long later
  - `python/pybind11/cucim_py.cpp:234` in `get_array_interface()`
- Check performance
  - `cpp/plugins/cucim.kit.cuslide/src/cuslide/tiff/ifd.cpp:122` in `IFD::read()` : string concatenation

### GPUDirect-Storage (GDS) Support

- {fa}`check,text-success mr-1` Develop a wrapper for cufile API -- [v0.2.0](../release_notes/v0.2.0.md)
- {fa}`check,text-success mr-1` Static link with cufile when [libcufile.a is available](https://docs.google.com/document/d/1DQ_T805dlTcDU9bGW32E2ak5InX8iUcNI7Tq_lXAtLc/edit?ts=5f90bc5f) -- [v0.3.0](../release_notes/v0.3.0.md)

### Interoperability

- {fa}`check,text-success mr-1` Provide an example/plan for the interoperability with DALI -- [v0.2.0](../release_notes/v0.2.0.md)
- {fa}`check,text-success mr-1` Showcase the interoperability with DALI -- [v0.3.0](../release_notes/v0.3.0.md)
- Support `__cuda_array_interface__` and DLPack object in Python API -- [v0.19.1](../release_notes/v0.19.1.md)
  - https://docs.cupy.dev/en/stable/reference/interoperability.html#dlpack
  - https://github.com/pytorch/pytorch/pull/11984
- Refactor the cupyimg package to incorporate it in the adaption layer of cuCIM. Change the namespaces -- [v0.19.0](../release_notes/v0.19.0.md)
  - Implement/expose `scikit-image`-like image loading APIs (such as `imread`) and filtering APIs for cuCIM library by using cuCIM's APIs
- Support DALI's CPU/GPU Tensor: <https://docs.nvidia.com/deeplearning/dali/user-guide/docs/data_types.html#tensor>
- Support loading data to CUDA memory -- [v0.19.1](../release_notes/v0.19.1.md)
- Consider adding `to_xxx()` methods in Python API
  - `examples/python/tiff_image/main.py:125`
- Support byte-like object for CuImage object so that the following method works -- [v0.19.1](../release_notes/v0.19.1.md)
    ```python
    from PIL import Image
    ...
    #np_img_arr = np.asarray(region)
    #Image.fromarray(np_img_arr)

    Image.fromarray(region)
    # /usr/local/lib/python3.6/dist-packages/PIL/Image.py in frombytes(self, data, decoder_name, *args)
    #     792         d = _getdecoder(self.mode, decoder_name, args)
    #     793         d.setimage(self.im)
    # --> 794         s = d.decode(data)
    #     795
    #     796         if s[0] >= 0:
    # TypeError: a bytes-like object is required, not 'cucim._cucim.CuImage'
    ```
- Provide universal cucim adaptors for DALI (for cucim::io::format::IImageFormat and cucim::filter::IImageFilter interfaces)
- Support pretty display for IPython(Jupyter Notebook)
  - https://ipython.readthedocs.io/en/stable/config/integrating.html#integrating-your-objects-with-ipython

### Pipeline

- Use app_dp_sample pipeline to convert input image(.svs) of Nuclei segmentation pipeline(app_dp_nuclei) to .tif image
  - Load .tif file using cuCIM for Nuclei segmentation pipeline

### Python API

- Feature parity with OpenSlide
- Add context manager for CuImage class (for `close()` method) -- [v0.19.1](../release_notes/v0.19.1.md)

### C++ API

- Design filtering API (which can embrace CuPy/CVCore/CuPyImg/OpenCV/scikit-image/dask-image)
- Feature parity with OpenSlide

- {fa}`check,text-success mr-1` Sort resolution levels (level 0: the largest resolution) for `CuImage::read_region()` method -- [v0.1.1](../release_notes/v0.1.1.md)
  - Add `TIFF::level_ifd(size_t level_index)` method
- {fa}`check,text-success mr-1` Support `metadata` and `raw_metadata` properties/methods -- [v0.3.0](../release_notes/v0.3.0.md)
  - Implement `CuImage::metadata()` with JSON library (Folly or Modern JSON)
    - `cpp/src/cucim.cpp:238`
- `ImageMetadataDesc` struct
  - {fa}`check,text-success mr-1` `resolution_dim_start` field: Reconsider its use (may not be needed) -- [v0.3.0](../release_notes/v0.3.0.md)
    - `cpp/include/cucim/io/format/image_format.h:53`
  - `channel_names` field : `S`, `T`, and other dimension can have names so need to be generalized
    - `cpp/include/cucim/io/format/image_format.h:51`
- `numpy_dtype()` method
  - Consider bfloat16: <https://github.com/dmlc/dlpack/issues/45>
  - Consider other byte-order (currently, we assume `little-endian`)
    - `cpp/plugins/cucim.kit.cuslide/src/cuslide/tiff/tiff.cpp:53)
  - `cpp/include/cucim/memory/dlpack.h:41`
- `checker_is_valid()` method
  - Add `buf_size` parameter and implement the method
  - `cpp/plugins/cucim.kit.cuslide/src/cuslide/cuslide.cpp:68`

- Consider default case (how to handle -1 index?)
  - `cpp/src/io/device.cpp` in `Device::Device()`
- Implement `Device::parse_type()`
  - `cpp/src/io/device.cpp:81`
- Implement `Device::validate_device()`
  - `cpp/src/io/device.cpp:116`

- Check illegal characters for `DimIndices::DimIndices()`
  - `cpp/src/cucim.cpp:35`
  - `cpp/src/cucim.cpp:46`

- Implement `detect_format()` method
  - `cpp/src/cucim.cpp:103`
- Detect available format for the file path
  - Also consider if the given file path is folder path (DICOM case)
  - `cpp/src/cucim.cpp:117` in `CuImage::CuImage()`
- Implement `CuImage::CuImage(const filesystem::Path& path, const std::string& plugin_name)`
  - `cpp/src/cucim.cpp:128`
- Implement `CuImage::dtype()`
  - Support string conversion like Device class
  - `cpp/src/cucim.cpp:283`

### Build

- Check if `CMAKE_EXPORT_PACKAGE_REGISTRY` is duplicate and remove it
  - `cucim/cpp/plugins/cucim.kit.cuslide/CMakeLists.txt:255`
- Install other dependencies for libtiff so that other compression methods are available
  - `cucim/Dockerfile:32`
- {fa}`check,text-success mr-1` Setup development environment with VSCode (in addition to CLion) -- [v0.3.0](../release_notes/v0.3.0.md)
- Use prebuilt libraries for dependencies

### Test

- {fa}`check,text-success mr-1` Parameterize input library/image -- [v0.3.0](../release_notes/v0.3.0.md)
  - `/ssd/repo/cucim/cpp/tests/test_read_region.cpp:69` in `Verify read_region`
  - `/ssd/repo/cucim/cpp/tests/test_cufile.cpp:79` in `Verify libcufile usage`
- {fa}`check,text-success mr-1` Use a VSCode plugin for local test execution -- [v0.3.0](../release_notes/v0.3.0.md)
  - `matepek.vscode-catch2-test-adapter` extension
    - <https://github.com/matepek/vscode-catch2-test-adapter/blob/master/documents/configuration/test.advancedExecutables.md>

### Platform

- Support Windows (currently only Linux package is available)

### Package & CI/CD

- {fa}`check,text-success mr-1` Make it work with various CUDA versions -- [v0.2.0](../release_notes/v0.2.0.md)
  - Currently, it is linked to CUDA 11.0 library
  - Refer to PyTorch's PyPi package
    - The PyPi package embeds CUDA runtime library.
    - https://github.com/pytorch/pytorch/issues/47268#issuecomment-721996861
- Move to Github Project
- Move `tox` setup from python folder to the project root folder
- Setup Conda recipe
- Setup automated test cases
- Provide a robust CI/CD system -- [v0.22.0](../release_notes/v0.22.0.md)
- Define KPIs and publish report -- [v0.22.0](../release_notes/v0.22.0.md)

- Add license files to the package
- Package a separate CXX11 ABI library
  - Currently, C++ library is forced to set `_GLIBCXX_USE_CXX11_ABI` to 0 due to [Dual ABI](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html) problem
  - `cpp/CMakeLists.txt:98`
- Support CPack
  - `CMakeLists.txt:177`

### Documentation

- {fa}`check,text-success mr-1` Pass SWIPAT -- [v0.1.1](../release_notes/v0.1.1.md)
- Refine README.md and relevant documents for the project
- Move Sphinx docs to the project root folder
- Add C++ API document
- Add C++ examples to Jupyter Notebook
  - Can install C++ Kernel: <https://xeus-cling.readthedocs.io/en/latest/installation.html#from-source-with-cmake>
- {fa}`check,text-success mr-1` Ignore link check for relative link with header that starts with `/` or `..` -- [v0.1.1](../release_notes/v0.1.1.md)
  - `python/cucim/docs/conf.py:71`
  - <https://www.sphinx-doc.org/en/master/usage/configuration.html?highlight=linkcheck#options-for-the-linkcheck-builder>
- {fa}`check,text-success mr-1` Force-reinstall cucim Python package in the Tox environment whenever `gen_docs` or `gen_docs_dev` command is executed -- [v0.2.0](../release_notes/v0.2.0.md)
  - <https://tox.readthedocs.io/en/latest/config.html#conf-usedevelop>
- Simplify method signatures in Python API Docs
  - `cucim._cucim.CuImage` -> `cucim.CuImage`
- Use new feature to reference a cross-link with header (from v0.13.0 of [myst-parser](https://pypi.org/project/myst-parser/))
  - <https://github.com/executablebooks/MyST-Parser/issues/149>
  - <https://myst-parser.readthedocs.io/en/v0.13.5/using/howto.html#automatically-create-targets-for-section-headers>
  - <https://myst-parser.readthedocs.io/en/v0.13.5/using/syntax-optional.html#auto-generated-header-anchors>

### Plugin-system (Carbonite)

- Update project to use the latest [Carbonite SDK](https://docs.omniverse.nvidia.com/prod_kit/prod_kit/developer_api.html#carbonite-sdk) for supporting plug-in architecture -- [v0.22.0](../release_notes/v0.22.0.md)
  - Migrate to use Carbonite SDK as it is
  - Update to use Minimal Carbonite SDK

- Handle errors and log error message once switched to use Carbonite SDK's built-in error routine
  - `cpp/plugins/cucim.kit.cuslide/src/cuslide/tiff/ifd.cpp` : when reading field info
  - `cpp/plugins/cucim.kit.cuslide/src/cuslide/tiff/ifd.cpp` in `IFD::read()` : memory size check if `out_buf->data` has high-enough memory
- Get plugin name from file_path
  - `cpp/src/core/cucim_plugin.cpp:53` in `Plugin::Plugin()`
- Generalize `CuImage::ensure_init()`
  - 'LINUX' path separator is used. Need to make it generalize once filesystem library is available
  - `cucim/cpp/src/cucim.cpp:520`

