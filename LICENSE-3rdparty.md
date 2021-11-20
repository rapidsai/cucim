cuCIM is licensed under the terms of the Apache-2.0 License.

However, cuCIM utilizes third-party software from various sources.
Portions of this software are copyrighted by their respective owners as indicated in the copyright
notices below.

The following acknowledgments pertain to this software license.

The full license text of the third-party software is available in `3rdparty` folder
in the repository/distribution.

---

libjpeg-turbo
- This software is based in part on the work of the Independent JPEG Group.
- License: libjpeg-turbo is covered by three compatible BSD-style open source licenses
  - The IJG (Independent JPEG Group) License
  - The Modified (3-clause) BSD License
  - The zlib License
  - https://github.com/libjpeg-turbo/libjpeg-turbo/blob/master/LICENSE.md
- Copyright:
  - D. R. Commander
  - Viktor Szathmáry
- Files:
  - cpp/plugins/cucim.kit.cuslide/src/cuslide/jpeg/libjpeg_turbo.cpp : Implementation of jpeg decoder.

libtiff
- License: BSD-like License
  - https://gitlab.com/libtiff/libtiff/-/blob/master/COPYRIGHT
- Copyright:
  - Sam Leffler
  - Silicon Graphics, Inc.
- Files:
  - cpp/plugins/cucim.kit.cuslide/src/cuslide/lzw/lzw_libtiff.cpp : Implementation of lzw decoder.

fmt
- License: MIT License
  - https://github.com/fmtlib/fmt/blob/master/LICENSE.rst
- Copyright: Victor Zverovich

spdlog
- License: MIT License
  - https://github.com/gabime/spdlog/blob/v1.x/LICENSE
- Copyright: Gabi Melman

Google Benchmark
- License: Apache-2.0 License
  - https://github.com/google/benchmark/blob/master/LICENSE
- Copyright: Google Inc.

Google Test
- License: BSD-3-Clause License
  - https://github.com/google/googletest/blob/master/LICENSE
- Copyright: Google Inc.

Catch2
- License: BSL-1.0 License
  - https://github.com/catchorg/Catch2/blob/devel/LICENSE.txt
- Copyright: Catch2 Authors

CLI11
- License: BSD-3-Clause License
  - https://github.com/CLIUtils/CLI11/blob/master/LICENSE
- Copyright: University of Cincinnati

pybind11
- License: BSD-3-Clause License
  - https://github.com/pybind/pybind11/blob/master/LICENSE
- Copyright: Wenzel Jakob
- Files:
  - python/pybind11/cucim_py.cpp : Implementation of `vector2pytuple()` method.

JSON for Modern C++
- License: MIT License
  - https://github.com/nlohmann/json/blob/develop/LICENSE.MIT
- Copyright: Niels Lohmann

pybind11_json
- License: BSD-3-Clause License
  - https://github.com/pybind/pybind11_json/blob/master/LICENSE
- Copyright: Martin Renou

DLPack
- License: Apache-2.0 License
  - https://github.com/dmlc/dlpack/blob/main/LICENSE
- Copyright: DLPack Contributors

NVIDIA CUDA TOOLKIT (including libcufile)
- License: NVIDIA License
  - https://docs.nvidia.com/cuda/pdf/EULA.pdf
- Copyright: NVIDIA Corporation

RAPIDS RMM
- License: Apache-2.0 License
  - https://github.com/rapidsai/rmm/blob/branch-0.17/LICENSE
- Copyright: NVIDIA Corporation

OpenJPEG
- License: BSD-2-Clause License
  - https://github.com/uclouvain/openjpeg/blob/master/LICENSE
- Copyright:
  - Universite catholique de Louvain (UCL), Belgium
  - Professor Benoit Macq
  - Antonin Descampe
  - Francois-Olivier Devaux
  - Herve Drolon, FreeImage Team
  - Yannick Verschueren
  - David Janssens
  - Centre National d'Etudes Spatiales (CNES), France
  - CS Systemes d'Information, France
- Files:
  - cpp/plugins/cucim.kit.cuslide/src/cuslide/jpeg2k/libopenjpeg.cpp : Implementation of jpeg2k decoder.
  - cpp/plugins/cucim.kit.cuslide/src/cuslide/jpeg2k/color_conversion.cpp : Implementation of color conversion methods.

NVIDIA nvJPEG
- License: NVIDIA License
  - https://developer.download.nvidia.com/compute/redist/libnvjpeg/EULA-nvjpeg.txt
- Copyright: NVIDIA Corporation

NVIDIA nvJPEG2000
- License: NVIDIA License
  - https://docs.nvidia.com/cuda/nvjpeg2000/license.html
- Copyright: NVIDIA Corporation

libspng
- License: BSD-2-Clause License
  - https://github.com/randy408/libspng/blob/master/LICENSE
- Copyright: Randy

PyTorch
- License: BSD-3-Clause License
  - https://github.com/pytorch/pytorch/blob/master/LICENSE
- Copyright: PyTorch Contributors (See above link for the detail)

Abseil
- License: Apache-2.0 License
  - https://github.com/abseil/abseil-cpp/blob/master/LICENSE
- Copyright: The Abseil Authors

Boost C++ Libraries
- License: BSL-1.0 License
  - https://github.com/boostorg/boost/blob/master/LICENSE_1_0.txt
- Copyright: The Boost Authors

Folly
- License: Apache-2.0 License
  - https://github.com/facebook/folly/blob/master/LICENSE
- Copyright: Facebook, Inc. and its affiliates.

NumPy
- License: BSD-3-Clause License
  - https://github.com/numpy/numpy/blob/master/LICENSE.txt
- Copyright: NumPy Developers.

pytest
- License: MIT License
  - https://github.com/pytest-dev/pytest/blob/master/LICENSE
- Copyright: Holger Krekel and others

CuPy
- License: MIT License
  - https://github.com/cupy/cupy/blob/master/LICENSE
- Copyright:
  - Preferred Infrastructure, Inc.
  - Preferred Networks, Inc.

OpenSlide
- License: GNU Lesser General Public License v2.1
  - https://github.com/openslide/openslide/blob/master/LICENSE.txt
- Copyright: Carnegie Mellon University and others
- Usage: For comparing performance in benchmark binaries

Click
- License: BSD-3-Clause License
  - https://github.com/pallets/click/blob/master/LICENSE.rst
- Copyright: Pallets

tifffile
- License: BSD-3-Clause License
  - https://github.com/cgohlke/tifffile/blob/master/LICENSE
- Copyright: Christoph Gohlke

Dask
- License: BSD-3-Clause License
  - https://github.com/dask/dask/blob/master/LICENSE.txt
- Copyright: Anaconda, Inc. and contributors

Dask CUDA
- License: Apache-2.0 License
  - https://github.com/rapidsai/dask-cuda/blob/branch-0.17/LICENSE
- Copyright: Dask CUDA Authors

Zarr
- License: MIT License
  - https://github.com/zarr-developers/zarr-python/blob/master/LICENSE
- Copyright: Zarr Developers

scikit-image
- License: BSD-3-Clause License
  - https://github.com/scikit-image/scikit-image/blob/master/LICENSE.txt
- Copyright: the scikit-image team

OpenCV (extra modules, opencv-contrib-python)
- License: Apache-2.0 License
  - https://github.com/opencv/opencv_contrib/blob/master/LICENSE
- Copyright:
  - Intel Corporation
  - Willow Garage Inc.
  - NVIDIA Corporation
  - Advanced Micro Devices, Inc.
  - OpenCV Foundation
  - Itseez Inc.
  - Xperience AI
  - Shenzhen Institute of Artificial Intelligence and Robotics for Society

SCIFIO
- License: BSD-2-Clause License
  - https://github.com/scifio/scifio/blob/master/LICENSE.txt
- Copyright: SCIFIO developers
- Usage: Image format interface is inspired by this library.

AICSImageIO
- License: BSD-3-Clause License
  - https://github.com/AllenCellModeling/aicsimageio/blob/master/LICENSE
- Copyright: Allen Institute for Cell Science
- Usage: Some Python API design is inspired by this library.

pugixml
- License: MIT License
  - https://github.com/zeux/pugixml/blob/master/LICENSE.md
- Copyright: Arseny Kapoulkine
- Usage: Parsing XML metadata for Philips TIFF file (@cuslide plugin)

libdeflate
- License: MIT License
  - https://github.com/ebiggers/libdeflate/blob/master/COPYING
- Copyright: Eric Biggers
- Usage: Extracting tile image (zlib/deflate compressed)for TIFF file (@cuslide plugin)

libcuckoo
- License: Apache-2.0 License
  - https://github.com/efficient/libcuckoo/blob/master/LICENSE
- Copyright: Carnegie Mellon University and Intel Corporation
- Usage: Using concurrent hash table implementation for cache mechanism.

pytest-lazy-fixture
- License: MIT License
  - https://github.com/TvoroG/pytest-lazy-fixture/blob/master/LICENSE
- Copyright: Marsel Zaripov
- Usage: Using lazy fixture feature in PyTest.

psutil
- License: BSD-3-Clause License
  - https://github.com/giampaolo/psutil/blob/master/LICENSE
- Copyright:
  - Jay Loden
  - Dave Daeschler
  - Giampaolo Rodola
- Usage: Checking memory usage in Python applications.

gputil
- License: MIT License
  - https://github.com/anderskm/gputil/blob/master/LICENSE.txt
- Copyright: Anders Krogh Mortensen
- Usage: Checking memory usage in Python applications.
