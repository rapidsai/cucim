# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(../cmake/rapids_config.cmake)

project(
  cucim-python
  VERSION "${RAPIDS_VERSION}"
  DESCRIPTION "cuCIM Python bindings"
  LANGUAGES C CXX
)

include(rapids-cmake)
include(rapids-cython-core)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

if(NOT CUCIM_SDK_PATH)
  get_filename_component(CUCIM_SDK_PATH "${CMAKE_SOURCE_DIR}/.." ABSOLUTE)
endif()

find_package(
  cucim CONFIG REQUIRED
  HINTS "${CUCIM_SDK_PATH}/install/lib/cmake/cucim" "$ENV{PREFIX}/lib/cmake/cucim"
)

rapids_cython_init()
rapids_cython_create_modules(
  CXX
  SOURCE_FILES cython/_cucim.pyx
  LINKED_LIBRARIES cucim::cucim
  INSTALL_DIR cucim/clara
)

target_sources(_cucim PRIVATE cython/binding.cpp cython/binding.hpp)
target_include_directories(_cucim PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/cython")
target_compile_definitions(_cucim PRIVATE _GLIBCXX_USE_CXX11_ABI=0)

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib/cucim")
set_target_properties(
  _cucim
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib/cucim"
)
