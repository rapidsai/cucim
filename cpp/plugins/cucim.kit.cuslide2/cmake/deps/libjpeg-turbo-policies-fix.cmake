#
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#

# The following cmake policies are set by `CMAKE_PROJECT_INCLUDE_BEFORE` variables
# when `FetchContent` command is used (see https://gitlab.kitware.com/cmake/cmake/-/issues/19854).
cmake_policy(SET CMP0048 NEW)  # project() command manages VERSION variables. for libjpeg-turbo
cmake_policy(SET CMP0054 NEW)  # cmake-build-debug/_deps/deps-libjpeg-turbo-src/cmakescripts/GNUInstallDirs.cmake:174 (elseif):
cmake_policy(SET CMP0063 NEW)  # Honor the visibility properties for all target types including static library.
cmake_policy(SET CMP0077 NEW)  # Use normal variable that is injected, instead of ignoring/clearing normal variable: REQUIRE_SIMD/CMAKE_ASM_NASM_COMPILER.
# https://cmake.org/cmake/help/v3.18/policy/CMP0065.html : Do not add flags to export symbols from executables without the ENABLE_EXPORTS target property.
#   : this policy is not handled yet so always enable exports.
