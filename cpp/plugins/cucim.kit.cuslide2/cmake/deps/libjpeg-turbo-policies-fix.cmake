# Apache License, Version 2.0
# Copyright 2020-2021 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The following cmake policies are set by `CMAKE_PROJECT_INCLUDE_BEFORE` variables
# when `FetchContent` command is used (see https://gitlab.kitware.com/cmake/cmake/-/issues/19854).
cmake_policy(SET CMP0048 NEW)  # project() command manages VERSION variables. for libjpeg-turbo
cmake_policy(SET CMP0054 NEW)  # cmake-build-debug/_deps/deps-libjpeg-turbo-src/cmakescripts/GNUInstallDirs.cmake:174 (elseif):
cmake_policy(SET CMP0063 NEW)  # Honor the visibility properties for all target types including static library.
cmake_policy(SET CMP0077 NEW)  # Use normal variable that is injected, instead of ignoring/clearing normal variable: REQUIRE_SIMD/CMAKE_ASM_NASM_COMPILER.
# https://cmake.org/cmake/help/v3.18/policy/CMP0065.html : Do not add flags to export symbols from executables without the ENABLE_EXPORTS target property.
#   : this policy is not handled yet so always enable exports.
