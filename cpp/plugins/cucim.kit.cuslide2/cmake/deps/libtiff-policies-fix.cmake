# Apache License, Version 2.0
# Copyright 2020 NVIDIA Corporation
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

# Workaround for libtiff v4.1.0's cmake_minimum_required(VERSION 2.8.11)
# This allows the old version requirement to be accepted by newer CMake
if(POLICY CMP0000)
    cmake_policy(SET CMP0000 OLD)  # Allow old minimum version
endif()

cmake_policy(SET CMP0072 NEW)  # FindOpenGL prefers GLVND by default when available. for libtiff
cmake_policy(SET CMP0048 NEW)  # project() command manages VERSION variables. for libtiff
cmake_policy(SET CMP0063 NEW)  # Honor the visibility properties for all target types including static library.
cmake_policy(SET CMP0077 NEW)  # Honor normal variables. Without this, `set(jpeg OFF)` trick to force using static libjpeg-turbo doesn't work.
