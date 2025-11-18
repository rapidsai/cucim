#
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#

# The following cmake policies are set by `CMAKE_PROJECT_INCLUDE_BEFORE` variables
# when `FetchContent` command is used (see https://gitlab.kitware.com/cmake/cmake/-/issues/19854).
cmake_policy(SET CMP0072 NEW)  # FindOpenGL prefers GLVND by default when available. for libtiff
cmake_policy(SET CMP0048 NEW)  # project() command manages VERSION variables. for libtiff
cmake_policy(SET CMP0063 NEW)  # Honor the visibility properties for all target types including static library.
cmake_policy(SET CMP0077 NEW)  # Honor normal variables. Without this, `set(jpeg OFF)` trick to force using static libjpeg-turbo doesn't work.
