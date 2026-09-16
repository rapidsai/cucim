# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

if(NOT (rapids-cmake-branch OR rapids-cmake-version))
  message(FATAL_ERROR "The CMake variable `rapids-cmake-branch` or `rapids-cmake-version` must be defined")
endif()

if(NOT rapids-cmake-repo)
  set(rapids-cmake-repo rapidsai/rapids-cmake)
endif()

if(NOT rapids-cmake-branch)
  set(rapids-cmake-branch "release/${rapids-cmake-version}")
endif()

if(NOT rapids-cmake-url)
  set(rapids-cmake-url "https://github.com/${rapids-cmake-repo}/")
  if(rapids-cmake-fetch-via-git)
    if(rapids-cmake-sha)
      set(rapids-cmake-value-to-clone "${rapids-cmake-sha}")
    elseif(rapids-cmake-tag)
      set(rapids-cmake-value-to-clone "${rapids-cmake-tag}")
    else()
      set(rapids-cmake-value-to-clone "${rapids-cmake-branch}")
    endif()
  else()
    if(rapids-cmake-sha)
      set(rapids-cmake-value-to-clone "archive/${rapids-cmake-sha}.zip")
    elseif(rapids-cmake-tag)
      set(rapids-cmake-value-to-clone "archive/refs/tags/${rapids-cmake-tag}.zip")
    else()
      set(rapids-cmake-value-to-clone "archive/refs/heads/${rapids-cmake-branch}.zip")
    endif()
  endif()
endif()

include(FetchContent)
if(rapids-cmake-fetch-via-git)
  FetchContent_Declare(
    rapids-cmake
    GIT_REPOSITORY "${rapids-cmake-url}"
    GIT_TAG "${rapids-cmake-value-to-clone}"
  )
  message(STATUS "Fetching rapids-cmake from ${rapids-cmake-url}@${rapids-cmake-value-to-clone}")
else()
  string(APPEND rapids-cmake-url "${rapids-cmake-value-to-clone}")
  FetchContent_Declare(rapids-cmake URL "${rapids-cmake-url}")
  message(STATUS "Fetching rapids-cmake from ${rapids-cmake-url}")
endif()
FetchContent_GetProperties(rapids-cmake)
if(rapids-cmake_POPULATED)
  if(NOT "${rapids-cmake-dir}" IN_LIST CMAKE_MODULE_PATH)
    list(APPEND CMAKE_MODULE_PATH "${rapids-cmake-dir}")
  endif()
else()
  FetchContent_MakeAvailable(rapids-cmake)
endif()
