#
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#

include(FetchContent)

# Local deps directory (for cuslide2-specific dependencies like nvimgcodec)
set(CMAKE_LOCAL_DEPS_DIR "${CMAKE_CURRENT_LIST_DIR}/../deps")
# Shared deps directory from cuslide plugin (for common dependencies)
set(CMAKE_SHARED_DEPS_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../cucim.kit.cuslide/cmake/deps")

function(superbuild_depend module_name)
    # Check local deps first (cuslide2-specific), then shared deps
    if(EXISTS "${CMAKE_LOCAL_DEPS_DIR}/${module_name}.cmake")
        include("${CMAKE_LOCAL_DEPS_DIR}/${module_name}.cmake")
    elseif(EXISTS "${CMAKE_SHARED_DEPS_DIR}/${module_name}.cmake")
        include("${CMAKE_SHARED_DEPS_DIR}/${module_name}.cmake")
    else()
        message(FATAL_ERROR "Dependency ${module_name}.cmake not found in local or shared deps")
    endif()
endfunction()
