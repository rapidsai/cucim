# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

include(FetchContent)

set(CMAKE_SUPERBUILD_DEPS_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/..")

if(NOT COMMAND superbuild_depend)
    # superbuild_depend: Include a dependency cmake file
    # Searches in SUPERBUILD_ADDITIONAL_DEPS_DIRS first (if set), then the default deps dir
    function(superbuild_depend module_name)
        # Check additional deps directories first (for plugin-specific deps)
        foreach(deps_dir IN LISTS SUPERBUILD_ADDITIONAL_DEPS_DIRS)
            if(EXISTS "${deps_dir}/${module_name}.cmake")
                include("${deps_dir}/${module_name}.cmake")
                return()
            endif()
        endforeach()
        # Fall back to default deps directory
        include("${CMAKE_SUPERBUILD_DEPS_ROOT_DIR}/deps/${module_name}.cmake")
    endfunction()
endif()
