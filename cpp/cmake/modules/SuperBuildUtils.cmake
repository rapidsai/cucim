#
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#

include(FetchContent)

set(CMAKE_SUPERBUILD_DEPS_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/..")

if(NOT COMMAND superbuild_depend)
    function(superbuild_depend module_name)
        include("${CMAKE_SUPERBUILD_DEPS_ROOT_DIR}/deps/${module_name}.cmake")
    endfunction()
endif()
