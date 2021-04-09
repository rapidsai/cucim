#
# Copyright (c) 2020-2021, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

if(NOT TARGET deps::gds)
    if(NOT GDS_SDK_PATH)
        get_filename_component(GDS_SDK_PATH "${CMAKE_SOURCE_DIR}/gds" ABSOLUTE)
        message("GDS_SDK_PATH is not set. Using '${GDS_SDK_PATH}'")
    else()
        message("GDS_SDK_PATH is set to ${GDS_SDK_PATH}")
    endif()

    if(EXISTS "${GDS_SDK_PATH}/lib64/libcufile.so")
        add_library(deps::gds SHARED IMPORTED GLOBAL)
        set_target_properties(deps::gds PROPERTIES
            IMPORTED_LOCATION "${GDS_SDK_PATH}/lib64/libcufile.so"
            INTERFACE_INCLUDE_DIRECTORIES "${GDS_SDK_PATH}/lib64/"
        )
    else()
        message("'${GDS_SDK_PATH}/lib64/libcufile.so' is not available. Set CUCIM_SUPPORT_GDS to OFF and import cufile.h only.")
        # Do not support GDS
        set(CUCIM_SUPPORT_GDS OFF PARENT_SCOPE)
        add_library(deps::gds INTERFACE IMPORTED GLOBAL)
        set_target_properties(deps::gds PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${GDS_SDK_PATH}/lib64/"
        )
    endif()
endif()