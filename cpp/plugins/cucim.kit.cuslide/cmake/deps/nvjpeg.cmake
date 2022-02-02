#
# Copyright (c) 2021, NVIDIA CORPORATION.
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

if (NOT TARGET deps::nvjpeg_static)

    add_library(deps::nvjpeg_static STATIC IMPORTED GLOBAL)

    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../../../temp/cuda/include/nvjpeg.h)
        set(NVJPEG_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../temp/cuda/include)
    else ()
        message(FATAL_ERROR "nvjpeg.h not found")
    endif ()

    message("Set NVJPEG_INCLUDE_PATH to '${NVJPEG_INCLUDE_PATH}'.")

    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../../../temp/cuda/lib64/libnvjpeg_static.a)
        set(NVJPEG_STATIC_LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../temp/cuda/lib64/libnvjpeg_static.a)
    else ()
        message(FATAL_ERROR "libnvjpeg_static.a not found")
    endif ()

    message("Set NVJPEG_STATIC_LIB_PATH to '${NVJPEG_STATIC_LIB_PATH}'.")

    set_target_properties(deps::nvjpeg_static PROPERTIES
        IMPORTED_LOCATION "${NVJPEG_STATIC_LIB_PATH}"
        INTERFACE_INCLUDE_DIRECTORIES "${NVJPEG_INCLUDE_PATH}"
    )

endif ()
