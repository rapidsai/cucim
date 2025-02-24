#
# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

# This module tries to find libculibos.a from /usr/local/cuda if
# CUDA::culibos is not available through `find_package(CUDAToolkit REQUIRED)`.
if (NOT TARGET CUDA::culibos)

    find_package(CUDAToolkit REQUIRED)

    if(NOT TARGET CUDA::culibos)
        find_library(CUDA_culibos_LIBRARY
            NAMES culibos
            HINTS /usr/local/cuda
                ENV CUDA_PATH
            PATH_SUFFIXES nvidia/current lib64 lib/x64 lib
        )

        mark_as_advanced(CUDA_culibos_LIBRARY)

        if (NOT TARGET CUDA::culibos AND CUDA_culibos_LIBRARY)
            add_library(CUDA::culibos STATIC IMPORTED GLOBAL)
            target_include_directories(CUDA::culibos SYSTEM INTERFACE "${CUDAToolkit_INCLUDE_DIRS}")
            set_property(TARGET CUDA::culibos PROPERTY IMPORTED_LOCATION "${CUDA_culibos_LIBRARY}")
            message("Set CUDA_culibos_LIBRARY to '${CUDA_culibos_LIBRARY}'.")
        else ()
            message(FATAL_ERROR "Could not find CUDA::culibos.")
        endif()
    endif ()

endif ()
