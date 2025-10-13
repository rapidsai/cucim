#
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#

################################################################################
# Add executable: tiff_image
################################################################################

add_executable(tiff_image tiff_image/main.cpp)

set_target_properties(tiff_image
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)
target_compile_features(tiff_image PRIVATE ${CUCIM_REQUIRED_FEATURES})
# Use generator expression to avoid `nvcc fatal   : Value '-std=c++17' is not defined for option 'Werror'`
target_compile_options(tiff_image PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror -Wall -Wextra>)
target_link_libraries(tiff_image
        PRIVATE
            ${CUCIM_PACKAGE_NAME}
            deps::fmt
        )

################################################################################
# Add executable: cuslide2_nvimagecodec_test
################################################################################

add_executable(cuslide2_nvimagecodec_test cuslide2_nvimagecodec_test/main.cpp)

set_target_properties(cuslide2_nvimagecodec_test
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)
target_compile_features(cuslide2_nvimagecodec_test PRIVATE ${CUCIM_REQUIRED_FEATURES})
# Use generator expression to avoid `nvcc fatal   : Value '-std=c++17' is not defined for option 'Werror'`
target_compile_options(cuslide2_nvimagecodec_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror -Wall -Wextra>)
target_link_libraries(cuslide2_nvimagecodec_test
        PRIVATE
            ${CUCIM_PACKAGE_NAME}
            deps::fmt
        )
