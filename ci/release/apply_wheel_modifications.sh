#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Usage: bash apply_wheel_modifications.sh <new_version> <cuda_suffix>

VERSION=${1}
CUDA_SUFFIX=${2}

package_name="cucim"
package_dir="python/cucim"

# Patch project metadata files to include the CUDA version suffix and version override.
pyproject_file="${package_dir}/pyproject.toml"

# pyproject.toml versions
sed -i "s/^version = .*/version = \"${VERSION}\"/g" ${pyproject_file}

# update package name to have the cuda suffix
sed -i "s/name = \"${package_name}\"/name = \"${package_name}${CUDA_SUFFIX}\"/g" ${pyproject_file}

if [[ $CUDA_SUFFIX == "-cu12" ]]; then
    # change pyproject.toml to use CUDA 12.x version of cupy
    sed -i "s/cupy-cuda11x/cupy-cuda12x/g" ${pyproject_file}
fi
