#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

package_name="cucim"
package_dir="python/cucim"

CMAKE_BUILD_TYPE="release"

source rapids-configure-sccache
source rapids-date-string

# Use gha-tools rapids-pip-wheel-version to generate wheel version then
# update the necessary files
version_override="$(rapids-pip-wheel-version ${RAPIDS_DATE_STRING})"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# Patch project metadata files to include the CUDA version suffix and version override.
pyproject_file="${package_dir}/pyproject.toml"

# pyproject.toml versions
sed -i "s/^version = .*/version = \"${VERSION}\"/g" ${pyproject_file}

CUDA_SUFFIX="-${RAPIDS_PY_CUDA_SUFFIX}"
# update package name to have the cuda suffix
sed -i "s/name = \"${package_name}\"/name = \"${package_name}${CUDA_SUFFIX}\"/g" ${pyproject_file}

if [[ $CUDA_SUFFIX == "-cu12" ]]; then
    # change pyproject.toml to use CUDA 12.x version of cupy
    sed -i "s/cupy-cuda11x/cupy-cuda12x/g" ${pyproject_file}
fi

echo "The package name and/or version was modified in the package source. The git diff is:"
git diff

pip install --upgrade pip

#CMake version in the container is too old, install a new version in the python venv
pip install "cmake>=3.26.4" ninja

# Building the libjpeg-turbo dependency requires YASM
# Also need to install openslide dev libraries on the system
if ! command -v apt &> /dev/null
then
    echo "apt package manager not found, attempting to use yum"
    yum install yasm openslide-devel -y
else
    echo "apt package manager was found"
    apt update
    apt install yasm libopenslide-dev -y
fi

# First build the C++ lib using CMake via the run script
./run build_local libcucim ${CMAKE_BUILD_TYPE}

# Current `pip wheel` build assumes all shared libraries exist in this path
PYTHON_CLARA_SO_PREFIX_PATH="./python/cucim/src/cucim/clara/"

# Build the C++ cuslide and cumed plugins
# (omit copying binaries as they don't go in the wheel)
./run build_local cuslide ${CMAKE_BUILD_TYPE}
cp -P -r cpp/plugins/cucim.kit.cuslide/install/lib/* ${PYTHON_CLARA_SO_PREFIX_PATH}
./run build_local cumed ${CMAKE_BUILD_TYPE}
cp -P -r cpp/plugins/cucim.kit.cumed/install/lib/* ${PYTHON_CLARA_SO_PREFIX_PATH}

# Compile the Python bindings
./run build_local cucim ${CMAKE_BUILD_TYPE}
cp -P install/lib/*.so ${PYTHON_CLARA_SO_PREFIX_PATH}

cd "${package_dir}"

python -m pip wheel . -w dist -vvv --no-deps --disable-pip-version-check

mkdir -p final_dist
python -m auditwheel repair -w final_dist dist/*

RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 final_dist
