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

# This is the version of the suffix with a preceding hyphen. It's used
# everywhere except in the final wheel name.
PACKAGE_CUDA_SUFFIX="-${RAPIDS_PY_CUDA_SUFFIX}"

# Patch project metadata files to include the CUDA version suffix and version override.
pyproject_file="${package_dir}/pyproject.toml"

sed -i "s/^version = .*/version = \"${version_override}\"/g" ${pyproject_file}
sed -i "s/name = \"${package_name}\"/name = \"${package_name}${PACKAGE_CUDA_SUFFIX}\"/g" ${pyproject_file}

pip install --upgrade pip

#CMake version in the container is too old, install a new version in the python venv
pip install "cmake>=3.26.4" ninja

echo `which cmake`

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
    dpkg -L libopenslide-dev

    git apply ci/omit-cuslide-benchmarks-and-tests.patch
fi

# First build the C++ lib using CMake via the run script
./run build_local libcucim ${CMAKE_BUILD_TYPE}

# Build the C++ cuslide and cumed plugins
./run build_local cuslide ${CMAKE_BUILD_TYPE}
cp -P -r cpp/plugins/cucim.kit.cuslide/install/lib/* ./install/lib/
# omit copying potentially non-existent binaries due to patch above.
# they don't go in the wheel anyways
# cp -P -r cpp/plugins/cucim.kit.cuslide/install/bin/* ./install/bin/

./run build_local cumed ${CMAKE_BUILD_TYPE}
cp -P -r cpp/plugins/cucim.kit.cumed/install/lib/* ./install/lib/
# cp -P -r cpp/plugins/cucim.kit.cumed/install/bin/* ./install/bin/

# problems: boost-header-only takes a long time to download

# Compile the Python bindings
./run build_local cucim ${CMAKE_BUILD_TYPE}

# Copy the resulting cucim pybind11 shared library into the Python package src folder
cp -P python/install/lib/* python/cucim/src/cucim/clara/
# also need these files in the clara wheel
cp -P install/lib/*.so python/cucim/src/cucim/clara/

cd "${package_dir}"

python -m pip wheel . -w dist -vvv --no-deps --disable-pip-version-check

mkdir -p final_dist
python -m auditwheel repair -w final_dist dist/*

RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 final_dist
