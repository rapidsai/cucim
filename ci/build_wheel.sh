#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

package_name="cucim"
package_dir="python/cucim"

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

python -m pip install --upgrade pip

#CMake version in the container is too old, install a new version in the python venv
python -m pip install "cmake>=3.26.4" ninja

# apply patch to omit building tests and benchmark binaries so libopenslide isn't required
git apply ci/cmake-omit-benchmarks-examples-tests.patch

# First build the C++ lib using CMake via the run script
./run build_local libcucim release

# problems: boost-header-only takes a long time to download
# Fails to build any files requiring libopenslide as it isn't on the system

# Compile the Python bindings
./run build_local cucim release

# Copy the resulting cucim pybind11 shared library into the Python package src folder
cp -P python/install/lib/* python/cucim/src/cucim/clara/

cd "${package_dir}"

python -m pip wheel . -w dist -vvv --no-deps --disable-pip-version-check

mkdir -p final_dist
python -m auditwheel repair -w final_dist dist/*

RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 final_dist
