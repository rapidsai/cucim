#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

package_name="cucim"
package_dir="python/cucim"
package_src_dir="${package_dir}/src/${package_name}"

CMAKE_BUILD_TYPE="release"

source rapids-configure-sccache
source rapids-date-string

version=$(rapids-generate-version)
commit=$(git rev-parse HEAD)

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# Patch project metadata files to include the CUDA version suffix and version override.
pyproject_file="${package_dir}/pyproject.toml"

PACKAGE_CUDA_SUFFIX="-${RAPIDS_PY_CUDA_SUFFIX}"
# update package name to have the cuda suffix
sed -i "s/name = \"${package_name}\"/name = \"${package_name}${PACKAGE_CUDA_SUFFIX}\"/g" ${pyproject_file}
echo "${version}" > VERSION
sed -i "/^__git_commit__/ s/= .*/= \"${commit}\"/g" "${package_src_dir}/_version.py"

if [[ ${PACKAGE_CUDA_SUFFIX} == "-cu12" ]]; then
    # change pyproject.toml to use CUDA 12.x version of cupy
    sed -i "s/cupy-cuda11x/cupy-cuda12x/g" ${pyproject_file}
fi

# Install pip build dependencies (not yet using pyproject.toml)
rapids-dependency-file-generator \
  --file_key "py_build" \
  --output "requirements" \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee build_requirements.txt
pip install -r build_requirements.txt

# First build the C++ lib using CMake via the run script
./run build_local all ${CMAKE_BUILD_TYPE}

cd "${package_dir}"

python -m pip wheel . -w dist -vvv --no-deps --disable-pip-version-check

mkdir -p final_dist
python -m auditwheel repair -w final_dist dist/*
ls -1 final_dist | grep -vqz 'none'

RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 final_dist
