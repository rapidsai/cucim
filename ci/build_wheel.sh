#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

package_name="cucim"
package_dir="python/cucim"

CMAKE_BUILD_TYPE="release"

source rapids-configure-sccache
source rapids-date-string

rapids-generate-version > ./VERSION

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

PACKAGE_CUDA_SUFFIX="-${RAPIDS_PY_CUDA_SUFFIX}"

# Install pip build dependencies (not yet using pyproject.toml)
rapids-dependency-file-generator \
  --file-key "py_build" \
  --output "requirements" \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee build_requirements.txt

python -m pip install -r build_requirements.txt

sccache --zero-stats

# First build the C++ lib using CMake via the run script
./run build_local all ${CMAKE_BUILD_TYPE}

sccache --show-adv-stats

cd "${package_dir}"

sccache --zero-stats

rapids-logger "Building '${package_name}' wheel"
python -m pip wheel \
    -w dist \
    -v \
    --no-deps \
    --disable-pip-version-check \
    .

sccache --show-adv-stats

mkdir -p final_dist
python -m auditwheel repair -w final_dist dist/*
ls -1 final_dist | grep -vqz 'none'

RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 final_dist
