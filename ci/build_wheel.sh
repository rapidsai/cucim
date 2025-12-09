#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

package_name="cucim"
package_dir="python/cucim"

CMAKE_BUILD_TYPE="release"

source rapids-configure-sccache
source rapids-date-string
source rapids-init-pip

rapids-generate-version > ./VERSION

rapids-logger "Generating build requirements"

rapids-dependency-file-generator \
  --output requirements \
  --file-key "py_build_${package_name}" \
  --file-key "py_rapids_build_${package_name}" \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
| tee /tmp/requirements-build.txt

rapids-logger "Installing build requirements"
rapids-pip-retry install \
    -v \
    --prefer-binary \
    -r /tmp/requirements-build.txt

sccache --zero-stats

rapids-logger "pyenv rehash"
pyenv rehash

# First build the C++ lib using CMake via the run script
./run build_local all ${CMAKE_BUILD_TYPE}

sccache --show-adv-stats

cd "${package_dir}"

sccache --zero-stats

rapids-logger "Building '${package_name}' wheel"
rapids-pip-retry wheel \
    -w dist \
    -v \
    --no-build-isolation \
    --no-deps \
    --disable-pip-version-check \
    .

sccache --show-adv-stats

# repair wheels and write to the location that artifact-uploading code expects to find them
python -m auditwheel repair -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" dist/*
# shellcheck disable=SC2010
ls -1 "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" | grep -vqz 'none'

../../ci/validate_wheel.sh "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
