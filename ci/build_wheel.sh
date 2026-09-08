#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

package_name="cucim"
package_dir="python/cucim"

CMAKE_BUILD_TYPE="release"
RAPIDS_PY_API="${RAPIDS_PY_API:-cp311}"
export RAPIDS_PY_API

source rapids-configure-sccache
source rapids-datetime-string
source rapids-init-pip

sccache --stop-server 2>/dev/null || true

export SCCACHE_S3_KEY_PREFIX="${package_name}/${RAPIDS_CONDA_ARCH}/cuda${RAPIDS_CUDA_VERSION%%.*}/object-cache"
export SCCACHE_S3_PREPROCESSOR_CACHE_KEY_PREFIX="${package_name}/${RAPIDS_CONDA_ARCH}/cuda${RAPIDS_CUDA_VERSION%%.*}/preprocessor-cache"
export SCCACHE_S3_USE_PREPROCESSOR_CACHE_MODE="true"

sccache --start-server

RAPIDS_VERSION_SUFFIX=".post${RAPIDS_DATETIME_STRING}" \
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

# Build the native libraries and plugins before scikit-build-core builds the
# ABI3 extension. Building the extension here as well would leave a
# CPython-versioned shared object in the wheel package tree.
./run build_local native ${CMAKE_BUILD_TYPE}

sccache --show-adv-stats

cd "${package_dir}"

sccache --zero-stats

rapids-logger "Building '${package_name}' wheel"
RAPIDS_PIP_WHEEL_ARGS=(
  -w dist
  -v
  --no-build-isolation
  --no-deps
  --disable-pip-version-check
)

if [[ -n "${RAPIDS_PY_API:-}" ]]; then
  RAPIDS_PIP_WHEEL_ARGS+=(--config-settings="skbuild.wheel.py-api=${RAPIDS_PY_API}")
fi

rapids-pip-retry wheel \
    "${RAPIDS_PIP_WHEEL_ARGS[@]}" \
    .

sccache --show-adv-stats

# repair wheels and write to the location that artifact-uploading code expects to find them
python -m auditwheel repair -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" dist/*
# shellcheck disable=SC2010
ls -1 "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" | grep -vqz 'none'

../../ci/validate_wheel.sh "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

RAPIDS_PACKAGE_NAME="$(rapids-artifact-name wheel_python cucim cucim --stable --cuda "$RAPIDS_CUDA_VERSION")"
export RAPIDS_PACKAGE_NAME
