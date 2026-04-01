#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

./ci/rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-generate-version > ./VERSION

rapids-logger "Begin py build"

sccache --stop-server 2>/dev/null || true

export SCCACHE_S3_KEY_PREFIX="cucim/${RAPIDS_CONDA_ARCH}/cuda${RAPIDS_CUDA_VERSION%%.*}/object-cache"
export SCCACHE_S3_PREPROCESSOR_CACHE_KEY_PREFIX="cucim/${RAPIDS_CONDA_ARCH}/cuda${RAPIDS_CUDA_VERSION%%.*}/preprocessor-cache"
export SCCACHE_S3_USE_PREPROCESSOR_CACHE_MODE="true"

sccache --start-server

# this can be set back to 'prevent' once the xorg-* migrations are completed
# ref: https://github.com/rapidsai/cucim/issues/800#issuecomment-2529593457
conda config --set path_conflict warn

CPP_CHANNEL=$(rapids-download-conda-from-github cpp)

# TODO: Remove `--no-test` flag once importing on a CPU
# node works correctly
RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION) rapids-conda-retry build \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  conda/recipes/cucim

sccache --show-adv-stats
