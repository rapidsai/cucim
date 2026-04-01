#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

./ci/rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin cpp build"

sccache --stop-server 2>/dev/null || true

export SCCACHE_S3_KEY_PREFIX="libcucim/${RAPIDS_CONDA_ARCH}/cuda${RAPIDS_CUDA_VERSION%%.*}/object-cache"
export SCCACHE_S3_PREPROCESSOR_CACHE_KEY_PREFIX="libcucim/${RAPIDS_CONDA_ARCH}/cuda${RAPIDS_CUDA_VERSION%%.*}/preprocessor-cache"
export SCCACHE_S3_USE_PREPROCESSOR_CACHE_MODE="true"

sccache --start-server

# this can be set back to 'prevent' once the xorg-* migrations are completed
# ref: https://github.com/rapidsai/cucim/issues/800#issuecomment-2529593457
conda config --set path_conflict warn

RAPIDS_PACKAGE_VERSION=$(rapids-generate-version) rapids-conda-retry build \
    conda/recipes/libcucim

sccache --show-adv-stats
