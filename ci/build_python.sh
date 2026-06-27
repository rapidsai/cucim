#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-generate-version > ./VERSION

RAPIDS_PACKAGE_VERSION=$(rapids-generate-version)
export RAPIDS_PACKAGE_VERSION

rapids-logger "Begin py build"

sccache --stop-server 2>/dev/null || true

export SCCACHE_S3_KEY_PREFIX="cucim/${RAPIDS_CONDA_ARCH}/cuda${RAPIDS_CUDA_VERSION%%.*}/object-cache"
export SCCACHE_S3_PREPROCESSOR_CACHE_KEY_PREFIX="cucim/${RAPIDS_CONDA_ARCH}/cuda${RAPIDS_CUDA_VERSION%%.*}/preprocessor-cache"
export SCCACHE_S3_USE_PREPROCESSOR_CACHE_MODE="true"

sccache --start-server

# this can be set back to 'prevent' once the xorg-* migrations are completed
# ref: https://github.com/rapidsai/cucim/issues/800#issuecomment-2529593457
conda config --set path_conflict warn

rapids-logger "Building cucim"

# TODO: remove `--test skip` when importing on a CPU node works correctly
# --no-build-id allows for caching with `sccache`
# more info is available at
# https://rattler.build/latest/tips_and_tricks/#using-sccache-or-ccache-with-rattler-build
rattler-build build --recipe conda/recipes/cucim \
                    --experimental \
                    --test skip \
                    "${RATTLER_ARGS[@]}" \
                    "${RATTLER_CHANNELS[@]}"

sccache --show-adv-stats

RAPIDS_PACKAGE_NAME="$(rapids-artifact-name conda_python cucim cucim --py "$RAPIDS_PY_VERSION" --cuda "$RAPIDS_CUDA_VERSION")"
export RAPIDS_PACKAGE_NAME
