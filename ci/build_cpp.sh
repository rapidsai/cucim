#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin cpp build"

# this can be set back to 'prevent' once the xorg-* migrations are completed
# ref: https://github.com/rapidsai/cucim/issues/800#issuecomment-2529593457
conda config --set path_conflict warn

sccache --zero-stats

RAPIDS_PACKAGE_VERSION=$(rapids-generate-version) rapids-conda-retry mambabuild conda/recipes/libcucim

sccache --show-adv-stats

rapids-upload-conda-to-s3 cpp
