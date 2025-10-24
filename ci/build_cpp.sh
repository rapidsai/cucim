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

# this can be set back to 'prevent' once the xorg-* migrations are completed
# ref: https://github.com/rapidsai/cucim/issues/800#issuecomment-2529593457
conda config --set path_conflict warn

sccache --zero-stats


# TODO: Remove when CUDA 12.1 is dropped.
# In most cases, the CTK has cuFile.
# However the CTK only added cuFile for ARM in 12.2.
# So for ARM users on CTK 12.0 & 12.1, relax the cuFile requirement.
# On x86_64 or CTK 13 or ARM with CTK 12.2+, always require cuFile.
cat > extra_variants.yaml <<EOF
has_cufile:
  - True
EOF
if [[ "$(arch)" == "aarch64" ]] && [[ "${RAPIDS_CUDA_VERSION%%.*}" == "12" ]]
then
  cat >> extra_variants.yaml <<EOF
  - False
EOF
fi

echo 'Contents of `extra_variants.yaml`:'
cat extra_variants.yaml
echo ''

RAPIDS_PACKAGE_VERSION=$(rapids-generate-version) rapids-conda-retry build \
    -m extra_variants.yaml \
    conda/recipes/libcucim

sccache --show-adv-stats
