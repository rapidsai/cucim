#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.

set -euo pipefail

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin cpp build"

sccache --zero-stats

RAPIDS_PACKAGE_VERSION=$(rapids-generate-version)
export RAPIDS_PACKAGE_VERSION

# populates `RATTLER_CHANNELS` array and `RATTLER_ARGS` array
source rapids-rattler-channel-string

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

# --no-build-id allows for caching with `sccache`
# more info is available at
# https://rattler.build/latest/tips_and_tricks/#using-sccache-or-ccache-with-rattler-build
rattler-build build --recipe conda/recipes/libcucim \
                    --variant-config extra_variants.yaml \
                    "${RATTLER_ARGS[@]}" \
                    "${RATTLER_CHANNELS[@]}"

sccache --show-adv-stats

# remove build_cache directory to avoid uploading the entire source tree
# tracked in https://github.com/prefix-dev/rattler-build/issues/1424
rm -rf "$RAPIDS_CONDA_BLD_OUTPUT_DIR"/build_cache
