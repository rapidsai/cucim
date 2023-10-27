#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -eou pipefail

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="cucim_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install $(echo ./dist/cucim*.whl)[test]

# Run smoke tests for aarch64 pull requests
if [ "$(arch)" == "aarch64" && ${RAPIDS_BUILD_TYPE} == "pull-request" ]; then
    python ./ci/wheel_smoke_test.py
else
    # Note: imagecodecs is currently not installed on aarch64, so a handful of
    #       test cases will be skipped in that case
    python -m pytest ./python/cucim
fi
