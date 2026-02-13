#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -eou pipefail

source rapids-init-pip

PYTHON_WHEELHOUSE=$(rapids-download-from-github "$(rapids-package-name wheel_python cucim --stable --cuda "$RAPIDS_CUDA_VERSION")")

# echo to expand wildcard before adding `[extra]` requires for pip
rapids-pip-retry install "$(echo "${PYTHON_WHEELHOUSE}"/cucim*.whl)[test]"

if type -f yum > /dev/null 2>&1; then
    yum update -y
    yum install -y openslide
else
    DEBIAN_FRONTEND=noninteractive apt update
    DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends libopenslide0
fi

python -m pytest \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cucim.xml" \
  --numprocesses=8 \
  --dist=worksteal \
  ./python/cucim
