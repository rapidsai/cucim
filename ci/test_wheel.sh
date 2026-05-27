#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -eou pipefail

source rapids-init-pip

PYTHON_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python cucim cucim --py "$RAPIDS_PY_VERSION" --cuda "$RAPIDS_CUDA_VERSION")")

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints test_python "${PIP_CONSTRAINT}"

# notes:
#
#   * echo to expand wildcard before adding `[test]` requires for pip
#   * just providing --constraint="${PIP_CONSTRAINT}" to be explicit, and because
#     that environment variable is ignored if any other --constraint are passed via the CLI
#
rapids-pip-retry install \
    --prefer-binary \
    --constraint "${PIP_CONSTRAINT}" \
    "$(echo ${PYTHON_WHEELHOUSE}/cucim*.whl)[test]"

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
