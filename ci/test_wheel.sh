#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -eou pipefail

source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
PYTHON_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cucim_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)

# echo to expand wildcard before adding `[extra]` requires for pip
rapids-pip-retry install "$(echo ${PYTHON_WHEELHOUSE}/cucim*.whl)[test]"

CUDA_MAJOR_VERSION=${RAPIDS_CUDA_VERSION:0:2}

if type -f yum > /dev/null 2>&1; then
    yum update -y
    yum install -y openslide
else
    DEBIAN_FRONTEND=noninteractive apt update
    DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends libopenslide0
fi

if [[ ${CUDA_MAJOR_VERSION} == "11" ]]; then
    # Omit I/O-related tests in ./python/cucim/tests due to known CUDA bug
    # with dynamic loading of libcufile.
    python -m pytest \
      --junitxml="${RAPIDS_TESTS_DIR}/junit-cucim.xml" \
      --numprocesses=8 \
      --dist=worksteal \
      ./python/cucim/src/
else
    python -m pytest \
      --junitxml="${RAPIDS_TESTS_DIR}/junit-cucim.xml" \
      --numprocesses=8 \
      --dist=worksteal \
      ./python/cucim
fi
