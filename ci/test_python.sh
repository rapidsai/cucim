#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

# Common setup steps shared by Python test jobs

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Generate Python testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file_key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"

rapids-print-env

rapids-mamba-retry install \
  "file://${CPP_CHANNEL}::libcucim" \
  "file://${PYTHON_CHANNEL}::cucim"

rapids-logger "Check GPU usage"
nvidia-smi

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest cucim"
pushd python/cucim
export CUPY_DUMP_CUDA_SOURCE_ON_ERROR=1
timeout 20m pytest \
  --cache-clear \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cucim.xml" \
  --numprocesses=8 \
  --dist=loadscope \
  --cov-config=.coveragerc \
  --cov=cucim \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cucim-coverage.xml" \
  --cov-report=term \
  --maxfail=5 \
  -v \
  src \
  tests/unit \
  tests/performance
popd

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
