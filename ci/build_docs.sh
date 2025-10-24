#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-github cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-github python)

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-dependency-file-generator \
    --output conda \
    --file-key docs \
    --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
    --prepend-channel "${CPP_CHANNEL}" \
    --prepend-channel "${PYTHON_CHANNEL}" \
    | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n docs
conda activate docs

rapids-print-env

RAPIDS_DOCS_DIR="$(mktemp -d)"
export RAPIDS_DOCS_DIR

rapids-logger "Build Python docs"
pushd docs
sphinx-build -b dirhtml ./source _html
mkdir -p "${RAPIDS_DOCS_DIR}/cucim/"html
mv _html/* "${RAPIDS_DOCS_DIR}/cucim/html"
popd

RAPIDS_VERSION_NUMBER="$(rapids-version-major-minor)" rapids-upload-docs
