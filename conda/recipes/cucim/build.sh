#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e -u -o pipefail

LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}

echo "CC          : ${CC}"
echo "CXX         : ${CXX}"

# CUDA needs to include $PREFIX/include as system include path
export CUDAFLAGS="-isystem $BUILD_PREFIX/include -isystem $PREFIX/include "
export LD_LIBRARY_PATH="$BUILD_PREFIX/lib:$PREFIX/lib:$LD_LIBRARY_PATH"

pushd python/cucim

echo "PYTHON: ${PYTHON}"
$PYTHON -m pip install \
  --config-settings rapidsai.disable-cuda=true \
  --config-settings skbuild.wheel.py-api=cp311 \
  . -vv

popd
