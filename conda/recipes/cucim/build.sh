#!/bin/bash
# Copyright (c) 2021-2025, NVIDIA CORPORATION.

set -e -u -o pipefail

CUCIM_BUILD_TYPE=${CUCIM_BUILD_TYPE:-release}

echo "CC          : ${CC}"
echo "CXX         : ${CXX}"

# CUDA needs to include $PREFIX/include as system include path
export CUDAFLAGS="-isystem $BUILD_PREFIX/include -isystem $PREFIX/include "
export LD_LIBRARY_PATH="$BUILD_PREFIX/lib:$PREFIX/lib:$LD_LIBRARY_PATH"

# It is assumed that this script is executed from the root of the repo directory by conda-build
# (https://conda-forge.org/docs/maintainer/knowledge_base.html#using-cmake)
./run build_local cucim "${CUCIM_BUILD_TYPE}"

cp -P python/install/lib/* python/cucim/src/cucim/clara/

pushd python/cucim

echo "PYTHON: ${PYTHON}"
$PYTHON -m pip install --config-settings rapidsai.disable-cuda=true . -vv

popd
